// Opt-in acceleration kernels (resectionlab._accel_cpp).
//
// Separate from _geometry_cpp so the qualified simulator binary, its build
// manifest and its live integrity checks are untouched. Each kernel mirrors a
// Python reference function element by element; callers keep that reference
// and use it whenever an input is outside what these kernels accept.
// Build flags match the main extension: -O3 -fno-fast-math -ffp-contract=off.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using Index = std::int64_t;
using Bool3 = py::array_t<bool, py::array::c_style>;

struct Grid {
    Index nx, ny, nz;
    Index flat(Index i, Index j, Index k) const { return (i * ny + j) * nz + k; }
    bool inside(Index i, Index j, Index k) const {
        return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz;
    }
};

constexpr int kNeighbors[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

Grid grid_of(const py::array& array) {
    if (array.ndim() != 3) throw std::invalid_argument("expected a 3-D mask");
    return Grid{array.shape(0), array.shape(1), array.shape(2)};
}

std::vector<std::array<Index, 3>> cells_of(const py::array_t<Index, py::array::c_style>& cells, const Grid& grid) {
    if (cells.ndim() != 2 || cells.shape(1) != 3) throw std::invalid_argument("expected an (n, 3) index array");
    auto view = cells.unchecked<2>();
    std::vector<std::array<Index, 3>> out(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t r = 0; r < view.shape(0); ++r) {
        out[r] = {view(r, 0), view(r, 1), view(r, 2)};
        if (!grid.inside(out[r][0], out[r][1], out[r][2])) throw py::index_error("cell outside the grid");
    }
    return out;
}

// native_resection._connected_surface_cells: candidate components reachable
// from a candidate with a face on existing connected free space (or the grid
// boundary). Returns the accepted cells sorted lexicographically.
py::array_t<Index> connected_surface_cells(const py::array_t<Index, py::array::c_style>& candidates,
                                           const Bool3& connected_free) {
    const Grid grid = grid_of(connected_free);
    const auto cells = cells_of(candidates, grid);
    const bool* free = connected_free.data();
    std::vector<Index> pending;
    pending.reserve(cells.size());
    for (const auto& c : cells) pending.push_back(grid.flat(c[0], c[1], c[2]));
    std::sort(pending.begin(), pending.end());
    pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
    auto in_pending = [&](Index key) { return std::binary_search(pending.begin(), pending.end(), key); };
    std::vector<char> accepted(pending.size(), 0);
    auto slot = [&](Index key) {
        return static_cast<std::size_t>(std::lower_bound(pending.begin(), pending.end(), key) - pending.begin());
    };
    std::vector<Index> queue;
    for (Index key : pending) {
        const Index i = key / (grid.ny * grid.nz), j = (key / grid.nz) % grid.ny, k = key % grid.nz;
        bool seed = false;
        for (const auto& d : kNeighbors) {
            const Index a = i + d[0], b = j + d[1], c = k + d[2];
            if (!grid.inside(a, b, c) || free[grid.flat(a, b, c)]) { seed = true; break; }
        }
        if (seed) queue.push_back(key);
    }
    while (!queue.empty()) {
        const Index key = queue.back();
        queue.pop_back();
        const std::size_t s = slot(key);
        if (accepted[s]) continue;
        accepted[s] = 1;
        const Index i = key / (grid.ny * grid.nz), j = (key / grid.nz) % grid.ny, k = key % grid.nz;
        for (const auto& d : kNeighbors) {
            const Index a = i + d[0], b = j + d[1], c = k + d[2];
            if (!grid.inside(a, b, c)) continue;
            const Index other = grid.flat(a, b, c);
            if (in_pending(other) && !accepted[slot(other)]) queue.push_back(other);
        }
    }
    std::vector<Index> keep;
    for (std::size_t s = 0; s < pending.size(); ++s)
        if (accepted[s]) keep.push_back(pending[s]);  // flat order == lexicographic order
    py::array_t<Index> result({static_cast<py::ssize_t>(keep.size()), static_cast<py::ssize_t>(3)});
    auto out = result.mutable_unchecked<2>();
    for (std::size_t r = 0; r < keep.size(); ++r) {
        const Index key = keep[r];
        out(r, 0) = key / (grid.ny * grid.nz);
        out(r, 1) = (key / grid.nz) % grid.ny;
        out(r, 2) = key % grid.nz;
    }
    return result;
}

// native_resection._extend_connected_free: same LIFO traversal and mutation
// order, including the partial update before an unknown-domain refusal.
void extend_connected_free(const py::array_t<Index, py::array::c_style>& new_cells, const Bool3& remaining,
                           py::array_t<bool, py::array::c_style> connected_free, const py::object& domain_object) {
    const Grid grid = grid_of(connected_free);
    const Grid other = grid_of(remaining);
    if (other.nx != grid.nx || other.ny != grid.ny || other.nz != grid.nz) throw std::invalid_argument("mask shapes differ");
    const bool* domain = nullptr;
    Bool3 domain_array;
    if (!domain_object.is_none()) {
        domain_array = domain_object.cast<Bool3>();
        const Grid d = grid_of(domain_array);
        if (d.nx != grid.nx || d.ny != grid.ny || d.nz != grid.nz) throw std::invalid_argument("mask shapes differ");
        domain = domain_array.data();
    }
    const bool* rest = remaining.data();
    bool* free = connected_free.mutable_data();
    const auto cells = cells_of(new_cells, grid);
    std::vector<Index> queue;
    queue.reserve(cells.size());
    for (const auto& c : cells) queue.push_back(grid.flat(c[0], c[1], c[2]));
    while (!queue.empty()) {
        const Index key = queue.back();
        queue.pop_back();
        if (domain != nullptr && !domain[key]) throw std::invalid_argument("Unknown domain cannot become connected free space");
        if (free[key]) continue;
        free[key] = true;
        const Index i = key / (grid.ny * grid.nz), j = (key / grid.nz) % grid.ny, k = key % grid.nz;
        for (const auto& d : kNeighbors) {
            const Index a = i + d[0], b = j + d[1], c = k + d[2];
            if (!grid.inside(a, b, c)) continue;
            const Index n = grid.flat(a, b, c);
            if (!rest[n] && !free[n] && (domain == nullptr || domain[n])) queue.push_back(n);
        }
    }
}

// --- surface_fairing._intersection_vertices ------------------------------

using Vec = std::array<double, 3>;
using Tri = std::array<Vec, 3>;

inline Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
// numpy.cross component order for 3-vectors.
inline Vec cross(const Vec& a, const Vec& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
// numpy.einsum('nki,ni->nk') reduces its three-term inner loop with SIMD lanes,
// so its summation order is a property of the NumPy build, not of the formula.
// The Python wrapper measures which of these orders the running NumPy uses and
// disables this kernel when none of them matches.
enum ProjectionOrder : int { kLeftFold = 0, kLanePair = 1, kLanePairFused = 2 };
inline double projection(const Vec& a, const Vec& b, int order) {
    switch (order) {
        case kLeftFold: return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
        case kLanePair: return (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1];
        case kLanePairFused: return std::fma(a[2], b[2], a[0] * b[0]) + a[1] * b[1];
        default: throw std::invalid_argument("unknown projection order");
    }
}
inline double norm(const Vec& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

// surface_fairing.intersecting for one pair (vectorized SAT, coplanar axes).
bool intersecting(Tri a, Tri b, int order) {
    const Vec origin = a[0];
    for (auto& p : a) p = sub(p, origin);
    for (auto& p : b) p = sub(p, origin);
    Tri ae, be;
    for (int k = 0; k < 3; ++k) { ae[k] = sub(a[(k + 1) % 3], a[k]); be[k] = sub(b[(k + 1) % 3], b[k]); }
    const Vec an = cross(ae[0], ae[1]), bn = cross(be[0], be[1]);
    std::array<Vec, 17> axes;
    int n = 0;
    axes[n++] = an; axes[n++] = bn;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) axes[n++] = cross(ae[i], be[j]);
    for (int i = 0; i < 3; ++i) axes[n++] = cross(an, ae[i]);
    for (int i = 0; i < 3; ++i) axes[n++] = cross(bn, be[i]);
    for (const auto& axis : axes) {
        double pa[3], pb[3];
        for (int k = 0; k < 3; ++k) { pa[k] = projection(a[k], axis, order); pb[k] = projection(b[k], axis, order); }
        const double length = norm(axis);
        const double amax = std::max({pa[0], pa[1], pa[2]}), amin = std::min({pa[0], pa[1], pa[2]});
        const double bmax = std::max({pb[0], pb[1], pb[2]}), bmin = std::min({pb[0], pb[1], pb[2]});
        const bool separated = (amax < bmin - 1e-9 * length) || (bmax < amin - 1e-9 * length);
        if (separated && length >= 1e-15) return false;
    }
    return true;
}

py::array_t<bool> intersecting_batch(const py::array_t<double, py::array::c_style>& first,
                                     const py::array_t<double, py::array::c_style>& second, int order) {
    if (first.ndim() != 3 || first.shape(1) != 3 || first.shape(2) != 3 || second.ndim() != 3 ||
        second.shape(0) != first.shape(0) || second.shape(1) != 3 || second.shape(2) != 3)
        throw std::invalid_argument("expected two (n, 3, 3) triangle arrays");
    auto a = first.unchecked<3>(); auto b = second.unchecked<3>();
    py::array_t<bool> result(first.shape(0));
    auto out = result.mutable_unchecked<1>();
    for (py::ssize_t r = 0; r < a.shape(0); ++r) {
        Tri ta, tb;
        for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) { ta[k][i] = a(r, k, i); tb[k][i] = b(r, k, i); }
        out(r) = intersecting(ta, tb, order);
    }
    return result;
}

// Self-check helpers: the kernel's projections and lengths, for comparison
// with numpy.einsum and numpy.linalg.norm on the running NumPy.
py::array_t<double> projections(const py::array_t<double, py::array::c_style>& points,
                                const py::array_t<double, py::array::c_style>& axes, int order) {
    if (points.ndim() != 3 || points.shape(2) != 3 || axes.ndim() != 2 || axes.shape(1) != 3 ||
        axes.shape(0) != points.shape(0))
        throw std::invalid_argument("expected (n, k, 3) points and (n, 3) axes");
    auto p = points.unchecked<3>(); auto x = axes.unchecked<2>();
    py::array_t<double> result({points.shape(0), points.shape(1)});
    auto out = result.mutable_unchecked<2>();
    for (py::ssize_t r = 0; r < p.shape(0); ++r) {
        const Vec axis{x(r, 0), x(r, 1), x(r, 2)};
        for (py::ssize_t k = 0; k < p.shape(1); ++k) out(r, k) = projection({p(r, k, 0), p(r, k, 1), p(r, k, 2)}, axis, order);
    }
    return result;
}

py::array_t<double> norms(const py::array_t<double, py::array::c_style>& vectors) {
    if (vectors.ndim() != 2 || vectors.shape(1) != 3) throw std::invalid_argument("expected (n, 3) vectors");
    auto v = vectors.unchecked<2>();
    py::array_t<double> result(vectors.shape(0));
    auto out = result.mutable_unchecked<1>();
    for (py::ssize_t r = 0; r < v.shape(0); ++r) out(r) = norm({v(r, 0), v(r, 1), v(r, 2)});
    return result;
}

Tri triangle(const double* points, const Index* face) {
    Tri t;
    for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) t[k][i] = points[3 * face[k] + i];
    return t;
}

// Pairs, filters and order of _intersection_vertices after its moved/dirty
// masks. A uniform grid replaces the k-d tree as the candidate generator and
// the same centre-distance bound is applied. The k-d tree rounds its own node
// bounds, so a pair whose squared distance lies within kBoundBand of the bound
// is "ambiguous"; if any ambiguous pair would add a vertex, the second return
// value is true and the caller reruns the reference audit instead.
constexpr double kBoundBand = 1e-9;

py::tuple intersection_vertices(const py::array_t<double, py::array::c_style>& source,
                                const py::array_t<double, py::array::c_style>& candidate,
                                const py::array_t<Index, py::array::c_style>& faces,
                                const py::array_t<bool, py::array::c_style>& moved,
                                const py::array_t<bool, py::array::c_style>& dirty,
                                const py::object& cancelled, int order) {
    if (source.ndim() != 2 || source.shape(1) != 3 || candidate.ndim() != 2 || candidate.shape(1) != 3 ||
        candidate.shape(0) != source.shape(0) || faces.ndim() != 2 || faces.shape(1) != 3 ||
        moved.ndim() != 1 || dirty.ndim() != 1 || moved.shape(0) != faces.shape(0) || dirty.shape(0) != faces.shape(0))
        throw std::invalid_argument("invalid indexed surface");
    const Index count = faces.shape(0), vertices = source.shape(0);
    const Index* f = faces.data();
    for (Index r = 0; r < 3 * count; ++r)
        if (f[r] < 0 || f[r] >= vertices) throw std::invalid_argument("face index out of range");
    const double* cand = candidate.data();
    const double* src = source.data();
    const bool* mv = moved.data();
    const bool* dt = dirty.data();
    std::vector<Vec> low(count), high(count), center(count);
    std::vector<double> radius(count);
    double max_radius = 0.;
    for (Index t = 0; t < count; ++t) {
        const Tri tri = triangle(cand, f + 3 * t);
        for (int i = 0; i < 3; ++i) {
            low[t][i] = std::min({tri[0][i], tri[1][i], tri[2][i]});
            high[t][i] = std::max({tri[0][i], tri[1][i], tri[2][i]});
            center[t][i] = (low[t][i] + high[t][i]) * .5;
        }
        radius[t] = norm(sub(high[t], low[t])) * .5;
        if (radius[t] > max_radius) max_radius = radius[t];
        if (!std::isfinite(radius[t])) throw std::invalid_argument("non-finite surface");
    }
    std::vector<Index> result_vertices;
    if (count == 0) return py::make_tuple(py::array_t<Index>(0), false);
    // Grid of triangle centres. A cell edge strictly above the largest query
    // radius (with the ambiguity band) keeps every candidate in the 27 cells.
    const double cell = std::max((2. * max_radius + 2e-9) * (1. + 1e-6), 1e-12);
    Vec origin = center[0];
    for (Index t = 1; t < count; ++t) for (int i = 0; i < 3; ++i) origin[i] = std::min(origin[i], center[t][i]);
    auto cell_of = [&](const Vec& c) {
        std::array<Index, 3> out;
        for (int i = 0; i < 3; ++i) out[i] = static_cast<Index>(std::floor((c[i] - origin[i]) / cell));
        return out;
    };
    std::array<Index, 3> extent{0, 0, 0};
    std::vector<std::array<Index, 3>> cells(count);
    for (Index t = 0; t < count; ++t) {
        for (int i = 0; i < 3; ++i) {
            const double scaled = std::floor((center[t][i] - origin[i]) / cell);
            if (!(scaled >= 0.) || scaled > 2e6) throw std::overflow_error("surface spans too many grid cells");
        }
        cells[t] = cell_of(center[t]);
        for (int i = 0; i < 3; ++i) extent[i] = std::max(extent[i], cells[t][i] + 1);
    }
    auto key = [&](const std::array<Index, 3>& c) { return (c[0] * extent[1] + c[1]) * extent[2] + c[2]; };
    std::vector<Index> by_cell(count);
    for (Index t = 0; t < count; ++t) by_cell[t] = t;
    std::vector<Index> keys(count);
    for (Index t = 0; t < count; ++t) keys[t] = key(cells[t]);
    std::stable_sort(by_cell.begin(), by_cell.end(), [&](Index a, Index b) { return keys[a] < keys[b]; });
    std::vector<Index> sorted_keys(count);
    for (Index r = 0; r < count; ++r) sorted_keys[r] = keys[by_cell[r]];
    std::vector<char> bad(static_cast<std::size_t>(vertices), 0);
    bool ambiguous = false;
    Index processed = 0;
    for (Index first_face = 0; first_face < count; ++first_face) {
        if (!dt[first_face]) continue;
        if (processed % 512 == 0) {
            // Same cadence as the reference: once before each 512 dirty faces.
            const py::object state = cancelled();
            const int stop = PyObject_IsTrue(state.ptr());
            if (stop < 0) throw py::error_already_set();
            if (stop) {
                PyErr_SetString(PyExc_InterruptedError, "Surface intersection audit cancelled");
                throw py::error_already_set();
            }
        }
        ++processed;
        const double bound = radius[first_face] + max_radius + 1e-9;
        const double bound2 = bound * bound;
        const auto home = cells[first_face];
        for (Index di = -1; di <= 1; ++di) for (Index dj = -1; dj <= 1; ++dj) for (Index dk = -1; dk <= 1; ++dk) {
            const std::array<Index, 3> c{home[0] + di, home[1] + dj, home[2] + dk};
            if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= extent[0] || c[1] >= extent[1] || c[2] >= extent[2]) continue;
            const Index k = key(c);
            auto range = std::equal_range(sorted_keys.begin(), sorted_keys.end(), k);
            for (auto it = range.first; it != range.second; ++it) {
                const Index s = by_cell[it - sorted_keys.begin()];
                if (s == first_face) continue;
                if (!(s > first_face || !dt[s])) continue;
                if (!(mv[first_face] || mv[s])) continue;
                const Vec d = sub(center[s], center[first_face]);
                const double distance2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if (distance2 > bound2 * (1. + kBoundBand)) continue;
                const bool uncertain = distance2 >= bound2 * (1. - kBoundBand);
                const Index a = std::min(first_face, s), b = std::max(first_face, s);
                bool overlap = true;
                for (int i = 0; i < 3; ++i)
                    overlap = overlap && low[a][i] <= high[b][i] + 1e-9 && high[a][i] >= low[b][i] - 1e-9;
                if (!overlap) continue;
                bool shared = false;
                for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y) shared = shared || f[3 * a + x] == f[3 * b + y];
                if (shared) continue;
                if (!intersecting(triangle(cand, f + 3 * a), triangle(cand, f + 3 * b), order)) continue;
                if (intersecting(triangle(src, f + 3 * a), triangle(src, f + 3 * b), order)) continue;
                if (uncertain) { ambiguous = true; continue; }
                for (int x = 0; x < 3; ++x) { bad[f[3 * a + x]] = 1; bad[f[3 * b + x]] = 1; }
            }
        }
    }
    for (Index v = 0; v < vertices; ++v) if (bad[v]) result_vertices.push_back(v);
    py::array_t<Index> result(static_cast<py::ssize_t>(result_vertices.size()));
    std::copy(result_vertices.begin(), result_vertices.end(), result.mutable_data());
    return py::make_tuple(result, ambiguous);
}

}  // namespace

PYBIND11_MODULE(_accel_cpp, module) {
    module.doc() = "Opt-in RessectionLab acceleration kernels; Python references stay authoritative.";
    module.attr("VERSION") = 1;
    module.def("connected_surface_cells", &connected_surface_cells, py::arg("candidates"), py::arg("connected_free"));
    // connected_free is mutated in place, so it is never silently copied.
    module.def("extend_connected_free", &extend_connected_free, py::arg("new_cells"), py::arg("remaining"),
               py::arg("connected_free").noconvert(), py::arg("interaction_domain"));
    module.def("intersecting", &intersecting_batch, py::arg("first"), py::arg("second"), py::arg("order"));
    module.def("projections", &projections, py::arg("points"), py::arg("axes"), py::arg("order"));
    module.def("norms", &norms, py::arg("vectors"));
    module.def("intersection_vertices", &intersection_vertices, py::arg("source"), py::arg("candidate"),
               py::arg("faces"), py::arg("moved"), py::arg("dirty"), py::arg("cancelled"), py::arg("order"));
}
