#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <stdexcept>
#include <tuple>
#include "state_native.hpp"
#include "executor_native.hpp"
#include "json_native.hpp"
#include "motion_native.hpp"
#include "validation_native.hpp"
#include "simulation_bindings_native.hpp"

namespace py = pybind11;
namespace {
constexpr std::int64_t max_primitives = 32768;
constexpr std::int64_t max_items = 64;
using Cell = std::array<std::int64_t, 3>;
using Triple = std::array<double, 3>;

// Copy small checked metadata under the GIL: writable/aliased caller buffers
// cannot change the primitive computation while another Python thread runs.
template <typename T>
std::vector<T> checked_array(const py::array& value, const std::vector<py::ssize_t>& shape,
                             const char* name) {
    if (!value.dtype().is(py::dtype::of<T>()) || !(value.flags() & py::array::c_style)
            || value.ndim() != static_cast<py::ssize_t>(shape.size()))
        throw py::value_error(std::string(name) + " must have exact native dtype and C-contiguous shape");
    for (py::ssize_t axis = 0; axis < value.ndim(); ++axis)
        if (shape[axis] >= 0 && value.shape(axis) != shape[axis])
            throw py::value_error(std::string(name) + " has invalid shape");
    std::vector<T> copied(static_cast<std::size_t>(value.size()));
    if (!copied.empty()) std::memcpy(copied.data(), value.data(), copied.size() * sizeof(T));
    return copied;
}

void finite_values(const std::vector<double>& values) {
    for (double value : values)
        if (!std::isfinite(value)) throw py::value_error("native geometry metadata must be finite");
}

double squared_norm(const Triple& value) {
    return (value[0] * value[0] + value[1] * value[1]) + value[2] * value[2];
}

double point_distance(const Triple& point, const double* query, double eps) {
    Triple delta, relative;
    for (int axis = 0; axis < 3; ++axis) {
        delta[axis] = query[3 + axis] - query[axis];
        relative[axis] = point[axis] - query[axis];
    }
    const double length2 = squared_norm(delta);
    double fraction = 0.;
    if (length2 > eps * eps) {
        const double dot = (relative[0] * delta[0] + relative[1] * delta[1]) + relative[2] * delta[2];
        fraction = std::clamp(dot / length2, 0., 1.);
    }
    Triple residual;
    for (int axis = 0; axis < 3; ++axis)
        residual[axis] = point[axis] - (query[axis] + fraction * delta[axis]);
    return std::sqrt(squared_norm(residual));
}

double box_distance(const Cell& cell, const double* query, const std::vector<double>& spacing,
                    double eps) {
    Triple origin, half, delta;
    int changing = 0, changing_axis = 0;
    for (int axis = 0; axis < 3; ++axis) {
        origin[axis] = (query[6 + axis] - static_cast<double>(cell[axis])) * spacing[axis];
        half[axis] = spacing[axis] / 2.;
        delta[axis] = query[9 + axis];
        if (delta[axis] != 0.) { ++changing; changing_axis = axis; }
    }
    auto at = [&](double time) {
        Triple excess;
        for (int axis = 0; axis < 3; ++axis)
            excess[axis] = std::max(std::abs(origin[axis] + time * delta[axis]) - half[axis], 0.);
        return squared_norm(excess);
    };
    if (!changing) return std::sqrt(at(0.));
    if (changing == 1 && std::abs(delta[changing_axis]) > eps)
        return std::sqrt(at(std::clamp(-origin[changing_axis] / delta[changing_axis], 0., 1.)));
    std::array<double, 8> times{0., 0., 0., 0., 0., 0., 0., 1.};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(delta[axis]) > eps) {
            times[1 + 2 * axis] = std::clamp((-half[axis] - origin[axis]) / delta[axis], 0., 1.);
            times[2 + 2 * axis] = std::clamp((half[axis] - origin[axis]) / delta[axis], 0., 1.);
        }
    }
    std::sort(times.begin(), times.end());
    double minimum = std::numeric_limits<double>::infinity();
    for (int interval = 0; interval < 7; ++interval) {
        const double midpoint = (times[interval] + times[interval + 1]) / 2.;
        Triple numerators{}, denominators{};
        for (int axis = 0; axis < 3; ++axis) {
            const double position = origin[axis] + midpoint * delta[axis];
            if (std::abs(position) > half[axis]) {
                const double offset = origin[axis] - (position > half[axis] ? half[axis] : -half[axis]);
                numerators[axis] = offset * delta[axis];
                denominators[axis] = delta[axis] * delta[axis];
            }
        }
        const double numerator = (numerators[0] + numerators[1]) + numerators[2];
        const double denominator = (denominators[0] + denominators[1]) + denominators[2];
        const double optimum = denominator > eps * eps ? -numerator / denominator : midpoint;
        minimum = std::min(minimum, at(std::clamp(optimum, times[interval], times[interval + 1])));
    }
    return std::sqrt(minimum);
}

py::array_t<std::int64_t> owned_cells(const std::vector<Cell>& cells) {
    py::array_t<std::int64_t> result({static_cast<py::ssize_t>(cells.size()), py::ssize_t(3)});
    if (!cells.empty()) std::memcpy(result.mutable_data(), cells.data(), cells.size() * sizeof(Cell));
    return result;
}

py::tuple capsule_chunk(const py::array& query_array, const py::array& span_array,
                         const py::array& affine_array, const py::array& spacing_array,
                         double cell_radius, double eps) {
    if (query_array.ndim() != 2 || query_array.shape(0) > max_items)
        throw py::value_error("native geometry batch shape or item cap exceeded");
    const auto count = query_array.shape(0);
    auto queries = checked_array<double>(query_array, {count, 14}, "queries");
    auto spans = checked_array<std::int64_t>(span_array, {count, 8}, "spans");
    auto affine = checked_array<double>(affine_array, {4, 4}, "affine");
    auto spacing = checked_array<double>(spacing_array, {3}, "spacing");
    finite_values(queries); finite_values(affine); finite_values(spacing);
    if (!std::isfinite(cell_radius) || cell_radius < 0. || !std::isfinite(eps) || eps <= 0.)
        throw py::value_error("invalid native geometry radius/epsilon");
    const bool shear = spacing[0] == 0. && spacing[1] == 0. && spacing[2] == 0.;
    if (!shear && std::any_of(spacing.begin(), spacing.end(), [](double v) { return v <= 0.; }))
        throw py::value_error("spacing must be positive or all zero for shear");
    if (std::abs(affine[12]) > 1e-10 || std::abs(affine[13]) > 1e-10
            || std::abs(affine[14]) > 1e-10 || std::abs(affine[15] - 1.) > 1e-10)
        throw py::value_error("affine must have homogeneous final row");
    const double determinant = affine[0] * (affine[5] * affine[10] - affine[6] * affine[9])
        - affine[1] * (affine[4] * affine[10] - affine[6] * affine[8])
        + affine[2] * (affine[4] * affine[9] - affine[5] * affine[8]);
    if (!std::isfinite(determinant) || std::abs(determinant) < 1e-12)
        throw py::value_error("affine must be invertible");
    std::int64_t primitives = 0;
    for (py::ssize_t item = 0; item < count; ++item) {
        const auto* span = spans.data() + item * 8;
        const auto* query = queries.data() + item * 14;
        std::int64_t volume = 1;
        for (int axis = 0; axis < 3; ++axis) {
            if (span[axis] < 0 || span[axis + 3] < span[axis]
                    || span[axis + 3] == std::numeric_limits<std::int64_t>::max())
                throw py::value_error("invalid candidate bounds");
            const auto width = span[axis + 3] - span[axis] + 1;
            if (volume > std::numeric_limits<std::int64_t>::max() / width)
                throw py::value_error("candidate volume overflow");
            volume *= width;
        }
        if (span[6] < 0 || span[7] < 0 || span[6] > volume || span[7] > volume - span[6]
                || span[7] > max_primitives - primitives)
            throw py::value_error("candidate span exceeds bounds or primitive cap");
        if (query[12] < 0. || query[13] < query[12])
            throw py::value_error("invalid distance thresholds");
        primitives += span[7];
    }
    std::vector<std::vector<Cell>> hits(count), ambiguous(count);
    {
        py::gil_scoped_release release;
        for (py::ssize_t item = 0; item < count; ++item) {
            const auto* span = spans.data() + item * 8;
            const auto* query = queries.data() + item * 14;
            const auto width_y = span[4] - span[1] + 1, width_z = span[5] - span[2] + 1;
            const auto plane = width_y * width_z;
            Triple delta;
            for (int axis = 0; axis < 3; ++axis) delta[axis] = query[3 + axis] - query[axis];
            const double length2 = squared_norm(delta);
            const bool degenerate_boundary = std::abs(length2 - eps * eps) <=
                128. * std::numeric_limits<double>::epsilon() * std::max(length2, eps * eps);
            for (std::int64_t offset = span[6]; offset < span[6] + span[7]; ++offset) {
                Cell cell{span[0] + offset / plane, span[1] + (offset % plane) / width_z,
                          span[2] + offset % width_z};
                if (degenerate_boundary) { ambiguous[item].push_back(cell); continue; }
                Triple world;
                double magnitude = 1.;
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* row = affine.data() + 4 * axis;
                    world[axis] = ((cell[0] * row[0] + cell[1] * row[1]) + cell[2] * row[2]) + row[3];
                    magnitude += std::abs(cell[0] * row[0]) + std::abs(cell[1] * row[1])
                               + std::abs(cell[2] * row[2]) + std::abs(row[3])
                               + std::abs(query[axis]) + std::abs(query[3 + axis]);
                    if (!shear) magnitude += std::abs(query[9 + axis]) + spacing[axis];
                }
                // Different BLAS dot evaluation is resolved by the unchanged
                // reference predicates only for cells near either threshold.
                const double uncertainty = 512. * std::numeric_limits<double>::epsilon() * magnitude;
                const double broad = point_distance(world, query, eps);
                if (!std::isfinite(broad)) throw std::runtime_error("native geometry distance overflow");
                if (std::abs(broad - query[13]) <= uncertainty) {
                    ambiguous[item].push_back(cell); continue;
                }
                if (broad > query[13]) continue;
                const double fine = shear ? broad - cell_radius : box_distance(cell, query, spacing, eps);
                if (!std::isfinite(fine)) throw std::runtime_error("native geometry distance overflow");
                if (std::abs(fine - query[12]) <= uncertainty) ambiguous[item].push_back(cell);
                else if (fine <= query[12]) hits[item].push_back(cell);
            }
        }
    }
    py::tuple result(count);
    for (py::ssize_t item = 0; item < count; ++item)
        result[item] = py::make_tuple(owned_cells(hits[item]), owned_cells(ambiguous[item]));
    return result;
}

// Return -1 at arithmetic branch/threshold ambiguity; Python then evaluates
// the original scalar predicate. No approximate distance enters physical state.
int segment_predicate_values(const double* points, double threshold, double eps) {
    Triple u, v, w;
    for (int axis = 0; axis < 3; ++axis) {
        u[axis] = points[3 + axis] - points[axis];
        v[axis] = points[9 + axis] - points[6 + axis];
        w[axis] = points[axis] - points[6 + axis];
    }
    auto dot = [](const Triple& a, const Triple& b) {
        return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
    };
    const double aa = dot(u, u), bb = dot(u, v), cc = dot(v, v);
    const double dd = dot(u, w), ee = dot(v, w);
    int u_axis = -1, v_axis = -1, u_count = 0, v_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (u[axis] != 0.) { u_axis = axis; ++u_count; }
        if (v[axis] != 0.) { v_axis = axis; ++v_count; }
    }
    // These dots contain one product, matching NumPy's evaluation exactly.
    const bool axis_parallel = u_count == 1 && v_count == 1 && u_axis == v_axis;
    const double machine_eps = std::numeric_limits<double>::epsilon();
    for (double length2 : {aa, cc})
        if (std::abs(length2 - eps * eps) <= 128. * machine_eps * std::max(length2, eps * eps))
            return -1;
    double minimum = std::numeric_limits<double>::infinity();
    for (int endpoint = 0; endpoint < 4; ++endpoint) {
        Triple point;
        std::copy_n(points + 3 * endpoint, 3, point.begin());
        const double distance = point_distance(point, points + (endpoint < 2 ? 6 : 0), eps);
        minimum = std::min(minimum, distance);
    }
    const double product = aa * cc, determinant = product - bb * bb;
    const double branch = eps * eps * std::max(product, 1.);
    double condition = 1.;
    if (aa > 0. && cc > 0.) {
        const double branch_error = 1024. * machine_eps * std::max(product, 1.);
        if (!axis_parallel && std::abs(determinant - branch) <= branch_error) return -1;
        if (determinant > branch) {
            condition = axis_parallel ? 1. : std::max(1., product / determinant);
            if (condition > 1e8) return -1;
            const double s = (bb * ee - cc * dd) / determinant;
            const double t = (aa * ee - bb * dd) / determinant;
            const double parameter_error = 1024. * machine_eps * condition * std::max({1., std::abs(s), std::abs(t)});
            if (!axis_parallel && std::min({std::abs(s), std::abs(t), std::abs(1. - s), std::abs(1. - t)}) <= parameter_error)
                return -1;
            if (s >= 0. && s <= 1. && t >= 0. && t <= 1.) {
                Triple residual;
                for (int axis = 0; axis < 3; ++axis) residual[axis] = (w[axis] + s * u[axis]) - t * v[axis];
                minimum = std::min(minimum, std::sqrt(squared_norm(residual)));
            }
        }
    }
    double magnitude = 1. + threshold;
    for (int index = 0; index < 12; ++index) magnitude += std::abs(points[index]);
    const double uncertainty = 1024. * machine_eps * magnitude * condition;
    if (!std::isfinite(minimum) || !std::isfinite(product) || !std::isfinite(determinant))
        throw std::runtime_error("native segment predicate overflow");
    if (std::abs(minimum - threshold) <= uncertainty) return -1;
    return minimum <= threshold ? 1 : 0;
}
int segment_predicate(const py::array& endpoints, double threshold, double eps) {
    auto points = checked_array<double>(endpoints, {4, 3}, "endpoints");
    finite_values(points);
    if (!std::isfinite(threshold) || threshold < 0. || !std::isfinite(eps) || eps <= 0.)
        throw py::value_error("invalid segment predicate threshold/epsilon");
    return segment_predicate_values(points.data(), threshold, eps);
}

std::tuple<int, int, int> segment_predicate_batch(
    const py::array& endpoints, const py::array& thresholds, double eps) {
    if (endpoints.ndim() != 3 || endpoints.shape(0) > max_items)
        throw py::value_error("segment batch shape or item cap exceeded");
    const auto count = endpoints.shape(0);
    auto points = checked_array<double>(endpoints, {count, 4, 3}, "endpoints");
    auto limits = checked_array<double>(thresholds, {count}, "thresholds");
    if (!std::isfinite(eps) || eps <= 0.)
        throw py::value_error("invalid segment predicate threshold/epsilon");
    int first = -1, status = 0, examined = 0;
    {
        py::gil_scoped_release release;
        for (int index = 0; index < count; ++index) {
            const double* row = points.data() + 12 * index;
            for (int coordinate = 0; coordinate < 12; ++coordinate)
                if (!std::isfinite(row[coordinate]))
                    throw std::invalid_argument("native geometry metadata must be finite");
            if (!std::isfinite(limits[index]) || limits[index] < 0.)
                throw std::invalid_argument("invalid segment predicate threshold/epsilon");
            ++examined;
            status = segment_predicate_values(row, limits[index], eps);
            if (status != 0) { first = index; break; }
        }
    }
    return std::make_tuple(first, status, examined);
}

}  // namespace

PYBIND11_MODULE(_geometry_cpp, module) {
    module.attr("BUILD_IDENTITY") = RSL_BUILD_IDENTITY;
    module.attr("MAX_PRIMITIVES") = max_primitives;
    module.attr("MAX_BATCH_ITEMS") = max_items;
    module.def("capsule_chunk", &capsule_chunk, py::arg("queries").noconvert(),
        py::arg("spans").noconvert(), py::arg("affine").noconvert(), py::arg("spacing").noconvert(),
        py::arg("cell_radius"), py::arg("eps"));
    module.def("segment_predicate", &segment_predicate, py::arg("endpoints").noconvert(),
               py::arg("threshold"), py::arg("eps"));
    module.def("segment_predicate_batch", &segment_predicate_batch,
               py::arg("endpoints").noconvert(), py::arg("thresholds").noconvert(), py::arg("eps"));
    bind_state_native(module);
    bind_executor_physical(module);
    bind_executor_trajectory(module);
    bind_executor_state(module);
    bind_executor_cache(module);
    bind_executor_occupancy(module);
    bind_canonical_json(module);
    bind_motion_native(module);
    bind_validation_native(module);
    resectionlab::sim::boundary::bind(module);
}
