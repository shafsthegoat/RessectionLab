#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "simulation_control_native.hpp"
#include "simulation_motion_native.hpp"
#include "simulation_state_native.hpp"

namespace resectionlab::sim {
namespace geometry_detail {

constexpr double epsilon = 1e-9;
constexpr std::uint64_t primitive_chunk = 32768;
using Cell = std::array<std::int64_t, 3>;

inline Point subtract(const Point& first, const Point& second) {
    return {first[0] - second[0], first[1] - second[1], first[2] - second[2]};
}
inline double dot(const Point& first, const Point& second) {
    return (first[0] * second[0] + first[1] * second[1]) + first[2] * second[2];
}
inline double norm(const Point& value) { return std::sqrt(dot(value, value)); }
inline void finite(const Point& value, const char* reason) {
    for (double coordinate : value) if (!std::isfinite(coordinate)) throw std::invalid_argument(reason);
}
inline Point transform(const std::array<double, 16>& affine, const Point& value) {
    Point result;
    for (int axis = 0; axis < 3; ++axis) {
        const auto row = 4 * axis;
        result[axis] = ((value[0] * affine[row] + value[1] * affine[row + 1]) +
            value[2] * affine[row + 2]) + affine[row + 3];
    }
    return result;
}
inline std::int64_t grid_size(const Scene& scene) {
    std::int64_t count = 1;
    for (auto width : scene.shape) {
        if (width == 0 || width > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
                count > std::numeric_limits<std::int64_t>::max() / static_cast<std::int64_t>(width))
            throw std::invalid_argument("Invalid native simulation grid shape");
        count *= static_cast<std::int64_t>(width);
    }
    if (count > 128000000)
        throw std::invalid_argument("Simulation grid exceeds 128 million cells; choose a smaller declared resolution");
    return count;
}
inline Cell coordinates(const Scene& scene, std::int64_t cell) {
    const auto plane = static_cast<std::int64_t>(scene.shape[1] * scene.shape[2]);
    const auto depth = static_cast<std::int64_t>(scene.shape[2]);
    return {cell / plane, (cell % plane) / depth, cell % depth};
}

struct Frame {
    Point spacing;
    double cell_radius = 0.;
    bool orthogonal = true;
};

inline Frame frame(const Scene& scene) {
    grid_size(scene);
    for (const auto* matrix : {&scene.affine, &scene.inverse}) {
        for (double value : *matrix)
            if (!std::isfinite(value)) throw std::invalid_argument("affine must be a finite 4x4 matrix");
        if (std::abs((*matrix)[12]) > 1e-10 || std::abs((*matrix)[13]) > 1e-10 ||
                std::abs((*matrix)[14]) > 1e-10 || std::abs((*matrix)[15] - 1.) > 1e-10)
            throw std::invalid_argument("affine must have homogeneous last row [0,0,0,1]");
    }
    Frame result{scene.spacing, 0., true};
    for (double value : result.spacing)
        if (!std::isfinite(value) || value <= 0.) throw std::invalid_argument("Invalid native simulation spacing");
    for (int corner = 0; corner < 8; ++corner) {
        Point offset;
        const Point sign{corner & 4 ? .5 : -.5, corner & 2 ? .5 : -.5, corner & 1 ? .5 : -.5};
        for (int axis = 0; axis < 3; ++axis) {
            const auto row = 4 * axis;
            offset[axis] = (sign[0] * scene.affine[row] + sign[1] * scene.affine[row + 1]) +
                sign[2] * scene.affine[row + 2];
        }
        result.cell_radius = std::max(result.cell_radius, norm(offset));
    }
    for (int first = 0; first < 3; ++first) for (int second = 0; second < 3; ++second) {
        Point a, b;
        for (int row = 0; row < 3; ++row) {
            a[row] = scene.affine[4 * row + first] / result.spacing[first];
            b[row] = scene.affine[4 * row + second] / result.spacing[second];
        }
        if (std::abs(dot(a, b) - (first == second ? 1. : 0.)) > 1e-10) result.orthogonal = false;
    }
    if (!std::isfinite(result.cell_radius)) throw std::overflow_error("Native cell radius overflow");
    return result;
}

inline double point_distance(const Point& point, const Point& start, const Point& end) {
    const auto delta = subtract(end, start);
    const double length2 = dot(delta, delta);
    double fraction = 0.;
    if (length2 > epsilon * epsilon)
        fraction = std::clamp(dot(subtract(point, start), delta) / length2, 0., 1.);
    Point residual;
    for (int axis = 0; axis < 3; ++axis)
        residual[axis] = point[axis] - (start[axis] + fraction * delta[axis]);
    return norm(residual);
}

inline double box_distance(const Cell& cell, const Point& start_voxel,
                           const Point& delta, const Point& spacing) {
    Point origin, half;
    int changing = 0, changing_axis = 0;
    for (int axis = 0; axis < 3; ++axis) {
        origin[axis] = (start_voxel[axis] - static_cast<double>(cell[axis])) * spacing[axis];
        half[axis] = spacing[axis] / 2.;
        if (delta[axis] != 0.) { ++changing; changing_axis = axis; }
    }
    auto at = [&](double time) {
        Point excess;
        for (int axis = 0; axis < 3; ++axis)
            excess[axis] = std::max(std::abs(origin[axis] + time * delta[axis]) - half[axis], 0.);
        return dot(excess, excess);
    };
    if (changing == 0) return std::sqrt(at(0.));
    if (changing == 1 && std::abs(delta[changing_axis]) > epsilon)
        return std::sqrt(at(std::clamp(-origin[changing_axis] / delta[changing_axis], 0., 1.)));
    std::array<double, 8> times{0., 0., 0., 0., 0., 0., 0., 1.};
    for (int axis = 0; axis < 3; ++axis) if (std::abs(delta[axis]) > epsilon) {
        times[1 + 2 * axis] = std::clamp((-half[axis] - origin[axis]) / delta[axis], 0., 1.);
        times[2 + 2 * axis] = std::clamp((half[axis] - origin[axis]) / delta[axis], 0., 1.);
    }
    std::sort(times.begin(), times.end());
    double minimum = std::numeric_limits<double>::infinity();
    for (int interval = 0; interval < 7; ++interval) {
        const double midpoint = (times[interval] + times[interval + 1]) / 2.;
        Point numerator{}, denominator{};
        for (int axis = 0; axis < 3; ++axis) {
            const double position = origin[axis] + midpoint * delta[axis];
            if (std::abs(position) > half[axis]) {
                const double offset = origin[axis] - (position > half[axis] ? half[axis] : -half[axis]);
                numerator[axis] = offset * delta[axis];
                denominator[axis] = delta[axis] * delta[axis];
            }
        }
        const double top = (numerator[0] + numerator[1]) + numerator[2];
        const double bottom = (denominator[0] + denominator[1]) + denominator[2];
        const double optimum = bottom > epsilon * epsilon ? -top / bottom : midpoint;
        minimum = std::min(minimum, at(std::clamp(optimum, times[interval], times[interval + 1])));
    }
    return std::sqrt(minimum);
}

}  // namespace geometry_detail

// shortcut: exact legacy admission currently uses signed-permutation affines; qualify wider transforms before admitting them.
inline bool geometry_scene_supported(const Scene& scene) noexcept {
    std::uint64_t cells = 1;
    for (auto width : scene.shape) {
        if (width == 0 || width > 128000000 || cells > 128000000 / width) return false;
        cells *= width;
    }
    for (double value : scene.spacing) if (!std::isfinite(value) || value <= 0.) return false;
    for (const auto* matrix : {&scene.affine, &scene.inverse}) {
        for (double value : *matrix) if (!std::isfinite(value)) return false;
        if (std::abs((*matrix)[12]) > 1e-10 || std::abs((*matrix)[13]) > 1e-10 ||
                std::abs((*matrix)[14]) > 1e-10 || std::abs((*matrix)[15] - 1.) > 1e-10) return false;
    }
    std::array<int, 3> columns{};
    for (int row = 0; row < 3; ++row) {
        int nonzero = 0;
        for (int column = 0; column < 3; ++column) {
            const double value = scene.affine[4 * row + column];
            if (!std::isfinite(value)) return false;
            if (value != 0.) { ++nonzero; ++columns[column]; }
        }
        if (nonzero != 1 || !std::isfinite(scene.affine[4 * row + 3])) return false;
    }
    for (int count : columns) if (count != 1) return false;
    return true;
}

inline Point world_point(const Scene& scene, std::int64_t cell) {
    if (cell < 0 || cell >= geometry_detail::grid_size(scene)) throw std::out_of_range("Cell outside source grid");
    const auto index = geometry_detail::coordinates(scene, cell);
    return geometry_detail::transform(scene.affine,
        Point{static_cast<double>(index[0]), static_cast<double>(index[1]), static_cast<double>(index[2])});
}

inline double point_segment_distance(const Point& point, const Point& start, const Point& end) {
    geometry_detail::finite(point, "point must be a finite three-vector");
    geometry_detail::finite(start, "start_mm must be a finite three-vector");
    geometry_detail::finite(end, "end_mm must be a finite three-vector");
    const double result = geometry_detail::point_distance(point, start, end);
    if (!std::isfinite(result)) throw std::overflow_error("Native segment distance overflow");
    return result;
}

inline double segment_distance(const Point& a0, const Point& a1, const Point& b0, const Point& b1) {
    geometry_detail::finite(a0, "native geometry metadata must be finite");
    geometry_detail::finite(a1, "native geometry metadata must be finite");
    geometry_detail::finite(b0, "native geometry metadata must be finite");
    geometry_detail::finite(b1, "native geometry metadata must be finite");
    const auto u = geometry_detail::subtract(a1, a0), v = geometry_detail::subtract(b1, b0),
        w = geometry_detail::subtract(a0, b0);
    const double aa = geometry_detail::dot(u, u), bb = geometry_detail::dot(u, v), cc = geometry_detail::dot(v, v),
        dd = geometry_detail::dot(u, w), ee = geometry_detail::dot(v, w);
    double minimum = std::min({geometry_detail::point_distance(a0, b0, b1), geometry_detail::point_distance(a1, b0, b1),
        geometry_detail::point_distance(b0, a0, a1), geometry_detail::point_distance(b1, a0, a1)});
    const double product = aa * cc;
    const double determinant = product - bb * bb;
    if (determinant > geometry_detail::epsilon * geometry_detail::epsilon * std::max(product, 1.)) {
        const double s = (bb * ee - cc * dd) / determinant;
        const double t = (aa * ee - bb * dd) / determinant;
        if (s >= 0. && s <= 1. && t >= 0. && t <= 1.) {
            Point residual;
            for (int axis = 0; axis < 3; ++axis) residual[axis] = (w[axis] + s * u[axis]) - t * v[axis];
            minimum = std::min(minimum, geometry_detail::norm(residual));
        }
    }
    if (!std::isfinite(minimum) || !std::isfinite(product) || !std::isfinite(determinant))
        throw std::overflow_error("Native segment distance overflow");
    return minimum;
}

inline bool segments_within_distance(const Point& a0, const Point& a1, const Point& b0, const Point& b1,
                                     double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.) throw std::invalid_argument("invalid segment predicate threshold/epsilon");
    return segment_distance(a0, a1, b0, b1) <= threshold;
}

// Full-cell orthogonal coverage; shear retains the established circumsphere lower bound.
// Callers admit geometry_scene_supported before beginning a native transaction.
inline std::vector<std::int64_t> capsule_cells(const Scene& scene, const Point& start,
                                              const Point& end, double radius, const StopToken& stop) {
    using namespace geometry_detail;
    stop.check();
    finite(start, "start_mm must be a finite three-vector"); finite(end, "end_mm must be a finite three-vector");
    if (!std::isfinite(radius) || radius < 0.) throw std::invalid_argument("radius_mm must be finite and nonnegative");
    const auto geometry = frame(scene);
    const double broad_radius = radius + geometry.cell_radius;
    const auto first = transform(scene.inverse, start), last = transform(scene.inverse, end);
    Point scaled_delta;
    Cell low, high;
    bool empty = false;
    for (int axis = 0; axis < 3; ++axis) {
        const auto row = 4 * axis;
        const Point inverse_row{scene.inverse[row], scene.inverse[row + 1], scene.inverse[row + 2]};
        const double extent = norm(inverse_row) * broad_radius;
        const double lower = std::ceil((std::min(first[axis], last[axis]) - extent) - epsilon);
        const double upper = std::floor((std::max(first[axis], last[axis]) + extent) + epsilon);
        if (!std::isfinite(lower) || !std::isfinite(upper)) throw std::overflow_error("Native capsule bounds overflow");
        const auto maximum = static_cast<std::int64_t>(scene.shape[axis] - 1);
        if (lower > static_cast<double>(maximum) || upper < 0.) { empty = true; continue; }
        low[axis] = lower <= 0. ? 0 : static_cast<std::int64_t>(lower);
        high[axis] = upper >= static_cast<double>(maximum) ? maximum : static_cast<std::int64_t>(upper);
        if (low[axis] > high[axis]) empty = true;
        scaled_delta[axis] = (last[axis] - first[axis]) * geometry.spacing[axis];
    }
    std::vector<std::int64_t> result;
    if (empty) { stop.check(); return result; }
    const auto width_y = high[1] - low[1] + 1, width_z = high[2] - low[2] + 1;
    const auto plane = width_y * width_z;
    const auto candidates = (high[0] - low[0] + 1) * plane;
    result.reserve(static_cast<std::size_t>(std::min<std::int64_t>(candidates, 4096)));
    std::uint64_t visited = 0;
    for (std::int64_t x = low[0]; x <= high[0]; ++x)
        for (std::int64_t y = low[1]; y <= high[1]; ++y)
            for (std::int64_t z = low[2]; z <= high[2]; ++z) {
                if ((visited++ % primitive_chunk) == 0) stop.check();
                const Cell index{x, y, z};
                const auto center = transform(scene.affine, Point{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)});
                const double broad = point_distance(center, start, end);
                if (!std::isfinite(broad)) throw std::overflow_error("Native capsule distance overflow");
                if (broad > broad_radius + epsilon) continue;
                const double fine = geometry.orthogonal
                    ? box_distance(index, first, scaled_delta, geometry.spacing) : broad - geometry.cell_radius;
                if (!std::isfinite(fine)) throw std::overflow_error("Native capsule distance overflow");
                if (fine <= radius + epsilon)
                    result.push_back((x * static_cast<std::int64_t>(scene.shape[1]) + y) *
                        static_cast<std::int64_t>(scene.shape[2]) + z);
            }
    stop.check();
    return result;
}

}  // namespace resectionlab::sim
