#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace resectionlab::sim {

using Point = std::array<double, 3>;
using Quat = std::array<double, 4>;
using Matrix3 = std::array<double, 9>;

// Match the qualified SciPy 1.18.1 arm64 build, including its explicit rounding sites.
// Source: scipy/spatial/transform/_rotation_cy.pyx and _rotation.py (BSD-3-Clause).
// Other numerical-library builds require their own exactness qualification at admission.
inline double quaternion_squared_norm3(const Point& value) noexcept {
    return std::fma(value[2], value[2], std::fma(value[0], value[0], value[1] * value[1]));
}

inline double point_squared_norm(const Point& value) noexcept {
    return (value[0] * value[0] + value[1] * value[1]) + value[2] * value[2];
}

inline double point_norm(const Point& value) noexcept {
    return std::sqrt(point_squared_norm(value));
}

inline Quat normalized_quaternion(Quat value) {
    const Point vector{value[0], value[1], value[2]};
    const double norm = std::sqrt(std::fma(value[3], value[3], quaternion_squared_norm3(vector)));
    if (norm == 0. || std::isnan(norm))
        throw std::invalid_argument("Found zero norm quaternions in `quat`.");
    for (double& component : value) component /= norm;
    return value;
}

inline Quat inverse_quaternion(const Quat& unit) noexcept {
    return {-unit[0], -unit[1], -unit[2], unit[3]};
}

inline Quat compose_quaternion(const Quat& first, const Quat& second) {
    const double cross_x = std::fma(first[1], second[2], (-first[2]) * second[1]);
    const double cross_y = std::fma(first[2], second[0], (-first[0]) * second[2]);
    const double cross_z = std::fma(first[0], second[1], (-first[1]) * second[0]);
    Quat product{
        std::fma(first[3], second[0], second[3] * first[0]) + cross_x,
        std::fma(first[3], second[1], second[3] * first[1]) + cross_y,
        std::fma(first[3], second[2], second[3] * first[2]) + cross_z,
        std::fma(-first[2], second[2], std::fma(-first[1], second[1],
            std::fma(first[3], second[3], (-first[0]) * second[0])))
    };
    return normalized_quaternion(product);
}

inline Matrix3 rotation_matrix(const Quat& unit) noexcept {
    const double x = unit[0], y = unit[1], z = unit[2], w = unit[3];
    const double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
    const double xy = x * y, zw = z * w, xz = x * z;
    const double yw = y * w, yz = y * z, xw = x * w;
    return {
        ((x2 - y2) - z2) + w2, 2. * (xy - zw), 2. * (xz + yw),
        2. * (xy + zw), ((-x2 + y2) - z2) + w2, 2. * (yz - xw),
        2. * (xz - yw), 2. * (yz + xw), ((-x2 - y2) + z2) + w2
    };
}

inline Point apply_rotation(const Matrix3& matrix, const Point& point) noexcept {
    Point result;
    for (unsigned axis = 0; axis != 3; ++axis) {
        const unsigned offset = axis * 3;
        // NumPy's 3x3 matvec starts at +0 and does not contract the products.
        result[axis] = ((0. + matrix[offset] * point[0]) + matrix[offset + 1] * point[1])
            + matrix[offset + 2] * point[2];
    }
    return result;
}

inline double rotation_magnitude(const Quat& unit) noexcept {
    const Point vector{unit[0], unit[1], unit[2]};
    return 2. * std::atan2(std::sqrt(quaternion_squared_norm3(vector)), std::abs(unit[3]));
}

inline Point rotation_vector(Quat unit) noexcept {
    if (unit[3] < 0. || (unit[3] == 0. && unit[0] < 0.)
            || (unit[3] == 0. && unit[0] == 0. && unit[1] < 0.)
            || (unit[3] == 0. && unit[0] == 0. && unit[1] == 0. && unit[2] < 0.)) {
        for (double& component : unit) component *= -1.;
    }
    const Point vector{unit[0], unit[1], unit[2]};
    const double angle = 2. * std::atan2(std::sqrt(quaternion_squared_norm3(vector)), unit[3]);
    double scale;
    if (angle <= 1e-3) {
        const double angle2 = angle * angle;
        scale = (2. + angle2 / 12.) + ((7. * angle2) * angle2) / 2880.;
    } else scale = angle / std::sin(angle / 2.);
    return {scale * unit[0], scale * unit[1], scale * unit[2]};
}

inline Quat quaternion_from_rotation_vector(const Point& vector) noexcept {
    const double angle = std::sqrt(quaternion_squared_norm3(vector));
    double scale;
    if (angle <= 1e-3) {
        const double angle2 = angle * angle;
        scale = (.5 - angle2 / 48.) + (angle2 * angle2) / 3840.;
    } else scale = std::sin(angle / 2.) / angle;
    return {scale * vector[0], scale * vector[1], scale * vector[2], std::cos(angle / 2.)};
}

struct QuaternionTrajectory {
    Quat start;
    Point relative_rotation;

    static QuaternionTrajectory between(const Quat& first, const Quat& second) {
        const auto start = normalized_quaternion(first);
        const auto end = normalized_quaternion(second);
        return {start, rotation_vector(compose_quaternion(inverse_quaternion(start), end))};
    }

    Quat at(double fraction) const {
        if (!(fraction >= 0. && fraction <= 1.))
            throw std::invalid_argument("Interpolation times must be within the range [0.0, 1.0], both inclusive.");
        const Point delta{relative_rotation[0] * fraction, relative_rotation[1] * fraction,
                          relative_rotation[2] * fraction};
        // Endpoint evaluation still composes and normalizes, just as SciPy Slerp does.
        return compose_quaternion(start, quaternion_from_rotation_vector(delta));
    }
};

inline Quat midpoint_quaternion(const Quat& first, const Quat& second) {
    return QuaternionTrajectory::between(first, second).at(.5);
}

inline Point midpoint_position(const Point& first, const Point& second) noexcept {
    return {(first[0] + second[0]) / 2., (first[1] + second[1]) / 2.,
            (first[2] + second[2]) / 2.};
}

inline double rotation_angle(const Quat& first, const Quat& second) {
    return rotation_magnitude(compose_quaternion(normalized_quaternion(second),
                                                inverse_quaternion(normalized_quaternion(first))));
}

}  // namespace resectionlab::sim
