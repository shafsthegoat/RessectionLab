#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <xlocale.h>
#endif
#include "simulation_control_native.hpp"
#include "simulation_motion_native.hpp"
#include "simulation_queries_native.hpp"
#include "simulation_state_native.hpp"

namespace resectionlab::sim {

namespace sparse_solver {
struct SparseFactor;
std::shared_ptr<const SparseFactor> factorize_csc(std::int32_t size,
    const std::vector<std::int32_t>& columns, const std::vector<std::int32_t>& rows,
    const std::vector<double>& values);
std::vector<double> solve(const SparseFactor& factor, const std::vector<double>& rhs);
double dot(const std::vector<double>& first, const std::vector<double>& second);
}  // namespace sparse_solver

namespace deformation { struct Diagnostics; }
namespace continuum {
void tick(State&, double, const StopToken&, deformation::Diagnostics*);
}
namespace position {
void activate_exposed_patch(State&, double radius_mm, std::size_t max_nodes, const StopToken&);
std::vector<Point> gravity_forces_n(State&, const std::vector<std::int64_t>&, const StopToken&);
}

namespace deformation {

struct Diagnostics {
    std::uint64_t ticks = 0, substeps = 0, targetCapNodeTicks = 0, forceCapNodeTicks = 0;
    std::uint64_t displacementCapNodeSteps = 0, strainProjectionSteps = 0, collisionRejectedSteps = 0;
    double maxPreStrainProjectionNeighborRatio = 0., maxFreeResidualN = 0.;
    double maxAbsDiscreteEnergyBalanceNmm = 0.;

    Value value() const {
        return Value::object({{"scope", "attempted numerical solves, including rolled-back trials"},
            {"ticks", ticks}, {"substeps", substeps}, {"targetCapNodeTicks", targetCapNodeTicks},
            {"forceCapNodeTicks", forceCapNodeTicks}, {"displacementCapNodeSteps", displacementCapNodeSteps},
            {"strainProjectionSteps", strainProjectionSteps}, {"collisionRejectedSteps", collisionRejectedSteps},
            {"maxPreStrainProjectionNeighborRatio", maxPreStrainProjectionNeighborRatio},
            {"maxFreeResidualN", maxFreeResidualN},
            {"maxAbsDiscreteEnergyBalanceNmm", maxAbsDiscreteEnergyBalanceNmm}});
    }
};

inline Value profile() {
    return Value::object({{"radiusMm", 4.}, {"maxNodes", 2000}, {"fixedStepSeconds", 1. / 120.},
        {"brainStiffnessNPerMm", .0006}, {"tumorStiffnessNPerMm", .0012},
        {"dragNsPerMm", .0003}, {"holdStiffnessNPerMm", .02}, {"bulkPenaltyRatio", 2.},
        {"maxDisplacementMm", 3.}, {"maxEdgeStrain", .65}, {"maxForceN", .08},
        {"model", "local small-strain Kelvin-Voigt lattice; uncalibrated engineering assumptions"},
        {"units", "displacement mm; stiffness N/mm; drag N s/mm; time s"}});
}

inline void initialize(State& engine) {
    if (!engine.state.contains("deformation")) engine.state.set("deformation", Value::object());
    if (!engine.state.contains("deformationModel"))
        engine.state.set("deformationModel", Value::object({{"active", Value::array()},
            {"anchors", Value::array()}, {"remainderSeconds", 0.}, {"velocities", Value::object()}}));
}

inline Value settings(const State& engine) {
    Value result = profile();
    const auto* physics = engine.scene->properties.find("tissuePhysicsV2");
    const auto* declared = physics == nullptr ? nullptr : physics->find("brain");
    if (declared != nullptr && !declared->is_null()) {
        if (!declared->is_object() || declared->as_object().size() != result.as_object().size())
            throw std::invalid_argument("INVALID_DEFORMATION_PROFILE: expected complete brain profile");
        for (auto& item : result.as_object()) {
            const auto* value = declared->find(item.first);
            if (value == nullptr)
                throw std::invalid_argument("INVALID_DEFORMATION_PROFILE: expected complete brain profile");
            item.second = *value;
        }
    }
    for (const char* name : {"radiusMm", "fixedStepSeconds", "brainStiffnessNPerMm", "tumorStiffnessNPerMm",
            "dragNsPerMm", "holdStiffnessNPerMm", "maxDisplacementMm", "maxEdgeStrain", "maxForceN"}) {
        const auto& value = result.at(name);
        if (!value.is_number() || !std::isfinite(value.as_number()) || value.as_number() <= 0.)
            throw std::invalid_argument(std::string("INVALID_DEFORMATION_PROFILE:") + name);
    }
    const auto& nodes = result.at("maxNodes");
    const auto& bulk = result.at("bulkPenaltyRatio");
    if (!nodes.is_int() || nodes.as_int() < 1 || nodes.as_int() > 2000 || !bulk.is_number() ||
            !(bulk.as_number() >= 0. && bulk.as_number() <= 100.))
        throw std::invalid_argument("INVALID_DEFORMATION_PROFILE");
    const double step = result.at("fixedStepSeconds").as_number(), radius = result.at("radiusMm").as_number();
    if (!(step >= .001 && step <= .1 && radius >= .1 && radius <= 12.))
        throw std::invalid_argument("DEFORMATION_PROFILE_OUTSIDE_NUMERICAL_DOMAIN");
    return result;
}

namespace detail {

#if defined(__APPLE__)
extern "C" std::to_chars_result apple_to_chars_fixed(char*, char*, double, std::chars_format, int)
    __asm__("__ZNSt3__18to_charsEPcS0_dNS_12chars_formatEi") __attribute__((weak_import));
inline locale_t decimal_locale() {
    static locale_t value = newlocale(LC_NUMERIC_MASK, "C", nullptr);
    if (value == nullptr) throw std::runtime_error("Could not create native decimal locale");
    return value;
}
#endif

using Cells = std::set<std::int64_t>;
using Bond = std::pair<std::int64_t, std::int64_t>;
using Bonds = std::set<Bond>;
using Pulse = queries_detail::Pulse;

inline const Value& empty_object() { static const Value value = Value::object(); return value; }
inline const Value& empty_array() { static const Value value = Value::array(); return value; }
inline const Value& get(const Value& owner, std::string_view key, const Value& fallback) {
    const auto* found = owner.find(key); return found == nullptr ? fallback : *found;
}
inline double number(const Value& owner, std::string_view key, double fallback = 0.) {
    const auto* found = owner.find(key); return found == nullptr ? fallback : found->as_number();
}
inline Value point_value(const Point& value) { return Value::array({value[0], value[1], value[2]}); }
inline Value cell_values(const std::vector<std::int64_t>& cells) {
    Value result = Value::array(); result.as_array().reserve(cells.size());
    for (auto cell : cells) result.append(cell);
    return result;
}
inline Value cell_values(const Cells& cells) { return cell_values(std::vector<std::int64_t>(cells.begin(), cells.end())); }
inline Cells cell_set(const Value& value, Pulse& pulse) {
    Cells result; for (const auto& cell : value.as_array()) { pulse(); result.insert(cell.as_int()); } return result;
}
inline Point validated_position(const Point& value) {
    for (double coordinate : value) if (!std::isfinite(coordinate))
        throw std::invalid_argument("INVALID_DEFORMATION_TARGET");
    return value;
}
inline bool soft(const Scene& scene, std::int64_t cell) {
    const auto material = scene.material(cell);
    return material == "soft_tissue" || material == "brain" || material == "tumor";
}
inline Bond bond(std::int64_t first, std::int64_t second) {
    return first < second ? Bond{first, second} : Bond{second, first};
}
inline Bonds blocked(const State& engine, Pulse& pulse) {
    Bonds result;
    for (const char* name : {"cutEdges", "transectedEdges"})
        for (const auto& item : get(engine.state, name, empty_array()).as_array()) {
            pulse();
            if (item.is_array() && item.as_array().size() == 2)
                result.insert(bond(item.at(0).as_int(), item.at(1).as_int()));
        }
    return result;
}
inline std::array<std::int64_t, 3> coordinates(const Scene& scene, std::int64_t cell) {
    scene.label_at(cell);
    const auto plane = static_cast<std::int64_t>(scene.shape[1] * scene.shape[2]);
    const auto depth = static_cast<std::int64_t>(scene.shape[2]);
    return {cell / plane, cell % plane / depth, cell % depth};
}
inline std::int64_t cell_id(const Scene& scene, const std::array<std::int64_t, 3>& index) {
    return (index[0] * static_cast<std::int64_t>(scene.shape[1]) + index[1]) *
        static_cast<std::int64_t>(scene.shape[2]) + index[2];
}
inline Point neighbor_lengths(const Scene& scene) {
    Point result;
    for (int axis = 0; axis < 3; ++axis)
        result[axis] = point_norm({scene.affine[axis], scene.affine[4 + axis], scene.affine[8 + axis]});
    return result;
}
template <class Visit>
inline void neighbors(const Scene& scene, std::int64_t cell, const Point& lengths, Visit visit) {
    const auto original = coordinates(scene, cell);
    for (int axis = 0; axis < 3; ++axis) for (int sign : {-1, 1}) {
        auto next = original; next[axis] += sign;
        if (next[axis] >= 0 && next[axis] < static_cast<std::int64_t>(scene.shape[axis]))
            visit(cell_id(scene, next), lengths[axis]);
    }
}
inline bool equal(const Value& first, const Value& second) {
    if (first.is_number() && second.is_number()) return first.as_number() == second.as_number();
    if (first.kind() != second.kind()) return false;
    switch (first.kind()) {
        case Value::Kind::Null: return true;
        case Value::Kind::Bool: return first.as_bool() == second.as_bool();
        case Value::Kind::Int: return first.as_int() == second.as_int();
        case Value::Kind::Double: return first.as_double() == second.as_double();
        case Value::Kind::String: return first.as_string() == second.as_string();
        case Value::Kind::BigInt: return first.as_bigint() == second.as_bigint();
        case Value::Kind::Array:
            if (first.as_array().size() != second.as_array().size()) return false;
            for (std::size_t i = 0; i < first.as_array().size(); ++i) if (!equal(first.at(i), second.at(i))) return false;
            return true;
        case Value::Kind::Object:
            if (first.as_object().size() != second.as_object().size()) return false;
            for (const auto& item : first.as_object()) {
                const auto* found = second.find(item.first);
                if (found == nullptr || !equal(item.second, *found)) return false;
            }
            return true;
    }
    return false;
}

// Adapted from SciPy 1.18.1 sparse/sparsetools/{coo,csr}.h (BSD-3-Clause).
// Keep the qualified stock build's sparse traversal and fused accumulation order.
struct Sparse {
    std::int32_t rows = 0, columns = 0;
    std::vector<std::int32_t> pointers{0}, indices;
    std::vector<double> values;
};
struct Entry { std::int32_t row, column; double value; };

inline Sparse from_coo(std::int32_t rows, std::int32_t columns, const std::vector<Entry>& entries, Pulse& pulse) {
    Sparse result{rows, columns, std::vector<std::int32_t>(rows + 1), {}, {}};
    result.indices.resize(entries.size()); result.values.resize(entries.size());
    for (const auto& item : entries) { pulse(); ++result.pointers[item.row + 1]; }
    for (std::int32_t i = 0; i < rows; ++i) result.pointers[i + 1] += result.pointers[i];
    auto positions = result.pointers;
    for (const auto& item : entries) {
        pulse(); const auto position = positions[item.row]++;
        result.indices[position] = item.column; result.values[position] = item.value;
    }
    bool sorted = true;
    for (std::int32_t row = 0; row < rows && sorted; ++row)
        for (auto i = result.pointers[row] + 1; i < result.pointers[row + 1]; ++i)
            if (result.indices[i] < result.indices[i - 1]) { sorted = false; break; }
    if (!sorted) {
        std::vector<std::pair<std::int32_t, double>> temporary;
        for (std::int32_t row = 0; row < rows; ++row) {
            pulse(); temporary.clear();
            for (auto i = result.pointers[row]; i < result.pointers[row + 1]; ++i)
                temporary.emplace_back(result.indices[i], result.values[i]);
            std::sort(temporary.begin(), temporary.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            auto position = result.pointers[row];
            for (const auto& item : temporary) { result.indices[position] = item.first; result.values[position++] = item.second; }
        }
    }
    std::int32_t count = 0, old_end = 0;
    for (std::int32_t row = 0; row < rows; ++row) {
        auto i = old_end; old_end = result.pointers[row + 1];
        while (i < old_end) {
            pulse(); const auto column = result.indices[i]; double value = result.values[i++];
            while (i < old_end && result.indices[i] == column) value += result.values[i++];
            result.indices[count] = column; result.values[count++] = value;
        }
        result.pointers[row + 1] = count;
    }
    result.indices.resize(count); result.values.resize(count);
    return result;
}
inline Sparse transpose(const Sparse& matrix, Pulse& pulse) {
    Sparse result{matrix.columns, matrix.rows, std::vector<std::int32_t>(matrix.columns + 1), {}, {}};
    result.indices.resize(matrix.indices.size()); result.values.resize(matrix.values.size());
    for (auto column : matrix.indices) { pulse(); ++result.pointers[column + 1]; }
    for (std::int32_t i = 0; i < matrix.columns; ++i) result.pointers[i + 1] += result.pointers[i];
    auto positions = result.pointers;
    for (std::int32_t row = 0; row < matrix.rows; ++row)
        for (auto i = matrix.pointers[row]; i < matrix.pointers[row + 1]; ++i) {
            pulse(); const auto position = positions[matrix.indices[i]]++;
            result.indices[position] = row; result.values[position] = matrix.values[i];
        }
    return result;
}
inline Sparse diagonal(const std::vector<double>& values) {
    const auto size = static_cast<std::int32_t>(values.size());
    Sparse result{size, size, std::vector<std::int32_t>(size + 1), {}, {}};
    for (std::int32_t i = 0; i < size; ++i) {
        if (values[i] != 0.) { result.indices.push_back(i); result.values.push_back(values[i]); }
        result.pointers[i + 1] = static_cast<std::int32_t>(result.indices.size());
    }
    return result;
}
inline Sparse multiply(const Sparse& a, const Sparse& b, Pulse& pulse) {
    Sparse result{a.rows, b.columns, std::vector<std::int32_t>(a.rows + 1), {}, {}};
    std::vector<std::int32_t> next(b.columns, -1); std::vector<double> sums(b.columns, 0.);
    for (std::int32_t row = 0; row < a.rows; ++row) {
        std::int32_t head = -2, length = 0;
        for (auto i = a.pointers[row]; i < a.pointers[row + 1]; ++i)
            for (auto j = b.pointers[a.indices[i]]; j < b.pointers[a.indices[i] + 1]; ++j) {
                pulse(); const auto column = b.indices[j];
                sums[column] = std::fma(a.values[i], b.values[j], sums[column]);
                if (next[column] == -1) { next[column] = head; head = column; ++length; }
            }
        for (std::int32_t i = 0; i < length; ++i) {
            pulse();
            if (sums[head] != 0.) { result.indices.push_back(head); result.values.push_back(sums[head]); }
            const auto previous = head; head = next[head]; next[previous] = -1; sums[previous] = 0.;
        }
        result.pointers[row + 1] = static_cast<std::int32_t>(result.indices.size());
    }
    return result;
}
inline Sparse add(const Sparse& a, const Sparse& b, Pulse& pulse) {
    Sparse result{a.rows, a.columns, std::vector<std::int32_t>(a.rows + 1), {}, {}};
    for (std::int32_t row = 0; row < a.rows; ++row) {
        auto i = a.pointers[row], j = b.pointers[row];
        while (i < a.pointers[row + 1] || j < b.pointers[row + 1]) {
            pulse(); std::int32_t column; double value;
            if (j == b.pointers[row + 1] || (i < a.pointers[row + 1] && a.indices[i] < b.indices[j])) {
                column = a.indices[i]; value = a.values[i++] + 0.;
            } else if (i == a.pointers[row + 1] || b.indices[j] < a.indices[i]) {
                column = b.indices[j]; value = 0. + b.values[j++];
            } else {
                column = a.indices[i]; value = a.values[i++] + b.values[j++];
            }
            if (value != 0.) { result.indices.push_back(column); result.values.push_back(value); }
        }
        result.pointers[row + 1] = static_cast<std::int32_t>(result.indices.size());
    }
    return result;
}
inline Sparse free_matrix(const Sparse& matrix, const std::vector<std::int32_t>& free, Pulse& pulse) {
    std::vector<std::int32_t> index(matrix.columns, -1);
    for (std::size_t i = 0; i < free.size(); ++i) index[free[i]] = static_cast<std::int32_t>(i);
    const auto size = static_cast<std::int32_t>(free.size());
    Sparse result{size, size, std::vector<std::int32_t>(size + 1), {}, {}};
    for (std::int32_t row = 0; row < size; ++row) {
        for (auto i = matrix.pointers[free[row]]; i < matrix.pointers[free[row] + 1]; ++i) {
            pulse(); const auto column = index[matrix.indices[i]];
            if (column >= 0) { result.indices.push_back(column); result.values.push_back(matrix.values[i]); }
        }
        result.pointers[row + 1] = static_cast<std::int32_t>(result.indices.size());
    }
    return result;
}
inline std::vector<double> matvec(const Sparse& matrix, const std::vector<double>& vector, Pulse& pulse) {
    std::vector<double> result(matrix.rows);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        double value = 0.;
        for (auto i = matrix.pointers[row]; i < matrix.pointers[row + 1]; ++i) {
            pulse(); value = std::fma(matrix.values[i], vector[matrix.indices[i]], value);
        }
        result[row] = value;
    }
    return result;
}
inline Point linear_transform(const std::array<double, 16>& matrix, const Point& value, std::size_t count) {
    Point result;
    for (int row = 0; row < 3; ++row) {
        const auto start = 4 * row;
        // Stock NumPy selects its short-vector loop for one row, BLAS otherwise.
        result[row] = count == 1 ? ((0. + value[0] * matrix[start]) + value[1] * matrix[start + 1]) + value[2] * matrix[start + 2] :
            std::fma(value[2], matrix[start + 2], std::fma(value[1], matrix[start + 1], std::fma(value[0], matrix[start], 0.)));
    }
    return result;
}
inline std::vector<Point> rest_points(const Scene& scene, const std::vector<std::int64_t>& cells, bool translation, Pulse& pulse) {
    std::vector<Point> result; result.reserve(cells.size());
    for (auto cell : cells) {
        pulse(); const auto ijk = coordinates(scene, cell);
        auto point = linear_transform(scene.affine, {static_cast<double>(ijk[0]), static_cast<double>(ijk[1]), static_cast<double>(ijk[2])}, cells.size());
        if (translation) for (int axis = 0; axis < 3; ++axis) point[axis] += scene.affine[4 * axis + 3];
        result.push_back(point);
    }
    return result;
}
inline double round_remainder(double value) {
    char buffer[512];
    std::to_chars_result text;
#if defined(__APPLE__)
    if (__builtin_available(macOS 13.3, *)) {
        if (apple_to_chars_fixed == nullptr)
            throw std::invalid_argument("Native deformation number formatting requires macOS 13.3");
        text = apple_to_chars_fixed(buffer, buffer + sizeof(buffer) - 1, value, std::chars_format::fixed, 14);
    } else throw std::invalid_argument("Native deformation number formatting requires macOS 13.3");
#else
    text = std::to_chars(buffer, buffer + sizeof(buffer) - 1, value, std::chars_format::fixed, 14);
#endif
    if (text.ec != std::errc{}) throw std::overflow_error("Deformation remainder exceeds native decimal domain");
    double result = 0.;
#if defined(__APPLE__)
    *text.ptr = '\0'; char* end = nullptr;
    result = strtod_l(buffer, &end, decimal_locale());
    if (end != text.ptr) throw std::overflow_error("Invalid native deformation remainder");
#else
    const auto parsed = std::from_chars(buffer, text.ptr, result);
    if (parsed.ec != std::errc{}) throw std::overflow_error("Invalid native deformation remainder");
#endif
    return std::max(0., result);
}

}  // namespace detail

inline detail::Cells rigid_cells(const State& engine, const StopToken& stop) {
    detail::Cells result; detail::Pulse pulse{stop};
    if (!queries_detail::revision(engine.state, "toolMechanicsRevision", 1.) &&
            !queries_detail::revision(engine.state, "toolMechanicsRevision", 2.)) return result;
    queries_detail::each_hold(engine, true, [&](const Value& hold) {
        if (!hold.contains("target")) for (const auto& cell : hold.at("cells").as_array()) { pulse(); result.insert(cell.as_int()); }
    });
    return result;
}
inline detail::Cells rigid_cells(const State& engine) { const StopToken stop; return rigid_cells(engine, stop); }

inline void begin_hold(State& engine, std::string_view key, const std::vector<std::int64_t>& requested,
        const Point& requested_position, const StopToken& stop) {
    initialize(engine); const auto p = settings(engine); detail::Pulse pulse{stop};
    auto removed = detail::cell_set(engine.state.at("removed"), pulse);
    detail::Cells selected; for (auto cell : requested) { pulse(); selected.insert(cell); }
    for (auto cell : removed) selected.erase(cell);
    const auto rigid = rigid_cells(engine, stop);
    for (auto cell : selected) if (rigid.count(cell)) throw std::invalid_argument("MATERIAL_ALREADY_HELD");
    removed.insert(rigid.begin(), rigid.end());
    std::vector<std::int64_t> cells;
    for (auto cell : selected) { pulse(); if (detail::soft(*engine.scene, cell)) cells.push_back(cell); }
    if (cells.empty()) return;
    detail::Cells other;
    for (const auto& item : engine.state.at("holds").as_object()) if (item.first != key)
        for (const auto& cell : item.second.at("cells").as_array()) { pulse(); other.insert(cell.as_int()); }
    for (auto cell : cells) if (other.count(cell)) throw std::invalid_argument("MATERIAL_ALREADY_HELD");
    const auto position = detail::validated_position(requested_position);
    auto& model = engine.state.at("deformationModel"); auto active = detail::cell_set(model.at("active"), pulse);
    for (auto cell : removed) active.erase(cell);
    const auto blocked = detail::blocked(engine, pulse);
    const auto lengths = detail::neighbor_lengths(*engine.scene);
    std::unordered_map<std::int64_t, double> distance;
    std::deque<std::int64_t> queue(cells.begin(), cells.end());
    for (auto cell : cells) distance[cell] = 0.;
    const auto max_nodes = static_cast<std::size_t>(p.at("maxNodes").as_int());
    while (!queue.empty()) {
        pulse(); const auto cell = queue.front(); queue.pop_front();
        if (!active.count(cell) && active.size() >= max_nodes) {
            if (engine.scene->properties.contains("brainContinuumV1"))
                throw std::invalid_argument("CONTINUUM_PATCH_TRUNCATED: source-cell budget cannot cover the declared physical radius");
            continue;
        }
        active.insert(cell);
        detail::neighbors(*engine.scene, cell, lengths, [&](auto next, double length) {
            pulse(); const double d = distance.at(cell) + length;
            if (d > p.at("radiusMm").as_number() || removed.count(next) || blocked.count(detail::bond(cell, next)) ||
                    !detail::soft(*engine.scene, next)) return;
            const auto found = distance.find(next);
            if (found == distance.end() || d < found->second) { distance[next] = d; queue.push_back(next); }
        });
    }
    for (auto cell : cells) if (!active.count(cell)) throw std::invalid_argument("DEFORMATION_NODE_LIMIT");
    detail::Cells anchors;
    for (auto cell : active) {
        pulse(); bool anchored = false;
        detail::neighbors(*engine.scene, cell, lengths, [&](auto next, double) {
            pulse();
            if (!active.count(next) && !removed.count(next) && detail::soft(*engine.scene, next) &&
                    !blocked.count(detail::bond(cell, next))) anchored = true;
        });
        if (anchored) anchors.insert(cell);
    }
    for (auto cell : other) anchors.erase(cell);
    for (auto cell : cells) anchors.erase(cell);
    model.set("active", detail::cell_values(active)); model.set("anchors", detail::cell_values(anchors));
    Value base = Value::object();
    for (auto cell : cells) {
        const auto name = std::to_string(cell); const auto* value = engine.state.at("deformation").find(name);
        base.as_object().emplace_back(name, value == nullptr ? detail::point_value({0., 0., 0.}) : *value);
    }
    engine.state.at("holds").set(std::string(key), Value::object({{"cells", detail::cell_values(cells)},
        {"origin", detail::point_value(position)}, {"target", detail::point_value(position)},
        {"offset", Value::array({0, 0, 0})}, {"baseDisplacements", std::move(base)}}));
    stop.check();
}
inline void set_target(State& engine, std::string_view key, const Point& position) {
    engine.state.at("holds").at(key).set("target", detail::point_value(detail::validated_position(position)));
}
inline void release(State& engine, std::string_view key) { engine.state.at("holds").erase(key); }

}  // namespace deformation

struct DeformationSystem {
    struct Edge { std::size_t first, second; double length; };
    std::shared_ptr<const Scene> scene;
    Value settings;
    std::vector<std::int64_t> ids, anchors, held;
    std::vector<std::string> materials;
    std::vector<deformation::detail::Bond> bonds;
    std::unordered_map<std::int64_t, std::size_t> index;
    std::vector<Edge> edges;
    std::vector<std::int32_t> free;
    deformation::detail::Sparse stiffness;
    std::shared_ptr<const sparse_solver::SparseFactor> factor;
};

namespace deformation {
namespace detail {

inline std::shared_ptr<const DeformationSystem> system(State& engine, const Value& p,
        const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& anchors,
        const std::vector<Bond>& bonds, Pulse& pulse) {
    const Cells retained(ids.begin(), ids.end()); Cells held_set;
    for (const auto& item : engine.state.at("holds").as_object()) if (item.second.contains("target"))
        for (const auto& cell : item.second.at("cells").as_array()) {
            pulse(); if (retained.count(cell.as_int())) held_set.insert(cell.as_int());
        }
    const std::vector<std::int64_t> held(held_set.begin(), held_set.end());
    std::vector<std::string> materials; materials.reserve(ids.size());
    for (auto cell : ids) { pulse(); materials.emplace_back(engine.scene->material(cell)); }
    const auto& cached = engine.deformation_system;
    if (cached && cached->scene == engine.scene && cached->ids == ids && cached->materials == materials &&
            cached->anchors == anchors && cached->bonds == bonds && cached->held == held && equal(cached->settings, p)) return cached;
    auto result = std::make_shared<DeformationSystem>();
    result->scene = engine.scene; result->settings = p; result->ids = ids; result->anchors = anchors;
    result->bonds = bonds; result->held = held; result->materials = std::move(materials);
    const auto count = static_cast<std::int32_t>(ids.size()); std::vector<double> stiffness(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        pulse(); result->index[ids[i]] = i;
        stiffness[i] = p.at(result->materials[i] == "tumor" ? "tumorStiffnessNPerMm" : "brainStiffnessNPerMm").as_number();
    }
    const auto rest = rest_points(*engine.scene, ids, false, pulse);
    std::vector<Entry> entries, divergence;
    entries.reserve(bonds.size() * 12); divergence.reserve(bonds.size() * 12); result->edges.reserve(bonds.size());
    for (const auto& bond : bonds) {
        pulse(); const auto i = result->index.at(bond.first), j = result->index.at(bond.second);
        Point vector; for (int axis = 0; axis < 3; ++axis) vector[axis] = rest[j][axis] - rest[i][axis];
        const double length = point_norm(vector);
        Point direction; for (int axis = 0; axis < 3; ++axis) direction[axis] = vector[axis] / length;
        const double k = (2. * stiffness[i]) * stiffness[j] / (stiffness[i] + stiffness[j]);
        result->edges.push_back({i, j, length});
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            const auto ai = static_cast<std::int32_t>(3 * i) + axis, bj = static_cast<std::int32_t>(3 * j) + axis;
            entries.insert(entries.end(), {{ai, ai, k}, {ai, bj, -k}, {bj, ai, -k}, {bj, bj, k}});
            for (auto node : {i, j}) {
                divergence.push_back({static_cast<std::int32_t>(node), ai, -direction[axis] / (2. * length)});
                divergence.push_back({static_cast<std::int32_t>(node), bj, direction[axis] / (2. * length)});
            }
        }
    }
    auto K = from_coo(3 * count, 3 * count, entries, pulse);
    const auto D = from_coo(count, 3 * count, divergence, pulse);
    std::vector<double> weights(stiffness.size());
    const double cell_length_squared = std::pow(engine.scene->voxel_mm3, 2. / 3.);
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = (stiffness[i] * p.at("bulkPenaltyRatio").as_number()) * cell_length_squared;
    // CSC multiplication reverses the two CSR views; retaining that order matters.
    const auto weighted_D = multiply(diagonal(weights), D, pulse);
    const auto bulk_columns = multiply(transpose(D, pulse), weighted_D, pulse);
    K = add(K, transpose(bulk_columns, pulse), pulse);
    std::vector<double> foundation(3 * ids.size()), diagonal_A(3 * ids.size());
    const Cells fixed(anchors.begin(), anchors.end());
    const double drag_step = p.at("dragNsPerMm").as_number() / p.at("fixedStepSeconds").as_number();
    for (std::size_t i = 0; i < ids.size(); ++i) for (int axis = 0; axis < 3; ++axis) {
        const auto k = 3 * i + axis;
        foundation[k] = stiffness[i] * .05;
        diagonal_A[k] = (held_set.count(ids[i]) ? p.at("holdStiffnessNPerMm").as_number() : 0.) + drag_step;
        if (!fixed.count(ids[i])) result->free.push_back(static_cast<std::int32_t>(k));
    }
    result->stiffness = add(K, diagonal(foundation), pulse);
    const auto A = add(result->stiffness, diagonal(diagonal_A), pulse);
    if (!result->free.empty()) {
        const auto csc = transpose(free_matrix(A, result->free, pulse), pulse);
        pulse.stop.check();
        result->factor = sparse_solver::factorize_csc(static_cast<std::int32_t>(result->free.size()), csc.pointers, csc.indices, csc.values);
        pulse.stop.check();
    }
    engine.deformation_system = result;
    return result;
}

inline std::vector<double> flatten(const std::vector<Point>& points) {
    std::vector<double> result; result.reserve(3 * points.size());
    for (const auto& point : points) result.insert(result.end(), point.begin(), point.end());
    return result;
}
inline std::unordered_map<std::string, const Value*> displacement_index(const Value& field, Pulse& pulse) {
    std::unordered_map<std::string, const Value*> result; result.reserve(field.as_object().size());
    for (const auto& item : field.as_object()) {
        pulse(); result.emplace(item.first, &item.second);
    }
    return result;
}
inline void cap(std::vector<Point>& values, double maximum, std::uint64_t* count, Pulse& pulse) {
    for (auto& value : values) {
        pulse(); const double length = point_norm(value);
        if (count != nullptr && length > maximum) ++*count;
        const double ratio = maximum / std::max(length, 1e-15);
        const double scale = std::isnan(ratio) ? ratio : std::min(1., ratio);
        for (double& component : value) component *= scale;
    }
}
inline double neighbor_ratio(const DeformationSystem& system, const std::vector<Point>& values, Pulse& pulse) {
    double result = 0.; bool first = true;
    for (const auto& edge : system.edges) {
        pulse(); Point delta;
        for (int axis = 0; axis < 3; ++axis) delta[axis] = values[edge.second][axis] - values[edge.first][axis];
        const double ratio = point_norm(delta) / edge.length;
        result = first ? ratio : std::max(result, ratio); first = false;
    }
    return result;
}
inline bool obstacle_collision(const Scene& scene, const std::vector<Point>& world,
        const Cells& removed, const std::array<bool, 65536>& blocked, Pulse& pulse) {
    bool collision = false;
    for (const auto& point : world) {
        pulse(); auto voxel = linear_transform(scene.inverse, point, world.size());
        std::array<std::int64_t, 3> ijk; bool inside = true;
        for (int axis = 0; axis < 3; ++axis) {
            const double rounded = std::nearbyint(voxel[axis] + scene.inverse[4 * axis + 3]);
            if (!(rounded >= 0. && rounded < static_cast<double>(scene.shape[axis]))) inside = false;
            ijk[axis] = inside ? static_cast<std::int64_t>(rounded) : 0;
        }
        if (!inside) { collision = true; continue; }
        const auto cell = cell_id(scene, ijk);
        if (blocked[scene.label_at(cell)] && !removed.count(cell)) collision = true;
    }
    return collision;
}
inline void diagnostics_step(Diagnostics& report, const DeformationSystem& system,
        const std::vector<Point>& u, const std::vector<Point>& old, const std::vector<Point>& forcing,
        const std::vector<double>& hold_diagonal, double drag_step, Pulse& pulse) {
    const auto vector = flatten(u), before = flatten(old), force = flatten(forcing);
    std::vector<double> delta(vector.size());
    for (std::size_t i = 0; i < delta.size(); ++i) delta[i] = vector[i] - before[i];
    auto ku = matvec(system.stiffness, vector, pulse), kold = matvec(system.stiffness, before, pulse);
    auto kdelta = matvec(system.stiffness, delta, pulse);
    for (std::size_t i = 0; i < delta.size(); ++i) {
        ku[i] += hold_diagonal[i] * vector[i]; kold[i] += hold_diagonal[i] * before[i];
        kdelta[i] += hold_diagonal[i] * delta[i];
    }
    for (auto i : system.free) {
        pulse(); const double residual = (ku[i] + drag_step * delta[i]) - force[i];
        report.maxFreeResidualN = std::max(report.maxFreeResidualN, std::abs(residual));
    }
    const double change = .5 * (sparse_solver::dot(vector, ku) - sparse_solver::dot(before, kold));
    auto half_delta = delta; for (double& value : half_delta) value *= .5;
    const double balance = ((sparse_solver::dot(force, delta) - change) - drag_step * sparse_solver::dot(delta, delta)) -
        sparse_solver::dot(half_delta, kdelta);
    report.maxAbsDiscreteEnergyBalanceNmm = std::max(report.maxAbsDiscreteEnergyBalanceNmm, std::abs(balance));
}

}  // namespace detail

inline void tick(State& engine, double dt, const StopToken& stop, Diagnostics* diagnostic = nullptr) {
    initialize(engine);
    if (!std::isfinite(dt) || dt < 0.) throw std::invalid_argument("INVALID_DEFORMATION_TIME");
    const auto p = settings(engine); auto& model = engine.state.at("deformationModel");
    detail::Pulse pulse{stop}; const auto removed = detail::cell_set(engine.state.at("removed"), pulse);
    if (diagnostic != nullptr) ++diagnostic->ticks;
    if (engine.scene->properties.contains("brainContinuumV1")) {
        continuum::tick(engine, dt, stop, diagnostic); return;
    }
    if (engine.state.contains("patientPosition"))
        position::activate_exposed_patch(engine, p.at("radiusMm").as_number(), static_cast<std::size_t>(p.at("maxNodes").as_int()), stop);
    const auto rigid = rigid_cells(engine, stop); std::vector<std::int64_t> ids;
    for (const auto& value : model.at("active").as_array()) {
        pulse(); const auto cell = value.as_int();
        if (!removed.count(cell) && !rigid.count(cell)) ids.push_back(cell);
    }
    model.set("active", detail::cell_values(ids));
    for (auto& item : engine.state.at("holds").as_object()) {
        Value retained = Value::array();
        for (const auto& cell : item.second.at("cells").as_array()) { pulse(); if (!removed.count(cell.as_int())) retained.append(cell); }
        item.second.set("cells", std::move(retained));
    }
    const detail::Cells retained(ids.begin(), ids.end());
    for (Value* field : {&engine.state.at("deformation"), &model.at("velocities")}) {
        auto& items = field->as_object();
        items.erase(std::remove_if(items.begin(), items.end(), [&](const auto& item) {
            pulse(); return !retained.count(std::stoll(item.first));
        }), items.end());
    }
    if (ids.empty()) return;
    const auto blocked = detail::blocked(engine, pulse);
    const auto lengths = detail::neighbor_lengths(*engine.scene);
    detail::Bonds bonds;
    for (auto cell : ids) detail::neighbors(*engine.scene, cell, lengths, [&](auto next, double) {
        pulse(); if (cell < next && retained.count(next) && !blocked.count({cell, next})) bonds.insert({cell, next});
    });
    std::vector<std::int64_t> anchors;
    for (auto cell : detail::cell_set(model.at("anchors"), pulse)) if (retained.count(cell)) anchors.push_back(cell);
    const auto system = detail::system(engine, p, ids, anchors, {bonds.begin(), bonds.end()}, pulse);
    const double h = p.at("fixedStepSeconds").as_number(), total = model.at("remainderSeconds").as_number() + dt;
    const double steps_double = std::floor((total + 1e-12) / h);
    if (!(steps_double >= 0. && steps_double < static_cast<double>(std::numeric_limits<std::uint64_t>::max())))
        throw std::overflow_error("Native deformation substep count exceeds uint64 range");
    const auto steps = static_cast<std::uint64_t>(steps_double);
    model.set("remainderSeconds", detail::round_remainder(total - static_cast<double>(steps) * h));
    std::vector<Point> u; u.reserve(ids.size());
    const auto current = detail::displacement_index(engine.state.at("deformation"), pulse);
    for (auto cell : ids) {
        pulse(); const auto found = current.find(std::to_string(cell));
        u.push_back(found == current.end() ? Point{} : queries_detail::point(*found->second));
    }
    std::vector<Point> targets(ids.size()); detail::Cells held;
    for (const auto& item : engine.state.at("holds").as_object()) if (item.second.contains("target")) {
        const auto& hold = item.second;
        const auto target = queries_detail::point(hold.at("target")), origin = queries_detail::point(hold.at("origin"));
        const auto base = detail::displacement_index(detail::get(hold, "baseDisplacements", detail::empty_object()), pulse);
        Point delta; for (int axis = 0; axis < 3; ++axis) delta[axis] = target[axis] - origin[axis];
        for (const auto& value : hold.at("cells").as_array()) {
            pulse(); const auto cell = value.as_int(); held.insert(cell);
            const auto found = system->index.find(cell);
            if (found != system->index.end()) {
                const auto initial = base.find(std::to_string(cell));
                const auto initial_point = initial == base.end() ? Point{} : queries_detail::point(*initial->second);
                for (int axis = 0; axis < 3; ++axis)
                    targets[found->second][axis] = delta[axis] + initial_point[axis];
            }
        }
    }
    detail::cap(targets, p.at("maxDisplacementMm").as_number(), diagnostic == nullptr ? nullptr : &diagnostic->targetCapNodeTicks, pulse);
    std::vector<Point> forcing(ids.size()); std::vector<double> hold_diagonal(3 * ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const double stiffness = held.count(ids[i]) ? p.at("holdStiffnessNPerMm").as_number() : 0.;
        for (int axis = 0; axis < 3; ++axis) { forcing[i][axis] = targets[i][axis] * stiffness; hold_diagonal[3 * i + axis] = stiffness; }
    }
    if (engine.state.contains("patientPosition")) {
        const auto gravity = position::gravity_forces_n(engine, ids, stop);
        if (gravity.size() != ids.size()) throw std::logic_error("Native gravity force count mismatch");
        for (std::size_t i = 0; i < forcing.size(); ++i) for (int axis = 0; axis < 3; ++axis) forcing[i][axis] += gravity[i][axis];
    }
    detail::cap(forcing, p.at("maxForceN").as_number(), diagnostic == nullptr ? nullptr : &diagnostic->forceCapNodeTicks, pulse);
    const auto rest = detail::rest_points(*engine.scene, ids, true, pulse);
    std::array<bool, 65536> blocking_labels{};
    for (const auto& region : engine.scene->regions.as_array()) {
        const auto& material = region.at("material").as_string();
        if (material == "bone" || material == "vessel" || material == "unknown")
            blocking_labels[static_cast<std::size_t>(region.at("id").as_int())] = true;
    }
    std::vector<Point> velocity(ids.size());
    std::vector<double> rhs(system->free.size());
    const double drag_step = p.at("dragNsPerMm").as_number() / h;
    for (std::uint64_t step = 0; step < steps; ++step) {
        stop.check(); const auto old = u;
        for (std::size_t i = 0; i < system->free.size(); ++i) {
            const auto index = system->free[i];
            rhs[i] = drag_step * u[index / 3][index % 3] + forcing[index / 3][index % 3];
        }
        const auto result = rhs.empty() ? std::vector<double>{} : sparse_solver::solve(*system->factor, rhs);
        if (result.size() != rhs.size()) throw std::logic_error("Native sparse solution count mismatch");
        u.assign(ids.size(), Point{});
        for (std::size_t i = 0; i < result.size(); ++i) u[system->free[i] / 3][system->free[i] % 3] = result[i];
        if (diagnostic != nullptr) ++diagnostic->substeps;
        detail::cap(u, p.at("maxDisplacementMm").as_number(), diagnostic == nullptr ? nullptr : &diagnostic->displacementCapNodeSteps, pulse);
        double scale = 1.; bool first_edge = true;
        for (const auto& edge : system->edges) {
            pulse(); Point delta;
            for (int axis = 0; axis < 3; ++axis) delta[axis] = u[edge.second][axis] - u[edge.first][axis];
            const double ratio = p.at("maxEdgeStrain").as_number() * edge.length / std::max(point_norm(delta), 1e-15);
            scale = first_edge ? ratio : std::min(scale, ratio); first_edge = false;
        }
        if (diagnostic != nullptr) {
            diagnostic->strainProjectionSteps += scale < 1.;
            diagnostic->maxPreStrainProjectionNeighborRatio = std::max(diagnostic->maxPreStrainProjectionNeighborRatio, detail::neighbor_ratio(*system, u, pulse));
        }
        if (scale < 1.) for (auto& value : u) for (double& component : value) component *= scale;
        bool collision = false; std::vector<Point> world(ids.size());
        for (double alpha : {.25, .5, .75, 1.}) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                pulse(); for (int axis = 0; axis < 3; ++axis)
                    world[i][axis] = (rest[i][axis] + old[i][axis]) + alpha * (u[i][axis] - old[i][axis]);
            }
            collision |= detail::obstacle_collision(*engine.scene, world, removed, blocking_labels, pulse);
        }
        if (collision) u = old;
        if (diagnostic != nullptr) {
            diagnostic->collisionRejectedSteps += collision;
            detail::diagnostics_step(*diagnostic, *system, u, old, forcing, hold_diagonal, drag_step, pulse);
        }
        for (std::size_t i = 0; i < ids.size(); ++i) for (int axis = 0; axis < 3; ++axis) velocity[i][axis] = (u[i][axis] - old[i][axis]) / h;
    }
    if (steps != 0) {
        Value displacements = Value::object(), velocities = Value::object();
        displacements.as_object().reserve(ids.size()); velocities.as_object().reserve(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            pulse(); const auto key = std::to_string(ids[i]);
            displacements.as_object().emplace_back(key, detail::point_value(u[i]));
            velocities.as_object().emplace_back(key, detail::point_value(velocity[i]));
        }
        engine.state.set("deformation", std::move(displacements)); model.set("velocities", std::move(velocities));
        const auto vector = detail::flatten(u); auto half = vector;
        for (double& value : half) value *= .5;
        model.set("elasticEnergyNmm", sparse_solver::dot(half, detail::matvec(system->stiffness, vector, pulse)));
        model.set("maxNeighborDisplacementRatio", detail::neighbor_ratio(*system, u, pulse));
        model.set("activeNodes", ids.size()); model.set("model", p.at("model"));
    }
    stop.check();
}

inline Value raster(const State& engine, const StopToken& stop) {
    detail::Pulse pulse{stop}; const auto removed = detail::cell_set(engine.state.at("removed"), pulse);
    const auto* field = engine.state.find("deformation");
    std::vector<std::pair<std::int64_t, const Value*>> ordered;
    if (field != nullptr) for (const auto& item : field->as_object()) { pulse(); ordered.emplace_back(std::stoll(item.first), &item.second); }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& first, const auto& second) { return first.first < second.first; });
    Value result = Value::array();
    for (const auto& item : ordered) {
        pulse(); if (removed.count(item.first)) continue;
        const auto original = detail::coordinates(*engine.scene, item.first);
        const auto offset = detail::linear_transform(engine.scene->inverse, queries_detail::point(*item.second), 1);
        std::array<std::int64_t, 3> ijk;
        for (int axis = 0; axis < 3; ++axis) {
            const double rounded = std::nearbyint(static_cast<double>(original[axis]) + offset[axis]);
            if (!(rounded >= 0. && rounded < static_cast<double>(engine.scene->shape[axis])))
                throw std::invalid_argument("DISPLACEMENT_OUTSIDE_GRID");
            ijk[axis] = static_cast<std::int64_t>(rounded);
        }
        result.append(Value::array({item.first, detail::cell_id(*engine.scene, ijk)}));
    }
    stop.check(); return result;
}

inline Value metrics(const State& engine, const StopToken& stop) {
    detail::Pulse pulse{stop};
    const auto& field = detail::get(engine.state, "deformation", detail::empty_object());
    const auto& model = detail::get(engine.state, "deformationModel", detail::empty_object());
    double maximum = 0.; bool first = true;
    for (const auto& item : field.as_object()) {
        pulse(); const double length = point_norm(queries_detail::point(item.second));
        maximum = first ? length : std::max(maximum, length); first = false;
    }
    return Value::object({{"brainActiveNodes", field.as_object().size()}, {"brainMaxDisplacementMm", maximum},
        {"brainElasticEnergyNmm", detail::number(model, "elasticEnergyNmm")},
        {"brainMaxNeighborDisplacementRatio", detail::number(model, "maxNeighborDisplacementRatio")}});
}

}  // namespace deformation
}  // namespace resectionlab::sim
