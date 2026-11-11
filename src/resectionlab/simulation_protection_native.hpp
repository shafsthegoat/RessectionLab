#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "simulation_queries_native.hpp"

namespace resectionlab::sim {
namespace protection_detail {

inline bool protected_cell(const Scene& scene, std::int64_t cell) {
    return std::binary_search(scene.protected_cells.begin(), scene.protected_cells.end(), cell);
}

inline std::unordered_map<std::int64_t, std::int64_t> positions(const State& engine, const StopToken& stop) {
    std::unordered_map<std::int64_t, std::int64_t> moved;
    queries_detail::Pulse pulse{stop};
    queries_detail::each_hold(engine, true, [&](const Value& hold) {
        pulse();
        if (hold.contains("target")) return;
        const auto offset = queries_detail::lattice_offset(hold.at("offset"));
        if (offset[0] == 0 && offset[1] == 0 && offset[2] == 0) return;
        for (const auto& value : hold.at("cells").as_array()) {
            pulse();
            const auto cell = value.as_int();
            if (!protected_cell(*engine.scene, cell)) continue;
            auto location = geometry_detail::coordinates(*engine.scene, cell);
            for (int axis = 0; axis < 3; ++axis) {
                const auto maximum = static_cast<std::int64_t>(engine.scene->shape[axis]) - 1;
                if (offset[axis] < -location[axis] || offset[axis] > maximum - location[axis])
                    throw std::invalid_argument("invalid entry in coordinates array");
                location[axis] += offset[axis];
            }
            moved[cell] = (location[0] * static_cast<std::int64_t>(engine.scene->shape[1]) + location[1]) *
                static_cast<std::int64_t>(engine.scene->shape[2]) + location[2];
        }
    });
    stop.check();
    return moved;
}

}  // namespace protection_detail

inline double protection_cell_radius(const Scene& scene) {
    return geometry_detail::frame(scene).cell_radius;
}

inline std::vector<std::int64_t> protected_capsule_hits(const State& engine, const Point& a,
                                                       const Point& b, double radius,
                                                       const StopToken& stop) {
    stop.check();
    std::vector<std::int64_t> result;
    if (engine.scene->protected_cells.empty()) return result;
    geometry_detail::finite(a, "Finite RAS-mm capsule and source-cell positions required");
    geometry_detail::finite(b, "Finite RAS-mm capsule and source-cell positions required");
    const auto* declared = engine.scene->protection_contract.find("clearanceMm");
    const double clearance = declared == nullptr ? 2. : declared->as_number();
    if (!std::isfinite(radius) || radius < 0. || !std::isfinite(clearance) || clearance < 0.)
        throw std::invalid_argument("Finite RAS-mm capsule and source-cell positions required");
    const auto moved = protection_detail::positions(engine, stop);
    const auto delta = geometry_detail::subtract(b, a);
    const double length2 = geometry_detail::dot(delta, delta);
    const double limit = ((radius + clearance) + protection_cell_radius(*engine.scene)) + 1e-9;
    queries_detail::Pulse pulse{stop};
    for (auto cell : engine.scene->protected_cells) {
        pulse();
        const auto current = moved.find(cell);
        const auto point = world_point(*engine.scene, current == moved.end() ? cell : current->second);
        const auto relative = geometry_detail::subtract(point, a);
        const double projection = ((0. + relative[0] * delta[0]) + relative[1] * delta[1]) +
            relative[2] * delta[2];
        const double fraction = length2 != 0. ? std::clamp(projection / length2, 0., 1.) : 0.;
        const Point residual{relative[0] - fraction * delta[0], relative[1] - fraction * delta[1],
                             relative[2] - fraction * delta[2]};
        if (point_norm(residual) <= limit) result.push_back(cell);
    }
    stop.check();
    return result;
}

inline void protected_guard_sweep(const State& engine, const std::string& key, const Point& position,
                                  const Quat& quaternion, const StopToken& stop) {
    if (engine.scene->protected_cells.empty()) return;
    for (const auto& row : engine.state.at("slots").as_object()) {
        stop.check();
        const auto& slot = row.second;
        const auto start = queries_detail::point(slot.at("position"));
        const auto orientation = queries_detail::quaternion(slot.at("quaternion"));
        const auto& instrument = tool(slot.at("instrument").as_string());
        const auto end = row.first == key ? position : start;
        const auto end_orientation = row.first == key ? quaternion : orientation;
        for (const auto& capsule : swept_envelopes(instrument, parameters(engine, slot), start, orientation,
                                                  end, end_orientation)) {
            if (!protected_capsule_hits(engine, capsule.a, capsule.b, capsule.radius, stop).empty())
                throw std::invalid_argument("PROTECTED_STRUCTURE: instrument or shaft enters the declared clearance");
        }
    }
    stop.check();
}

}  // namespace resectionlab::sim
