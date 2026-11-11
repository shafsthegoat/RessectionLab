#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "simulation_cranial_native.hpp"

namespace resectionlab::sim::guards {

inline bool posed(const State& engine, std::string_view key, const StopToken& stop) {
    stop.check();
    const auto& slot = engine.state.at("slots").at(key);
    if (slot.at("active").truthy() || slot.at("inserted").truthy() || slot.at("grip").truthy() ||
        engine.state.at("holds").contains(key)) return true;
    for (const auto& action : engine.actions.as_array()) {
        stop.check();
        if (cranial::detail::same(cranial::detail::get(action, "slot"), Value(key))) {
            const auto& op = cranial::detail::get(action, "operation");
            if (op.is_string() && cranial::detail::in_names(op.as_string(),
                    {"place", "move", "apply", "withdraw", "hold", "grasp", "bladeManeuver"})) return true;
        }
    }
    return false;
}

inline bool tool_is_exterior(State& engine, std::string_view key, const StopToken& stop) {
    stop.check();
    const auto& exit = engine.scene->cranial_access.at("exit");
    const auto axis = static_cast<std::size_t>(exit.at("axis").as_int());
    const auto side = exit.at("side").as_int();
    const double plane = exit.at("planeVoxelBoundary").as_number();
    const auto& inverse = engine.scene->inverse;
    const Point row{inverse[4 * axis], inverse[4 * axis + 1], inverse[4 * axis + 2]};
    const double margin = point_norm(row);
    const auto& slot = engine.state.at("slots").at(key);
    for (const auto& capsule : capsule_envelopes(tool(slot.at("instrument").as_string()), parameters(engine, slot),
             queries_detail::point(slot.at("position")), queries_detail::quaternion(slot.at("quaternion")))) {
        stop.check();
        const double a = geometry_detail::dot(capsule.a, row) + inverse[4 * axis + 3];
        const double b = geometry_detail::dot(capsule.b, row) + inverse[4 * axis + 3];
        if (side == -1 && std::max(a, b) + capsule.radius * margin >= plane - 1e-9) return false;
        if (side == 1 && std::min(a, b) - capsule.radius * margin <= plane + 1e-9) return false;
    }
    return true;
}

inline void require_exterior_withdrawal(State& engine, std::string_view key, const StopToken& stop) {
    if (!tool_is_exterior(engine, key, stop))
        throw std::invalid_argument("CRANIAL_ACCESS_WITHDRAWAL_NOT_AT_EXTERIOR_EXIT");
}

inline bool outside(const State& engine, const Value& workspace, const std::vector<Capsule>& segments) {
    const auto minimum = queries_detail::point(workspace.at("minimumVoxelBoundary"));
    const auto maximum = queries_detail::point(workspace.at("maximumVoxelBoundary"));
    Point margins;
    for (int axis = 0; axis < 3; ++axis) margins[axis] = point_norm(Point{
        engine.scene->inverse[4 * axis], engine.scene->inverse[4 * axis + 1], engine.scene->inverse[4 * axis + 2]});
    for (const auto& segment : segments) {
        const auto first = geometry_detail::transform(engine.scene->inverse, segment.a);
        const auto second = geometry_detail::transform(engine.scene->inverse, segment.b);
        for (int axis = 0; axis < 3; ++axis) {
            const double low = std::min(first[axis], second[axis]) - segment.radius * margins[axis];
            const double high = std::max(first[axis], second[axis]) + segment.radius * margins[axis];
            if (low < minimum[axis] - 1e-9 || high > maximum[axis] + 1e-9) return true;
        }
    }
    return false;
}

inline bool inside_entry(const Value& contract, const std::vector<Capsule>& segments) {
    const auto& entry = contract.at("entry");
    const auto center = queries_detail::point(entry.at("centerRasMm"));
    const auto normal = queries_detail::point(entry.at("normalRasUnit"));
    const auto u = queries_detail::point(entry.at("uRasUnit"));
    const Point v{normal[1] * u[2] - normal[2] * u[1], normal[2] * u[0] - normal[0] * u[2],
                  normal[0] * u[1] - normal[1] * u[0]};
    const auto& half = entry.at("halfExtentsMm");
    bool enters = false;
    for (const auto& segment : segments) {
        auto a = geometry_detail::subtract(segment.a, center);
        auto b = geometry_detail::subtract(segment.b, center);
        const double sa = geometry_detail::dot(a, normal), sb = geometry_detail::dot(b, normal);
        if (std::max(sa, sb) + segment.radius <= 1e-9) continue;
        enters = true;
        if (std::max(sa, sb) + segment.radius > entry.at("maximumInwardMm").as_number() + 1e-9)
            throw std::invalid_argument("CRANIAL_ACCESS_DEPTH_OUTSIDE");
        if (sa < -segment.radius && sb > sa) {
            const double fraction = (-segment.radius - sa) / (sb - sa);
            for (int axis = 0; axis < 3; ++axis) a[axis] = a[axis] + fraction * (b[axis] - a[axis]);
        } else if (sb < -segment.radius && sa > sb) {
            const double fraction = (-segment.radius - sb) / (sa - sb);
            for (int axis = 0; axis < 3; ++axis) b[axis] = b[axis] + fraction * (a[axis] - b[axis]);
        }
        for (const auto& point : {a, b}) {
            if (std::abs(geometry_detail::dot(point, u)) + segment.radius > half.at(0).as_number() + 1e-9 ||
                std::abs(geometry_detail::dot(point, v)) + segment.radius > half.at(1).as_number() + 1e-9)
                throw std::invalid_argument("CRANIAL_ACCESS_WINDOW_OUTSIDE");
        }
    }
    return enters;
}

inline void check_transition(State& engine, std::string_view key, std::string_view operation,
                             const Point& position, const Quat& quaternion, const StopToken& stop) {
    stop.check();
    const auto& contract = engine.scene->cranial_access;
    if (contract.is_null()) return;
    const bool milling = cranial::is_milling(contract);
    if (milling && (!queries_detail::revision(engine.state, "toolMechanicsRevision", 2.) ||
                    !queries_detail::revision(engine.state, "physicalClockRevision", 1.)))
        throw std::invalid_argument("CRANIAL_ACCESS_MILLING_STOCK_AND_CLOCK_REQUIRED");
    const bool evacuated = !milling && bone_transport::aperture_evacuated(engine, stop);
    const auto& entry = contract.at("entry");
    const auto normal = queries_detail::point(entry.at("normalRasUnit"));
    const auto center = queries_detail::point(entry.at("centerRasMm"));
    for (const auto& hand : engine.state.at("slots").as_object()) {
        stop.check();
        const auto& slot = hand.second;
        if (!posed(engine, hand.first, stop) && (hand.first != key ||
            cranial::detail::in_names(operation, {"exchange", "inspect", "deactivate"}))) continue;
        const auto& name = slot.at("instrument").as_string();
        const auto& instrument = tool(name);
        const auto params = parameters(engine, slot);
        const auto end = hand.first == key ? position : queries_detail::point(slot.at("position"));
        const auto q = hand.first == key ? quaternion : queries_detail::quaternion(slot.at("quaternion"));
        const auto segments = hand.first == key
            ? swept_envelopes(instrument, params, queries_detail::point(slot.at("position")),
                queries_detail::quaternion(slot.at("quaternion")), end, q)
            : capsule_envelopes(instrument, params, end, q);
        if (outside(engine, contract.at("exit"), segments))
            throw std::invalid_argument("CRANIAL_ACCESS_OUTSIDE_REVIEWED_WORKSPACE");
        if (!inside_entry(contract, segments)) continue;
        cranial::require_access_evidence(engine, stop);
        const bool preparation = milling ? cranial::detail::in_names(name, {"cranial_drill", "frazier_suction"})
            : cranial::detail::in_names(name, {"cranial_drill", "craniotome", "periosteal_elevator", "bone_flap_holder", "dissector"});
        if (milling && !preparation) throw std::invalid_argument("CRANIAL_ACCESS_MILLING_TREATMENT_UNQUALIFIED");
        const auto axis = apply_rotation(rotation_matrix(normalized_quaternion(q)), Point{0., 0., 1.});
        const double radians = entry.at("maximumAngleDegrees").as_number() * (3.14159265358979323846 / 180.);
        if (geometry_detail::dot(axis, normal) < std::cos(radians) - 1e-9)
            throw std::invalid_argument("CRANIAL_ACCESS_ATTITUDE_OUTSIDE");
        if (!evacuated && !preparation) throw std::invalid_argument("CRANIAL_ACCESS_REQUIRES_EVACUATED_FLAP");
        if (!evacuated && preparation) {
            const auto& window = contract.at(milling ? "apertureCells" : "windowCells");
            const double inner = cranial::detail::inner_boundary(engine, window, stop);
            for (const auto& segment : segments) {
                const double depth = std::max(geometry_detail::dot(geometry_detail::subtract(segment.a, center), normal),
                    geometry_detail::dot(geometry_detail::subtract(segment.b, center), normal)) + segment.radius;
                if (depth > inner + 1e-9)
                    throw std::invalid_argument("CRANIAL_ACCESS_BEYOND_INNER_TABLE_BEFORE_EVACUATION");
            }
        }
    }
    stop.check();
}

inline void check_working(State& engine, std::string_view key,
                          const std::vector<std::int64_t>& working, const StopToken& stop) {
    stop.check();
    const auto& contract = engine.scene->cranial_access;
    if (contract.is_null()) return;
    const bool milling = cranial::is_milling(contract);
    for (const auto& hand : engine.state.at("slots").as_object()) {
        stop.check();
        const auto& slot = hand.second;
        if (hand.first != key && !slot.at("active").truthy()) continue;
        const auto& name = slot.at("instrument").as_string();
        const bool preparation = cranial::detail::in_names(name, {"cranial_drill", "frazier_suction"});
        if (milling && preparation) {
            const auto params = parameters(engine, slot);
            if (params.at("radiusMm").as_number() != tool(name).radius || params.at("depthMm").as_number() != 2.)
                throw std::invalid_argument("CRANIAL_ACCESS_MILLING_STOCK_GEOMETRY_REQUIRED");
        }
        if (!cranial::detail::in_names(name, {"cranial_drill", "craniotome", "bone_flap_holder"}) &&
                !(milling && name == "frazier_suction")) continue;
        auto contacts = hand.first == key ? working : std::vector<std::int64_t>{};
        if (hand.first != key) {
            contacts = cranial::detail::contacts(engine, slot, capsule_envelopes(tool(name), parameters(engine, slot),
                queries_detail::point(slot.at("position")), queries_detail::quaternion(slot.at("quaternion"))), stop);
        }
        if (milling && name == "frazier_suction") {
            if (!contacts.empty()) throw std::invalid_argument("CRANIAL_ACCESS_MILLING_SUCTION_REQUIRES_CLEARED_VOID");
            continue;
        }
        static const Value empty = Value::array();
        const auto& allowed = milling ? (name == "bone_flap_holder" ? empty : contract.at("apertureCells"))
            : contract.at(name == "bone_flap_holder" ? "flapCells" : "kerfCells");
        queries_detail::Pulse pulse{stop};
        for (auto cell : contacts) {
            pulse();
            if (!cranial::detail::has_cell(allowed, cell))
                throw std::invalid_argument("CRANIAL_ACCESS_BONE_WORK_OUTSIDE_DECLARED_SCOPE");
        }
    }
    stop.check();
}

inline void enforce(State& engine, std::string_view key, std::string_view operation,
                     const Point& position, const Quat& quaternion,
                     const std::vector<std::int64_t>& working, const StopToken& stop) {
    stop.check();
    if (cranial::enforce(engine, key, operation, position, quaternion, working, stop)) return;
    if (engine.scene->cranial_access.is_null()) return;
    check_transition(engine, key, operation, position, quaternion, stop);
    check_working(engine, key, working, stop);
}

inline void enforce(State& engine, const std::string& key, const std::string& operation,
                     const Point& position, const Quat& quaternion,
                     const std::set<std::int64_t>& working, const StopToken& stop) {
    enforce(engine, std::string_view(key), std::string_view(operation), position, quaternion,
        std::vector<std::int64_t>(working.begin(), working.end()), stop);
}

}  // namespace resectionlab::sim::guards
