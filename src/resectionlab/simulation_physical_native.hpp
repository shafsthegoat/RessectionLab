#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_effects_native.hpp"
#include "simulation_protection_native.hpp"
#include "simulation_stats_native.hpp"

namespace resectionlab::sim {

namespace cusa_v3 {
inline const Value* profile(const State&);
inline std::pair<Point, std::vector<std::int64_t>> contact_limited_pose(
    const State&, const Point&, const Point&, double, const StopToken&);
inline Value apply(State&, const std::vector<std::int64_t>&, double, const Value&, const Point&, bool,
                   const StopToken&);
}
namespace bone_transport {
inline void validate_hold(State&, const std::string&, const StopToken&);
inline Value grasp(State&, const std::string&, const std::vector<std::int64_t>&, const Point&, const Quat&,
                   const StopToken&);
inline void check_tool_sweep(State&, const std::string&, const Value&, const std::string&, const StopToken&);
inline void check_sweep(State&, const std::string&, const std::array<std::int64_t, 3>&, const Value&,
                        const StopToken&);
inline void check_detached(State&, const StopToken&);
}
namespace membrane_transport {
inline void validate_hold(State&, const std::string&, const StopToken&);
inline Value grasp(State&, const std::string&, const std::vector<std::int64_t>&, const Point&, const Quat&,
                   const StopToken&);
inline void check_tool_sweep(State&, const std::string&, const Value&, const std::string&, const StopToken&);
inline void check_sweep(State&, const std::string&, const std::array<std::int64_t, 3>&, const Value&,
                        const StopToken&);
inline void check_detached(State&, const StopToken&);
}
namespace guards {
inline void enforce(State&, const std::string&, const std::string&, const Point&, const Quat&,
                    const std::set<std::int64_t>&, const StopToken&);
}
namespace cranial { inline void validate_effects(State&, const StopToken&); }

namespace physical_detail {

using Cells = std::set<std::int64_t>;
using Exposure = std::map<std::pair<std::string, std::string>, double>;

inline bool revised(const State& engine) {
    return queries_detail::revision(engine.state, "toolMechanicsRevision", 1.) ||
        queries_detail::revision(engine.state, "toolMechanicsRevision", 2.);
}
inline bool revised_contact(const State& engine, const Tool& instrument) {
    return revised(engine) && instrument.effect != "remove" && instrument.effect != "suction" &&
        instrument.effect != "irrigate" && instrument.effect != "probe";
}
inline bool exempt_operation(const std::string& operation) {
    return operation == "withdraw" || operation == "release" || operation == "deactivate" ||
        operation == "inspect" || operation == "exchange";
}
inline Value point_value(const Point& point) { return Value::array({point[0], point[1], point[2]}); }
inline Value quaternion_value(const Quat& quaternion) {
    return Value::array({quaternion[0], quaternion[1], quaternion[2], quaternion[3]});
}
inline Value pose_value(const Point& point, const Quat& quaternion) {
    return Value::object({{"position", point_value(point)}, {"quaternion", quaternion_value(quaternion)}});
}
inline Cells cell_set(const Value& value, const StopToken& stop) {
    Cells result;
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : value.as_array()) { pulse(); result.insert(cell.as_int()); }
    return result;
}
inline Cells cell_set(const std::vector<std::int64_t>& value, const StopToken& stop) {
    Cells result;
    queries_detail::Pulse pulse{stop};
    for (auto cell : value) { pulse(); result.insert(cell); }
    return result;
}
inline void add_cells(Cells& result, const std::vector<std::int64_t>& cells, const StopToken& stop) {
    queries_detail::Pulse pulse{stop};
    for (auto cell : cells) { pulse(); result.insert(cell); }
}
inline std::vector<std::int64_t> sorted_cells(const Cells& cells, const StopToken& stop) {
    std::vector<std::int64_t> result;
    result.reserve(cells.size());
    queries_detail::Pulse pulse{stop};
    for (auto cell : cells) { pulse(); result.push_back(cell); }
    return result;
}
inline bool difference(const Cells& cells, const Cells& other, const StopToken& stop) {
    queries_detail::Pulse pulse{stop};
    for (auto cell : cells) { pulse(); if (other.find(cell) == other.end()) return true; }
    return false;
}
inline Cells held_cells(const State& engine, const std::string& key, const StopToken& stop) {
    const auto* hold = engine.state.at("holds").find(key);
    return hold == nullptr ? Cells{} : cell_set(hold->at("cells"), stop);
}
inline Cells shaft_held_cells(const State& engine, const std::string& key, const StopToken& stop) {
    const auto& name = engine.state.at("slots").at(key).at("instrument").as_string();
    return name == "bone_flap_holder" || name == "membrane_forceps" ? Cells{} : held_cells(engine, key, stop);
}
inline std::vector<std::string> keys(const Value& mapping) {
    std::vector<std::string> result;
    result.reserve(mapping.as_object().size());
    for (const auto& item : mapping.as_object()) result.push_back(item.first);
    return result;
}
inline bool has_kind(const State& engine, const std::string& key, const char* kind) {
    const auto* hold = engine.state.at("holds").find(key);
    return hold != nullptr && legacy_effects::text_is(*hold, "kind", kind);
}
inline std::vector<std::int64_t> eligible(const State& engine, const Tool& instrument,
                                         const std::vector<std::int64_t>& cells, const StopToken& stop) {
    std::vector<std::int64_t> result;
    queries_detail::Pulse pulse{stop};
    for (auto cell : cells) { pulse(); if (instrument.accepts(engine.scene->material(cell))) result.push_back(cell); }
    return result;
}
inline void append_events(Value& destination, Value source) {
    for (auto& event : source.as_array()) destination.append(std::move(event));
}

inline Value effect(State& engine, const std::string& key, const std::string& operation,
                     const std::vector<std::int64_t>& cells, const Point& position,
                     const Quat& quaternion, double dt, const StopToken& stop) {
    const auto name = engine.state.at("slots").at(key).at("instrument").as_string();
    if (name == "cusa" && cusa_v3::profile(engine) != nullptr) {
        const auto params = parameters(engine, engine.state.at("slots").at(key));
        const auto contacts = touched(engine, position, position, params.at("radiusMm").as_number(), stop);
        return cusa_v3::apply(engine, contacts, dt, params, position, true, stop);
    }
    if (name == "bone_flap_holder") return bone_transport::grasp(engine, key, cells, position, quaternion, stop);
    if (name == "membrane_forceps") return membrane_transport::grasp(engine, key, cells, position, quaternion, stop);
    return apply_legacy_effect(engine, key, operation, cells, position, quaternion, dt, stop);
}

inline Value shared_clock(State& engine, const std::string& key, const std::string& operation,
                           const Point& position, const Quat& quaternion, bool active, double dt,
                           const Cells& working, const std::vector<std::int64_t>& acting_eligible,
                           const StopToken& stop) {
    struct Record {
        std::string hand;
        const Tool* instrument;
        Point position;
        Quat quaternion;
        bool effective;
        Cells contacts;
        std::vector<std::int64_t> effect_cells;
    };
    std::vector<Record> records;
    queries_detail::Pulse pulse{stop};
    auto hands = keys(engine.state.at("slots"));
    std::sort(hands.begin(), hands.end());
    for (const auto& hand : hands) {
        stop.check();
        const auto slot = engine.state.at("slots").at(hand);
        const bool acting = hand == key;
        const bool effective = acting ? active : slot.at("active").truthy();
        if (!(effective || slot.at("inserted").truthy() ||
                (acting && operation != "inspect" && operation != "place" && operation != "exchange"))) continue;
        const auto& instrument = tool(slot.at("instrument").as_string());
        const auto params = parameters(engine, slot);
        const auto p = acting ? position : queries_detail::point(slot.at("position"));
        const auto q = acting ? quaternion : queries_detail::quaternion(slot.at("quaternion"));
        Cells contacts;
        std::vector<std::int64_t> effect_cells;
        if (acting) {
            for (auto cell : working) { pulse(); contacts.insert(cell); }
            effect_cells = acting_eligible;
        }
        else {
            const auto tip = touched(engine, p, p, params.at("radiusMm").as_number(), stop);
            const bool revised = revised_contact(engine, instrument);
            if (!revised) contacts = cell_set(tip, stop);
            for (const auto& capsule : capsule_envelopes(instrument, params, p, q)) {
                const auto cells = touched(engine, capsule.a, capsule.b, capsule.radius, stop);
                if (capsule.part == "shaft") {
                    if (difference(cell_set(cells, stop), shaft_held_cells(engine, hand, stop), stop))
                        throw std::invalid_argument("SHAFT_COLLISION:" + hand);
                } else add_cells(contacts, cells, stop);
            }
            effect_cells = revised ? sorted_cells(contacts, stop) : tip;
        }
        if (effective) for (auto cell : contacts) {
            pulse();
            if (!instrument.materials[0].empty() && !instrument.accepts(engine.scene->material(cell)))
                throw std::invalid_argument("ACTIVE_TOOL_UNSUPPORTED_MATERIAL:" + hand);
        }
        effect_cells = eligible(engine, instrument, effect_cells, stop);
        records.push_back({hand, &instrument, p, q, effective, std::move(contacts), std::move(effect_cells)});
    }
    Value events = Value::array();
    Cells contacts, vessels;
    const bool milling = legacy_effects::text_is(engine.scene->cranial_access, "method", "milling");
    for (const auto& record : records) {
        stop.check();
        if (record.effective) {
            const std::string op = record.hand == key ? operation : "apply";
            auto effects = effect(engine, record.hand, op, record.effect_cells, record.position,
                                  record.quaternion, dt, stop);
            if (milling) for (auto& event : effects.as_array()) {
                event.set("sourceSlot", record.hand);
                event.set("instrumentId", engine.state.at("slots").at(record.hand).at("instrument"));
                event.set("effectOperation", op);
            }
            append_events(events, std::move(effects));
        }
        for (auto cell : record.contacts) {
            pulse();
            const std::string material(engine.scene->material(cell));
            legacy_effects::add_number(engine.state.at("materialExposure"), material, dt);
            if (material == "vessel") vessels.insert(cell);
        }
        if (record.instrument->effect == "elevate" || record.instrument->effect == "dissect")
            for (auto cell : record.contacts) { pulse(); contacts.insert(cell); }
    }
    auto encounters = cell_set(engine.state.at("vesselEncounters"), stop);
    for (auto cell : vessels) { pulse(); encounters.insert(cell); }
    engine.state.set("vesselEncounters", legacy_effects::integers(encounters));
    for (const auto& hold : engine.state.at("holds").as_object())
        for (const auto& cell : hold.second.at("cells").as_array()) { pulse(); contacts.insert(cell.as_int()); }
    for (const auto& cell : engine.state.at("removed").as_array()) { pulse(); contacts.erase(cell.as_int()); }
    for (auto cell : contacts) { pulse(); legacy_effects::add_number(engine.state.at("contact"), std::to_string(cell), dt); }
    stop.check();
    return events;
}

inline std::array<std::int64_t, 3> moved_offset(const State& engine, const Value& hold, const Point& position) {
    const auto delta = geometry_detail::subtract(position, queries_detail::point(hold.at("origin")));
    const auto* base_value = hold.find("baseOffset");
    const auto base = base_value == nullptr ? std::array<std::int64_t, 3>{0, 0, 0} :
        queries_detail::lattice_offset(*base_value);
    std::array<std::int64_t, 3> result;
    for (int axis = 0; axis < 3; ++axis) {
        const auto row = axis * 4;
        const double displacement = ((0. + engine.scene->inverse[row] * delta[0]) +
            engine.scene->inverse[row + 1] * delta[1]) + engine.scene->inverse[row + 2] * delta[2];
        const double rounded = std::nearbyint(displacement);
        const auto integer = rounded >= -0x1p63 && rounded < 0x1p63 ? static_cast<std::int64_t>(rounded) :
            std::numeric_limits<std::int64_t>::min();
        const auto sum = static_cast<std::uint64_t>(integer) + static_cast<std::uint64_t>(base[axis]);
        std::memcpy(&result[axis], &sum, sizeof(sum));
    }
    return result;
}

inline Exposure exposure(const State& engine, const StopToken& stop) {
    Exposure result;
    queries_detail::Pulse pulse{stop};
    const auto add = [&](const char* name, const Value* field) {
        if (field == nullptr) return;
        for (const auto& value : field->as_object()) {
            pulse();
            if (protection_detail::protected_cell(*engine.scene, std::stoll(value.first)))
                result[{name, value.first}] = value.second.as_number();
        }
    };
    add("thermal", engine.state.find("thermal"));
    const auto* bipolar = engine.state.find("bipolarThermalV1");
    add("bipolarTemperatureRise", bipolar ? bipolar->find("temperatureRiseK") : nullptr);
    add("bipolarTemperatureIntegral", bipolar ? bipolar->find("excessTemperatureIntegralKSeconds") : nullptr);
    const auto* drilling = engine.state.find("drillingV2");
    add("boneTemperature", drilling ? drilling->find("temperatureC") : nullptr);
    add("boneDose", drilling ? drilling->find("thermalDoseCSeconds") : nullptr);
    return result;
}

inline void protected_after(const State& engine, const Exposure& before, const StopToken& stop) {
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : engine.state.at("removed").as_array()) {
        pulse();
        if (protection_detail::protected_cell(*engine.scene, cell.as_int()))
            throw std::invalid_argument("PROTECTED_STRUCTURE: indirect removal or cutting is blocked");
    }
    for (const char* name : {"cutEdges", "transectedEdges"}) if (const auto* edges = engine.state.find(name))
        for (const auto& edge : edges->as_array()) for (const auto& cell : edge.as_array()) {
            pulse();
            if (protection_detail::protected_cell(*engine.scene, cell.as_int()))
                throw std::invalid_argument("PROTECTED_STRUCTURE: indirect removal or cutting is blocked");
        }
    const auto* config = engine.scene->guard_config.find("critical_max_displacement_mm");
    const double limit = (config ? config->as_number() : 0.) + 1e-9;
    if (const auto* deformation = engine.state.find("deformation")) for (const auto& item : deformation->as_object()) {
        pulse();
        if (protection_detail::protected_cell(*engine.scene, std::stoll(item.first)) &&
                point_norm(queries_detail::point(item.second)) > limit)
            throw std::invalid_argument("PROTECTED_STRUCTURE: indirect deformation exceeds the declared limit");
    }
    for (const auto& row : engine.state.at("holds").as_object()) {
        pulse();
        const auto& hold = row.second;
        if (hold.contains("target")) continue;
        bool protected_membership = false;
        for (const auto& cell : hold.at("cells").as_array()) {
            pulse();
            protected_membership = protected_membership || protection_detail::protected_cell(*engine.scene, cell.as_int());
        }
        if (protected_membership && point_norm(queries_detail::offset_world(*engine.scene,
                queries_detail::lattice_offset(hold.at("offset")))) > limit)
            throw std::invalid_argument("PROTECTED_STRUCTURE: grasp displaces protected fragment membership");
    }
    for (const auto& item : exposure(engine, stop)) {
        pulse();
        const auto previous = before.find(item.first);
        const double baseline = previous == before.end() ? (item.first.first == "boneTemperature" ? 37. : 0.) : previous->second;
        if (item.second > baseline)
            throw std::invalid_argument("PROTECTED_STRUCTURE: indirect thermal exposure is blocked");
    }
    if (const auto* vessels = engine.state.find("vesselsV2")) if (const auto* faces = vessels->find("faces"))
        for (const auto& item : faces->as_object()) {
            pulse();
            if (item.second.at("damage").as_number() <= 0.) continue;
            for (const auto& cell : item.second.at("cells").as_array()) {
                pulse();
                if (protection_detail::protected_cell(*engine.scene, cell.as_int()))
                    throw std::invalid_argument("PROTECTED_STRUCTURE: indirect vessel injury is blocked");
            }
        }
    if (const auto* graph = engine.state.find("vascularGraphV1"); graph && graph->truthy())
        for (const auto& edge : engine.scene->properties.at("vascularGraphV1").at("edges").as_array()) {
            pulse();
            const auto& row = graph->at("edges").at(edge.at("id").as_string());
            if (!row.at("cutFace").is_null() || row.at("compressionFraction").as_number() > 0. ||
                    row.at("occlusionFraction").as_number() > 0.)
                for (const auto& cell : edge.at("sourceCells").as_array()) {
                    pulse();
                    if (protection_detail::protected_cell(*engine.scene, cell.as_int()))
                        throw std::invalid_argument("PROTECTED_STRUCTURE: indirect vascular conductance change is blocked");
                }
        }
    stop.check();
}

}  // namespace physical_detail

// Admission excludes V2, continuum, planar and surface-contact profiles before any transition.
inline Value physical_substep(State& engine, std::string_view key_name, std::string_view operation_name,
                               Point position, Quat quaternion, bool active, double dt,
                               const StopToken& stop) {
    execution_stats.physical_substeps.fetch_add(1, std::memory_order_relaxed);
    using namespace physical_detail;
    const std::string key(key_name), operation(operation_name);
    stop.check();
    queries_detail::Pulse pulse{stop};
    const auto requested_position = position;
    const bool protected_state = !engine.scene->protected_cells.empty();
    Exposure previous_exposure;
    if (protected_state) {
        protected_guard_sweep(engine, key, position, quaternion, stop);
        previous_exposure = exposure(engine, stop);
    }
    const Value slot = engine.state.at("slots").at(key);
    const auto& instrument = tool(slot.at("instrument").as_string());
    const Value params = parameters(engine, slot);
    const auto previous_quaternion = queries_detail::quaternion(slot.at("quaternion"));
    if (has_kind(engine, key, "membrane-patch-v1") && rotation_angle(previous_quaternion, quaternion) > 1e-8)
        throw std::invalid_argument("MEMBRANE_PATCH_ROTATION_UNSUPPORTED");
    if (has_kind(engine, key, "bone-flap-v1") && rotation_angle(previous_quaternion, quaternion) > 1e-8)
        throw std::invalid_argument("BONE_FLAP_ROTATION_UNSUPPORTED: translate the source-bound fragment");
    (void)normalized_quaternion(quaternion);
    const auto previous = queries_detail::point(slot.at("position"));
    const bool cusa = instrument.id == "cusa" && cusa_v3::profile(engine) != nullptr;
    if (cusa && active) position = cusa_v3::contact_limited_pose(engine, previous, position,
        params.at("radiusMm").as_number(), stop).first;
    auto tip = touched(engine, previous, position, params.at("radiusMm").as_number(), stop);
    const auto end_pose = pose_value(position, quaternion);
    const auto swept = swept_envelopes(instrument, params, previous, previous_quaternion, position, quaternion);
    if (!engine.scene->cranial_access.is_null()) bone_transport::check_tool_sweep(engine, key, end_pose, operation, stop);
    if (!engine.scene->cranial_stages.is_null()) membrane_transport::check_tool_sweep(engine, key, end_pose, operation, stop);
    const bool revised_work = revised_contact(engine, instrument);
    Cells shafts, working = revised_work ? Cells{} : cell_set(tip, stop);
    PositiveIntSet working_order(&stop);
    if (!revised_work) working_order.update(tip.begin(), tip.end());
    const auto covers = touched_many(engine, swept, stop);
    for (std::size_t index = 0; index < swept.size(); ++index) {
        const auto& capsule = swept[index];
        const auto& cells = covers[index];
        add_cells(capsule.part == "shaft" ? shafts : working, cells, stop);
        if (capsule.part != "shaft") working_order.update(cells.begin(), cells.end());
    }
    if (revised_work) tip = sorted_cells(working, stop);
    if (!engine.scene->cranial_access.is_null()) guards::enforce(engine, key, operation, position, quaternion, working, stop);
    const auto held = held_cells(engine, key, stop);
    if (difference(shafts, shaft_held_cells(engine, key, stop), stop)) throw std::invalid_argument("SHAFT_COLLISION");
    if (cusa && active && !queries_detail::revision(engine.state, "toolMechanicsRevision", 2.)) {
        const auto tip_contacts = cell_set(touched(engine, position, position, params.at("radiusMm").as_number(), stop), stop);
        for (auto cell : working) {
            pulse();
            if (!held.count(cell) && !tip_contacts.count(cell))
                throw std::invalid_argument("RETAINED_TISSUE_COLLISION: CUSA body requires a cleared corridor");
        }
    }
    for (const auto& other_key : keys(engine.state.at("slots"))) {
        stop.check();
        if (other_key == key) continue;
        const auto other = engine.state.at("slots").at(other_key);
        const auto& other_tool = tool(other.at("instrument").as_string());
        if (pair_conflict(swept, capsule_envelopes(other_tool, parameters(engine, other),
                queries_detail::point(other.at("position")), queries_detail::quaternion(other.at("quaternion"))), stop))
            throw std::invalid_argument("TOOL_COLLISION:" + other_key);
    }
    if (instrument.effect != "suction" && instrument.effect != "irrigate" &&
            difference(working, held, stop) && !exempt_operation(operation)) {
        for (auto cell : working) {
            pulse();
            if (!held.count(cell) && !instrument.accepts(engine.scene->material(cell)))
                throw std::invalid_argument("UNSUPPORTED_MATERIAL");
        }
        if (!active && !engine.state.at("holds").contains(key)) throw std::invalid_argument("RETAINED_TISSUE_COLLISION");
    }
    const auto compatible = eligible(engine, instrument, revised_work ? sorted_cells(working, stop) : tip, stop);
    const bool field_action = active && (instrument.effect == "suction" || instrument.effect == "irrigate");
    if (field_action && point_norm(geometry_detail::subtract(position, previous)) > 1e-8) {
        const auto current = cell_set(touched(engine, position, position, instrument.shaft_radius, stop), stop);
        const auto before = cell_set(touched(engine, previous, previous, instrument.shaft_radius, stop), stop);
        if (difference(current, before, stop)) throw std::invalid_argument("RETAINED_TISSUE_COLLISION");
    }
    if (operation == "withdraw") {
        Cells before = cell_set(touched(engine, previous, previous, params.at("radiusMm").as_number(), stop), stop);
        Cells current = cell_set(tip, stop);
        if (revised_work) {
            before.clear(); current.clear();
            for (const auto& capsule : capsule_envelopes(instrument, params, previous, previous_quaternion))
                if (capsule.part != "shaft") add_cells(before, touched(engine, capsule.a, capsule.b, capsule.radius, stop), stop);
            for (const auto& capsule : capsule_envelopes(instrument, params, position, quaternion))
                if (capsule.part != "shaft") add_cells(current, touched(engine, capsule.a, capsule.b, capsule.radius, stop), stop);
        }
        if (difference(current, before, stop)) throw std::invalid_argument("WITHDRAWAL_PATH_BLOCKED");
    }
    bool unsupported_tip = false;
    for (auto cell : tip) { pulse(); unsupported_tip = unsupported_tip || !instrument.accepts(engine.scene->material(cell)); }
    if (!tip.empty() && !field_action && ((!active && !engine.state.at("holds").contains(key)) || unsupported_tip) &&
            !exempt_operation(operation))
        throw std::invalid_argument(active ? "UNSUPPORTED_MATERIAL" : "RETAINED_TISSUE_COLLISION");
    const bool shared = queries_detail::revision(engine.state, "physicalClockRevision", 1.);
    Value events = Value::array();
    if (shared) events = shared_clock(engine, key, operation, position, quaternion, active, dt, working, compatible, stop);
    else if (active) events = effect(engine, key, operation, compatible, position, quaternion, dt, stop);
    if (engine.state.at("holds").contains(key)) (void)occupancy(engine, stop);
    if (engine.state.at("holds").contains(key) && (operation == "move" || has_kind(engine, key, "bone-flap-v1") ||
            has_kind(engine, key, "membrane-patch-v1"))) {
        if (instrument.id == "bone_flap_holder") bone_transport::validate_hold(engine, key, stop);
        else if (instrument.id == "membrane_forceps") membrane_transport::validate_hold(engine, key, stop);
        else validate_legacy_hold_motion(engine, key, stop);
        if (revised(engine) && rotation_angle(previous_quaternion, quaternion) > 1e-8)
            throw std::invalid_argument("FRAGMENT_ROTATION_UNSUPPORTED: translate the source-bound fragment");
        const auto offset = moved_offset(engine, engine.state.at("holds").at(key), position);
        if (instrument.id == "bone_flap_holder") bone_transport::check_sweep(engine, key, offset, end_pose, stop);
        if (instrument.id == "membrane_forceps") membrane_transport::check_sweep(engine, key, offset, end_pose, stop);
        engine.state.at("holds").at(key).set("offset", Value::array({offset[0], offset[1], offset[2]}));
        (void)occupancy(engine, stop);
    }
    for (const auto& other_key : keys(engine.state.at("slots"))) {
        stop.check();
        const auto other = engine.state.at("slots").at(other_key);
        if (shared || other_key == key || !other.at("active").truthy()) continue;
        const auto& other_tool = tool(other.at("instrument").as_string());
        const auto other_params = parameters(engine, other);
        const auto p = queries_detail::point(other.at("position"));
        const auto q = queries_detail::quaternion(other.at("quaternion"));
        auto cells = touched(engine, p, p, other_params.at("radiusMm").as_number(), stop);
        if (revised_contact(engine, other_tool)) {
            Cells contacts;
            for (const auto& capsule : capsule_envelopes(other_tool, other_params, p, q))
                if (capsule.part != "shaft") add_cells(contacts, touched(engine, capsule.a, capsule.b, capsule.radius, stop), stop);
            cells = sorted_cells(contacts, stop);
        }
        auto accepted = eligible(engine, other_tool, cells, stop);
        if (!other_tool.materials[0].empty() && accepted.size() != cells.size())
            throw std::invalid_argument("OTHER_ACTIVE_TOOL_UNSUPPORTED_MATERIAL:" + other_key);
        append_events(events, effect(engine, other_key, "apply", accepted, p, q, dt, stop));
    }
    if (!shared) {
        auto encounters = cell_set(engine.state.at("vesselEncounters"), stop);
        working_order.for_each([&](std::int64_t cell) {
            pulse();
            const std::string material(engine.scene->material(cell));
            if (material == "vessel") encounters.insert(cell);
            legacy_effects::add_number(engine.state.at("materialExposure"), material, dt);
        });
        engine.state.set("vesselEncounters", legacy_effects::integers(encounters));
    }
    auto contacts = instrument.effect == "elevate" || instrument.effect == "dissect" ?
        working_order.copy() : PositiveIntSet(&stop);
    for (const auto& hold : engine.state.at("holds").as_object())
        for (const auto& cell : hold.second.at("cells").as_array()) { pulse(); contacts.insert(cell.as_int()); }
    const auto& removed = engine.state.at("removed").as_array();
    contacts.difference_update(removed.begin(), removed.end(), [](const Value& value) { return value.as_int(); });
    if (!shared) contacts.for_each([&](std::int64_t cell) {
        pulse();
        legacy_effects::add_number(engine.state.at("contact"), std::to_string(cell), dt);
    });
    auto& next_slot = engine.state.at("slots").at(key);
    next_slot.set("position", point_value(position)); next_slot.set("quaternion", quaternion_value(quaternion));
    if (operation != "place" && operation != "exchange" && operation != "release" && operation != "inspect" &&
            operation != "deactivate") next_slot.set("inserted", true);
    legacy_effects::add_number(engine.state, "elapsedSeconds", dt);
    legacy_effects::add_number(engine.state, "travelMm", point_norm(geometry_detail::subtract(position, previous)));
    for (const auto& hold_key : keys(engine.state.at("holds"))) if (has_kind(engine, hold_key, "bone-flap-v1")) {
        const auto offset = queries_detail::lattice_offset(engine.state.at("holds").at(hold_key).at("offset"));
        bone_transport::check_sweep(engine, hold_key, offset, engine.state.at("slots").at(hold_key), stop);
    }
    if (!engine.scene->cranial_access.is_null()) bone_transport::check_detached(engine, stop);
    tick_legacy_fields(engine, dt, stop);
    if (!engine.scene->cranial_stages.is_null()) {
        for (const auto& hold_key : keys(engine.state.at("holds"))) if (has_kind(engine, hold_key, "membrane-patch-v1")) {
            const auto offset = queries_detail::lattice_offset(engine.state.at("holds").at(hold_key).at("offset"));
            membrane_transport::check_sweep(engine, hold_key, offset, engine.state.at("slots").at(hold_key), stop);
        }
        membrane_transport::check_detached(engine, stop);
        cranial::validate_effects(engine, stop);
    }
    if (protected_state) {
        protected_after(engine, previous_exposure, stop);
        const auto current_positions = protection_detail::positions(engine, stop);
        if (std::any_of(current_positions.begin(), current_positions.end(), [](const auto& entry) { return entry.first != entry.second; }))
            protected_guard_sweep(engine, key, requested_position, quaternion, stop);
    }
    stop.check();
    return events;
}

}  // namespace resectionlab::sim
