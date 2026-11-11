#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_hash_native.hpp"
#include "simulation_queries_native.hpp"
#include "simulation_protection_native.hpp"

namespace resectionlab::sim {
namespace cranial {
const Value& validate_access(State&, const StopToken&);
void require_access_evidence(State&, const StopToken&);
const Value& validate_stage_contract(State&, const StopToken&);
void require_evidence(State&, const StopToken&);
bool verified_bone_prefix(State&, bool, const StopToken&);
bool tool_is_exterior(State&, std::string_view, const StopToken&);
}
namespace guards {
bool posed(const State&, std::string_view, const StopToken&);
}
namespace transport_detail {
using Cells = std::set<std::int64_t>;
using Index = std::array<std::int64_t, 3>;
using Edge = std::array<std::int64_t, 2>;
using Capsules = std::vector<std::pair<std::string, std::vector<Capsule>>>;
enum class Family { Bone, Membrane };
inline bool membrane(Family family) { return family == Family::Membrane; }
inline const char* holder(Family family) { return membrane(family) ? "membrane_forceps" : "bone_flap_holder"; }
inline const char* kind(Family family) { return membrane(family) ? "membrane-patch-v1" : "bone-flap-v1"; }
inline const char* field(Family family) { return membrane(family) ? "membraneCells" : "flapCells"; }
inline std::string prefix(Family family) { return membrane(family) ? "MEMBRANE_PATCH_" : "BONE_FLAP_"; }
[[noreturn]] inline void fail(Family family, const std::string& reason) {
    throw std::invalid_argument(prefix(family) + reason);
}
[[noreturn]] inline void bone_fail(const std::string& reason) {
    throw std::invalid_argument("CRANIOTOMY_" + reason);
}
inline const Value& empty_array() { static const auto value = Value::array(); return value; }
inline const Value& get(const Value& owner, std::string_view name, const Value& fallback = empty_array()) {
    const auto* value = owner.find(name); return value ? *value : fallback;
}
inline bool text(const Value& owner, std::string_view name, std::string_view expected) {
    const auto* value = owner.find(name); return value && value->is_string() && value->as_string() == expected;
}
inline bool equal(const Value& first, const Value& second) {
    if ((first.is_number() || first.is_bool()) && (second.is_number() || second.is_bool())) {
        if (first.is_double() && second.is_double()) return first.as_double() == second.as_double();
        if (!first.is_double() && !second.is_double())
            return (first.is_bool() ? std::int64_t(first.as_bool()) : first.as_int()) ==
                (second.is_bool() ? std::int64_t(second.as_bool()) : second.as_int());
        const auto& integer = first.is_double() ? second : first;
        const double real = first.is_double() ? first.as_double() : second.as_double();
        if (!std::isfinite(real) || real < -9223372036854775808. || real >= 9223372036854775808. || std::trunc(real) != real)
            return false;
        return std::int64_t(real) == (integer.is_bool() ? std::int64_t(integer.as_bool()) : integer.as_int());
    }
    if (first.kind() != second.kind()) return false;
    if (first.is_null()) return true;
    if (first.is_string()) return first.as_string() == second.as_string();
    if (first.is_array()) {
        if (first.as_array().size() != second.as_array().size()) return false;
        for (std::size_t i = 0; i < first.as_array().size(); ++i)
            if (!equal(first.at(i), second.at(i))) return false;
        return true;
    }
    if (first.as_object().size() != second.as_object().size()) return false;
    for (const auto& item : first.as_object()) {
        const auto* value = second.find(item.first); if (!value || !equal(item.second, *value)) return false;
    }
    return true;
}
inline Cells cells(const Value& value) {
    Cells result; for (const auto& item : value.as_array()) result.insert(item.as_int()); return result;
}
inline Value values(const Cells& source) {
    Value result = Value::array(); for (auto cell : source) result.append(cell); return result;
}
inline Value point_value(const Point& point) { return Value::array({point[0], point[1], point[2]}); }
inline Value quat_value(const Quat& value) { return Value::array({value[0], value[1], value[2], value[3]}); }
inline Value index_value(const Index& value) { return Value::array({value[0], value[1], value[2]}); }
inline bool intersects(const Value& value, const Cells& selected) {
    for (const auto& item : value.as_array()) if (selected.count(item.as_int())) return true;
    return false;
}
inline bool subset(const Cells& first, const Cells& second) {
    return std::includes(second.begin(), second.end(), first.begin(), first.end());
}
inline bool includes_string(const Value& values, std::string_view text) {
    for (const auto& value : values.as_array()) if (value.is_string() && value.as_string() == text) return true;
    return false;
}
inline bool equal_cells(const Value& first, const Value& second) { return equal(first, second); }
inline Value& slot(State& engine, const std::string& key) { return engine.state.at("slots").at(key); }
inline const Value& contract(State& engine, Family family, const StopToken& stop) {
    if (!membrane(family)) {
        if (engine.scene->cranial_access.is_null()) fail(family, "CONTRACT_REQUIRED");
        const auto& result = cranial::validate_access(engine, stop);
        cranial::require_access_evidence(engine, stop);
        return result;
    }
    if (engine.scene->cranial_stages.is_null()) fail(family, "STAGE_CONTRACT_REQUIRED");
    const auto& result = cranial::validate_stage_contract(engine, stop);
    cranial::require_evidence(engine, stop);
    cranial::verified_bone_prefix(engine, !engine.in_transition, stop);
    return result;
}
inline void validate_holder(State& engine, const std::string& key, Family family) {
    const auto& selected = slot(engine, key);
    if (!text(selected, "instrument", holder(family)) ||
            !queries_detail::revision(engine.state, "toolMechanicsRevision", 2.))
        fail(family, "REQUIRES_STOCK_HOLDER_REVISION2");
    const auto params = parameters(engine, selected);
    const double jaws = params.at("jawOpeningMm").as_number();
    if (params.at("radiusMm").as_number() != 1.5 || params.at("depthMm").as_number() != 2. || !(0. <= jaws && jaws <= 2.))
        fail(family, "HOLDER_GEOMETRY_FIXED");
    if (engine.state.at("holds").contains(key) && jaws > .5) fail(family, "RELEASE_BEFORE_OPENING_JAWS");
}
inline std::pair<const Value*, Value*> validate_hold(State& engine, const std::string& key,
                                                   Family family, const StopToken& stop) {
    validate_holder(engine, key, family);
    const auto& declaration = contract(engine, family, stop);
    auto* hold = engine.state.at("holds").find(key);
    if (!hold || !text(*hold, "kind", kind(family)) ||
            !equal(get(*hold, "contractHash", Value()), declaration.at("contentHash")) ||
            !equal_cells(get(*hold, "cells"), declaration.at(field(family))) ||
            !text(*hold, "componentHash", digest(declaration.at(field(family))))) fail(family, "HOLD_BINDING_INVALID");
    const auto own = cells(hold->at("cells"));
    if (intersects(engine.state.at("removed"), own)) fail(family, "HELD_MEMBERSHIP_CHANGED");
    for (auto cell : engine.scene->protected_cells) if (own.count(cell))
        throw std::invalid_argument(membrane(family) ? "PROTECTED_STRUCTURE: membrane patch contains protected membership" :
            "PROTECTED_STRUCTURE: bone flap contains protected membership");
    const auto& released = engine.state.at("separatedAttachments");
    for (const auto& attachment : engine.scene->properties.at("attachments").as_array())
        if (intersects(attachment.at("cells"), own) && !includes_string(released, attachment.at("id").as_string()))
            fail(family, "ATTACHMENTS_NOT_RELEASED");
    return {&declaration, hold};
}
inline std::set<Edge> edges(const Value& values) {
    std::set<Edge> result;
    for (const auto& edge : values.as_array()) {
        auto first = edge.at(0).as_int(), second = edge.at(1).as_int();
        result.insert({std::min(first, second), std::max(first, second)});
    }
    return result;
}
inline bool boundary_cut(const State& engine, const Value& declaration) {
    const auto required = edges(declaration.at("boundaryEdges"));
    const auto cut = edges(get(engine.state, "cutEdges"));
    return std::includes(cut.begin(), cut.end(), required.begin(), required.end());
}
inline std::vector<const Value*> fragments(const State& engine) {
    std::vector<const Value*> result;
    for (const auto& item : engine.state.at("holds").as_object()) result.push_back(&item.second);
    for (const auto& item : get(engine.state, "detachedFragments").as_array()) result.push_back(&item);
    return result;
}
inline void eligible_patch(State& engine, const Value& declaration, const Value* allowed) {
    const auto patch = cells(declaration.at("membraneCells"));
    if (intersects(engine.state.at("removed"), patch)) fail(Family::Membrane, "MEMBERSHIP_ALREADY_REMOVED");
    if (!boundary_cut(engine, declaration)) fail(Family::Membrane, "COMPLETE_BOUNDARY_CUT_REQUIRED");
    for (const auto& attachment : engine.scene->properties.at("attachments").as_array())
        if (includes_string(declaration.at("attachmentIds"), attachment.at("id").as_string()) &&
                !includes_string(engine.state.at("separatedAttachments"), attachment.at("id").as_string()))
            fail(Family::Membrane, "ATTACHMENTS_NOT_RELEASED");
    for (auto cell : engine.scene->protected_cells) if (patch.count(cell))
        throw std::invalid_argument("PROTECTED_STRUCTURE: membrane patch contains protected source cells");
    for (const auto& fragment : get(engine.state, "detachedFragments").as_array())
        if (&fragment != allowed && intersects(fragment.at("cells"), patch)) fail(Family::Membrane, "MATERIAL_ALREADY_DETACHED");
}
inline Index shifted(const Scene& source, std::int64_t cell, const Index& offset) {
    auto index = geometry_detail::coordinates(source, cell);
    for (int axis = 0; axis < 3; ++axis) index[axis] += offset[axis];
    return index;
}
inline Point world_index(const Scene& source, const Index& index) {
    return geometry_detail::transform(source.affine, {double(index[0]), double(index[1]), double(index[2])});
}
inline void inside_bounds(const State& engine, const Value& declaration, const std::vector<Index>& indices, Family family) {
    Point low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
    for (const auto& index : indices) for (int axis = 0; axis < 3; ++axis) {
        low[axis] = std::min(low[axis], double(index[axis]) - .5);
        high[axis] = std::max(high[axis], double(index[axis]) + .5);
    }
    for (int axis = 0; axis < 3; ++axis)
        if (low[axis] < -.5 || high[axis] > double(engine.scene->shape[axis]) - .5)
            fail(family, "OUTSIDE_OBSERVED_GRID");
    const auto& access = membrane(family) ? engine.scene->cranial_access : declaration;
    auto check = [&](const Value& bounds) {
        const auto minimum = queries_detail::point(bounds.at("minimumVoxelBoundary"));
        const auto maximum = queries_detail::point(bounds.at("maximumVoxelBoundary"));
        for (int axis = 0; axis < 3; ++axis)
            if (low[axis] < minimum[axis] - 1e-9 || high[axis] > maximum[axis] + 1e-9)
                fail(family, "OUTSIDE_REVIEWED_WORKSPACE");
    };
    check(access.at("exit"));
    const auto* adapter = engine.scene->source.find("adapter");
    if (adapter && adapter->is_object()) if (const auto* domain = adapter->find("reviewedTaskDomainV1")) check(*domain);
}
inline std::vector<Capsule> jaw_envelopes(const State& engine, const std::string& key,
                                         const Point& point, const Quat& quaternion, Family family) {
    auto result = capsule_envelopes(tool(holder(family)), parameters(engine, engine.state.at("slots").at(key)), point, quaternion);
    result.erase(std::remove_if(result.begin(), result.end(), [](const Capsule& value) { return value.part == "shaft"; }), result.end());
    return result;
}
inline bool jaw_contact(const Point& point, const std::vector<Capsule>& jaws, double radius) {
    for (const auto& jaw : jaws)
        if (segments_within_distance(point, point, jaw.a, jaw.b, 2 * radius + jaw.radius + 1e-9)) return true;
    return false;
}
inline Capsules capsules(const State& engine, const std::string& key, const Value& end_pose) {
    Capsules result;
    for (const auto& entry : engine.state.at("slots").as_object()) {
        const auto& selected = entry.second;
        const auto& instrument = tool(selected.at("instrument").as_string());
        auto params = parameters(engine, selected);
        result.emplace_back(entry.first, entry.first == key ? swept_envelopes(instrument, params, selected, end_pose) :
            capsule_envelopes(instrument, params, queries_detail::point(selected.at("position")), queries_detail::quaternion(selected.at("quaternion"))));
    }
    return result;
}
inline bool swept_cell_interior_intersects(const Index& start, const Index& end, const Index& stationary) {
    double lower = 0., upper = 1.;
    for (int axis = 0; axis < 3; ++axis) {
        const auto a = start[axis], d = end[axis] - a, q = stationary[axis];
        const auto low = q - 1, high = q + 1;
        if (d == 0) { if (a <= low || a >= high) return false; }
        else {
            const double first = double(low - a) / double(d), second = double(high - a) / double(d);
            lower = std::max(lower, std::min(first, second)); upper = std::min(upper, std::max(first, second));
            if (lower >= upper) return false;
        }
    }
    return lower < upper;
}
inline void lift_rule(const Value& declaration, const Index& before, const Index& after) {
    const auto axis = declaration.at("membraneLiftAxis").as_int(), sign = declaration.at("membraneOutwardSign").as_int();
    const auto first = before[axis] * sign, second = after[axis] * sign;
    if (first < 0 || second < 0) fail(Family::Membrane, "INWARD_LIFT_UNSUPPORTED");
    if (std::min(first, second) < 2) for (int direction = 0; direction < 3; ++direction)
        if (direction != axis && before[direction] != after[direction]) fail(Family::Membrane, "NORMAL_LIFT_TWO_CELLS_REQUIRED");
}
inline bool rim_contact_only(const State& engine, const Value& declaration, std::int64_t source_id,
        std::int64_t candidate, const Index& before, const Index& after, bool cut, const Cells& retained) {
    if (!retained.count(candidate)) return false;
    const auto source = geometry_detail::coordinates(*engine.scene, source_id);
    const auto rim = geometry_detail::coordinates(*engine.scene, candidate);
    std::int64_t maximum = 0;
    for (int axis = 0; axis < 3; ++axis) maximum = std::max(maximum, std::abs(source[axis] - rim[axis]));
    if (maximum != 1 || !cut) return false;
    const auto removed = cells(engine.state.at("removed"));
    if (removed.count(candidate)) fail(Family::Membrane, "RETAINED_RIM_CHANGED");
    for (const auto* fragment : fragments(engine)) if (cells(fragment->at("cells")).count(candidate))
        fail(Family::Membrane, "RETAINED_RIM_CHANGED");
    queries_detail::no_deformation(engine);
    return !swept_cell_interior_intersects(before, after, rim);
}

inline void validate_bone_scope(State& engine, const Value& declaration, const StopToken& stop) {
    stop.check();
    auto intended = cells(declaration.at("flapCells"));
    const auto kerf = cells(declaration.at("kerfCells")); intended.insert(kerf.begin(), kerf.end());
    if (engine.scene->original_target_cells.empty()) bone_fail("NONEMPTY_TARGET_EVIDENCE_REQUIRED");
    for (auto cell : engine.scene->original_target_cells) if (intended.count(cell)) bone_fail("TARGET_OVERLAP");
    Cells protected_cells(engine.scene->protected_cells.begin(), engine.scene->protected_cells.end());
    for (const auto& row : get(engine.scene->source, "protectedAnatomyV1").as_array()) {
        const auto listed = cells(row.at("cells")); protected_cells.insert(listed.begin(), listed.end());
    }
    queries_detail::Pulse pulse{stop};
    for (std::size_t cell = 0; cell < engine.scene->labels.size(); ++cell) {
        pulse(); const auto id = engine.scene->labels[cell]; if (id == 0) continue;
        const auto& region = engine.scene->region(id);
        const auto* protected_flag = region.find("protected"), *critical = region.find("critical");
        if ((protected_flag && protected_flag->is_bool() && protected_flag->as_bool()) ||
                (critical && critical->is_bool() && critical->as_bool())) protected_cells.insert(std::int64_t(cell));
    }
    for (auto cell : intended) if (protected_cells.count(cell)) bone_fail("PROTECTED_STRUCTURE");
}
inline void eligible_bone_component(State& engine, const Value& declaration, const Value* allowed,
                                    const StopToken& stop) {
    validate_bone_scope(engine, declaration, stop);
    cranial::require_access_evidence(engine, stop);
    const auto& source = *engine.scene;
    const auto flap = cells(declaration.at("flapCells")), kerf = cells(declaration.at("kerfCells")),
        retained = cells(declaration.at("retainedSkullCells"));
    const auto& removed_values = engine.state.at("removed").as_array();
    std::int64_t previous = -1;
    for (const auto& value : removed_values) {
        if (!value.is_int() || value.as_int() <= previous || value.as_int() < 0 ||
                static_cast<std::uint64_t>(value.as_int()) >= source.labels.size())
            throw std::invalid_argument("CRANIOTOMY_CONTRACT_INVALID: removed state requires sorted distinct source integer cells");
        previous = value.as_int();
    }
    const auto removed = cells(engine.state.at("removed"));
    if (!subset(kerf, removed)) bone_fail("KERF_INCOMPLETE");
    for (auto cell : removed) if (flap.count(cell) || retained.count(cell)) bone_fail("INTENDED_BONE_MEMBERSHIP_CHANGED");
    const auto& detached = get(engine.state, "detachedFragments").as_array();
    if (allowed) {
        std::size_t instances = 0; for (const auto& value : detached) if (&value == allowed) ++instances;
        if (!allowed->is_object() || instances != 1 || !text(*allowed, "kind", "bone-flap-v1") ||
                !equal(get(*allowed, "contractHash", Value()), declaration.at("contentHash")) ||
                !text(*allowed, "componentHash", digest(declaration.at("flapCells"))) ||
                !equal_cells(get(*allowed, "cells"), declaration.at("flapCells"))) bone_fail("DETACHED_BINDING_INVALID");
        Index offset{};
        for (const auto* name : {"offset", "baseOffset"}) {
            const auto* values = allowed->find(name);
            if (!values || !values->is_array() || values->as_array().size() != 3) bone_fail("DETACHED_TRANSFORM_INVALID");
            for (const auto& value : values->as_array()) if (!value.is_int()) bone_fail("DETACHED_TRANSFORM_INVALID");
            if (std::string_view(name) == "offset") offset = queries_detail::lattice_offset(*values);
        }
        const auto minimum = queries_detail::point(declaration.at("exit").at("minimumVoxelBoundary"));
        const auto maximum = queries_detail::point(declaration.at("exit").at("maximumVoxelBoundary"));
        for (auto cell : flap) {
            const auto index = shifted(source, cell, offset);
            for (int axis = 0; axis < 3; ++axis)
                if (double(index[axis]) - .5 < minimum[axis] - 1e-9 || double(index[axis]) + .5 > maximum[axis] + 1e-9)
                    bone_fail("DETACHED_TRANSFORM_INVALID");
        }
    }
    Cells held;
    for (const auto* fragment : fragments(engine)) if (fragment != allowed && !fragment->contains("target")) {
        const auto selected = cells(fragment->at("cells")); held.insert(selected.begin(), selected.end());
    }
    for (auto cell : held) if (flap.count(cell)) bone_fail("MATERIAL_ALREADY_HELD");
    for (auto cell : held) if (retained.count(cell)) bone_fail("RETAINED_SKULL_SUPPORT_CHANGED");
    const auto anchors = cells(declaration.at("boneAnchorCells"));
    if (anchors.empty()) bone_fail("SUPPORT_ANCHORS_REQUIRED");
    if (!source.properties.contains("attachments")) bone_fail("ATTACHMENT_COVERAGE_REQUIRED");
    for (const auto& attachment : source.properties.at("attachments").as_array()) if (intersects(attachment.at("cells"), flap)) {
        if (!text(attachment, "kind", "periosteal") && !text(attachment, "kind", "soft")) bone_fail("ATTACHMENT_MODEL_UNSUPPORTED");
        if (!includes_string(engine.state.at("separatedAttachments"), attachment.at("id").as_string()))
            bone_fail("SEPARATE_ATTACHMENTS_BEFORE_GRASP");
    }
    std::vector<std::int32_t> components(source.labels.size(), 0);
    queries_detail::Pulse pulse{stop};
    std::array<bool, 65536> bone_labels{};
    for (const auto& region : source.regions.as_array()) if (text(region, "material", "bone"))
        bone_labels[region.at("id").as_int()] = true;
    for (std::size_t cell = 0; cell < source.labels.size(); ++cell) {
        pulse(); if (!bone_labels[source.labels[cell]] || removed.count(std::int64_t(cell))) components[cell] = -1;
    }
    std::int32_t identifier = 0;
    std::vector<std::int64_t> stack;
    for (std::size_t seed = 0; seed < components.size(); ++seed) {
        pulse(); if (components[seed] != 0) continue;
        ++identifier; components[seed] = identifier; stack.push_back(std::int64_t(seed));
        while (!stack.empty()) {
            pulse(); const auto cell = stack.back(); stack.pop_back();
            const auto index = geometry_detail::coordinates(source, cell);
            for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) for (int z = -1; z <= 1; ++z) {
                const Index neighbor{index[0] + x, index[1] + y, index[2] + z};
                bool valid = true;
                for (int axis = 0; axis < 3; ++axis)
                    valid &= neighbor[axis] >= 0 && static_cast<std::uint64_t>(neighbor[axis]) < source.shape[axis];
                if (!valid) continue;
                const auto next = (neighbor[0] * std::int64_t(source.shape[1]) + neighbor[1]) * std::int64_t(source.shape[2]) + neighbor[2];
                if (components[next] == 0) { components[next] = identifier; stack.push_back(next); }
            }
        }
    }
    const auto selected_id = components[*flap.begin()];
    Cells component;
    for (std::size_t cell = 0; cell < components.size(); ++cell) {
        pulse(); if (components[cell] == selected_id) component.insert(std::int64_t(cell));
    }
    if (component != flap) bone_fail("BONE_BRIDGE_OR_FLAP_COMPONENT_MISMATCH");
    Cells remaining = flap;
    stack.push_back(*remaining.begin()); remaining.erase(stack.back());
    while (!stack.empty()) {
        pulse(); const auto index = geometry_detail::coordinates(source, stack.back()); stack.pop_back();
        for (int axis = 0; axis < 3; ++axis) for (int direction : {-1, 1}) {
            auto neighbor = index; neighbor[axis] += direction;
            if (neighbor[axis] < 0 || std::uint64_t(neighbor[axis]) >= source.shape[axis]) continue;
            const auto next = (neighbor[0] * std::int64_t(source.shape[1]) + neighbor[1]) * std::int64_t(source.shape[2]) + neighbor[2];
            if (remaining.erase(next)) stack.push_back(next);
        }
    }
    if (!remaining.empty()) bone_fail("FLAP_NOT_FACE_CONNECTED");
    std::set<std::int32_t> anchored;
    for (auto anchor : anchors) anchored.insert(components[anchor]);
    if (anchored.count(-1)) bone_fail("RETAINED_SKULL_SUPPORT_INCOMPLETE");
    for (auto cell : retained) if (!anchored.count(components[cell])) bone_fail("RETAINED_SKULL_SUPPORT_INCOMPLETE");
    const auto& exit = declaration.at("exit");
    const auto axis = exit.at("axis").as_int(), side = exit.at("side").as_int();
    const double plane = exit.at("planeVoxelBoundary").as_number();
    std::int64_t minimum = INT64_MAX, maximum = INT64_MIN, low = INT64_MAX, high = INT64_MIN;
    for (std::size_t cell = 0; cell < source.labels.size(); ++cell) {
        pulse(); if (!source.labels[cell]) continue;
        const auto index = geometry_detail::coordinates(source, std::int64_t(cell));
        minimum = std::min(minimum, index[axis]); maximum = std::max(maximum, index[axis]);
    }
    for (auto cell : component) { const auto index = geometry_detail::coordinates(source, cell); low = std::min(low, index[axis]); high = std::max(high, index[axis]); }
    const double bound = side > 0 ? double(maximum) + .5 : double(minimum) - .5;
    if (double(side) * (plane - bound) < -1e-9) bone_fail("EXIT_NOT_SOURCE_EXTERIOR");
    const double room = side > 0 ? exit.at("maximumVoxelBoundary").at(axis).as_number() - plane :
        plane - exit.at("minimumVoxelBoundary").at(axis).as_number();
    if (room <= double(high - low + 1) + 1e-9) bone_fail("EXIT_WORKSPACE_INSUFFICIENT");
    stop.check();
}
inline Value grasp(State& engine, const std::string& key, const std::vector<std::int64_t>& contacts_input,
                   const Point& position, const Quat& quaternion, Family family, const StopToken& stop) {
    validate_holder(engine, key, family);
    const auto& declaration = contract(engine, family, stop);
    if (parameters(engine, slot(engine, key)).at("jawOpeningMm").as_number() > .5) return Value::array();
    if (engine.state.at("holds").contains(key)) { validate_hold(engine, key, family, stop); return Value::array(); }
    const auto own = cells(declaration.at(field(family)));
    Cells contacts; for (auto cell : contacts_input) if (own.count(cell)) contacts.insert(cell);
    if (contacts.empty()) return Value::array();
    for (auto cell : engine.scene->protected_cells) if (own.count(cell))
        throw std::invalid_argument(membrane(family) ? "PROTECTED_STRUCTURE: membrane patch contains protected membership" :
            "PROTECTED_STRUCTURE: bone flap contains protected membership");
    for (const auto& held : engine.state.at("holds").as_object())
        if (intersects(held.second.at("cells"), own)) fail(family, "MATERIAL_ALREADY_HELD");
    auto* detached_values = engine.state.find("detachedFragments");
    std::vector<std::size_t> detached;
    if (detached_values) for (std::size_t i = 0; i < detached_values->as_array().size(); ++i)
        if (intersects(detached_values->at(i).at("cells"), own)) detached.push_back(i);
    if (!detached.empty() && (detached.size() != 1 || !text(detached_values->at(detached[0]), "kind", kind(family)) ||
            !equal(get(detached_values->at(detached[0]), "contractHash", Value()), declaration.at("contentHash"))))
        fail(family, "DETACHED_BINDING_INVALID");
    const Value* allowed = detached.empty() ? nullptr : &detached_values->at(detached[0]);
    if (membrane(family)) eligible_patch(engine, declaration, allowed);
    else eligible_bone_component(engine, declaration, allowed, stop);
    Value hold;
    if (allowed) {
        hold = *allowed;
        detached_values->as_array().erase(detached_values->as_array().begin() + detached[0]);
        hold.set("origin", point_value(position)); hold.set("baseOffset", hold.at("offset"));
    } else hold = Value::object({{"kind", kind(family)}, {"contractHash", declaration.at("contentHash")},
        {"componentHash", digest(declaration.at(field(family)))}, {"cells", declaration.at(field(family))},
        {"origin", point_value(position)}, {"offset", Value::array({0, 0, 0})}, {"baseOffset", Value::array({0, 0, 0})}});
    const auto offset = queries_detail::offset_world(*engine.scene, queries_detail::lattice_offset(hold.at("offset")));
    const auto radius = protection_cell_radius(*engine.scene);
    const auto jaws = jaw_envelopes(engine, key, position, quaternion, family);
    Value region = Value::array();
    std::size_t index = 0;
    for (const auto& cell : declaration.at(field(family)).as_array()) {
        if ((index++ % 32) == 0) stop.check();
        auto point = world_point(*engine.scene, cell.as_int());
        for (int axis = 0; axis < 3; ++axis) point[axis] += offset[axis];
        if (jaw_contact(point, jaws, radius)) region.append(cell);
    }
    hold.set("gripCells", std::move(region)); hold.set("gripContactCells", values(contacts));
    hold.set("holderSlot", key); hold.set("gripPose", Value::object({{"position", point_value(position)}, {"quaternion", quat_value(quaternion)}}));
    const auto component_hash = hold.at("componentHash");
    engine.state.at("holds").set(key, std::move(hold)); slot(engine, key).set("grip", declaration.at(field(family)));
    return Value::array({Value::object({{"kind", membrane(family) ? "membrane_patch_grasp" : "bone_flap_grasp"},
        {"componentHash", component_hash}, {"cells", own.size()}, {"regrasp", !detached.empty()}, {"removalMm3", 0.}})});
}
inline void check_sweep(State& engine, const std::string& key, const Index& new_offset,
                        const Value& end_pose, Family family, const StopToken& stop) {
    const auto checked = validate_hold(engine, key, family, stop);
    const auto& declaration = *checked.first; const auto& hold = *checked.second;
    const auto old_offset = queries_detail::lattice_offset(hold.at("offset"));
    std::vector<Index> before, after, bounds;
    std::vector<Point> first, last;
    for (const auto& cell : hold.at("cells").as_array()) {
        const auto a = shifted(*engine.scene, cell.as_int(), old_offset), b = shifted(*engine.scene, cell.as_int(), new_offset);
        before.push_back(a); after.push_back(b); bounds.push_back(a); bounds.push_back(b);
        first.push_back(world_index(*engine.scene, a)); last.push_back(world_index(*engine.scene, b));
    }
    if (membrane(family)) lift_rule(declaration, old_offset, new_offset);
    inside_bounds(engine, declaration, bounds, family);
    const double radius = protection_cell_radius(*engine.scene);
    const auto own = cells(hold.at("cells")), grip = cells(hold.at("gripCells")), witness = cells(hold.at("gripContactCells"));
    const auto jaws = jaw_envelopes(engine, key, queries_detail::point(end_pose.at("position")),
        queries_detail::quaternion(end_pose.at("quaternion")), family);
    bool contact = false;
    for (std::size_t i = 0; i < last.size() && !contact; ++i)
        contact = witness.count(hold.at("cells").at(i).as_int()) && jaw_contact(last[i], jaws, radius);
    if (witness.empty() || !subset(witness, grip) || !subset(grip, own) || !contact) fail(family, "GRIP_CONTACT_LOST");
    const bool cut = membrane(family) && boundary_cut(engine, declaration);
    const auto retained = membrane(family) ? cells(declaration.at("retainedMembraneCells")) : Cells{};
    const auto owner = occupancy(engine, stop);
    const auto tools = capsules(engine, key, end_pose);
    for (std::size_t index = 0; index < first.size(); ++index) {
        if ((index % 32) == 0) stop.check();
        const auto source_id = hold.at("cells").at(index).as_int();
        if (!protected_capsule_hits(engine, first[index], last[index], radius, stop).empty())
            throw std::invalid_argument(membrane(family) ? "PROTECTED_STRUCTURE: entire membrane patch sweep enters declared clearance" :
                "PROTECTED_STRUCTURE: entire bone flap sweep enters declared clearance");
        const auto occupied = owner.material_ids(capsule_cells(*engine.scene, first[index], last[index], radius, stop), stop);
        for (auto candidate : occupied) if (!own.count(candidate)) {
            if (!membrane(family) || !rim_contact_only(engine, declaration, source_id, candidate, before[index], after[index], cut, retained))
                fail(family, "SWEPT_ANATOMY_COLLISION");
        }
        for (const auto& hand : tools) for (const auto& segment : hand.second) {
            if (hand.first == key && segment.part != "shaft" && grip.count(source_id)) continue;
            if (segments_within_distance(first[index], last[index], segment.a, segment.b, radius + segment.radius + 1e-9))
                fail(family, hand.first == key ? "SHAFT_COLLISION" : "OTHER_TOOL_COLLISION:" + hand.first);
        }
    }
}
inline void check_tool_sweep(State& engine, const std::string& key, const Value& end_pose,
                             const std::string& operation, Family family, const StopToken& stop) {
    const auto& selected = slot(engine, key);
    const auto moving = swept_envelopes(tool(selected.at("instrument").as_string()), parameters(engine, selected), selected, end_pose);
    const double radius = protection_cell_radius(*engine.scene);
    std::vector<std::pair<std::string, const Value*>> held;
    for (const auto& item : engine.state.at("holds").as_object()) held.emplace_back(item.first, &item.second);
    for (const auto& item : get(engine.state, "detachedFragments").as_array()) held.emplace_back("", &item);
    for (const auto& item : held) {
        const auto& hold = *item.second;
        if (!text(hold, "kind", kind(family)) || item.first == key) continue;
        const auto offset = queries_detail::offset_world(*engine.scene, queries_detail::lattice_offset(hold.at("offset")));
        const auto grip = cells(hold.at("gripCells"));
        std::size_t index = 0;
        for (const auto& cell : hold.at("cells").as_array()) {
            if ((index++ % 32) == 0) stop.check();
            auto point = world_point(*engine.scene, cell.as_int());
            for (int axis = 0; axis < 3; ++axis) point[axis] += offset[axis];
            for (const auto& segment : moving) {
                if (item.first.empty() && text(selected, "instrument", holder(family)) && segment.part != "shaft") {
                    if (operation == "grasp" || (text(hold, "holderSlot", key) &&
                            (operation == "release" || operation == "deactivate" || operation == "withdraw" || operation == "inspect") &&
                            grip.count(cell.as_int()))) continue;
                }
                if (segments_within_distance(point, point, segment.a, segment.b, radius + segment.radius + 1e-9))
                    fail(family, "MOVING_TOOL_COLLISION:" + key);
            }
        }
    }
}
inline void check_detached(State& engine, Family family, const StopToken& stop) {
    std::vector<const Value*> held;
    for (const auto& item : get(engine.state, "detachedFragments").as_array()) if (text(item, "kind", kind(family))) held.push_back(&item);
    if (held.empty()) return;
    const auto& declaration = contract(engine, family, stop);
    const bool cut = membrane(family) && boundary_cut(engine, declaration);
    const auto retained = membrane(family) ? cells(declaration.at("retainedMembraneCells")) : Cells{};
    const double radius = protection_cell_radius(*engine.scene);
    const auto owner = occupancy(engine, stop);
    auto forbidden = cells(engine.state.at("removed")); forbidden.insert(engine.scene->protected_cells.begin(), engine.scene->protected_cells.end());
    for (const auto* item : held) {
        const auto& hold = *item;
        if (!equal(get(hold, "contractHash", Value()), declaration.at("contentHash")) ||
                !equal_cells(hold.at("cells"), declaration.at(field(family))) || intersects(hold.at("cells"), forbidden))
            fail(family, "DETACHED_BINDING_INVALID");
        const auto offset = queries_detail::lattice_offset(hold.at("offset"));
        std::vector<Index> indices;
        for (const auto& cell : hold.at("cells").as_array()) indices.push_back(shifted(*engine.scene, cell.as_int(), offset));
        inside_bounds(engine, declaration, indices, family);
        const auto own = cells(hold.at("cells")), grip = cells(hold.at("gripCells"));
        for (std::size_t index = 0; index < indices.size(); ++index) {
            if ((index % 32) == 0) stop.check();
            const auto cell = hold.at("cells").at(index).as_int(); const auto point = world_index(*engine.scene, indices[index]);
            if (!protected_capsule_hits(engine, point, point, radius, stop).empty())
                throw std::invalid_argument(membrane(family) ? "PROTECTED_STRUCTURE: retained detached membrane patch contacts declared clearance" :
                    "PROTECTED_STRUCTURE: retained detached bone flap contacts declared clearance");
            const auto occupied = owner.material_ids(capsule_cells(*engine.scene, point, point, radius, stop), stop);
            for (auto candidate : occupied) if (!own.count(candidate)) {
                if (!membrane(family) || !rim_contact_only(engine, declaration, cell, candidate, indices[index], indices[index], cut, retained))
                    fail(family, "DETACHED_ANATOMY_COLLISION");
            }
            for (const auto& hand : engine.state.at("slots").as_object()) {
                const auto& selected = hand.second;
                const auto segments = capsule_envelopes(tool(selected.at("instrument").as_string()), parameters(engine, selected),
                    queries_detail::point(selected.at("position")), queries_detail::quaternion(selected.at("quaternion")));
                for (const auto& segment : segments) {
                    if (text(hold, "holderSlot", hand.first) && text(selected, "instrument", holder(family)) && segment.part != "shaft" && grip.count(cell)) continue;
                    if (segments_within_distance(point, point, segment.a, segment.b, radius + segment.radius + 1e-9))
                        fail(family, "DETACHED_TOOL_COLLISION:" + hand.first);
                }
            }
        }
    }
}
inline Value extract(State& engine, const std::string& key, Family family, const StopToken& stop) {
    const auto checked = validate_hold(engine, key, family, stop);
    const auto& declaration = *checked.first;
    const auto hold = *checked.second;
    const auto offset = queries_detail::lattice_offset(hold.at("offset"));
    if (offset[0] == 0 && offset[1] == 0 && offset[2] == 0) fail(family, "MOVE_THROUGH_EXIT_BEFORE_EXTRACTION");
    check_sweep(engine, key, offset, slot(engine, key), family, stop);
    const auto& exit = (membrane(family) ? engine.scene->cranial_access : declaration).at("exit");
    const auto axis = exit.at("axis").as_int(), side = exit.at("side").as_int();
    const double plane = exit.at("planeVoxelBoundary").as_number();
    std::int64_t low = INT64_MAX, high = INT64_MIN, occupied_low = INT64_MAX, occupied_high = INT64_MIN;
    for (const auto& cell : hold.at("cells").as_array()) {
        const auto position = shifted(*engine.scene, cell.as_int(), offset);
        low = std::min(low, position[axis]); high = std::max(high, position[axis]);
    }
    queries_detail::Pulse pulse{stop};
    for (std::size_t cell = 0; cell < engine.scene->labels.size(); ++cell) {
        pulse(); if (!engine.scene->labels[cell]) continue;
        const auto index = geometry_detail::coordinates(*engine.scene, std::int64_t(cell));
        occupied_low = std::min(occupied_low, index[axis]); occupied_high = std::max(occupied_high, index[axis]);
    }
    const bool exterior = side == -1 ? double(high) + .5 < std::min(plane, double(occupied_low) - .5) - 1e-9 :
        double(low) - .5 > std::max(plane, double(occupied_high) + .5) + 1e-9;
    if (!exterior) fail(family, "NOT_AT_DECLARED_EXTERIOR_EXIT");
    const auto count = hold.at("cells").as_array().size();
    Value receipt = Value::object({{"kind", membrane(family) ? "membrane_patch_evacuation" : "bone_flap_evacuation"},
        {"contractHash", declaration.at("contentHash")}, {"componentHash", hold.at("componentHash")}, {"cells", count},
        {"volumeMm3", double(count) * engine.scene->voxel_mm3}, {"offset", hold.at("offset")},
        {"exitAxis", axis}, {"exitSide", side}, {"checkedWholeFragmentSweep", true}});
    auto removed = cells(engine.state.at("removed")); const auto own = cells(hold.at("cells")); removed.insert(own.begin(), own.end());
    engine.state.set("removed", values(removed));
    const std::string evacuation_field = membrane(family) ? "membranePatchEvacuations" : "boneFlapEvacuations";
    if (!engine.state.contains(evacuation_field)) engine.state.set(evacuation_field, Value::array());
    engine.state.at(evacuation_field).append(receipt);
    engine.state.at("holds").erase(key); slot(engine, key).set("grip", Value::array());
    if (!membrane(family)) engine.state.set("boneTopologyDirty", true);
    return Value::array({receipt});
}
}  // namespace transport_detail

namespace bone_transport {
inline void validate_holder(State& engine, const std::string& key) { transport_detail::validate_holder(engine, key, transport_detail::Family::Bone); }
inline void validate_hold(State& engine, const std::string& key, const StopToken& stop) { transport_detail::validate_hold(engine, key, transport_detail::Family::Bone, stop); }
inline Value grasp(State& engine, const std::string& key, const std::vector<std::int64_t>& cells,
                   const Point& position, const Quat& quaternion, const StopToken& stop) {
    return transport_detail::grasp(engine, key, cells, position, quaternion, transport_detail::Family::Bone, stop);
}
inline void check_sweep(State& engine, const std::string& key, const std::array<std::int64_t, 3>& offset,
                        const Value& end_pose, const StopToken& stop) {
    transport_detail::check_sweep(engine, key, offset, end_pose, transport_detail::Family::Bone, stop);
}
inline void check_tool_sweep(State& engine, const std::string& key, const Value& end_pose,
                             const std::string& operation, const StopToken& stop) {
    transport_detail::check_tool_sweep(engine, key, end_pose, operation, transport_detail::Family::Bone, stop);
}
inline void check_detached(State& engine, const StopToken& stop) { transport_detail::check_detached(engine, transport_detail::Family::Bone, stop); }
inline Value extract(State& engine, const std::string& key, const StopToken& stop) { return transport_detail::extract(engine, key, transport_detail::Family::Bone, stop); }
inline bool aperture_evacuated(State& engine, const Value& declaration, const StopToken& stop) {
    stop.check();
    const auto removed = transport_detail::cells(engine.state.at("removed"));
    if (!transport_detail::subset(transport_detail::cells(declaration.at("flapCells")), removed) ||
            !transport_detail::subset(transport_detail::cells(declaration.at("kerfCells")), removed)) return false;
    for (const auto& event : transport_detail::get(engine.state, "boneFlapEvacuations").as_array())
        if (transport_detail::equal(transport_detail::get(event, "contractHash", Value()), declaration.at("contentHash"))) return true;
    return false;
}
inline bool aperture_evacuated(State& engine, const StopToken& stop) {
    return aperture_evacuated(engine, transport_detail::contract(engine, transport_detail::Family::Bone, stop), stop);
}
inline Value describe(State& engine, const StopToken& stop) {
    const auto& declaration = engine.scene->cranial_access;
    const auto removed = transport_detail::cells(engine.state.at("removed"));
    std::size_t kerf = 0, flap = 0;
    for (const auto& cell : declaration.at("kerfCells").as_array()) kerf += removed.count(cell.as_int());
    for (const auto& cell : declaration.at("flapCells").as_array()) flap += removed.count(cell.as_int());
    bool held = false, detached = false;
    for (const auto& item : engine.state.at("holds").as_object()) held |= transport_detail::text(item.second, "kind", "bone-flap-v1");
    for (const auto& item : transport_detail::get(engine.state, "detachedFragments").as_array()) detached |= transport_detail::text(item, "kind", "bone-flap-v1");
    const bool evacuated = aperture_evacuated(engine, declaration, stop);
    bool withdrawn = !engine.state.at("holds").truthy() && !transport_detail::get(engine.state, "detachedFragments").truthy();
    for (const auto& item : engine.state.at("slots").as_object()) {
        const auto& selected = item.second;
        withdrawn &= !selected.at("inserted").truthy() && !selected.at("active").truthy() && !selected.at("grip").truthy();
    }
    if (withdrawn) for (const auto& item : engine.state.at("slots").as_object()) {
        if (guards::posed(engine, item.first, stop) && !cranial::tool_is_exterior(engine, item.first, stop)) {
            withdrawn = false; break;
        }
    }
    const char* phase = evacuated && withdrawn ? "HOLDER_WITHDRAWN" : evacuated ? "EVACUATED" : held ? "GRASPED_OR_TRANSPORTED" :
        detached ? "DETACHED_IN_FIELD" : kerf ? "KERF_IN_PROGRESS" : "BOUND";
    return Value::object({{"contractHash", declaration.at("contentHash")}, {"phase", phase},
        {"kerfRemovedMm3", double(kerf) * engine.scene->voxel_mm3}, {"flapEvacuatedMm3", double(flap) * engine.scene->voxel_mm3},
        {"declaredBoneRemovedMm3", double(kerf + flap) * engine.scene->voxel_mm3}, {"apertureEvacuated", evacuated},
        {"openingComplete", evacuated && withdrawn}, {"transport", "translation-only snapped affine cells with conservative whole-volume sweeps"},
        {"clinicalValidation", "none"}});
}
}  // namespace bone_transport

namespace membrane_transport {
inline void validate_holder(State& engine, const std::string& key) { transport_detail::validate_holder(engine, key, transport_detail::Family::Membrane); }
inline void validate_hold(State& engine, const std::string& key, const StopToken& stop) { transport_detail::validate_hold(engine, key, transport_detail::Family::Membrane, stop); }
inline Value grasp(State& engine, const std::string& key, const std::vector<std::int64_t>& cells,
                   const Point& position, const Quat& quaternion, const StopToken& stop) {
    return transport_detail::grasp(engine, key, cells, position, quaternion, transport_detail::Family::Membrane, stop);
}
inline void check_sweep(State& engine, const std::string& key, const std::array<std::int64_t, 3>& offset,
                        const Value& end_pose, const StopToken& stop) {
    transport_detail::check_sweep(engine, key, offset, end_pose, transport_detail::Family::Membrane, stop);
}
inline void check_tool_sweep(State& engine, const std::string& key, const Value& end_pose,
                             const std::string& operation, const StopToken& stop) {
    transport_detail::check_tool_sweep(engine, key, end_pose, operation, transport_detail::Family::Membrane, stop);
}
inline void check_detached(State& engine, const StopToken& stop) { transport_detail::check_detached(engine, transport_detail::Family::Membrane, stop); }
inline Value extract(State& engine, const std::string& key, const StopToken& stop) { return transport_detail::extract(engine, key, transport_detail::Family::Membrane, stop); }
inline bool swept_cell_interior_intersects(const std::array<std::int64_t, 3>& before,
        const std::array<std::int64_t, 3>& after, const std::array<std::int64_t, 3>& stationary) {
    return transport_detail::swept_cell_interior_intersects(before, after, stationary);
}
inline Value clearance(State& engine, const StopToken& stop) {
    const auto& declaration = transport_detail::contract(engine, transport_detail::Family::Membrane, stop);
    const auto patch = transport_detail::cells(declaration.at("membraneCells")), retained = transport_detail::cells(declaration.at("retainedMembraneCells"));
    const auto removed = transport_detail::cells(engine.state.at("removed"));
    Value events = Value::array();
    std::map<std::string, std::string> selected{{"left", "cranial_drill"}, {"right", "cusa"}};
    const auto count = std::min(engine.actions.as_array().size(), engine.history.as_array().size());
    for (std::size_t index = 0; index < count; ++index) {
        stop.check(); const auto& action = engine.actions.at(index); const auto& event = engine.history.at(index);
        const auto& hand = action.at("slot").as_string();
        if (transport_detail::text(action, "operation", "exchange")) selected[hand] = action.at("instrument").as_string();
        for (const auto& effect : transport_detail::get(event, "events").as_array()) {
            if (!transport_detail::text(effect, "kind", "membrane_patch_evacuation")) continue;
            const auto* fraction = event.find("completedFraction"), *effect_count = effect.find("cells"), *volume = effect.find("volumeMm3");
            if (!transport_detail::text(event, "status", "accepted") || !fraction || !transport_detail::equal(*fraction, Value(1.)) ||
                    !transport_detail::text(action, "operation", "extract") || selected[hand] != "membrane_forceps" ||
                    !transport_detail::equal(transport_detail::get(effect, "contractHash", Value()), declaration.at("contentHash")) ||
                    !transport_detail::text(effect, "componentHash", digest(declaration.at("membraneCells"))) ||
                    !effect_count || !transport_detail::equal(*effect_count, Value(patch.size())) ||
                    !volume || !transport_detail::equal(*volume, Value(double(patch.size()) * engine.scene->voxel_mm3)))
                throw std::invalid_argument("MEMBRANE_PATCH_NATIVE_EXTRACTION_HISTORY_REQUIRED");
            events.append(effect);
        }
    }
    bool rim_intact = true, held = false;
    std::size_t removed_patch = 0;
    for (auto cell : patch) removed_patch += removed.count(cell);
    for (auto cell : retained) if (removed.count(cell)) rim_intact = false;
    auto membrane = patch; membrane.insert(retained.begin(), retained.end());
    for (const auto* fragment : transport_detail::fragments(engine)) held |= transport_detail::intersects(fragment->at("cells"), membrane);
    queries_detail::no_deformation(engine);
    const bool valid = events.as_array().size() == 1 && transport_detail::equal(transport_detail::get(engine.state, "membranePatchEvacuations"), events) &&
        removed_patch == patch.size() && rim_intact && !held;
    return Value::object({{"cleared", valid}, {"patchSourceCells", patch.size()}, {"removedPatchCells", removed_patch},
        {"removedMembraneMm3", double(removed_patch) * engine.scene->voxel_mm3}, {"retainedRimIntact", rim_intact},
        {"actualNativeExtractionEvents", events.as_array().size()}, {"clinicalValidation", "none"},
        {"method", "source-bound categorical whole-patch cut/translate/extract surrogate; no membrane continuum or clinical calibration"}});
}
}  // namespace membrane_transport
}  // namespace resectionlab::sim
