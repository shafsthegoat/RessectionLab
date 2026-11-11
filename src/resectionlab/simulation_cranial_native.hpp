#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include "simulation_queries_native.hpp"
#include "simulation_snapshot_native.hpp"

namespace resectionlab::sim {
namespace bone_transport {
inline Value describe(State&, const StopToken&);
inline bool aperture_evacuated(State&, const StopToken&);
}
namespace membrane_transport {
inline Value clearance(State&, const StopToken&);
}
namespace guards {
inline bool tool_is_exterior(State&, std::string_view, const StopToken&);
inline bool posed(const State&, std::string_view, const StopToken&);
inline bool inside_entry(const Value&, const std::vector<Capsule>&);
inline bool outside(const State&, const Value&, const std::vector<Capsule>&);
}

namespace cranial {
namespace detail {

[[noreturn]] inline void fail(std::string_view reason) {
    throw std::invalid_argument("CRANIAL_STAGE_" + std::string(reason));
}
inline const Value& get(const Value& value, std::string_view key) {
    static const Value missing;
    const auto* found = value.find(key);
    return found ? *found : missing;
}
inline const Value& array(const Value& value, std::string_view key) {
    static const Value empty = Value::array();
    const auto* found = value.find(key);
    return found ? *found : empty;
}
inline bool same(const Value& a, const Value& b) {
    // Python container equality treats integral floats and bools as numbers.
    const auto numeric = [](const Value& v) { return v.is_number() || v.is_bool(); };
    if (numeric(a) && numeric(b)) {
        const auto integer = [](const Value& v) { return v.is_bool() ? std::int64_t(v.as_bool()) : v.as_int(); };
        if (!a.is_double() && !b.is_double()) return integer(a) == integer(b);
        if (a.is_double() && b.is_double()) return a.as_double() == b.as_double();
        const double d = a.is_double() ? a.as_double() : b.as_double();
        const auto i = a.is_double() ? integer(b) : integer(a);
        return std::isfinite(d) && d >= -9223372036854775808. && d < 9223372036854775808.
            && std::trunc(d) == d && static_cast<std::int64_t>(d) == i;
    }
    if (a.kind() != b.kind()) return false;
    if (a.is_null()) return true;
    if (a.is_bigint()) return a.as_bigint() == b.as_bigint();
    if (a.is_string()) return a.as_string() == b.as_string();
    if (a.is_array()) {
        if (a.as_array().size() != b.as_array().size()) return false;
        for (std::size_t i = 0; i < a.as_array().size(); ++i)
            if (!same(a.at(i), b.at(i))) return false;
        return true;
    }
    if (a.is_object()) {
        if (a.as_object().size() != b.as_object().size()) return false;
        for (const auto& member : a.as_object()) {
            const auto* found = b.find(member.first);
            if (!found || !same(member.second, *found)) return false;
        }
        return true;
    }
    return false;
}
inline bool has_cell(const Value& sorted, std::int64_t cell) {
    const auto& values = sorted.as_array();
    const auto found = std::lower_bound(values.begin(), values.end(), cell,
        [](const Value& a, std::int64_t b) { return a.as_int() < b; });
    return found != values.end() && found->as_int() == cell;
}
inline std::set<std::int64_t> cells(const Value& values, const StopToken& stop) {
    std::set<std::int64_t> result;
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : values.as_array()) { pulse(); result.insert(cell.as_int()); }
    stop.check();
    return result;
}
inline Value cell_array(const std::set<std::int64_t>& values, const StopToken& stop) {
    Value result = Value::array();
    queries_detail::Pulse pulse{stop};
    for (auto cell : values) { pulse(); result.append(cell); }
    stop.check();
    return result;
}
inline bool any_cells(const Value& values, const Value& scope, const StopToken& stop) {
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : values.as_array()) { pulse(); if (has_cell(scope, cell.as_int())) return true; }
    stop.check();
    return false;
}
inline bool fragments_empty(const State& engine) {
    return !engine.state.at("holds").truthy() && !get(engine.state, "detachedFragments").truthy();
}
inline bool in_names(std::string_view value, std::initializer_list<std::string_view> names) {
    for (auto name : names) if (name == value) return true;
    return false;
}
inline bool anchored_components(const Scene& scene, const Value& retained, const Value& anchors,
                                 bool nonempty, const StopToken& stop) {
    auto remaining = cells(retained, stop);
    const auto anchored = cells(anchors, stop);
    if (remaining.empty()) return !nonempty;
    queries_detail::Pulse pulse{stop};
    while (!remaining.empty()) {
        bool supported = false;
        std::vector<std::int64_t> pending{*remaining.begin()};
        remaining.erase(remaining.begin());
        while (!pending.empty()) {
            pulse();
            const auto cell = pending.back(); pending.pop_back();
            supported = supported || anchored.count(cell) != 0;
            const auto at = geometry_detail::coordinates(scene, cell);
            for (int axis = 0; axis < 3; ++axis) for (std::int64_t direction : {-1, 1}) {
                auto neighbor = at; neighbor[axis] += direction;
                if (neighbor[axis] < 0 || static_cast<std::uint64_t>(neighbor[axis]) >= scene.shape[axis]) continue;
                const auto id = (neighbor[0] * static_cast<std::int64_t>(scene.shape[1]) + neighbor[1])
                    * static_cast<std::int64_t>(scene.shape[2]) + neighbor[2];
                const auto found = remaining.find(id);
                if (found != remaining.end()) { pending.push_back(id); remaining.erase(found); }
            }
        }
        if (!supported) return false;
    }
    stop.check();
    return true;
}
inline bool boundary_edge(const Value& boundary, std::int64_t first, std::int64_t second) {
    if (second < first) std::swap(first, second);
    const std::pair<std::int64_t, std::int64_t> edge{first, second};
    const auto& edges = boundary.as_array();
    const auto found = std::lower_bound(edges.begin(), edges.end(), edge, [](const Value& a, const auto& b) {
        return std::make_pair(a.at(0).as_int(), a.at(1).as_int()) < b;
    });
    return found != edges.end() && found->at(0).as_int() == first && found->at(1).as_int() == second;
}
inline bool matching_fragment(const Value& value, const Value& contract) {
    return same(get(value, "kind"), Value("membrane-patch-v1")) &&
        same(get(value, "contractHash"), contract.at("contentHash"));
}
inline Value prefix_slice(const Value& values, std::size_t size) {
    return Value::array(Array(values.as_array().begin(), values.as_array().begin() + size));
}
inline std::vector<std::int64_t> contacts(const State& engine, const Value& slot,
                                         const std::vector<Capsule>& segments, const StopToken& stop) {
    std::set<std::int64_t> result;
    for (const auto& segment : segments) if (segment.part != "shaft") {
        const auto hit = touched(engine, segment.a, segment.b, segment.radius, stop);
        result.insert(hit.begin(), hit.end());
    }
    return {result.begin(), result.end()};
}
inline double inner_boundary(const State& engine, const Value& source_cells, const StopToken& stop) {
    const auto& entry = engine.scene->cranial_access.at("entry");
    const auto normal = queries_detail::point(entry.at("normalRasUnit"));
    const auto center = queries_detail::point(entry.at("centerRasMm"));
    double inner = -std::numeric_limits<double>::infinity();
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : source_cells.as_array()) {
        pulse();
        inner = std::max(inner, geometry_detail::dot(geometry_detail::subtract(
            world_point(*engine.scene, cell.as_int()), center), normal));
    }
    double support = 0.;
    for (int axis = 0; axis < 3; ++axis) {
        const auto& a = engine.scene->affine;
        support += std::abs((a[axis] * normal[0] + a[4 + axis] * normal[1]) + a[8 + axis] * normal[2]);
    }
    stop.check();
    return inner + .5 * support;
}

}  // namespace detail

inline bool is_milling(const Value& contract) {
    return detail::same(detail::get(contract, "schema"), Value("resectionlab.cranial-milling.v1"));
}

inline const Value& validate_access(State& engine, const StopToken& stop) {
    stop.check();
    const auto& contract = engine.scene->cranial_access;
    if (contract.is_null()) throw std::invalid_argument("CRANIOTOMY_CONTRACT_REQUIRED");
    if (!is_milling(contract) && !detail::same(detail::get(contract, "schema"), Value("resectionlab.cranial-access.v1")))
        throw std::invalid_argument("CRANIOTOMY_CONTRACT_INVALID: schema");
    return contract;
}

inline void require_access_evidence(State& engine, const StopToken& stop) {
    const auto& contract = validate_access(engine, stop);
    const bool milling = is_milling(contract);
    const std::string prefix = milling ? "MILLING_" : "CRANIOTOMY_";
    for (auto name : (milling ? std::vector<std::string>{"window", "boneScope", "innerBoundary", "exit"}
                             : std::vector<std::string>{"window", "support", "attachments", "exit"})) {
        stop.check();
        const auto* proof = contract.at("evidence").find(name);
        if (!proof) throw std::invalid_argument(prefix + "COVERAGE_PROOFS_REQUIRED");
        if (!detail::same(detail::get(*proof, "status"), Value("reviewed"))) {
            for (char& c : name) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
            throw std::invalid_argument(prefix + name + "_EVIDENCE_REQUIRED");
        }
    }
    if (milling && contract.at("retainedSkullCells").truthy() && !contract.at("boneAnchorCells").truthy())
        throw std::invalid_argument("MILLING_SUPPORT_EVIDENCE_REQUIRED");
}

inline const Value& validate_stage_contract(State& engine, const StopToken& stop) {
    stop.check();
    const auto& contract = engine.scene->cranial_stages;
    if (contract.is_null() || !detail::same(detail::get(contract, "schema"), Value("resectionlab.cranial-stages.v1")))
        detail::fail("CONTRACT_SCHEMA");
    const auto protection = engine.scene->protection_contract.truthy()
        ? engine.scene->protection_contract.at("contentHash") : Value();
    if (!detail::same(contract.at("sourceSceneId"), Value(engine.scene->identity)) ||
        !detail::same(contract.at("protectionBinding"), protection) ||
        engine.scene->cranial_access.is_null() ||
        !detail::same(contract.at("boneContractHash"), engine.scene->cranial_access.at("contentHash")))
        detail::fail("CONTRACT_SOURCE_CHANGED");
    return contract;
}

inline void require_evidence(State& engine, const StopToken& stop) {
    const auto& contract = validate_stage_contract(engine, stop);
    if (!detail::same(contract.at("startingExposure"), Value("conditional_post_scalp_exposure")))
        detail::fail("INTACT_SCALP_STAGE_UNSUPPORTED");
    for (const auto& proof : contract.at("evidence").as_object()) {
        stop.check();
        if (!detail::same(detail::get(proof.second, "status"), Value("reviewed")) ||
            detail::get(proof.second, "evidenceSha256").is_null())
            detail::fail("REVIEWED_STAGE_EVIDENCE_REQUIRED");
    }
    if (!detail::anchored_components(*engine.scene, contract.at("retainedMembraneCells"),
            contract.at("membraneAnchorCells"), false, stop)) detail::fail("RETAINED_RIM_SUPPORT_REQUIRED");
    for (const auto& attachment : detail::array(engine.scene->properties, "attachments").as_array()) {
        stop.check();
        for (const auto& id : contract.at("attachmentIds").as_array())
            if (detail::same(attachment.at("id"), id) && !detail::same(attachment.at("kind"), Value("soft")))
                detail::fail("ATTACHMENT_KIND_UNSUPPORTED");
    }
    if (!queries_detail::revision(engine.state, "toolMechanicsRevision", 2.) ||
        !queries_detail::revision(engine.state, "physicalClockRevision", 1.)) detail::fail("MECHANICS2_CLOCK1_REQUIRED");
}

inline void history_current(State& engine, const StopToken& stop) {
    if (engine.actions.as_array().size() != engine.history.as_array().size()) detail::fail("NATIVE_HISTORY_REQUIRED");
    if (engine.scene->initial_state_id.empty()) detail::fail("CANONICAL_STAGE_SOURCE_REQUIRED");
    Value previous(engine.scene->initial_state_id);
    for (std::size_t i = 0; i < engine.actions.as_array().size(); ++i) {
        stop.check();
        const auto& receipt = engine.history.at(i);
        if (!detail::same(detail::get(receipt, "action"), engine.actions.at(i)) ||
            !detail::same(receipt.at("stateBefore"), previous)) detail::fail("NATIVE_HISTORY_REQUIRED");
        previous = receipt.at("stateAfter");
    }
    if (!detail::same(previous, Value(state_id(engine)))) detail::fail("NATIVE_CURRENT_HISTORY_REQUIRED");
    stop.check();
}

inline bool verified_bone_prefix(State& engine, bool full, const StopToken& stop) {
    stop.check();
    const auto& contract = engine.scene->cranial_stages;
    const auto& proof = engine.bone_prefix;
    if (contract.is_null() || proof.is_null() ||
        !detail::same(detail::get(proof, "stage_hash"), contract.at("contentHash"))) detail::fail("ACCEPTED_BONE_PREFIX_REQUIRED");
    const auto& position = proof.at("position");
    if (!position.is_int() || position.as_int() <= 0 ||
        static_cast<std::uint64_t>(position.as_int()) > engine.actions.as_array().size() ||
        static_cast<std::uint64_t>(position.as_int()) > engine.history.as_array().size()) detail::fail("ACCEPTED_BONE_PREFIX_REQUIRED");
    const auto size = static_cast<std::size_t>(position.as_int());
    if (!detail::same(engine.history.at(size - 1).at("stateAfter"), proof.at("state_id")) ||
        (full && (digest(detail::prefix_slice(engine.actions, size)) != proof.at("actions_hash").as_string() ||
                  digest(detail::prefix_slice(engine.history, size)) != proof.at("history_hash").as_string())))
        detail::fail("ACCEPTED_BONE_PREFIX_REQUIRED");
    const auto removed = detail::cells(engine.state.at("removed"), stop);
    const auto proof_removed = detail::cells(proof.at("removed_bone_cells"), stop);
    for (const auto& cell : contract.at("boneCells").as_array()) {
        stop.check();
        if (!proof_removed.count(cell.as_int())) detail::fail("ACCEPTED_BONE_PREFIX_REQUIRED");
    }
    for (auto cell : proof_removed) { stop.check(); if (!removed.count(cell)) detail::fail("BONE_REMOVAL_REGRESSED"); }
    return true;
}

inline void prepare_permission(State& engine, const StopToken& stop) {
    const auto status = membrane_transport::clearance(engine, stop);
    engine.stage_permission = Value::object({{"stage_hash", engine.scene->cranial_stages.at("contentHash")},
        {"position", engine.actions.as_array().size()}, {"state_id", state_id(engine)},
        {"dura_cleared", status.at("cleared")}});
    stop.check();
}

inline void prepare_transition(State& engine, const StopToken& stop) {
    stop.check();
    if (engine.scene->cranial_stages.is_null()) return;
    const auto& contract = validate_stage_contract(engine, stop);
    history_current(engine, stop);
    if (!engine.bone_prefix.is_null()) {
        verified_bone_prefix(engine, true, stop);
        prepare_permission(engine, stop);
        return;
    }
    if (engine.actions.as_array().empty()) return;
    const auto status = bone_transport::describe(engine, stop);
    if (!status.truthy() || !detail::get(status, "openingComplete").truthy()) return;
    const auto removed = detail::cells(engine.state.at("removed"), stop);
    std::set<std::int64_t> bone;
    for (auto cell : removed) {
        stop.check();
        if (!detail::has_cell(contract.at("boneCells"), cell)) detail::fail("NONBONE_BEFORE_BONE_BOUNDARY");
        bone.insert(cell);
    }
    if (!detail::fragments_empty(engine)) detail::fail("BONE_BOUNDARY_WITHDRAWAL_REQUIRED");
    Value proof = Value::object({{"stage_hash", contract.at("contentHash")},
        {"position", engine.actions.as_array().size()}, {"state_id", state_id(engine)},
        {"actions_hash", digest(engine.actions)}, {"history_hash", digest(engine.history)},
        {"removed_bone_cells", detail::cell_array(bone, stop)},
        {"opening_summary_json", canonical_json(status)}, {"opening_summary", status}});
    stop.check();
    engine.bone_prefix = std::move(proof);
    prepare_permission(engine, stop);
}

inline bool enforce(State& engine, std::string_view key, std::string_view operation,
                    const Point& position, const Quat& quaternion,
                    const std::vector<std::int64_t>& working, const StopToken& stop) {
    stop.check();
    if (engine.scene->cranial_stages.is_null()) return false;
    const auto& contract = validate_stage_contract(engine, stop);
    require_evidence(engine, stop);
    const auto& slots = engine.state.at("slots").as_object();
    if (engine.bone_prefix.is_null()) {
        for (const auto& hand : slots) {
            stop.check();
            auto contact = hand.first == key ? working : std::vector<std::int64_t>{};
            if (hand.first != key && hand.second.at("active").truthy()) {
                const auto& slot = hand.second;
                contact = detail::contacts(engine, slot, capsule_envelopes(tool(slot.at("instrument").as_string()),
                    parameters(engine, slot), queries_detail::point(slot.at("position")),
                    queries_detail::quaternion(slot.at("quaternion"))), stop);
            }
            for (auto cell : contact) {
                stop.check();
                if (!detail::has_cell(contract.at("boneCells"), cell)) detail::fail("NONBONE_BEFORE_BONE_BOUNDARY");
            }
        }
        return false;
    }
    verified_bone_prefix(engine, false, stop);
    const auto& permission = engine.stage_permission;
    if (permission.is_null() ||
        !detail::same(detail::get(permission, "stage_hash"), contract.at("contentHash")) ||
        !detail::same(detail::get(permission, "position"), Value(engine.actions.as_array().size())) ||
        engine.history.as_array().empty() ||
        !detail::same(detail::get(permission, "state_id"), engine.history.as_array().back().at("stateAfter")))
        detail::fail("NATIVE_STAGE_PREPARATION_REQUIRED");
    const bool cleared = permission.at("dura_cleared").truthy();
    const auto& bone = engine.scene->cranial_access;
    const auto& entry = bone.at("entry");
    const auto normal = queries_detail::point(entry.at("normalRasUnit"));
    const auto center = queries_detail::point(entry.at("centerRasMm"));
    const double membrane_inner = detail::inner_boundary(engine, contract.at("membraneCells"), stop);
    for (const auto& hand : slots) {
        stop.check();
        const auto& slot = hand.second;
        if (!guards::posed(engine, hand.first, stop) && (hand.first != key ||
            detail::in_names(operation, {"exchange", "inspect", "deactivate"}))) continue;
        const auto& name = slot.at("instrument").as_string();
        const auto& instrument = tool(name);
        const auto params = parameters(engine, slot);
        const double expected_depth = name == "dural_scissors" ? contract.at("scissorsDepthMm").as_number() : 2.;
        const bool family = params.at("radiusMm").as_number() == instrument.radius &&
            params.at("depthMm").as_number() == expected_depth;
        const auto end = hand.first == key ? position : queries_detail::point(slot.at("position"));
        const auto q = hand.first == key ? quaternion : queries_detail::quaternion(slot.at("quaternion"));
        const auto segments = hand.first == key
            ? swept_envelopes(instrument, params, queries_detail::point(slot.at("position")),
                queries_detail::quaternion(slot.at("quaternion")), end, q)
            : capsule_envelopes(instrument, params, end, q);
        if (guards::outside(engine, bone.at("exit"), segments)) detail::fail("OUTSIDE_REVIEWED_WORKSPACE");
        if (guards::inside_entry(bone, segments)) {
            if (!family) detail::fail("FROZEN_STOCK_PARAMETER_FAMILY");
            const auto axis = apply_rotation(rotation_matrix(normalized_quaternion(q)), Point{0., 0., 1.});
            const double radians = entry.at("maximumAngleDegrees").as_number() * (3.14159265358979323846 / 180.);
            if (geometry_detail::dot(axis, normal) < std::cos(radians) - 1e-9) detail::fail("ATTITUDE_OUTSIDE");
            const bool membrane_tool = detail::in_names(name, {"dural_scissors", "dissector", "membrane_forceps", "frazier_suction"});
            const bool work_tool = detail::in_names(name, {"cusa", "brain_retractor", "frazier_suction"});
            if (!membrane_tool && !(cleared && work_tool)) detail::fail(cleared ? "TOOL_SCOPE" : "MEMBRANE_CLEARANCE_REQUIRED");
            const double maximum = cleared ? entry.at("maximumInwardMm").as_number() : membrane_inner;
            for (const auto& segment : segments)
                if (std::max(geometry_detail::dot(geometry_detail::subtract(segment.a, center), normal),
                    geometry_detail::dot(geometry_detail::subtract(segment.b, center), normal)) + segment.radius > maximum + 1e-9)
                    detail::fail("BEYOND_STAGE_INNER_BOUNDARY");
        }
        auto contact = hand.first == key ? working : std::vector<std::int64_t>{};
        if (hand.first != key && slot.at("active").truthy()) contact = detail::contacts(engine, slot, segments, stop);
        if (contact.empty()) continue;
        if (detail::in_names(name, {"dural_scissors", "dissector", "membrane_forceps"})) {
            queries_detail::Pulse pulse{stop};
            for (auto cell : contact) {
                pulse();
                if (!detail::has_cell(contract.at("membraneCells"), cell) &&
                    !detail::has_cell(contract.at("retainedMembraneCells"), cell)) detail::fail("MEMBRANE_TOOL_SOURCE_SCOPE");
            }
        } else if (detail::in_names(name, {"cusa", "brain_retractor"})) {
            if (!cleared) detail::fail("MEMBRANE_CLEARANCE_REQUIRED");
            queries_detail::Pulse pulse{stop};
            for (auto cell : contact) { pulse(); if (!detail::has_cell(contract.at("workCells"), cell)) detail::fail("WORK_SOURCE_SCOPE"); }
        } else if (name == "frazier_suction") detail::fail("SUCTION_REQUIRES_RETAINED_VOID");
        else detail::fail("TOOL_SOURCE_SCOPE");
    }
    stop.check();
    return true;
}

inline void validate_effects(State& engine, const StopToken& stop) {
    stop.check();
    if (engine.scene->cranial_stages.is_null()) return;
    const auto& contract = validate_stage_contract(engine, stop);
    const auto& rim = contract.at("retainedMembraneCells");
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : engine.state.at("removed").as_array()) {
        pulse(); const auto id = cell.as_int();
        if (!detail::has_cell(contract.at("boneCells"), id) && !detail::has_cell(contract.at("membraneCells"), id) &&
            !detail::has_cell(contract.at("workCells"), id)) detail::fail("REMOVAL_SOURCE_SCOPE");
    }
    if (detail::any_cells(engine.state.at("removed"), rim, stop)) detail::fail("RETAINED_RIM_REMOVAL");
    const auto* deformation = engine.state.find("deformation");
    if (deformation) for (const auto& item : deformation->as_object()) {
        pulse();
        if (detail::has_cell(rim, std::stoll(item.first)) && point_norm(queries_detail::point(item.second)) > 0.)
            detail::fail("RETAINED_RIM_DEFORMATION");
    }
    queries_detail::each_hold(engine, true, [&](const Value& hold) {
        pulse();
        if (detail::any_cells(hold.at("cells"), rim, stop)) detail::fail("RETAINED_RIM_HOLD");
    });
    for (auto field : {"cutEdges", "transectedEdges"}) for (const auto& edge : detail::array(engine.state, field).as_array()) {
        pulse();
        bool membrane = false;
        for (const auto& id : edge.as_array()) membrane = membrane || detail::has_cell(contract.at("membraneCells"), id.as_int()) || detail::has_cell(rim, id.as_int());
        if (membrane && (edge.as_array().size() != 2 || !detail::boundary_edge(contract.at("boundaryEdges"), edge.at(0).as_int(), edge.at(1).as_int())))
            detail::fail("RETAINED_MEMBRANE_TOPOLOGY");
    }
    std::set<std::string> allowed;
    for (const auto& id : contract.at("attachmentIds").as_array()) { pulse(); allowed.insert(id.as_string()); }
    for (const auto& attachment : detail::array(engine.scene->properties, "attachments").as_array()) {
        pulse();
        if (detail::in_names(attachment.at("kind").as_string(), {"soft", "periosteal"}) &&
            detail::any_cells(attachment.at("cells"), contract.at("boneCells"), stop)) allowed.insert(attachment.at("id").as_string());
    }
    for (const auto& id : engine.state.at("separatedAttachments").as_array()) {
        pulse(); if (!allowed.count(id.as_string())) detail::fail("ATTACHMENT_RELEASE_SOURCE_SCOPE");
    }
    stop.check();
}

inline bool tool_is_exterior(State& engine, std::string_view key, const StopToken& stop) {
    return guards::tool_is_exterior(engine, key, stop);
}

inline Value stage_readiness(State& engine, const StopToken& stop) {
    stop.check();
    const auto& contract = engine.scene->cranial_stages;
    if (contract.is_null()) return Value();
    Value result = Value::object({{"schema", "resectionlab.cranial-stages.v1"}, {"contractHash", contract.at("contentHash")},
        {"stateId", state_id(engine)}, {"sceneId", engine.scene->identity}, {"phase", "BOUND"}, {"assessmentCurrent", false},
        {"startingExposure", contract.at("startingExposure")}, {"sourceUnknowns", Value::array({"Uncalibrated membrane mechanics", "Clinical outcomes unknown"})},
        {"patchCells", contract.at("membraneCells").as_array().size()}, {"requiredBoundaryEdges", contract.at("boundaryEdges").as_array().size()},
        {"cutBoundaryEdges", 0}, {"boundaryCutComplete", false}, {"patchHeld", false}, {"patchDetached", false},
        {"patchExtracted", false}, {"bonePrefixVerified", false}, {"duraCleared", false}, {"workEntrySupported", false},
        {"wholeExitComplete", false}, {"clinicalValidation", "none"}, {"reason", Value()}});
    try {
        validate_stage_contract(engine, stop);
        std::set<std::pair<std::int64_t, std::int64_t>> cuts;
        for (const auto& edge : detail::array(engine.state, "cutEdges").as_array()) {
            stop.check();
            if (edge.as_array().size() == 2) cuts.emplace(std::min(edge.at(0).as_int(), edge.at(1).as_int()), std::max(edge.at(0).as_int(), edge.at(1).as_int()));
        }
        std::size_t count = 0;
        for (const auto& edge : contract.at("boundaryEdges").as_array()) {
            stop.check(); if (cuts.count({edge.at(0).as_int(), edge.at(1).as_int()})) ++count;
        }
        result.set("cutBoundaryEdges", count);
        result.set("boundaryCutComplete", count == contract.at("boundaryEdges").as_array().size());
        for (const auto& hold : engine.state.at("holds").as_object()) {
            stop.check(); if (detail::matching_fragment(hold.second, contract)) result.set("patchHeld", true);
        }
        for (const auto& hold : detail::array(engine.state, "detachedFragments").as_array()) {
            stop.check(); if (detail::matching_fragment(hold, contract)) result.set("patchDetached", true);
        }
        for (const auto& proof : contract.at("evidence").as_object()) if (!detail::same(proof.second.at("status"), Value("reviewed")))
            result.at("sourceUnknowns").append(proof.first + " evidence " + proof.second.at("status").as_string());
        require_evidence(engine, stop); history_current(engine, stop); result.set("assessmentCurrent", true);
        prepare_transition(engine, stop); verified_bone_prefix(engine, true, stop); result.set("bonePrefixVerified", true);
        const auto membrane = membrane_transport::clearance(engine, stop);
        result.set("membrane", membrane);
        const bool cleared = membrane.at("cleared").truthy();
        result.set("duraCleared", cleared); result.set("patchExtracted", cleared); result.set("workEntrySupported", cleared);
        bool withdrawn = cleared && detail::fragments_empty(engine);
        if (withdrawn) for (const auto& hand : engine.state.at("slots").as_object()) {
            stop.check();
            const auto& slot = hand.second;
            if (slot.at("inserted").truthy() || slot.at("active").truthy() || slot.at("grip").truthy() ||
                (guards::posed(engine, hand.first, stop) && !guards::tool_is_exterior(engine, hand.first, stop))) { withdrawn = false; break; }
        }
        result.set("wholeExitComplete", withdrawn);
        if (!cleared) result.set("reason", "MEMBRANE_CLEARANCE_REQUIRED");
    } catch (const std::invalid_argument& error) { result.set("reason", error.what()); }
    result.set("phase", result.at("wholeExitComplete").truthy() ? "WITHDRAWN" : result.at("patchExtracted").truthy() ? "EXTRACTED" :
        result.at("patchDetached").truthy() ? "RELEASED_IN_FIELD" : result.at("patchHeld").truthy() ? "HELD" :
        result.at("boundaryCutComplete").truthy() ? "CUT_RETAINED" : result.at("cutBoundaryEdges").truthy() ? "MEMBRANE_CUTTING" :
        result.at("bonePrefixVerified").truthy() ? "MEMBRANE_READY" : engine.actions.truthy() ? "BONE_WORK" : "BOUND");
    stop.check();
    return result;
}

inline void source_validate(State& engine, const StopToken& stop) {
    const auto& contract = validate_access(engine, stop);
    const bool milling = is_milling(contract);
    const std::string prefix = milling ? "MILLING_" : "CRANIOTOMY_";
    if (milling) {
        require_access_evidence(engine, stop);
        const auto& v2 = detail::get(engine.scene->properties, "tissuePhysicsV2");
        if (!detail::get(v2, "bone").is_object()) throw std::invalid_argument("MILLING_V2_BONE_PROFILE_REQUIRED");
    }
    const auto& first = contract.at(milling ? "apertureCells" : "flapCells");
    const auto& second = milling ? first : contract.at("kerfCells");
    const auto intended = [&](std::int64_t cell) { return detail::has_cell(first, cell) || detail::has_cell(second, cell); };
    queries_detail::Pulse pulse{stop};
    for (auto cell : engine.scene->original_target_cells) {
        pulse(); if (intended(cell)) throw std::invalid_argument(prefix + "TARGET_OVERLAP");
    }
    if (engine.scene->original_target_cells.empty()) throw std::invalid_argument("CRANIOTOMY_NONEMPTY_TARGET_EVIDENCE_REQUIRED");
    for (auto cell : engine.scene->protected_cells) {
        pulse(); if (intended(cell)) throw std::invalid_argument(prefix + "PROTECTED_STRUCTURE");
    }
    for (const auto& row : detail::array(engine.scene->source, "protectedAnatomyV1").as_array())
        for (const auto& cell : row.at("cells").as_array()) { pulse(); if (intended(cell.as_int())) throw std::invalid_argument(prefix + "PROTECTED_STRUCTURE"); }
    for (std::size_t cell = 0; cell < engine.scene->labels.size(); ++cell) {
        pulse(); const auto label = engine.scene->labels[cell];
        if (!label) continue;
        const auto& region = engine.scene->region(label);
        const auto& a = detail::get(region, "protected"); const auto& b = detail::get(region, "critical");
        if (((a.is_bool() && a.as_bool()) || (b.is_bool() && b.as_bool())) && intended(static_cast<std::int64_t>(cell)))
            throw std::invalid_argument(prefix + "PROTECTED_STRUCTURE");
    }
    if (!milling) { stop.check(); return; }
    const auto& retained = contract.at("retainedSkullCells");
    std::set<std::int64_t> removed;
    for (const auto& cell : engine.state.at("removed").as_array()) {
        pulse();
        if (!cell.is_int() || cell.as_int() < 0 || static_cast<std::uint64_t>(cell.as_int()) >= engine.scene->labels.size() || !removed.insert(cell.as_int()).second)
            throw std::invalid_argument("MILLING_REMOVED_LEDGER_INVALID");
    }
    if (detail::any_cells(engine.state.at("removed"), retained, stop)) throw std::invalid_argument("MILLING_RETAINED_SKULL_CHANGED");
    queries_detail::each_hold(engine, true, [&](const Value& hold) {
        if (detail::any_cells(detail::array(hold, "cells"), retained, stop)) throw std::invalid_argument("MILLING_RETAINED_SKULL_CHANGED");
    });
    const auto* deformation = engine.state.find("deformation");
    if (deformation) for (const auto& item : deformation->as_object()) {
        pulse();
        if (detail::has_cell(retained, std::stoll(item.first)) && item.second.truthy())
            for (const auto& component : item.second.as_array()) if (component.truthy()) throw std::invalid_argument("MILLING_RETAINED_SKULL_CHANGED");
    }
    if (!detail::anchored_components(*engine.scene, retained, contract.at("boneAnchorCells"), true, stop))
        throw std::invalid_argument("MILLING_RETAINED_SUPPORT_REQUIRED");
    stop.check();
}

}  // namespace cranial
}  // namespace resectionlab::sim
