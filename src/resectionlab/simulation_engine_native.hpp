#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_cusa_native.hpp"
#include "simulation_effects_native.hpp"
#include "simulation_snapshot_native.hpp"

namespace resectionlab::sim {

namespace cranial {
void prepare_transition(State&, const StopToken&);
void source_validate(State&, const StopToken&);
}
namespace guards { void require_exterior_withdrawal(State&, std::string_view, const StopToken&); }
namespace bone_transport {
void validate_holder(State&, const std::string&);
Value extract(State&, const std::string&, const StopToken&);
}
namespace membrane_transport {
void validate_holder(State&, const std::string&);
Value extract(State&, const std::string&, const StopToken&);
}
Value physical_substep(State&, std::string_view, std::string_view, Point, Quat, bool, double, const StopToken&);

namespace engine_detail {

inline void extend(Value& destination, Value source) {
    auto& target = destination.as_array();
    auto& items = source.as_array();
    target.insert(target.end(), std::make_move_iterator(items.begin()), std::make_move_iterator(items.end()));
}
inline bool is_text(const Value& owner, std::string_view key, std::string_view text) {
    const auto* value = owner.find(key);
    return value && value->is_string() && value->as_string() == text;
}
inline bool flag(const Value& owner, std::string_view key) {
    const auto* value = owner.find(key); return value && value->truthy();
}
inline bool revised(const State& engine) {
    const auto* value = engine.state.find("toolMechanicsRevision");
    return value && value->is_int() && (value->as_int() == 1 || value->as_int() == 2);
}
inline Point point(const Value& value) {
    if (!value.is_array() || value.as_array().size() != 3)
        throw std::invalid_argument("Finite coordinates with the declared dimension are required");
    Point result;
    for (int i = 0; i < 3; ++i) {
        if (!value.at(i).is_number() || !std::isfinite(value.at(i).as_number()))
            throw std::invalid_argument("Finite coordinates with the declared dimension are required");
        result[i] = value.at(i).as_number();
    }
    return result;
}
inline Quat quaternion(const Value& value) {
    if (!value.is_array() || value.as_array().size() != 4)
        throw std::invalid_argument("Finite coordinates with the declared dimension are required");
    Quat result;
    double norm_squared = 0.;
    for (int i = 0; i < 4; ++i) {
        if (!value.at(i).is_number() || !std::isfinite(value.at(i).as_number()))
            throw std::invalid_argument("Finite coordinates with the declared dimension are required");
        result[i] = value.at(i).as_number(); norm_squared += result[i] * result[i];
    }
    if (std::abs(std::sqrt(norm_squared) - 1.) > 1e-5)
        throw std::invalid_argument("Orientation must be a unit xyzw quaternion");
    return result;
}
inline Value value(const Point& p) { return Value::array({p[0], p[1], p[2]}); }
inline Value value(const Quat& q) { return Value::array({q[0], q[1], q[2], q[3]}); }
inline Point difference(const Point& a, const Point& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline double distance(const Value& a, const Value& b) { return point_norm(difference(point(a), point(b))); }
inline bool one_of(std::string_view value, std::initializer_list<std::string_view> values) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
inline Value metrics(const State& engine, const StopToken& stop) {
    auto base = legacy_metrics(engine, stop);
    Value result = Value::object();
    for (auto& item : base.as_object()) {
        if (item.first == "removedByLabelMm3" && engine.state.contains("cusaV3")) {
            auto extra = cusa_v3::metrics(engine);
            for (auto& field : extra.as_object()) result.set(field.first, std::move(field.second));
        }
        result.set(item.first, std::move(item.second));
    }
    return result;
}
inline Value reason_code(const Value& reason, bool invalid = false) {
    if (reason.is_null() || !reason.truthy()) return {};
    const auto& text = reason.as_string();
    const auto code = text.substr(0, text.find(':'));
    if (invalid) {
        bool cased = false;
        for (unsigned char ch : code) {
            if (ch >= 'a' && ch <= 'z') return "INVALID_ACTION";
            if (ch >= 'A' && ch <= 'Z') cased = true;
        }
        if (!cased) return "INVALID_ACTION";
    }
    return code;
}
struct ValidatedAction { double duration; Value points; };
inline ValidatedAction validate(const State& engine, const Value& action) {
    if (!action.is_object()) throw std::invalid_argument("Unsupported action fields");
    for (const auto& item : action.as_object()) if (!one_of(item.first,
        {"slot", "operation", "instrument", "pose", "path", "parameters", "duration", "position",
         "mechanicsRevision", "clockRevision", "maneuverPhase"}))
        throw std::invalid_argument("Unsupported action fields");
    const auto* key = action.find("slot");
    if (!key || !key->is_string() || !engine.state.at("slots").contains(key->as_string()))
        throw std::invalid_argument("Choose left or right instrument slot");
    const auto* operation = action.find("operation");
    if (!operation || !operation->is_string() || !one_of(operation->as_string(),
        {"place", "move", "activate", "deactivate", "apply", "hold", "grasp", "release", "exchange",
         "withdraw", "extract", "inspect", "configure", "position", "bladeManeuver"}))
        throw std::invalid_argument("Unsupported instrument operation");
    const auto& op = operation->as_string();
    if (action.contains("maneuverPhase") && op != "bladeManeuver")
        throw std::invalid_argument("PLANAR_MANEUVER_PHASE_REQUIRES_MANEUVER_ACTION");
    const auto* raw_duration = action.find("duration");
    const double duration = raw_duration ? (raw_duration->is_bool() ? double(raw_duration->as_bool()) : raw_duration->as_number()) : 1.;
    if (!std::isfinite(duration) || !(0. < duration && duration <= 30.))
        throw std::invalid_argument("Duration must be in (0, 30] seconds");
    if (engine.actions.as_array().size() >= 8192)
        throw std::invalid_argument("Action limit reached; save and start a new episode");
    Value points = Value::array();
    if (const auto* path = action.find("path")) points = *path;
    else if (const auto* pose = action.find("pose")) points.append(*pose);
    if (const auto* rev = action.find("mechanicsRevision"))
        if (op != "configure" || !rev->is_int() || (rev->as_int() != 1 && rev->as_int() != 2))
            throw std::invalid_argument("INVALID_INSTRUMENT_MECHANICS_REVISION");
    if (const auto* rev = action.find("clockRevision"))
        if (op != "configure" || !rev->is_int() || rev->as_int() != 1)
            throw std::invalid_argument("INVALID_PHYSICAL_CLOCK_REVISION");
    if (action.contains("position") && op != "position")
        throw std::invalid_argument("POSITION_PAYLOAD_REQUIRES_POSITION_OPERATION");
    if (op == "configure" || op == "position") {
        if (points.truthy() || flag(action, "parameters") || flag(action, "instrument"))
            throw std::invalid_argument("SETUP_CANNOT_MOVE_INSTRUMENT");
    }
    if (!points.is_array() || points.as_array().size() > 512)
        throw std::invalid_argument("Path requires at most 512 poses");
    for (const auto& pose : points.as_array()) {
        if (!pose.is_object() || pose.as_object().size() != 2 || !pose.contains("position") || !pose.contains("quaternion"))
            throw std::invalid_argument("Each pose needs position and quaternion");
        point(pose.at("position")); quaternion(pose.at("quaternion"));
    }
    const auto* params = action.find("parameters");
    if (params) {
        if (!params->is_object()) throw std::invalid_argument("Unsupported instrument parameters");
        for (const auto& item : params->as_object()) if (!one_of(item.first,
            {"radiusMm", "depthMm", "flowMm3PerSecond", "power", "jawOpeningMm", "amplitudeMa", "frequencyHz", "pulseWidthUs"}))
            throw std::invalid_argument("Unsupported instrument parameters");
        for (const auto& item : params->as_object()) {
            double lo = 0., hi = 1000.;
            if (item.first == "radiusMm") { lo = .1; hi = 12.; }
            else if (item.first == "depthMm") { lo = .1; hi = 20.; }
            else if (item.first == "power") hi = 10.;
            else if (item.first == "jawOpeningMm") hi = 12.;
            else if (item.first == "amplitudeMa") hi = 20.;
            else if (item.first == "frequencyHz") hi = 500.;
            if (!item.second.is_number() || !std::isfinite(item.second.as_number()) ||
                item.second.as_number() < lo || item.second.as_number() > hi)
                throw std::invalid_argument("Invalid parameter: " + item.first);
        }
    }
    return {duration, std::move(points)};
}

inline Value execute(State& engine, const Value& action, double duration, Value points, const StopToken& stop) {
    const auto key = action.at("slot").as_string(), op = action.at("operation").as_string();
    auto& slot = engine.state.at("slots").at(key);
    const double previous_jaw = parameters(engine, slot).at("jawOpeningMm").as_number();
    Value events = Value::array();
    if (op == "configure" || op == "position") {
        for (const auto& hand : engine.state.at("slots").as_object())
            if (flag(hand.second, "active") || flag(hand.second, "inserted"))
                throw std::invalid_argument("WITHDRAW_BOTH_TOOLS_BEFORE_SETUP");
        if (engine.state.at("holds").truthy()) throw std::invalid_argument("WITHDRAW_BOTH_TOOLS_BEFORE_SETUP");
        if (op == "position") throw std::invalid_argument("POSITION_REQUIRES_V2_MECHANICS");
        if (action.contains("mechanicsRevision") || !action.contains("clockRevision")) {
            const auto* revision = action.find("mechanicsRevision");
            engine.state.set("toolMechanicsRevision", revision ? *revision : Value(1));
            cusa_v3::initialize(engine);
            events.append(Value::object({{"kind", "instrument_mechanics_enabled"}, {"revision", engine.state.at("toolMechanicsRevision")}}));
        }
        if (action.contains("clockRevision")) {
            engine.state.set("physicalClockRevision", 1);
            events.append(Value::object({{"kind", "physical_clock_enabled"}, {"revision", 1}}));
        }
        return Value::object({{"status", "accepted"}, {"reason", {}}, {"completedFraction", 1.}, {"events", events}});
    }
    if (op == "extract" && revised(engine)) {
        const auto& name = slot.at("instrument").as_string();
        if (!one_of(name, {"tumor_forceps", "bone_flap_holder", "membrane_forceps"}) || points.truthy())
            throw std::invalid_argument("UNSUPPORTED_EXTRACTION");
        if (!engine.state.at("holds").contains(key)) throw std::invalid_argument("GRASP_BEFORE_EXTRACTION");
        events = name == "bone_flap_holder" ? bone_transport::extract(engine, key, stop)
            : name == "membrane_forceps" ? membrane_transport::extract(engine, key, stop)
            : extract_legacy_fragment(engine, key, stop);
        slot.set("active", false);
        const auto count = static_cast<std::size_t>(std::max(1., std::ceil(duration / .1)));
        for (std::size_t i = 0; i < count; ++i) {
            stop.check(); extend(events, physical_substep(engine, key, "deactivate", point(slot.at("position")),
                quaternion(slot.at("quaternion")), false, duration / count, stop));
        }
        return Value::object({{"status", "accepted"}, {"reason", {}}, {"completedFraction", 1.}, {"events", events}});
    }
    if (op == "exchange") {
        if (flag(slot, "inserted") || flag(slot, "active") || flag(slot, "grip") || engine.state.at("holds").contains(key))
            throw std::invalid_argument("WITHDRAW_BEFORE_EXCHANGE: deactivate, release, and withdraw first");
        const auto* name = action.find("instrument");
        bool known = false;
        if (name && name->is_string()) for (const auto& t : tools) if (t.id == name->as_string()) known = true;
        if (!known) throw std::invalid_argument("UNKNOWN_INSTRUMENT");
        slot.set("instrument", *name); slot.set("parameters", Value::object());
    }
    if (const auto* settings = action.find("parameters"))
        for (const auto& item : settings->as_object()) slot.at("parameters").set(item.first, item.second);
    const auto& instrument = tool(slot.at("instrument").as_string());
    const auto params = parameters(engine, slot);
    if (op != "exchange") {
        if (instrument.id == "bone_flap_holder") bone_transport::validate_holder(engine, key);
        if (instrument.id == "membrane_forceps") membrane_transport::validate_holder(engine, key);
    }
    const double jaw_target = params.at("jawOpeningMm").as_number();
    const bool articulate = revised(engine) && one_of(instrument.geometry, {"forceps", "scissors"}) && previous_jaw != jaw_target;
    if (articulate) slot.at("parameters").set("jawOpeningMm", previous_jaw);
    const bool cusa = instrument.id == "cusa" && cusa_v3::profile(engine) != nullptr;
    if (cusa) cusa_v3::validate_parameters(params);
    const Value progress_reason = cusa ? Value("TISSUE_ASPIRATION_IN_PROGRESS") : Value();
    if ((op == "hold" && instrument.effect != "retract") ||
        (one_of(op, {"grasp", "extract"}) && instrument.effect != "grasp"))
        throw std::invalid_argument("UNSUPPORTED_OPERATION_FOR_TOOL");
    if (one_of(op, {"exchange", "release", "deactivate", "activate", "inspect", "extract"}) && points.truthy())
        throw std::invalid_argument("This operation cannot also move the instrument");
    if (op == "deactivate") slot.set("active", false);
    if (op == "activate") slot.set("active", true);
    if (op == "release") {
        if (const auto* retained = engine.state.at("holds").find(key)) {
            if (!retained->contains("target") && revised(engine)) {
                auto* detached = engine.state.find("detachedFragments");
                if (!detached) detached = &engine.state.set("detachedFragments", Value::array());
                detached->append(*retained);
            }
        }
        engine.state.at("holds").erase(key); slot.set("grip", Value::array()); slot.set("active", false);
        occupancy(engine, stop);
    }
    if (op == "place") {
        if (flag(slot, "inserted") || flag(slot, "active") || flag(slot, "grip"))
            throw std::invalid_argument("WITHDRAW_BEFORE_PLACE");
        if (points.as_array().size() != 1) throw std::invalid_argument("Placement requires one pose in free space");
        const auto& pose = points.at(0); const auto p = point(pose.at("position"));
        const auto axis = apply_rotation(rotation_matrix(normalized_quaternion(quaternion(pose.at("quaternion")))), {0.,0.,1.});
        const Point before{p[0]-axis[0]*40.,p[1]-axis[1]*40.,p[2]-axis[2]*40.};
        if (!touched(engine, before, p, std::max(params.at("radiusMm").as_number(), instrument.shaft_radius), stop).empty())
            throw std::invalid_argument("PLACE_OUTSIDE_MATERIAL: insert by a checked move or active stroke");
        slot.set("position", pose.at("position")); slot.set("quaternion", pose.at("quaternion"));
        points = Value::array();
    }
    if (op == "withdraw" && !points.truthy()) {
        const auto p = point(slot.at("position"));
        const auto axis = apply_rotation(rotation_matrix(normalized_quaternion(quaternion(slot.at("quaternion")))), {0.,0.,1.});
        points.append(Value::object({{"position", value(Point{p[0]-50.*axis[0],p[1]-50.*axis[1],p[2]-50.*axis[2]})},
            {"quaternion", slot.at("quaternion")}}));
    }
    if (op == "withdraw") {
        if (flag(slot, "grip") || engine.state.at("holds").contains(key)) throw std::invalid_argument("RELEASE_BEFORE_WITHDRAWAL");
        slot.set("active", false);
    }
    bool active = one_of(op, {"apply", "hold", "grasp", "extract", "activate"}) ||
        (flag(slot, "active") && op == "move") || (op == "inspect" && instrument.effect == "probe");
    if (const auto* clock = engine.state.find("physicalClockRevision"))
        if (clock->is_int() && clock->as_int() == 1) active = active || flag(slot, "active");
    Value poses = Value::array({Value::object({{"position", slot.at("position")}, {"quaternion", slot.at("quaternion")}})});
    for (const auto& pose : points.as_array()) poses.append(pose);
    if (poses.as_array().size() == 1) poses.append(poses.at(0));
    struct Segment { Value first, last; std::size_t count; double length; QuaternionTrajectory rotation; };
    std::vector<Segment> segments;
    std::size_t total = 0;
    double length = 0.;
    const double step = std::min(.4, *std::min_element(engine.scene->spacing.begin(), engine.scene->spacing.end()) / 3.);
    for (std::size_t i = 1; i < poses.as_array().size(); ++i) {
        const auto& a = poses.at(i-1); const auto& b = poses.at(i);
        const double d = distance(b.at("position"), a.at("position"));
        const double angle = rotation_angle(quaternion(a.at("quaternion")), quaternion(b.at("quaternion")));
        const double raw_count = std::max({1., std::ceil((d+40.*angle)/step),
            std::ceil(duration / (poses.as_array().size()-1) / .1),
            articulate ? std::ceil(std::abs(jaw_target-previous_jaw)/step/(poses.as_array().size()-1)) : 1.});
        if (!std::isfinite(raw_count)) throw std::overflow_error("cannot convert float infinity to integer");
        const auto count = static_cast<std::size_t>(std::min(raw_count,20001.));
        total += count; length += d;
        segments.push_back({a,b,count,d,QuaternionTrajectory::between(quaternion(a.at("quaternion")),quaternion(b.at("quaternion")))});
    }
    if (total > 20000) throw std::invalid_argument("PATH_TOO_LONG: split this stroke into shorter actions");
    for (const auto& segment : segments) if (segment.length > 250.) throw std::invalid_argument("PATH_TOO_LONG");
    std::size_t done = 0;
    ++execution_stats.trajectories;
    double completed_length = 0.;
    Value reason, contact, path_fraction;
    for (const auto& segment : segments) {
        const auto a = point(segment.first.at("position")), b = point(segment.last.at("position"));
        for (std::size_t j = 0; j < segment.count; ++j) {
            stop.check(); const double t = double(j+1)/segment.count;
            const Point p{(1.-t)*a[0]+t*b[0],(1.-t)*a[1]+t*b[1],(1.-t)*a[2]+t*b[2]};
            const auto q = segment.rotation.at(t);
            Value old = engine.state;
            try {
                if (articulate) engine.state.at("slots").at(key).at("parameters").set("jawOpeningMm",
                    done+1 == total ? jaw_target : previous_jaw+(jaw_target-previous_jaw)*(done+1)/total);
                extend(events, physical_substep(engine, key, op, p, q, active, duration/total, stop));
            } catch (const std::invalid_argument& error) {
                stop.check(); const std::string text = error.what(); const auto code = text.substr(0,text.find(':'));
                if (code == "PROTECTED_STRUCTURE" || !engine.scene->cranial_access.is_null() ||
                    code.rfind("CRANIAL_ACCESS_",0)==0 || code.rfind("CRANIOTOMY_",0)==0 || code.rfind("BONE_FLAP_",0)==0) throw;
                engine.state = std::move(old); ++execution_stats.rollbacks; reason = text; contact = value(p); break;
            }
            ++done;
        }
        if (!reason.is_null()) break;
        if (segments.size() > 1 && active && progress_reason.truthy()) {
            const double remaining = distance(engine.state.at("slots").at(key).at("position"),segment.last.at("position"));
            if (remaining > 1e-5) {
                path_fraction = (completed_length+std::max(0.,segment.length-remaining))/std::max(length,1e-15);
                reason = progress_reason; contact = engine.state.at("slots").at(key).at("position");
                events.append(Value::object({{"kind", "cusa_contact_dwell"}, {"position", contact},
                    {"remainingPathMm", length*(1.-path_fraction.as_number())}})); break;
            }
        }
        completed_length += segment.length;
    }
    if (!reason.is_null()) events.append(Value::object({{"kind", "contact_stop"}, {"reason", reason}, {"position", contact}}));
    if (done == 0) return Value::object({{"status", "rejected"}, {"reason", reason}, {"completedFraction", 0.}, {"events", events}});
    auto& final_slot = engine.state.at("slots").at(key);
    if (op == "withdraw" && reason.is_null()) {
        if (!engine.scene->cranial_access.is_null()) guards::require_exterior_withdrawal(engine,key,stop);
        final_slot.set("inserted", false);
    }
    if (one_of(op,{"hold","grasp"})) if (auto* hold = engine.state.at("holds").find(key))
        if (!hold->contains("target") && !is_text(*hold,"kind","bone-flap-v1")) hold->set("origin",final_slot.at("position"));
    if (active && !events.truthy()) events.append(Value::object({{"kind","no_material_effect"},
        {"reason","No compatible structure intersected by the working tip"}}));
    events.as_array().insert(events.as_array().begin(), Value::object({{"kind",op},{"instrument",tool_name(instrument.id)},{"slot",key}}));
    double fraction = path_fraction.is_null() ? double(done)/total : path_fraction.as_number();
    if (reason.is_null() && points.truthy() && active && progress_reason.truthy()) {
        const double remaining = distance(final_slot.at("position"),points.as_array().back().at("position"));
        if (remaining > 1e-5) {
            fraction = std::max(0.,1.-remaining/std::max(length,remaining)); reason = progress_reason; contact = final_slot.at("position");
            events.append(Value::object({{"kind","cusa_contact_dwell"},{"position",contact},{"remainingPathMm",remaining}}));
        }
    }
    return Value::object({{"status",reason.is_null()?"accepted":"partial"},{"reason",reason},
        {"completedFraction",fraction},{"contactPosition",contact},{"events",events}});
}

}  // namespace engine_detail

}  // namespace resectionlab::sim
#include "simulation_domain_native.hpp"
namespace resectionlab::sim {

class Engine {
public:
    explicit Engine(State initial) : current_(std::move(initial)), initial_(current_) {}
    Engine(const Engine& other) {
        ++execution_stats.clones;
        std::lock_guard<std::recursive_mutex> lock(other.mutex_);
        current_=other.current_; initial_=other.initial_;
    }
    std::unique_lock<std::recursive_mutex> lock() const { return std::unique_lock<std::recursive_mutex>(mutex_); }
    const State& state() const noexcept { return current_; }
    State& boundary_state() noexcept { return current_; }
    Engine clone() const { return *this; }
    Engine fresh() const { return Engine(initial_); }
    std::string identity() const { auto held=lock(); return state_id(current_); }
    Value metrics(const StopToken& stop) const { auto held=lock(); return engine_detail::metrics(current_,stop); }
    Value snapshot() const { auto held=lock(); return sim::snapshot(current_); }

    std::pair<std::unique_ptr<Engine>,Value> prepare(const Value& action,const StopToken& stop) const {
        auto held=lock();
        ++execution_stats.actions;
        stop.check(); const auto before = identity();
        auto rejected = [&](std::string message) {
            return Value::object({{"status","rejected"},{"reason",message},
                {"reasonCode",engine_detail::reason_code(Value(message),true)},
                {"completedFraction",0.},{"elapsedSeconds",0.},{"stateId",before},
                {"events",Value::array()},{"deltas",Value::object()},{"metrics",metrics(stop)}});
        };
        if (current_.scene->source_inspection_only)
            return {nullptr,rejected("SOURCE_INSPECTION_ONLY")};
        auto candidate = std::make_unique<Engine>(*this);
        const double before_time = current_.state.at("elapsedSeconds").as_number();
        std::set<std::int64_t> old_removed;
        for (const auto& cell : current_.state.at("removed").as_array()) old_removed.insert(cell.as_int());
        Value result;
        try {
            candidate->current_.in_transition = true;
            if (!current_.scene->cranial_stages.is_null()) cranial::prepare_transition(candidate->current_,stop);
            if (!current_.scene->cranial_access.is_null()) cranial::source_validate(candidate->current_,stop);
            if (task_domain::reviewed_domain(*current_.scene)) {
                auto refusal=task_domain::action_rejection(candidate->current_,action,stop,candidate->metrics(stop));
                if (!refusal.is_null()) return {nullptr,std::move(refusal)};
            }
            const auto validated = engine_detail::validate(candidate->current_,action);
            result = engine_detail::execute(candidate->current_,action,validated.duration,validated.points,stop);
        } catch (const std::invalid_argument& error) { stop.check(); return {nullptr,rejected(error.what())}; }
          catch (const std::out_of_range& error) { stop.check(); return {nullptr,rejected(std::string("'")+error.what()+"'")}; }
          catch (const std::bad_variant_access&) { stop.check(); return {nullptr,rejected("Invalid action value type")}; }
          catch (const std::overflow_error& error) { stop.check(); return {nullptr,rejected(error.what())}; }
        candidate->current_.in_transition = false;
        stop.check(); result.set("reasonCode",engine_detail::reason_code(result.at("reason")));
        const bool accepted = !engine_detail::is_text(result,"status","rejected");
        result.set("elapsedSeconds",accepted ? candidate->current_.state.at("elapsedSeconds").as_number()-before_time : 0.);
        if (!accepted) { result.set("stateId",before); result.set("metrics",metrics(stop)); return {nullptr,std::move(result)}; }
        cusa_v3::reconcile_removed(candidate->current_);
        finalize_legacy_action(candidate->current_,stop); stop.check();
        result.set("elapsedSeconds",candidate->current_.state.at("elapsedSeconds").as_number()-before_time);
        const auto after = state_id(candidate->current_);
        Value removed = Value::array();
        for (const auto& cell : candidate->current_.state.at("removed").as_array())
            if (!old_removed.count(cell.as_int())) removed.append(cell);
        result.set("deltas",Value::object({{"removedCells",removed},{"geometryRevision",after}}));
        candidate->current_.actions.append(action);
        Value history = Value::object({{"action",action},{"stateBefore",before},{"stateAfter",after}});
        for (const auto& item : result.as_object()) history.set(item.first,item.second);
        candidate->current_.history.append(std::move(history)); ++candidate->current_.version;
        stop.check(); result.set("stateId",after); return {std::move(candidate),std::move(result)};
    }
    Value step(const Value& action,const StopToken& stop) {
        auto held=lock();
        auto prepared = prepare(action,stop);
        if (prepared.first) {
            prepared.second.set("metrics",prepared.first->metrics(stop)); stop.check();
            current_ = std::move(prepared.first->current_);
        }
        return std::move(prepared.second);
    }
private:
    mutable std::recursive_mutex mutex_;
    State current_, initial_;
};

}  // namespace resectionlab::sim
