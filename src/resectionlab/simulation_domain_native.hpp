#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_control_native.hpp"
#include "simulation_queries_native.hpp"
#include "simulation_snapshot_native.hpp"

// Included after engine_detail::validate is defined, before the public Engine.
namespace resectionlab::sim::task_domain {

inline constexpr std::string_view PROPERTY = "reviewedTaskDomainV1";
inline constexpr int MAX_INTERVALS = 20000;

struct TaskDomain {
    Point minimum, maximum;
    std::string review_sha256, mask_sha256;
};

inline bool pose_operation(std::string_view operation) {
    for (auto value : {"place", "move", "apply", "withdraw", "hold", "grasp", "bladeManeuver"})
        if (operation == value) return true;
    return false;
}

inline std::optional<TaskDomain> reviewed_domain(const Scene& scene) {
    const auto* adapter = scene.source.find("adapter");
    if (adapter == nullptr || !adapter->is_object()) return std::nullopt;
    const auto* raw = adapter->find(PROPERTY);
    if (raw == nullptr) return std::nullopt;
    constexpr const char* prefix = "REVIEWED_TASK_DOMAIN_INVALID: ";
    constexpr std::array<std::string_view, 6> fields{"mode", "shape", "minimumVoxelBoundary",
        "maximumVoxelBoundary", "reviewEvidenceSha256", "maskSha256"};
    bool exact = raw->is_object() && raw->as_object().size() == fields.size();
    if (exact) for (auto field : fields) if (!raw->contains(field)) { exact = false; break; }
    const auto* mode = raw->find("mode");
    if (!exact || mode == nullptr || !mode->is_string() || mode->as_string() != "restrict-unknown")
        throw std::invalid_argument(std::string(prefix) + "exact restrict-unknown declaration required");
    const auto& shape = raw->at("shape");
    bool matches = shape.is_array() && shape.as_array().size() == 3;
    if (matches) for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto& width = shape.at(axis);
        if (!width.is_int() || width.as_int() <= 0 ||
                static_cast<std::uint64_t>(width.as_int()) != scene.shape[axis]) { matches = false; break; }
    }
    if (!matches) throw std::invalid_argument(std::string(prefix) + "shape differs from the current saved native grid");
    TaskDomain result;
    for (const auto& binding : {std::pair<std::string_view, Point*>{"minimumVoxelBoundary", &result.minimum},
                              std::pair<std::string_view, Point*>{"maximumVoxelBoundary", &result.maximum}}) {
        const auto& value = raw->at(binding.first);
        bool valid = value.is_array() && value.as_array().size() == 3;
        if (valid) for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto& coordinate = value.at(axis);
            if (!coordinate.is_number() || !std::isfinite(coordinate.as_number())) { valid = false; break; }
            (*binding.second)[axis] = coordinate.as_number();
        }
        if (!valid) throw std::invalid_argument(std::string(prefix) + "finite three-axis voxel-face bounds required");
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (result.minimum[axis] < -.5 || result.maximum[axis] > static_cast<double>(scene.shape[axis]) - .5 ||
                result.minimum[axis] >= result.maximum[axis])
            throw std::invalid_argument(std::string(prefix) + "box must lie inside observed native voxel faces");
    for (auto name : {"reviewEvidenceSha256", "maskSha256"}) {
        const auto& value = raw->at(name);
        bool valid = value.is_string() && value.as_string().size() == 64;
        if (valid) for (unsigned char ch : value.as_string())
            if (!((ch >= 'a' && ch <= 'f') || (ch >= '0' && ch <= '9'))) { valid = false; break; }
        if (!valid) throw std::invalid_argument(std::string(prefix) + "raw SHA256 evidence bindings required");
    }
    result.review_sha256 = raw->at("reviewEvidenceSha256").as_string();
    result.mask_sha256 = raw->at("maskSha256").as_string();
    return result;
}

inline bool _posed(const State& engine, std::string_view key, const StopToken& stop) {
    stop.check();
    const auto& slot = engine.state.at("slots").at(key);
    if (slot.at("active").truthy() || slot.at("inserted").truthy() || slot.at("grip").truthy() ||
            engine.state.at("holds").contains(key)) return true;
    queries_detail::Pulse pulse{stop};
    for (const auto& action : engine.actions.as_array()) {
        pulse();
        const auto* hand = action.find("slot");
        const auto* operation = action.find("operation");
        if (hand != nullptr && hand->is_string() && hand->as_string() == key &&
                operation != nullptr && operation->is_string() && pose_operation(operation->as_string())) return true;
    }
    stop.check();
    return false;
}

inline std::optional<std::string> _outside(const Scene& scene, const TaskDomain& domain,
                                          const std::vector<Capsule>& segments, const StopToken& stop) {
    Point margins;
    for (int axis = 0; axis < 3; ++axis) {
        const auto row = 4 * axis;
        margins[axis] = geometry_detail::norm(Point{scene.inverse[row], scene.inverse[row + 1], scene.inverse[row + 2]});
    }
    queries_detail::Pulse pulse{stop};
    for (const auto& segment : segments) {
        pulse();
        const auto first = geometry_detail::transform(scene.inverse, segment.a);
        const auto second = geometry_detail::transform(scene.inverse, segment.b);
        for (int axis = 0; axis < 3; ++axis) {
            const double low = std::min(first[axis], second[axis]) - segment.radius * margins[axis];
            const double high = std::max(first[axis], second[axis]) + segment.radius * margins[axis];
            if (low < domain.minimum[axis] - 1e-9 || high > domain.maximum[axis] + 1e-9)
                return segment.part + " capsule extends outside the reviewed task box; outside is unknown";
        }
    }
    stop.check();
    return std::nullopt;
}

inline Value _rejection(const State& engine, const std::string& reason, const Value& metrics,
                        std::string_view code = "REVIEWED_TASK_DOMAIN_OUTSIDE") {
    return Value::object({{"status", "rejected"}, {"reason", std::string(code) + ": " + reason},
        {"reasonCode", code}, {"completedFraction", 0.}, {"elapsedSeconds", 0.}, {"stateId", state_id(engine)},
        {"events", Value::array()}, {"deltas", Value::object()}, {"metrics", metrics}});
}

inline Value action_rejection(State& engine, const Value& action, const StopToken& stop, const Value& metrics) {
    const auto domain = reviewed_domain(queries_detail::scene(engine));
    if (!domain) return {};
    stop.check();
    Value poses;
    try {
        poses = engine_detail::validate(engine, action).points;
    } catch (const std::invalid_argument&) { return {}; }
      catch (const std::out_of_range&) { return {}; }
      catch (const std::bad_variant_access&) { return {}; }
      catch (const std::overflow_error&) { return {}; }
    const auto& key = action.at("slot").as_string();
    const auto& operation = action.at("operation").as_string();
    if (operation == "bladeManeuver")
        return _rejection(engine, "internally generated planar paths are not admitted in V1", metrics,
            "REVIEWED_TASK_DOMAIN_UNSUPPORTED_OPERATION");
    const auto& current = engine.state.at("slots").at(key);
    Value proposed = current;
    if (operation == "exchange") {
        const auto* replacement = action.find("instrument");
        bool known = false;
        if (replacement != nullptr && replacement->is_string())
            for (const auto& candidate : tools) if (candidate.id == replacement->as_string()) { known = true; break; }
        if (!known) return {};
        proposed.set("instrument", *replacement);
        proposed.set("parameters", Value::object());
    }
    if (const auto* pending = action.find("parameters"))
        for (const auto& item : pending->as_object()) proposed.at("parameters").set(item.first, item.second);
    auto params = parameters(engine, proposed);
    const auto current_params = parameters(engine, current);
    params.set("jawOpeningMm", std::max(params.at("jawOpeningMm").as_number(), current_params.at("jawOpeningMm").as_number()));
    const auto& instrument = tool(proposed.at("instrument").as_string());
    std::vector<Capsule> stationary;
    for (const auto& row : engine.state.at("slots").as_object()) {
        if (row.first == key || !_posed(engine, row.first, stop)) continue;
        auto capsules = capsule_envelopes(tool(row.second.at("instrument").as_string()), parameters(engine, row.second),
            queries_detail::point(row.second.at("position")), queries_detail::quaternion(row.second.at("quaternion")));
        stationary.insert(stationary.end(), capsules.begin(), capsules.end());
    }
    if (_posed(engine, key, stop)) {
        auto capsules = capsule_envelopes(tool(current.at("instrument").as_string()), parameters(engine, current),
            queries_detail::point(current.at("position")), queries_detail::quaternion(current.at("quaternion")));
        stationary.insert(stationary.end(), capsules.begin(), capsules.end());
    }
    if (const auto reason = _outside(*engine.scene, *domain, stationary, stop)) return _rejection(engine, *reason, metrics);
    if ((operation == "configure" || operation == "position" || operation == "exchange") && !_posed(engine, key, stop)) return {};
    if (operation == "place") {
        if (poses.as_array().size() != 1) return {};
        const auto& pose = poses.at(0);
        const auto reason = _outside(*engine.scene, *domain, capsule_envelopes(instrument, params,
            queries_detail::point(pose.at("position")), queries_detail::quaternion(pose.at("quaternion"))), stop);
        return reason ? _rejection(engine, *reason, metrics) : Value{};
    }
    if (operation == "withdraw" && poses.as_array().empty()) {
        const auto matrix = rotation_matrix(normalized_quaternion(queries_detail::quaternion(current.at("quaternion"))));
        const auto axis = apply_rotation(matrix, Point{0., 0., 1.});
        auto position = queries_detail::point(current.at("position"));
        for (int index = 0; index < 3; ++index) position[index] -= 50. * axis[index];
        poses = Value::array({Value::object({{"position", Value::array({position[0], position[1], position[2]})},
                                           {"quaternion", current.at("quaternion")}})});
    }
    if (poses.as_array().empty()) poses.append(Value::object({{"position", current.at("position")}, {"quaternion", current.at("quaternion")}}));
    auto previous_position = queries_detail::point(current.at("position"));
    auto previous_quaternion = queries_detail::quaternion(current.at("quaternion"));
    int intervals = 0;
    for (const auto& end : poses.as_array()) {
        const auto a = previous_position, b = queries_detail::point(end.at("position"));
        const auto end_quaternion = queries_detail::quaternion(end.at("quaternion"));
        const double angle = rotation_angle(previous_quaternion, end_quaternion);
        const double translation = point_norm(geometry_detail::subtract(b, a)) / .25;
        const double rotation = angle / 0.017453292519943295;
        if (!std::isfinite(translation) || !std::isfinite(rotation))
            throw std::overflow_error("cannot convert float infinity to integer");
        const double count_number = std::max({1., std::ceil(translation), std::ceil(rotation)});
        if (count_number > MAX_INTERVALS - intervals)
            return _rejection(engine, "split the path into shorter reviewed actions", metrics, "REVIEWED_TASK_DOMAIN_PATH_BUDGET");
        const int count = static_cast<int>(count_number);
        intervals += count;
        const auto interpolation = QuaternionTrajectory::between(previous_quaternion, end_quaternion);
        const double step = 1. / static_cast<double>(count);
        for (int index = 1; index <= count; ++index) {
            stop.check();
            const double fraction = index == count ? 1. : static_cast<double>(index) * step;
            Point position;
            for (int axis = 0; axis < 3; ++axis) position[axis] = (1. - fraction) * a[axis] + fraction * b[axis];
            const auto quaternion = interpolation.at(fraction);
            const auto reason = _outside(*engine.scene, *domain, swept_envelopes(instrument, params,
                previous_position, previous_quaternion, position, quaternion), stop);
            if (reason) return _rejection(engine, *reason, metrics);
            previous_position = position;
            previous_quaternion = quaternion;
        }
    }
    stop.check();
    return {};
}

}  // namespace resectionlab::sim::task_domain
