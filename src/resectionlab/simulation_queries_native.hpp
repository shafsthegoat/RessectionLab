#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "simulation_control_native.hpp"
#include "simulation_geometry_native.hpp"
#include "simulation_motion_native.hpp"
#include "simulation_state_native.hpp"
#include "simulation_stats_native.hpp"
#include "simulation_tools_native.hpp"

namespace resectionlab::sim {
namespace deformation { inline Value raster(const State&, const StopToken&); }
namespace queries_detail {

struct Pulse {
    const StopToken& stop;
    std::uint64_t work = 0;
    void operator()() { if ((work++ % 32768) == 0) stop.check(); }
};

inline const Scene& scene(const State& engine) {
    if (!engine.scene) throw std::invalid_argument("Native state has no source scene");
    return *engine.scene;
}
inline void no_deformation(const State& engine) {
    const auto* value = engine.state.find("deformation");
    if (value != nullptr && value->truthy())
        throw std::invalid_argument("UNSUPPORTED_SIMULATION_DEFORMATION");
}
inline double number(const Value& object, std::string_view key, double fallback) {
    const auto* value = object.find(key);
    return value == nullptr ? fallback : value->as_number();
}
inline bool revision(const Value& object, std::string_view key, double expected) {
    const auto* value = object.find(key);
    if (value == nullptr) return false;
    if (value->is_bool()) return static_cast<double>(value->as_bool()) == expected;
    return value->is_number() && value->as_number() == expected;
}
inline Point point(const Value& value) {
    const auto& items = value.as_array();
    if (items.size() != 3) throw std::invalid_argument("Expected a three-vector");
    return {items[0].as_number(), items[1].as_number(), items[2].as_number()};
}
inline Quat quaternion(const Value& value) {
    const auto& items = value.as_array();
    if (items.size() != 4) throw std::invalid_argument("Expected an xyzw quaternion");
    return {items[0].as_number(), items[1].as_number(), items[2].as_number(), items[3].as_number()};
}
inline std::array<std::int64_t, 3> lattice_offset(const Value& value) {
    const auto& items = value.as_array();
    if (items.size() != 3) throw std::invalid_argument("INVALID_LATTICE_OFFSET");
    std::array<std::int64_t, 3> result;
    for (int axis = 0; axis < 3; ++axis) {
        if (!items[axis].is_int()) throw std::invalid_argument("INVALID_LATTICE_OFFSET");
        result[axis] = items[axis].as_int();
    }
    return result;
}
template <class Visit>
inline void each_hold(const State& engine, bool detached, Visit visit) {
    for (const auto& row : engine.state.at("holds").as_object()) visit(row.second);
    const auto* fragments = engine.state.find("detachedFragments");
    if (detached && fragments != nullptr) for (const auto& value : fragments->as_array()) visit(value);
}
inline Point offset_world(const Scene& source, const std::array<std::int64_t, 3>& offset) {
    Point result;
    for (int row = 0; row < 3; ++row) {
        const auto index = 4 * row;
        result[row] = ((0. + source.affine[index] * static_cast<double>(offset[0])) +
            source.affine[index + 1] * static_cast<double>(offset[1])) +
            source.affine[index + 2] * static_cast<double>(offset[2]);
    }
    return result;
}

}  // namespace queries_detail

struct Occupancy {
    std::shared_ptr<const Scene> scene;
    std::unordered_set<std::int64_t> cleared;
    std::unordered_map<std::int64_t, std::int64_t> moved;
    std::unordered_map<std::int64_t, std::vector<std::int64_t>> memberships;

    std::int64_t owner(std::int64_t cell) const {
        if (!scene) throw std::invalid_argument("Native occupancy has no source scene");
        const auto count = static_cast<std::int64_t>(scene->labels.size());
        if (cell < -count || cell >= count) throw std::out_of_range("Cell outside source grid");
        const auto target = moved.find(cell);
        if (target != moved.end()) return target->second;
        if (cleared.find(cell) != cleared.end()) return 0;
        const auto wrapped = cell < 0 ? cell + count : cell;
        return scene->labels[static_cast<std::size_t>(wrapped)] != 0 ? cell + 1 : 0;
    }
    std::int64_t operator[](std::int64_t cell) const { return owner(cell); }
    bool occupied(std::int64_t cell) const { return owner(cell) != 0; }

    std::vector<std::int64_t> labels(const std::vector<std::int64_t>& ids, const StopToken& stop) const {
        queries_detail::Pulse pulse{stop};
        std::vector<std::int64_t> result;
        result.reserve(ids.size());
        for (auto cell : ids) { pulse(); result.push_back(owner(cell)); }
        stop.check();
        return result;
    }
    std::vector<std::int64_t> material_ids(const std::vector<std::int64_t>& ids, const StopToken& stop) const {
        queries_detail::Pulse pulse{stop};
        std::set<std::int64_t> selected;
        for (auto cell : ids) {
            pulse();
            const auto value = owner(cell);
            const auto shared = memberships.find(cell);
            if (shared != memberships.end()) {
                for (auto source : shared->second) { pulse(); selected.insert(source); }
            } else if (value != 0) selected.insert(value - 1);
        }
        std::vector<std::int64_t> result;
        result.reserve(selected.size());
        for (auto cell : selected) { pulse(); result.push_back(cell); }
        stop.check();
        return result;
    }
};

inline Occupancy occupancy(const State& engine, const StopToken& stop) {
    stop.check();
    const auto& source = queries_detail::scene(engine);
    const auto count = geometry_detail::grid_size(source);
    if (static_cast<std::uint64_t>(count) != source.labels.size())
        throw std::invalid_argument("Source label count differs from its grid");
    Occupancy result;
    result.scene = engine.scene;
    std::unordered_set<std::int64_t> removed, held;
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : engine.state.at("removed").as_array()) {
        pulse();
        removed.insert(cell.as_int());
        result.cleared.insert(cell.as_int());
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> moves;
    queries_detail::each_hold(engine, true, [&](const Value& hold) {
        pulse();
        if (hold.contains("target")) return;
        std::vector<std::int64_t> retained;
        for (const auto& value : hold.at("cells").as_array()) {
            pulse();
            const auto cell = value.as_int();
            if (removed.find(cell) == removed.end()) retained.push_back(cell);
        }
        if (retained.empty()) return;
        for (auto cell : retained) {
            pulse();
            if (held.find(cell) != held.end()) throw std::invalid_argument("MATERIAL_ALREADY_HELD");
        }
        for (auto cell : retained) { pulse(); held.insert(cell); }
        const auto offset = queries_detail::lattice_offset(hold.at("offset"));
        for (auto cell : retained) {
            pulse();
            if (cell < 0 || cell >= count) throw std::out_of_range("Cell outside source grid");
            auto index = geometry_detail::coordinates(source, cell);
            for (int axis = 0; axis < 3; ++axis) {
                const auto maximum = static_cast<std::int64_t>(source.shape[axis]) - 1;
                if (offset[axis] < -index[axis] || offset[axis] > maximum - index[axis])
                    throw std::invalid_argument("DISPLACEMENT_OUTSIDE_GRID");
                index[axis] += offset[axis];
            }
            const auto destination = (index[0] * static_cast<std::int64_t>(source.shape[1]) + index[1]) *
                static_cast<std::int64_t>(source.shape[2]) + index[2];
            result.cleared.insert(cell);
            moves.emplace_back(cell, destination);
        }
    });
    for (const auto& move : moves) {
        pulse();
        if (result.moved.find(move.second) != result.moved.end() ||
                (source.labels[static_cast<std::size_t>(move.second)] != 0 &&
                 result.cleared.find(move.second) == result.cleared.end()))
            throw std::invalid_argument("DISPLACEMENT_COLLISION");
        result.moved.emplace(move.second, move.first + 1);
    }
    if (engine.state.contains("deformation")) {
        const auto pairs = deformation::raster(engine, stop);
        for (const auto& pair : pairs.as_array()) { pulse(); result.cleared.insert(pair.at(0).as_int()); }
        for (const auto& pair : pairs.as_array()) {
            pulse();
            const auto original = pair.at(0).as_int(), target = pair.at(1).as_int();
            if (source.label_at(target) != 0 && result.cleared.find(target) == result.cleared.end()) {
                const auto material = source.material(target);
                if (material != "soft_tissue" && material != "tumor")
                    throw std::invalid_argument("DISPLACEMENT_COLLISION");
                result.memberships[target].push_back(target);
            }
            result.memberships[target].push_back(original);
        }
        for (auto& item : result.memberships) {
            pulse();
            auto& originals = item.second;
            std::sort(originals.begin(), originals.end(), [&](auto first, auto second) { pulse(); return first < second; });
            originals.erase(std::unique(originals.begin(), originals.end(), [&](auto first, auto second) {
                pulse(); return first == second;
            }), originals.end());
            auto owner = originals.front();
            auto rank = source.region_index[source.label_at(owner)];
            if (rank < 0) throw std::out_of_range("Region ID is not declared");
            for (auto cell : originals) {
                pulse();
                const auto candidate = source.region_index[source.label_at(cell)];
                if (candidate < 0) throw std::out_of_range("Region ID is not declared");
                if (candidate > rank || (candidate == rank && cell > owner)) { owner = cell; rank = candidate; }
            }
            result.moved[item.first] = owner + 1;
        }
    }
    stop.check();
    return result;
}

inline std::vector<std::int64_t> touched(const State& engine, const Point& start, const Point& end,
                                        double radius, const StopToken& stop) {
    auto cells = capsule_cells(queries_detail::scene(engine), start, end, radius, stop);
    return occupancy(engine, stop).material_ids(cells, stop);
}

// False selects CUSA's historical holds-only contact positions; true follows effects.
inline std::vector<Point> current_points(const State& engine, const std::vector<std::int64_t>& cells,
                                         const StopToken& stop, bool include_detached = true) {
    stop.check();
    const auto& source = queries_detail::scene(engine);
    queries_detail::Pulse pulse{stop};
    const auto* v2 = source.properties.find("tissuePhysicsV2");
    const bool raster_positions = include_detached && v2 != nullptr && v2->truthy();
    std::unordered_map<std::int64_t, std::int64_t> raster;
    if (raster_positions) {
        const auto pairs = deformation::raster(engine, stop);
        for (const auto& pair : pairs.as_array()) {
            pulse(); raster[pair.at(0).as_int()] = pair.at(1).as_int();
        }
    }
    std::vector<Point> result;
    result.reserve(cells.size());
    std::unordered_map<std::int64_t, std::size_t> locations;
    for (std::size_t index = 0; index < cells.size(); ++index) {
        pulse();
        const auto position = raster.find(cells[index]);
        result.push_back(world_point(source, position == raster.end() ? cells[index] : position->second));
        locations[cells[index]] = index;
    }
    if (raster_positions &&
            !queries_detail::revision(engine.state, "toolMechanicsRevision", 1.) &&
            !queries_detail::revision(engine.state, "toolMechanicsRevision", 2.)) {
        stop.check();
        return result;
    }
    queries_detail::each_hold(engine, include_detached, [&](const Value& hold) {
        pulse();
        if (hold.contains("target")) return;
        const auto offset = queries_detail::offset_world(source, queries_detail::lattice_offset(hold.at("offset")));
        for (const auto& cell : hold.at("cells").as_array()) {
            pulse();
            const auto position = locations.find(cell.as_int());
            if (position == locations.end()) continue;
            auto& point = result[position->second];
            for (int axis = 0; axis < 3; ++axis) point[axis] += offset[axis];
        }
    });
    if (!include_detached) {
        const auto* field = engine.state.find("deformation");
        if (field != nullptr) {
            std::unordered_map<std::string_view, const Value*> displacements;
            displacements.reserve(field->as_object().size());
            for (const auto& item : field->as_object()) { pulse(); displacements.emplace(item.first, &item.second); }
            for (std::size_t index = 0; index < cells.size(); ++index) {
                pulse();
                const auto found = displacements.find(std::to_string(cells[index]));
                if (found == displacements.end() || found->second->is_null()) continue;
                const auto original = world_point(source, cells[index]);
                const auto offset = queries_detail::point(*found->second);
                for (int axis = 0; axis < 3; ++axis) result[index][axis] = original[axis] + offset[axis];
            }
        }
    }
    stop.check();
    return result;
}

struct Capsule {
    Point a, b;
    double radius;
    std::string part;
};

inline std::vector<std::vector<std::int64_t>> touched_many(const State& engine,
        const std::vector<Capsule>& queries, const StopToken& stop) {
    stop.check();
    if (queries.size() > 64) throw std::invalid_argument("Native geometry batch exceeds 64 items");
    execution_stats.geometry_batches.fetch_add(1, std::memory_order_relaxed);
    execution_stats.geometry_batch_items.fetch_add(queries.size(), std::memory_order_relaxed);
    auto maximum = execution_stats.geometry_batch_max_items.load(std::memory_order_relaxed);
    while (maximum < queries.size() && !execution_stats.geometry_batch_max_items.compare_exchange_weak(
            maximum, queries.size(), std::memory_order_relaxed)) {}
    std::vector<std::vector<std::int64_t>> covers;
    covers.reserve(queries.size());
    for (const auto& query : queries) {
        stop.check();
        covers.push_back(capsule_cells(queries_detail::scene(engine), query.a, query.b, query.radius, stop));
    }
    if (queries.empty()) return covers;
    const auto owner = occupancy(engine, stop);
    for (auto& cells : covers) { stop.check(); cells = owner.material_ids(cells, stop); }
    stop.check();
    return covers;
}

inline std::vector<Capsule> capsule_envelopes(const Tool& instrument, const Value& params,
                                             const Point& position, const Quat& quaternion,
                                             double inflation = 0.) {
    geometry_detail::finite(position, "position must be a finite three-vector");
    for (double value : quaternion)
        if (!std::isfinite(value)) throw std::invalid_argument("Orientation must be finite");
    if (!std::isfinite(inflation) || inflation < 0.) throw std::invalid_argument("Invalid capsule inflation");
    const auto rotation = rotation_matrix(normalized_quaternion(quaternion));
    const double radius = params.at("radiusMm").as_number();
    const double depth = queries_detail::number(params, "depthMm", 2.);
    if (instrument.geometry == "blade" && queries_detail::revision(params, "retractorFaceContactRevision", 1.))
        throw std::invalid_argument("UNSUPPORTED_SIMULATION_RETRACTOR_PAD");
    std::vector<Capsule> result;
    auto capsule = [&](const Point& a, const Point& b, double width, const char* part) {
        auto first = apply_rotation(rotation, a), second = apply_rotation(rotation, b);
        for (int axis = 0; axis < 3; ++axis) { first[axis] = position[axis] + first[axis]; second[axis] = position[axis] + second[axis]; }
        result.push_back({first, second, width + inflation, part});
    };
    if (instrument.geometry == "aspirator" && queries_detail::revision(params, "contactGeometryRevision", 2.)) {
        capsule({0., 0., -40.}, {0., 0., 0.}, instrument.shaft_radius, "shaft");
        capsule({0., 0., 0.}, {0., 0., 0.}, radius, "working");
        return result;
    }
    capsule({0., 0., -40.}, {0., 0., -std::max(1., std::min(depth, 5.))}, instrument.shaft_radius, "shaft");
    if (instrument.geometry == "scissors" || instrument.geometry == "forceps") {
        const double half = queries_detail::number(params, "jawOpeningMm", 2.) / 2.;
        for (double sign : {-1., 1.}) capsule({sign * half, 0., -depth}, {sign * half, 0., 0.}, .35, "working");
    } else if (instrument.geometry == "blade" || instrument.geometry == "elevator" || instrument.geometry == "kerf") {
        for (double z : {0., std::min(depth, 5.)}) capsule({-radius, 0., -z}, {radius, 0., -z}, .4, "working");
    } else capsule({0., 0., -std::min(depth, 3.)}, {0., 0., 0.}, radius, "working");
    return result;
}

inline std::vector<Capsule> swept_envelopes(const Tool& instrument, const Value& params,
                                           const Point& start, const Quat& start_quaternion,
                                           const Point& end, const Quat& end_quaternion) {
    const auto delta = geometry_detail::subtract(end, start);
    const double angle = rotation_angle(start_quaternion, end_quaternion);
    const double inflation = point_norm(delta) / 2. + 84. * std::sin(angle / 4.);
    return capsule_envelopes(instrument, params, midpoint_position(start, end),
        midpoint_quaternion(start_quaternion, end_quaternion), inflation);
}
inline std::vector<Capsule> swept_envelopes(const Tool& instrument, const Value& params,
                                           const Value& start, const Value& end) {
    return swept_envelopes(instrument, params, queries_detail::point(start.at("position")),
        queries_detail::quaternion(start.at("quaternion")), queries_detail::point(end.at("position")),
        queries_detail::quaternion(end.at("quaternion")));
}

inline bool pair_conflict(const std::vector<Capsule>& first, const std::vector<Capsule>& second,
                           const StopToken& stop) {
    queries_detail::Pulse pulse{stop};
    for (const auto& a : first) for (const auto& b : second) {
        pulse();
        if (segments_within_distance(a.a, a.b, b.a, b.b, (a.radius + b.radius) + 1e-9)) {
            stop.check();
            return true;
        }
    }
    stop.check();
    return false;
}

}  // namespace resectionlab::sim
