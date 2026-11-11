#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_control_native.hpp"
#include "simulation_intset_native.hpp"
#include "simulation_queries_native.hpp"

namespace resectionlab::sim {
namespace legacy_effects {

using Edge = std::pair<std::int64_t, std::int64_t>;
using Cells = std::set<std::int64_t>;

inline const Value& empty_array() { static const Value value = Value::array(); return value; }
inline const Value& empty_object() { static const Value value = Value::object(); return value; }
inline const Value& get(const Value& owner, std::string_view key, const Value& fallback) {
    const auto* value = owner.find(key); return value == nullptr ? fallback : *value;
}
inline double number(const Value& owner, std::string_view key, double fallback = 0.) {
    const auto* value = owner.find(key); return value == nullptr ? fallback : value->as_number();
}
inline bool flag(const Value& owner, std::string_view key, bool fallback = false) {
    const auto* value = owner.find(key); return value == nullptr ? fallback : value->truthy();
}
inline bool text_is(const Value& owner, std::string_view key, std::string_view expected) {
    const auto* value = owner.find(key); return value && value->is_string() && value->as_string() == expected;
}
inline Value& ensure(Value& owner, std::string key, Value fallback) {
    if (auto* value = owner.find(key)) return *value;
    return owner.set(std::move(key), std::move(fallback));
}
inline Value point_value(const Point& point) { return Value::array({point[0], point[1], point[2]}); }
inline Point point(const Value& value) {
    return {value.at(0).as_number(), value.at(1).as_number(), value.at(2).as_number()};
}
inline Value integers(const std::vector<std::int64_t>& cells) {
    Value result = Value::array();
    for (auto cell : cells) result.append(cell);
    return result;
}
inline Value integers(const Cells& cells) {
    return integers(std::vector<std::int64_t>(cells.begin(), cells.end()));
}
inline Cells cell_set(const Value& values) {
    Cells result;
    for (const auto& value : values.as_array()) result.insert(value.as_int());
    return result;
}
inline bool intersects(const Value& values, const Cells& cells) {
    for (const auto& value : values.as_array()) if (cells.count(value.as_int())) return true;
    return false;
}
inline bool equal(const Value& first, const Value& second) {
    if (first.is_number() && second.is_number()) return first.as_number() == second.as_number();
    if (first.kind() != second.kind()) return false;
    switch (first.kind()) {
        case Value::Kind::Null: return true;
        case Value::Kind::Bool: return first.as_bool() == second.as_bool();
        case Value::Kind::Int: return first.as_int() == second.as_int();
        case Value::Kind::Double: return first.as_double() == second.as_double();
        case Value::Kind::BigInt: return first.as_bigint() == second.as_bigint();
        case Value::Kind::String: return first.as_string() == second.as_string();
        case Value::Kind::Array:
            if (first.as_array().size() != second.as_array().size()) return false;
            for (std::size_t i = 0; i < first.as_array().size(); ++i)
                if (!equal(first.at(i), second.at(i))) return false;
            return true;
        case Value::Kind::Object:
            if (first.as_object().size() != second.as_object().size()) return false;
            for (const auto& item : first.as_object()) {
                const auto* found = second.find(item.first);
                if (!found || !equal(item.second, *found)) return false;
            }
            return true;
    }
    return false;
}

// CPython 3.12 sum: integer zero for empty inputs, then Neumaier float accumulation.
inline Value sum(const std::vector<Value>& values) {
    std::int64_t integer = 0;
    std::size_t i = 0;
    for (; i < values.size() && values[i].is_int(); ++i) {
        const auto next = values[i].as_int();
        if ((next > 0 && integer > std::numeric_limits<std::int64_t>::max() - next) ||
            (next < 0 && integer < std::numeric_limits<std::int64_t>::min() - next))
            throw std::overflow_error("Native legacy sum integer overflow");
        integer += next;
    }
    if (i == values.size()) return integer;
    double result = static_cast<double>(integer) + values[i++].as_number();
    double compensation = 0.;
    for (; i < values.size(); ++i) {
        const auto& value = values[i];
        if (value.is_int()) { result += static_cast<double>(value.as_int()); continue; }
        const double x = value.as_number(), total = result + x;
        compensation += std::abs(result) >= std::abs(x) ? (result - total) + x : (x - total) + result;
        result = total;
    }
    if (compensation != 0. && std::isfinite(compensation)) result += compensation;
    return result;
}
inline Value sum_values(const Value& object) {
    std::vector<Value> values;
    values.reserve(object.as_object().size());
    for (const auto& item : object.as_object()) values.push_back(item.second);
    return sum(values);
}
inline void add_number(Value& owner, std::string key, double amount, double fallback = 0.) {
    const double before = number(owner, key, fallback);
    owner.set(std::move(key), before + amount);
}
inline std::array<std::int64_t, 3> coordinates(const Scene& scene, std::int64_t cell) {
    scene.label_at(cell);
    const auto plane = static_cast<std::int64_t>(scene.shape[1] * scene.shape[2]);
    const auto depth = static_cast<std::int64_t>(scene.shape[2]);
    return {cell / plane, (cell % plane) / depth, cell % depth};
}
inline std::int64_t cell_id(const Scene& scene, const std::array<std::int64_t, 3>& index) {
    return (index[0] * static_cast<std::int64_t>(scene.shape[1]) + index[1]) *
        static_cast<std::int64_t>(scene.shape[2]) + index[2];
}
inline std::vector<std::int64_t> neighbors(const Scene& scene, std::int64_t cell) {
    const auto origin = coordinates(scene, cell);
    std::vector<std::int64_t> result;
    result.reserve(6);
    for (int axis = 0; axis < 3; ++axis) for (int direction : {-1, 1}) {
        auto next = origin; next[axis] += direction;
        if (next[axis] >= 0 && next[axis] < static_cast<std::int64_t>(scene.shape[axis]))
            result.push_back(cell_id(scene, next));
    }
    return result;
}
inline bool revised(const State& engine) {
    const auto* revision = engine.state.find("toolMechanicsRevision");
    return revision && revision->is_int() && (revision->as_int() == 1 || revision->as_int() == 2);
}
inline std::vector<const Value*> holds(const State& engine) {
    std::vector<const Value*> result;
    for (const auto& item : engine.state.at("holds").as_object()) result.push_back(&item.second);
    for (const auto& item : get(engine.state, "detachedFragments", empty_array()).as_array()) result.push_back(&item);
    return result;
}

inline std::int64_t free_outlet(const State& engine, std::int64_t cell, const Occupancy& occupied) {
    if (!occupied.occupied(cell)) return cell;
    for (auto neighbor : neighbors(*engine.scene, cell)) if (!occupied.occupied(neighbor)) return neighbor;
    return -1;
}
inline Point current_point(const State& engine, std::int64_t cell, const StopToken& stop) {
    return current_points(engine, {cell}, stop).front();
}
inline Point subtract(const Point& first, const Point& second) {
    return {first[0] - second[0], first[1] - second[1], first[2] - second[2]};
}
inline double side(const Point& point, const Point& origin, const Point& normal) {
    const auto delta = subtract(point, origin);
    return ((0. + delta[0] * normal[0]) + delta[1] * normal[1]) + delta[2] * normal[2];
}
inline bool crosses(double first, double second) { return std::min(first, second) <= 0. && 0. < std::max(first, second); }
inline std::vector<Edge> edges(const State& engine, const std::vector<std::int64_t>& cells,
                               const Cells* allowed, const StopToken& stop) {
    const auto removed = cell_set(engine.state.at("removed"));
    std::set<Edge> result;
    std::size_t visited = 0;
    for (auto first : cells) {
        if ((visited++ % 32768) == 0) stop.check();
        for (auto second : neighbors(*engine.scene, first))
            if (!removed.count(second) && engine.scene->label_at(second) && (!allowed || allowed->count(second)))
                result.emplace(std::min(first, second), std::max(first, second));
    }
    return {result.begin(), result.end()};
}
inline Value edge_values(const std::vector<Edge>& edges) {
    Value result = Value::array();
    for (const auto& edge : edges) result.append(Value::array({edge.first, edge.second}));
    return result;
}
inline std::size_t record_edges(State& engine, std::string field, const std::vector<Edge>& additions) {
    std::set<Edge> before;
    for (const auto& edge : engine.state.at(field).as_array()) before.emplace(edge.at(0).as_int(), edge.at(1).as_int());
    auto after = before; after.insert(additions.begin(), additions.end());
    engine.state.set(std::move(field), edge_values({after.begin(), after.end()}));
    return after.size() - before.size();
}
inline std::vector<Edge> blade_edges(const State& engine, const std::vector<std::int64_t>& cells,
                                     const Point& p, const Quat& unit, const Value& params,
                                     const Tool& instrument, const StopToken& stop) {
    const auto normal = apply_rotation(rotation_matrix(unit), {0., 1., 0.});
    const auto inverse = rotation_matrix(inverse_quaternion(unit));
    std::vector<Edge> result;
    for (const auto& edge : edges(engine, cells, nullptr, stop)) {
        if (!instrument.accepts(engine.scene->material(edge.first)) || !instrument.accepts(engine.scene->material(edge.second))) continue;
        const auto a = current_point(engine, edge.first, stop), b = current_point(engine, edge.second, stop);
        const double sa = side(a, p, normal), sb = side(b, p, normal);
        if (!crosses(sa, sb)) continue;
        const double fraction = -sa / (sb - sa);
        Point intersection;
        for (int axis = 0; axis < 3; ++axis) intersection[axis] = a[axis] + (b[axis] - a[axis]) * fraction;
        const auto local = apply_rotation(inverse, subtract(intersection, p));
        const double width = instrument.geometry == "scissors" ? number(params, "jawOpeningMm") / 2. + .35 : number(params, "radiusMm") + .4;
        if (std::abs(local[0]) <= width && -std::min(number(params, "depthMm"), 5.) - .35 <= local[2] && local[2] <= .35)
            result.push_back(edge);
    }
    return result;
}
inline const Value* vessel_profile(const State& engine) {
    const auto* profile = engine.scene->properties.find("tissuePhysics");
    return profile && profile->is_object() ? profile->find("vessel") : nullptr;
}
inline double wound_flow(const Value& profile) {
    if (number(profile, "pressurePa") == 0.) return 0.;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double log_flow = (((std::log(pi / 8.) + 4. * std::log(number(profile, "radiusMm"))) +
        std::log(number(profile, "pressurePa"))) - std::log(number(profile, "viscosityPaS"))) -
        std::log(number(profile, "pathLengthMm"));
    return std::exp(std::min(log_flow, std::log(number(profile, "maxFlowMm3PerSecond"))));
}
inline void validate_physics(const Scene& scene) {
    const auto* profile = scene.properties.find("tissuePhysics");
    if (!profile || profile->is_null()) return;
    if (!profile->is_object() || !text_is(*profile, "schema", "resectionlab.tissue-physics.v1"))
        throw std::invalid_argument("TISSUE_PHYSICS_SCHEMA_REQUIRED");
    for (const auto& item : profile->as_object())
        if (item.first != "schema" && item.first != "provenance" && item.first != "vessel" && item.first != "bone")
            throw std::invalid_argument("UNKNOWN_TISSUE_PHYSICS_FIELD");
    const auto* provenance = profile->find("provenance");
    const auto* description = provenance ? provenance->find("description") : nullptr;
    if (!provenance || !provenance->is_object() || !text_is(*provenance, "kind", "assumed-research") ||
        !description || !description->is_string() || description->as_string().find_first_not_of(" \t\r\n\v\f") == std::string::npos)
        throw std::invalid_argument("EXPLICIT_ASSUMED_RESEARCH_PROVENANCE_REQUIRED");
    if (const auto* bone = profile->find("bone"))
        if (!bone->is_object() || bone->as_object().size() != 1 || !text_is(*bone, "model", "rigid-cell-removal-v1"))
            throw std::invalid_argument("UNSUPPORTED_BONE_PHYSICS_MODEL");
    if (const auto* vessel = profile->find("vessel")) {
        if (!vessel->is_object() || vessel->as_object().size() != 7 || !text_is(*vessel, "model", "local-poiseuille-v1"))
            throw std::invalid_argument("COMPLETE_EXPLICIT_VESSEL_PROFILE_REQUIRED");
        for (const char* name : {"radiusMm", "pressurePa", "viscosityPaS", "pathLengthMm", "maxFlowMm3PerSecond", "coagulationExposure"})
            if (!vessel->contains(name)) throw std::invalid_argument("COMPLETE_EXPLICIT_VESSEL_PROFILE_REQUIRED");
        for (const char* name : {"radiusMm", "pressurePa", "viscosityPaS", "pathLengthMm", "maxFlowMm3PerSecond", "coagulationExposure"}) {
            const auto* value = vessel->find(name);
            if (!value || !value->is_number() || !std::isfinite(value->as_number()) || value->as_number() < 0. ||
                (std::string_view(name) != "pressurePa" && value->as_number() == 0.))
                throw std::invalid_argument(std::string("INVALID_VESSEL_PARAMETER:") + name);
        }
        wound_flow(*vessel);
    }
    if (!profile->contains("bone") && !profile->contains("vessel"))
        throw std::invalid_argument("TISSUE_PHYSICS_PROFILE_HAS_NO_MODEL");
}
inline void record_vessel_transections(State& engine, const std::vector<Edge>& edges) {
    const auto* profile = vessel_profile(engine);
    if (!profile) return;
    std::set<Edge> ordered;
    for (const auto& edge : edges) ordered.emplace(std::min(edge.first, edge.second), std::max(edge.first, edge.second));
    for (const auto& edge : ordered) {
        if (engine.scene->material(edge.first) != "vessel" || engine.scene->material(edge.second) != "vessel") continue;
        const auto key = std::to_string(edge.first) + ":" + std::to_string(edge.second);
        if (engine.state.at("vesselConnections").contains(key)) continue;
        const auto source_id = "vessel-wound:" + key;
        const auto cells = Value::array({edge.first, edge.second});
        engine.state.at("vesselConnections").set(key, Value::object({{"cells", cells}, {"continuity", "severed"}, {"patent", false}, {"perfusion", "unknown"}}));
        engine.state.at("vesselWounds").set(key, Value::object({{"cells", cells}, {"sourceId", source_id}, {"model", profile->at("model")},
            {"flowInterpretation", "one equivalent outlet per severed voxel adjacency"}}));
        engine.state.at("sources").append(Value::object({{"id", source_id}, {"cell", edge.first}, {"kind", "modeled-vessel-wound"},
            {"connectionId", key}, {"rateMm3PerSecond", wound_flow(*profile)}, {"coagulationExposure", profile->at("coagulationExposure")},
            {"patent", false}, {"bleedingActive", true}, {"exposure", 0.}, {"reservoirMm3", 0.}, {"emittedMm3", 0.}}));
    }
}

}  // namespace legacy_effects

inline void initialize_legacy_effect_state(State& engine) {
    using namespace legacy_effects;
    validate_physics(*engine.scene);
    if (vessel_profile(engine)) {
        ensure(engine.state, "vesselConnections", Value::object());
        ensure(engine.state, "vesselWounds", Value::object());
    }
    for (const char* name : {"cutEdges", "separatedAttachments", "transectedEdges", "flapComponents", "meshCutEdges", "meshSeams"})
        ensure(engine.state, name, Value::array());
    ensure(engine.state, "debris", Value::object());
    for (auto& source : engine.state.at("sources").as_array()) {
        ensure(source, "patent", Value()); ensure(source, "exposure", 0.); ensure(source, "bleedingActive", true);
    }
}

// Actor admission calls this before mutation; separate native actors own the newer profiles.
inline std::string legacy_effects_unsupported(const State& engine) {
    for (const char* field : {"tissuePhysicsV2", "brainContinuumV1", "vascularGraphV1", "bipolarThermalV1"})
        if (engine.scene->properties.contains(field)) return std::string("NATIVE_LEGACY_UNSUPPORTED:") + field;
    for (const char* field : {"patientPosition", "deformation", "deformationModel", "vascularGraphV1", "bipolarThermalV1"})
        if (engine.state.contains(field)) return std::string("NATIVE_LEGACY_UNSUPPORTED:") + field;
    for (const auto& source : engine.state.at("sources").as_array())
        if (source.contains("modelVersion")) return "NATIVE_LEGACY_UNSUPPORTED:versioned-vessel-source";
    return {};
}

namespace legacy_effects {

inline const Value& source_geometry(const State& engine) {
    const auto* adapter = engine.scene->source.find("adapter");
    return adapter && adapter->is_object() && (text_is(*adapter, "type", "mesh") || text_is(*adapter, "type", "surface_buffers"))
        ? *adapter : engine.scene->source;
}
inline Value release(State& engine, const Cells& cells, std::string_view kind) {
    const auto* attachments = engine.scene->properties.find("attachments");
    if (!attachments) throw std::invalid_argument("MISSING_ATTACHMENT_PROPERTIES");
    std::set<std::string> done;
    for (const auto& item : engine.state.at("separatedAttachments").as_array()) done.insert(item.as_string());
    Value changed = Value::array();
    for (const auto& attachment : attachments->as_array()) {
        const auto& id = attachment.at("id").as_string();
        if (text_is(attachment, "kind", kind) && !done.count(id) && intersects(attachment.at("cells"), cells)) {
            done.insert(id); changed.append(id);
        }
    }
    Value separated = Value::array();
    for (const auto& id : done) separated.append(id);
    engine.state.set("separatedAttachments", separated);
    engine.state.set("separated", separated);
    return changed;
}
inline void bone_components(State& engine, const StopToken& stop) {
    const auto removed = cell_set(engine.state.at("removed"));
    std::vector<std::int64_t> identifiers(engine.scene->labels.size(), 0);
    const auto& anchors = get(engine.scene->properties, "boneAnchors", empty_array());
    const auto anchor_cells = cell_set(anchors);
    const bool compact = flag(engine.scene->properties, "tissuePhysics") || flag(engine.scene->properties, "tissuePhysicsV2");
    Value summaries = Value::array();
    std::int64_t identifier = 0;
    for (std::size_t cell = 0; cell < identifiers.size(); ++cell) {
        if (cell % 32768 == 0) stop.check();
        if (identifiers[cell] || removed.count(static_cast<std::int64_t>(cell)) || engine.scene->material(cell) != "bone") continue;
        ++identifier;
        std::vector<std::int64_t> cells{static_cast<std::int64_t>(cell)};
        identifiers[cell] = identifier;
        auto low = coordinates(*engine.scene, cell), high = low;
        bool anchored = false;
        for (std::size_t index = 0; index < cells.size(); ++index) {
            if (index % 32768 == 0) stop.check();
            const auto current = cells[index];
            anchored = anchored || anchor_cells.count(current);
            const auto ijk = coordinates(*engine.scene, current);
            for (int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(low[axis], ijk[axis]); high[axis] = std::max(high[axis], ijk[axis]);
            }
            for (auto next : neighbors(*engine.scene, current)) {
                if (!identifiers[next] && !removed.count(next) && engine.scene->material(next) == "bone") {
                    identifiers[next] = identifier; cells.push_back(next);
                }
            }
        }
        const auto count = cells.size();
        std::sort(cells.begin(), cells.end());
        if (compact && count > 4096) cells.clear();
        Value separable = anchors.as_array().empty() ? Value() : Value(!anchored);
        if (!compact) summaries.append(Value::object({{"cells", integers(cells)}, {"separable", separable}}));
        else summaries.append(Value::object({{"componentId", identifier}, {"cellCount", count}, {"cells", integers(cells)},
            {"cellsOmitted", count > 4096}, {"boundsIjk", Value::array({Value::array({low[0], low[1], low[2]}),
                Value::array({high[0], high[1], high[2]})})}, {"separable", separable}}));
    }
    engine.state.set("flapComponents", std::move(summaries));
}
inline std::size_t mesh_seam(State& engine, const Point& p, const Quat& unit, const Value& params,
                              const StopToken& stop) {
    const auto& source = source_geometry(engine);
    if (!text_is(source, "topology", "sheet") && !text_is(source, "topology", "strand")) return 0;
    for (const auto& item : engine.state.at("holds").as_object())
        for (const auto& value : item.second.at("offset").as_array())
            if (value.truthy()) throw std::invalid_argument("MESH_SEAM_ON_DISPLACED_GEOMETRY_UNSUPPORTED");
    double scale;
    if (text_is(source, "units", "mm")) scale = 1.;
    else if (text_is(source, "units", "meter")) scale = 1000.;
    else if (text_is(source, "units", "micron")) scale = .001;
    else throw std::invalid_argument("MESH_SEAM_REQUIRES_DECLARED_FRAME_UNITS");
    if (!text_is(source, "frame", "RAS+") && !text_is(source, "frame", "LPS+"))
        throw std::invalid_argument("MESH_SEAM_REQUIRES_DECLARED_FRAME_UNITS");
    std::vector<Point> vertices, local;
    const auto inverse = rotation_matrix(inverse_quaternion(unit));
    for (const auto& value : source.at("vertices").as_array()) {
        Point vertex = point(value);
        for (double& component : vertex) component *= scale;
        if (text_is(source, "frame", "LPS+")) { vertex[0] *= -1.; vertex[1] *= -1.; }
        vertices.push_back(vertex); local.push_back(apply_rotation(inverse, subtract(vertex, p)));
    }
    std::set<Edge> edges;
    for (const auto& line : get(source, "lines", empty_array()).as_array()) {
        const auto a = line.at(0).as_int(), b = line.at(1).as_int();
        edges.emplace(std::min(a, b), std::max(a, b));
    }
    const auto& faces = get(source, "faces", empty_array());
    for (const auto& face : faces.as_array()) for (std::size_t i = 0; i < 3; ++i) {
        const auto a = face.at(i).as_int(), b = face.at((i + 1) % 3).as_int();
        edges.emplace(std::min(a, b), std::max(a, b));
    }
    std::set<Edge> old;
    for (const auto& edge : engine.state.at("meshCutEdges").as_array()) old.emplace(edge.at(0).as_int(), edge.at(1).as_int());
    std::map<Edge, Point> crossed;
    std::size_t visited = 0;
    for (const auto& edge : edges) {
        if ((visited++ % 32768) == 0) stop.check();
        const auto& a = local.at(edge.first); const auto& b = local.at(edge.second);
        if (!crosses(a[1], b[1])) continue;
        const double fraction = -a[1] / (b[1] - a[1]);
        Point intersection;
        for (int axis = 0; axis < 3; ++axis) intersection[axis] = a[axis] + fraction * (b[axis] - a[axis]);
        const double norm = std::sqrt(intersection[0] * intersection[0] + intersection[2] * intersection[2]);
        if (std::abs(intersection[2]) > number(params, "depthMm") || norm > number(params, "radiusMm")) continue;
        Point world;
        for (int axis = 0; axis < 3; ++axis)
            world[axis] = vertices[edge.first][axis] + fraction * (vertices[edge.second][axis] - vertices[edge.first][axis]);
        crossed.emplace(edge, world);
    }
    std::vector<Edge> additions;
    for (const auto& item : crossed) additions.push_back(item.first);
    const auto count = record_edges(engine, "meshCutEdges", additions);
    for (std::size_t face_id = 0; face_id < faces.as_array().size(); ++face_id) {
        Cells face = cell_set(faces.at(face_id));
        Value hits = Value::array();
        for (const auto& item : crossed)
            if (face.count(item.first.first) && face.count(item.first.second)) hits.append(point_value(item.second));
        if (hits.as_array().size() >= 2) {
            hits.as_array().resize(2);
            Value seam = Value::object({{"faceId", face_id}, {"points", hits}, {"kind", "retained_sheet_seam"}});
            bool present = false;
            for (const auto& old_seam : engine.state.at("meshSeams").as_array()) if (equal(old_seam, seam)) { present = true; break; }
            if (!present) engine.state.at("meshSeams").append(std::move(seam));
        }
    }
    if (text_is(source, "topology", "strand")) for (const auto& item : crossed)
        if (!old.count(item.first)) engine.state.at("meshSeams").append(Value::object({
            {"edge", Value::array({item.first.first, item.first.second})},
            {"points", Value::array({point_value(item.second)})}, {"kind", "retained_strand_break"}}));
    return count;
}

}  // namespace legacy_effects

inline void validate_legacy_hold_motion(State& engine, std::string_view key, const StopToken& stop) {
    using namespace legacy_effects;
    stop.check();
    const auto& instrument = engine.state.at("slots").at(key).at("instrument").as_string();
    if (instrument == "bone_flap_holder" || instrument == "membrane_forceps")
        throw std::invalid_argument("NATIVE_TRANSPORT_DISPATCH_REQUIRED");
    if (instrument != "tumor_forceps") return;
    const auto* attachments = engine.scene->properties.find("attachments");
    if (!attachments) throw std::invalid_argument("MISSING_ATTACHMENT_PROPERTIES");
    const auto cells = cell_set(engine.state.at("holds").at(key).at("cells"));
    std::set<std::string> separated;
    for (const auto& value : engine.state.at("separatedAttachments").as_array()) separated.insert(value.as_string());
    std::vector<const Value*> attached;
    for (const auto& attachment : attachments->as_array())
        if (!separated.count(attachment.at("id").as_string()) && intersects(attachment.at("cells"), cells)) attached.push_back(&attachment);
    if (!attached.empty() && revised(engine)) throw std::invalid_argument("SEPARATE_ATTACHMENTS_BEFORE_DISPLACEMENT");
    for (const auto* attachment : attached) if (!flag(*attachment, "breakable"))
        throw std::invalid_argument("ATTACHMENT_PREVENTS_DISPLACEMENT");
    for (const auto* attachment : attached) separated.insert(attachment->at("id").as_string());
    Value result = Value::array(); for (const auto& id : separated) result.append(id);
    engine.state.set("separatedAttachments", std::move(result));
}

inline Value extract_legacy_fragment(State& engine, std::string_view key, const StopToken& stop) {
    using namespace legacy_effects;
    const Value hold = engine.state.at("holds").at(key);
    auto cells = cell_set(hold.at("cells"));
    for (const auto& cell : engine.state.at("removed").as_array()) cells.erase(cell.as_int());
    if (cells.empty()) throw std::invalid_argument("NO_RETAINED_FRAGMENT");
    for (auto cell : engine.scene->protected_cells) if (cells.count(cell))
        throw std::invalid_argument("PROTECTED_STRUCTURE: fragment evacuation includes protected source membership");
    std::set<std::string> separated;
    for (const auto& value : engine.state.at("separatedAttachments").as_array()) separated.insert(value.as_string());
    for (const auto& attachment : get(engine.scene->properties, "attachments", empty_array()).as_array())
        if (!separated.count(attachment.at("id").as_string()) && intersects(attachment.at("cells"), cells))
            throw std::invalid_argument("SEPARATE_ATTACHMENTS_BEFORE_EXTRACTION");
    const auto offset = point(hold.at("offset"));
    if (offset[0] == 0. && offset[1] == 0. && offset[2] == 0.)
        throw std::invalid_argument("MOVE_FRAGMENT_THROUGH_EXIT_BEFORE_EXTRACTION");
    std::array<std::int64_t, 3> low{INT64_MAX, INT64_MAX, INT64_MAX}, high{-1, -1, -1};
    for (std::size_t cell = 0; cell < engine.scene->labels.size(); ++cell) {
        if (cell % 32768 == 0) stop.check();
        if (!engine.scene->labels[cell]) continue;
        const auto ijk = coordinates(*engine.scene, cell);
        for (int axis = 0; axis < 3; ++axis) { low[axis] = std::min(low[axis], ijk[axis]); high[axis] = std::max(high[axis], ijk[axis]); }
    }
    bool exterior = false;
    for (int axis = 0; axis < 3; ++axis) {
        bool below = true, above = true;
        for (auto cell : cells) {
            const auto ijk = coordinates(*engine.scene, cell);
            below = below && ijk[axis] + offset[axis] < low[axis];
            above = above && ijk[axis] + offset[axis] > high[axis];
        }
        exterior = exterior || below || above;
    }
    if (!exterior) throw std::invalid_argument("FRAGMENT_NOT_AT_SOURCE_EXTERIOR_EXIT");
    occupancy(engine, stop);
    auto removed = cell_set(engine.state.at("removed")); removed.insert(cells.begin(), cells.end());
    engine.state.set("removed", integers(removed));
    const double volume = static_cast<double>(cells.size()) * engine.scene->voxel_mm3;
    add_number(engine.state, "evacuatedForcepsMm3", volume);
    engine.state.at("holds").erase(key); engine.state.at("slots").at(key).set("grip", Value::array());
    return Value::array({Value::object({{"kind", "detached_extraction"}, {"cells", cells.size()},
        {"volumeMm3", volume}, {"exit", "source-anatomy-bounds"}, {"checkedFragmentMotion", true}})});
}

inline std::string legacy_effect_action_unsupported(const State& engine, std::string_view key) {
    const auto reason = legacy_effects_unsupported(engine);
    if (!reason.empty()) return reason;
    const auto& name = engine.state.at("slots").at(key).at("instrument").as_string();
    if (name == "bone_flap_holder" || name == "membrane_forceps") return "NATIVE_TRANSPORT_DISPATCH_REQUIRED";
    if (name == "cusa" && (legacy_effects::revised(engine) || legacy_effects::flag(engine.scene->properties, "cusaV3")))
        return "NATIVE_CUSA_V3_DISPATCH_REQUIRED";
    return {};
}

inline Value apply_legacy_effect(State& engine, std::string_view key, std::string_view op,
    const std::vector<std::int64_t>& input_cells, const Point& p, const Quat& q, double dt,
    const StopToken& stop) {
    using namespace legacy_effects;
    const auto unsupported = legacy_effect_action_unsupported(engine, key);
    if (!unsupported.empty()) throw std::invalid_argument(unsupported);
    stop.check();
    initialize_legacy_effect_state(engine);
    const Value slot = engine.state.at("slots").at(key);
    const auto& instrument = tool(slot.at("instrument").as_string());
    const auto effect = instrument.effect;
    const Value params = parameters(engine, slot);
    const auto unit = normalized_quaternion(q);
    const auto inverse = rotation_matrix(inverse_quaternion(unit));
    std::vector<std::int64_t> cells = input_cells;
    auto event = [](Value value) { return Value::array({std::move(value)}); };
    if (effect == "remove") {
        PositiveIntSet candidates(cells.begin(), cells.end(), &stop), before(&stop);
        for (const auto& value : engine.state.at("removed").as_array()) before.insert(value.as_int());
        const auto added = candidates.difference(before);
        auto removed = cell_set(engine.state.at("removed"));
        added.for_each([&](std::int64_t cell) { removed.insert(cell); });
        engine.state.set("removed", integers(removed));
        if (instrument.id == "cranial_drill") {
            added.for_each([&](std::int64_t cell) {
                add_number(engine.state.at("debris"), std::to_string(cell), engine.scene->voxel_mm3);
            });
            engine.state.set("debrisMm3", sum_values(engine.state.at("debris")));
            if (!added.empty() && (flag(engine.scene->properties, "tissuePhysics") || !engine.scene->cranial_access.is_null()))
                engine.state.set("boneTopologyDirty", true);
        }
        return added.empty() ? Value::array() : event(Value::object({{"kind", "removal"}, {"cells", added.size()},
            {"volumeMm3", static_cast<double>(added.size()) * engine.scene->voxel_mm3}}));
    }
    if (effect == "retract") {
        if (!engine.state.at("holds").contains(key) && !cells.empty()) {
            engine.state.at("holds").set(std::string(key), Value::object({{"cells", integers(cells)},
                {"origin", point_value(p)}, {"offset", Value::array({0, 0, 0})}}));
            return event(Value::object({{"kind", "hold_created"}, {"cells", cells.size()}}));
        }
        return Value::array();
    }
    if (effect == "incise" || effect == "sheet_cut" || effect == "kerf" || effect == "transect") {
        std::vector<std::int64_t> filtered;
        for (auto cell : cells) {
            const auto local = apply_rotation(inverse, subtract(current_point(engine, cell, stop), p));
            if (std::abs(local[2]) <= number(params, "depthMm")) filtered.push_back(cell);
        }
        cells = std::move(filtered);
    }
    const Cells selected(cells.begin(), cells.end());
    if (effect == "incise" || effect == "sheet_cut") {
        if (effect == "sheet_cut" && number(params, "jawOpeningMm") > .5) return Value::array();
        const auto normal = apply_rotation(rotation_matrix(unit), {0., 1., 0.});
        std::vector<Edge> cut;
        for (const auto& edge : edges(engine, cells, &selected, stop)) {
            if (crosses(side(current_point(engine, edge.first, stop), p, normal), side(current_point(engine, edge.second, stop), p, normal)))
                cut.push_back(edge);
        }
        if (revised(engine)) cut = blade_edges(engine, cells, p, unit, params, instrument, stop);
        const auto count = record_edges(engine, "cutEdges", cut);
        const auto mesh_count = effect == "sheet_cut" ? mesh_seam(engine, p, unit, params, stop) : 0;
        if (count) engine.state.at("cuts").append(Value::object({{"tool", instrument.id}, {"cells", integers(selected)},
            {"edges", edge_values(cut)}, {"normal", point_value(normal)}}));
        return count || mesh_count ? event(Value::object({{"kind", effect == "sheet_cut" ? "sheet_opening" : "incision"},
            {"edges", count}, {"meshEdges", mesh_count}})) : Value::array();
    }
    if (effect == "elevate" || effect == "dissect") {
        Value released = release(engine, selected, effect == "elevate" ? "periosteal" : "soft");
        return released.as_array().empty() ? Value::array() : event(Value::object({{"kind", "attachment_release"}, {"attachments", released}}));
    }
    if (effect == "kerf") {
        auto removed = cell_set(engine.state.at("removed"));
        PositiveIntSet candidates(&stop), before(&stop);
        const double half_width = *std::min_element(engine.scene->spacing.begin(), engine.scene->spacing.end()) * .5;
        for (auto cell : cells) {
            const auto local = apply_rotation(inverse, subtract(current_point(engine, cell, stop), p));
            if (std::abs(local[1]) <= half_width) candidates.insert(cell);
        }
        for (const auto& value : engine.state.at("removed").as_array()) before.insert(value.as_int());
        const auto added = candidates.difference(before);
        added.for_each([&](std::int64_t cell) { removed.insert(cell); });
        engine.state.set("removed", integers(removed));
        added.for_each([&](std::int64_t cell) {
            add_number(engine.state.at("debris"), std::to_string(cell), engine.scene->voxel_mm3);
        });
        engine.state.set("debrisMm3", sum_values(engine.state.at("debris")));
        if (!added.empty()) engine.state.set("boneTopologyDirty", true);
        return added.empty() ? Value::array() : event(Value::object({{"kind", "bone_kerf"}, {"cells", added.size()},
            {"components", engine.state.at("flapComponents").as_array().size()}}));
    }
    if (effect == "transect") {
        if (number(params, "jawOpeningMm") > .5) return Value::array();
        const auto* profile = vessel_profile(engine);
        if (!engine.scene->properties.contains("strands") && !text_is(source_geometry(engine), "topology", "strand") && !profile)
            throw std::invalid_argument("MISSING_STRAND_PROPERTIES");
        std::vector<Edge> candidate;
        for (const auto& strand : get(engine.scene->properties, "strands", empty_array()).as_array()) {
            const auto eligible = cell_set(strand.at("cells"));
            std::vector<std::int64_t> local;
            std::set_intersection(selected.begin(), selected.end(), eligible.begin(), eligible.end(), std::back_inserter(local));
            const auto part = edges(engine, local, &eligible, stop); candidate.insert(candidate.end(), part.begin(), part.end());
        }
        if (profile) {
            const auto removed = cell_set(engine.state.at("removed"));
            for (auto cell : cells) if (engine.scene->material(cell) == "vessel")
                for (auto next : neighbors(*engine.scene, cell))
                    if (engine.scene->label_at(next) && !removed.count(next) && engine.scene->material(next) == "vessel")
                        candidate.emplace_back(std::min(cell, next), std::max(cell, next));
        }
        const auto normal = apply_rotation(rotation_matrix(unit), {0., 1., 0.});
        std::vector<Edge> cut;
        for (const auto& edge : candidate)
            if (crosses(side(current_point(engine, edge.first, stop), p, normal), side(current_point(engine, edge.second, stop), p, normal))) cut.push_back(edge);
        const auto count = record_edges(engine, "transectedEdges", cut);
        std::size_t soft_count = 0;
        if (revised(engine)) {
            std::vector<std::int64_t> soft;
            for (auto cell : cells) if (engine.scene->material(cell) == "soft_tissue" || engine.scene->material(cell) == "tumor") soft.push_back(cell);
            std::vector<Edge> soft_edges;
            for (const auto& edge : blade_edges(engine, soft, p, unit, params, instrument, stop)) {
                const auto a = engine.scene->material(edge.first), b = engine.scene->material(edge.second);
                if ((a == "soft_tissue" || a == "tumor") && (b == "soft_tissue" || b == "tumor")) soft_edges.push_back(edge);
            }
            soft_count = record_edges(engine, "cutEdges", soft_edges);
        }
        record_vessel_transections(engine, cut);
        const auto mesh_count = text_is(source_geometry(engine), "topology", "strand") ? mesh_seam(engine, p, unit, params, stop) : 0;
        return count || mesh_count || soft_count ? event(Value::object({{"kind", "strand_transection"}, {"edges", count},
            {"softEdges", soft_count}, {"meshEdges", mesh_count}, {"piecesRetained", true}})) : Value::array();
    }
    if (effect == "grasp") {
        if (op == "extract") {
            const auto* held = engine.state.at("holds").find(key);
            if (!held || !held->truthy()) throw std::invalid_argument("GRASP_BEFORE_EXTRACTION");
            if (revised(engine)) return extract_legacy_fragment(engine, key, stop);
            const auto* attachments = engine.scene->properties.find("attachments");
            if (!attachments) throw std::invalid_argument("MISSING_ATTACHMENT_PROPERTIES");
            const Value removed_cells = held->at("cells");
            const auto held_cells = cell_set(removed_cells);
            std::set<std::string> separated;
            for (const auto& value : engine.state.at("separatedAttachments").as_array()) separated.insert(value.as_string());
            std::vector<const Value*> attached;
            for (const auto& attachment : attachments->as_array())
                if (!separated.count(attachment.at("id").as_string()) && intersects(attachment.at("cells"), held_cells)) attached.push_back(&attachment);
            for (const auto* attachment : attached) if (!flag(*attachment, "breakable")) throw std::invalid_argument("ATTACHMENT_PREVENTS_EXTRACTION");
            for (const auto* attachment : attached) separated.insert(attachment->at("id").as_string());
            Value separated_values = Value::array(); for (const auto& value : separated) separated_values.append(value);
            engine.state.set("separatedAttachments", separated_values);
            auto removed = cell_set(engine.state.at("removed")); removed.insert(held_cells.begin(), held_cells.end());
            engine.state.set("removed", integers(removed)); engine.state.at("holds").erase(key);
            engine.state.at("slots").at(key).set("grip", Value::array());
            return event(Value::object({{"kind", "detached_extraction"}, {"cells", removed_cells.as_array().size()}}));
        }
        if (cells.empty() || engine.state.at("holds").contains(key)) return Value::array();
        if (number(params, "jawOpeningMm") > .5) return Value::array();
        if (revised(engine)) {
            auto* detached = engine.state.find("detachedFragments");
            if (detached) for (std::size_t i = 0; i < detached->as_array().size(); ++i) {
                if (!intersects(detached->at(i).at("cells"), selected)) continue;
                Value fragment = detached->at(i);
                detached->as_array().erase(detached->as_array().begin() + i);
                fragment.set("origin", point_value(p)); fragment.set("baseOffset", fragment.at("offset"));
                const Value grip = fragment.at("cells");
                engine.state.at("holds").set(std::string(key), fragment); engine.state.at("slots").at(key).set("grip", grip);
                return event(Value::object({{"kind", "grasp_created"}, {"cells", grip.as_array().size()}, {"regrasp", true}}));
            }
        }
        std::set<Edge> blocked;
        for (const char* field : {"cutEdges", "transectedEdges"}) for (const auto& edge : engine.state.at(field).as_array())
            blocked.emplace(edge.at(0).as_int(), edge.at(1).as_int());
        const auto removed = cell_set(engine.state.at("removed"));
        Cells fragment{cells.front()}; std::vector<std::int64_t> stack{cells.front()};
        std::size_t visited = 0;
        while (!stack.empty()) {
            if ((visited++ % 32768) == 0) stop.check();
            const auto cell = stack.back(); stack.pop_back();
            for (auto next : neighbors(*engine.scene, cell)) {
                if (fragment.count(next) || removed.count(next) || !engine.scene->label_at(next) || engine.scene->material(next) != "tumor" ||
                    blocked.count({std::min(cell, next), std::max(cell, next)})) continue;
                fragment.insert(next); stack.push_back(next);
            }
        }
        if (revised(engine)) for (const auto* hold : holds(engine))
            if (intersects(hold->at("cells"), fragment)) throw std::invalid_argument("MATERIAL_ALREADY_HELD");
        Value hold = Value::object({{"cells", integers(fragment)}, {"origin", point_value(p)}, {"offset", Value::array({0, 0, 0})}});
        if (revised(engine)) hold.set("baseOffset", Value::array({0, 0, 0}));
        engine.state.at("holds").set(std::string(key), std::move(hold));
        engine.state.at("slots").at(key).set("grip", integers(fragment));
        return event(Value::object({{"kind", "grasp_created"}, {"cells", fragment.size()}}));
    }
    if (effect == "suction" || effect == "irrigate") {
        const auto spatial = capsule_cells(*engine.scene, p, p, number(params, "radiusMm"), stop);
        double budget = number(params, "flowMm3PerSecond") * dt;
        if (effect == "suction") {
            Value amounts = Value::object();
            for (const char* field : {"fluid", "debris"}) {
                double total = 0.;
                for (auto cell : spatial) {
                    const auto name = std::to_string(cell);
                    const double take = std::min(budget, number(engine.state.at(field), name));
                    if (take != 0.) {
                        add_number(engine.state.at(field), name, -take); total += take; budget -= take;
                        if (number(engine.state.at(field), name) <= 1e-12) engine.state.at(field).erase(name);
                    }
                }
                amounts.set(field, total);
            }
            add_number(engine.state, "fluidRemovedMm3", number(amounts, "fluid"));
            engine.state.set("debrisMm3", sum_values(engine.state.at("debris")));
            if (number(amounts, "debris") != 0.) add_number(engine.state, "debrisRemovedMm3", number(amounts, "debris"));
            return event(Value::object({{"kind", "aspiration"}, {"fluid", amounts.at("fluid")}, {"debris", amounts.at("debris")}}));
        }
        Point outlet = apply_rotation(rotation_matrix(unit), {0., 0., number(params, "depthMm")});
        for (int axis = 0; axis < 3; ++axis) outlet[axis] = p[axis] + outlet[axis];
        std::array<std::int64_t, 3> index;
        bool outside = false;
        for (int axis = 0; axis < 3; ++axis) {
            const auto row = axis * 4;
            const double coordinate = (((0. + engine.scene->inverse[row] * outlet[0]) + engine.scene->inverse[row + 1] * outlet[1]) +
                engine.scene->inverse[row + 2] * outlet[2]) + engine.scene->inverse[row + 3];
            const double rounded = std::nearbyint(coordinate);
            if (!std::isfinite(rounded) || rounded < 0. || rounded >= static_cast<double>(engine.scene->shape[axis])) outside = true;
            index[axis] = outside ? 0 : static_cast<std::int64_t>(rounded);
        }
        if (outside) add_number(engine.state, "fluidEscapedMm3", budget);
        else {
            const auto occupied = occupancy(engine, stop);
            const auto destination = free_outlet(engine, cell_id(*engine.scene, index), occupied);
            if (destination < 0) throw std::invalid_argument("FLUID_OUTLET_BLOCKED");
            add_number(engine.state.at("fluid"), std::to_string(destination), budget);
        }
        add_number(engine.state, "fluidAddedMm3", budget);
        return event(Value::object({{"kind", "irrigation"}, {"volumeMm3", budget}, {"outlet", point_value(outlet)}}));
    }
    if (effect == "coagulate") {
        if (revised(engine)) {
            if (number(params, "jawOpeningMm") > .5) return Value::array();
            std::vector<std::int64_t> filtered;
            for (auto cell : cells) {
                const auto local = apply_rotation(inverse, subtract(current_point(engine, cell, stop), p));
                if (std::abs(local[0]) <= .35 + number(params, "jawOpeningMm") / 2. && std::abs(local[1]) <= .35 &&
                    -number(params, "depthMm") - .35 <= local[2] && local[2] <= .35) filtered.push_back(cell);
            }
            cells = std::move(filtered);
            if (cells.empty()) return Value::array();
        }
        const double exposure = number(params, "power") * dt;
        for (auto cell : cells) add_number(engine.state.at("thermal"), std::to_string(cell), exposure);
        Value changed = Value::array();
        const Cells contacts(cells.begin(), cells.end());
        for (auto& source : engine.state.at("sources").as_array()) {
            if (!contacts.count(source.at("cell").as_int())) continue;
            if (!source.contains("coagulationExposure")) throw std::invalid_argument("MISSING_SOURCE_COAGULATION_PROPERTY");
            add_number(source, "exposure", exposure);
            if (number(source, "exposure") >= number(source, "coagulationExposure") && flag(source, "bleedingActive", true)) {
                source.set("bleedingActive", false); changed.append(source.at("id"));
            }
        }
        return event(Value::object({{"kind", "local_energy"}, {"exposure", exposure}, {"controlledBleedingSources", changed}}));
    }
    if (effect == "probe") {
        const std::string probe(instrument.id.substr(0, instrument.id.size() - 6));
        Value evidence = Value::array();
        for (const auto& row : get(engine.scene->properties, "functionalEvidence", empty_array()).as_array())
            if (text_is(row, "probe", probe) && selected.count(row.at("cell").as_int())) evidence.append(row);
        Value supplied = Value::object();
        for (const char* field : {"amplitudeMa", "frequencyHz", "pulseWidthUs"}) if (params.contains(field)) supplied.set(field, params.at(field));
        const bool known = !evidence.as_array().empty();
        Value observation = Value::object({{"kind", "mapping"}, {"probe", probe}, {"position", point_value(p)}, {"cells", integers(selected)},
            {"parameters", supplied}, {"coverage", known ? "supplied sites only" : "unknown"}, {"evidence", evidence},
            {"result", known ? "supplied_evidence" : "unknown"}, {"uncertainty", known ? "source supplied" : "no matching supplied functional evidence"}});
        const auto& observations = engine.state.at("observations").as_array();
        if (observations.empty() || !equal(observations.back(), observation)) engine.state.at("observations").append(observation);
        return event(std::move(observation));
    }
    throw std::invalid_argument("UNSUPPORTED_MECHANISM");
}

inline void tick_legacy_fields(State& engine, double dt, const StopToken& stop) {
    using namespace legacy_effects;
    const auto unsupported = legacy_effects_unsupported(engine);
    if (!unsupported.empty()) throw std::invalid_argument(unsupported);
    stop.check();
    std::optional<Occupancy> occupied;
    for (auto& source : engine.state.at("sources").as_array()) {
        if (flag(source, "bleedingActive", true)) {
            if (!source.contains("rateMm3PerSecond") || !source.contains("cell"))
                throw std::invalid_argument("MISSING_BLEEDING_SOURCE_PROPERTIES");
            const double amount = number(source, "rateMm3PerSecond") * dt;
            if (text_is(source, "kind", "modeled-vessel-wound"))
                source.set("emittedMm3", source.at("emittedMm3").as_number() + amount);
            add_number(source, "reservoirMm3", amount);
            add_number(engine.state, "fluidAddedMm3", amount);
        }
        const auto cell = source.at("cell").as_int();
        if (!occupied) occupied.emplace(occupancy(engine, stop));
        const auto destination = free_outlet(engine, cell, *occupied);
        if (destination >= 0) {
            const double amount = number(source, "reservoirMm3");
            if (amount != 0.) {
                add_number(engine.state.at("fluid"), std::to_string(destination), amount);
                source.set("reservoirMm3", 0.);
            }
        }
    }
    if (!occupied) occupied.emplace(occupancy(engine, stop));
    const double capacity = engine.scene->voxel_mm3;
    std::vector<std::string> keys;
    for (const auto& item : engine.state.at("fluid").as_object()) keys.push_back(item.first);
    std::size_t visited = 0;
    for (const auto& key : keys) {
        if ((visited++ % 32768) == 0) stop.check();
        const auto cell = std::stoll(key);
        if (occupied->occupied(cell)) {
            const auto destination = free_outlet(engine, cell, *occupied);
            const double amount = engine.state.at("fluid").at(key).as_number();
            engine.state.at("fluid").erase(key);
            if (destination < 0) add_number(engine.state, "fluidEscapedMm3", amount);
            else add_number(engine.state.at("fluid"), std::to_string(destination), amount);
        }
    }
    keys.clear();
    for (const auto& item : engine.state.at("fluid").as_object()) keys.push_back(item.first);
    std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return std::stoll(a) < std::stoll(b); });
    for (const auto& key : keys) {
        if ((visited++ % 32768) == 0) stop.check();
        const auto cell = std::stoll(key);
        double excess = std::max(0., number(engine.state.at("fluid"), key) - capacity);
        for (auto neighbor : neighbors(*engine.scene, cell)) {
            if (excess == 0.) break;
            if (occupied->occupied(neighbor)) continue;
            const auto next_key = std::to_string(neighbor);
            const double transfer = std::min(excess, std::max(0., capacity - number(engine.state.at("fluid"), next_key)));
            if (transfer != 0.) {
                const double sediment = number(engine.state.at("debris"), key) *
                    std::min(1., transfer / number(engine.state.at("fluid"), key));
                if (sediment != 0.) {
                    add_number(engine.state.at("debris"), key, -sediment);
                    add_number(engine.state.at("debris"), next_key, sediment);
                }
                add_number(engine.state.at("fluid"), next_key, transfer);
                add_number(engine.state.at("fluid"), key, -transfer); excess -= transfer;
            }
        }
        const auto index = coordinates(*engine.scene, cell);
        bool boundary = false;
        for (int axis = 0; axis < 3; ++axis)
            boundary = boundary || index[axis] == 0 || index[axis] == static_cast<std::int64_t>(engine.scene->shape[axis]) - 1;
        if (excess != 0. && boundary) {
            const double sediment = number(engine.state.at("debris"), key) *
                std::min(1., excess / number(engine.state.at("fluid"), key));
            if (sediment != 0.) {
                add_number(engine.state.at("debris"), key, -sediment); add_number(engine.state, "debrisEscapedMm3", sediment);
            }
            add_number(engine.state.at("fluid"), key, -excess); add_number(engine.state, "fluidEscapedMm3", excess);
        }
    }
    engine.state.set("debrisMm3", sum_values(engine.state.at("debris")));
}

inline void finalize_legacy_action(State& engine, const StopToken& stop) {
    // Native CUSA reconciliation is dispatched by the actor before this legacy topology pass.
    const bool dirty = legacy_effects::flag(engine.state, "boneTopologyDirty");
    engine.state.erase("boneTopologyDirty");
    if (dirty) legacy_effects::bone_components(engine, stop);
}

inline Value legacy_physics_metrics(const State& engine) {
    using namespace legacy_effects;
    std::int64_t active = 0;
    std::vector<Value> generated, reservoir;
    for (const auto& source : engine.state.at("sources").as_array()) if (text_is(source, "kind", "modeled-vessel-wound")) {
        active += source.at("bleedingActive").truthy();
        generated.push_back(get(source, "emittedMm3", Value(0.)));
        reservoir.push_back(get(source, "reservoirMm3", Value(0.)));
    }
    return Value::object({{"severedVesselConnections", get(engine.state, "vesselConnections", empty_object()).as_object().size()},
        {"activeVesselWounds", active}, {"modeledVesselFluidGeneratedMm3", sum(generated)},
        {"modeledVesselReservoirMm3", sum(reservoir)}, {"globalPerfusion", "unknown"}});
}

inline Value legacy_physics_descriptor(const State& engine) {
    const auto* profile = engine.scene->properties.find("tissuePhysics");
    return Value::object({{"enabled", profile && !profile->is_null()}, {"profile", profile ? *profile : Value()},
        {"capabilities", Value::object({{"rigidBoneRemoval", true}, {"damageTriggeredVesselWounds", legacy_effects::vessel_profile(engine) != nullptr},
            {"patientCalibratedPhysiology", false}, {"globalPerfusion", false}})},
        {"limits", Value::array({"local voxel adjacency, not anatomical vessel centerlines",
            "assumed equivalent outlet flow, not patient blood loss prediction",
            "one modeled outlet per cut adjacency; total flow depends on grid resolution",
            "no pressure network, pulsatility, collateral territories, or tissue perfusion"})}});
}

inline Value legacy_metrics(const State& engine, const StopToken& stop) {
    using namespace legacy_effects;
    Value result = legacy_physics_metrics(engine), by_label = Value::object();
    std::vector<Value> target_volumes;
    for (const auto& value : engine.state.at("removed").as_array()) {
        const auto cell = value.as_int();
        const auto& region = engine.scene->region(engine.scene->label_at(cell));
        add_number(by_label, region.at("name").as_string(), engine.scene->voxel_mm3);
        if (flag(region, "target")) target_volumes.emplace_back(engine.scene->voxel_mm3);
    }
    const Value target = sum(target_volumes);
    result.set("removedByLabelMm3", by_label); result.set("targetRemovedMm3", target);
    result.set("nonTargetRemovedMm3", static_cast<double>(engine.state.at("removed").as_array().size()) * engine.scene->voxel_mm3 - target.as_number());
    for (const char* field : {"elapsedSeconds", "travelMm"})
        result.set(field, engine.state.at(field));
    result.set("contactCellSeconds", sum_values(engine.state.at("contact")));
    for (const char* field : {"debrisMm3", "fluidAddedMm3", "fluidRemovedMm3"})
        result.set(field, engine.state.at(field));
    result.set("debrisRemovedMm3", get(engine.state, "debrisRemovedMm3", Value(0.)));
    result.set("debrisEscapedMm3", get(engine.state, "debrisEscapedMm3", Value(0.)));
    std::vector<Value> source_reservoir;
    for (const auto& source : engine.state.at("sources").as_array()) source_reservoir.push_back(get(source, "reservoirMm3", Value(0.)));
    const auto fluid_sum = sum_values(engine.state.at("fluid")); const auto reservoir_sum = sum(source_reservoir);
    if (fluid_sum.is_int() && reservoir_sum.is_int()) result.set("fluidPresentMm3", fluid_sum.as_int() + reservoir_sum.as_int());
    else result.set("fluidPresentMm3", fluid_sum.as_number() + reservoir_sum.as_number());
    result.set("fluidEscapedMm3", engine.state.at("fluidEscapedMm3"));
    result.set("thermalExposure", sum_values(engine.state.at("thermal")));
    result.set("vesselEncounters", engine.state.at("vesselEncounters").as_array().size());
    result.set("exposureByMaterialCellSeconds", engine.state.at("materialExposure"));
    std::int64_t unknown_sites = 0, collisions = 0, unknown_cells = 0;
    for (const auto& observation : engine.state.at("observations").as_array()) if (text_is(observation, "result", "unknown")) ++unknown_sites;
    for (const auto& row : engine.history.as_array()) if (text_is(row, "status", "partial") && !text_is(row, "reason", "TISSUE_ASPIRATION_IN_PROGRESS")) ++collisions;
    for (std::size_t cell = 0; cell < engine.scene->labels.size(); ++cell) {
        if (cell % 32768 == 0) stop.check();
        if (engine.scene->material(cell) == "unknown") ++unknown_cells;
    }
    result.set("unknownStimulationSites", unknown_sites); result.set("collisions", collisions); result.set("unknownMaterialCells", unknown_cells);
    return result;
}

}  // namespace resectionlab::sim



