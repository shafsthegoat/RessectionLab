#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include "simulation_cranial_native.hpp"

namespace resectionlab::sim {
namespace admission_detail {

using cranial::detail::get;
using cranial::detail::same;

[[noreturn]] inline void invalid(std::string_view reason, bool milling = false) {
    throw std::invalid_argument(std::string(milling ? "MILLING_CONTRACT_INVALID: " : "CRANIOTOMY_CONTRACT_INVALID: ") + std::string(reason));
}
[[noreturn]] inline void stage_fail(std::string_view reason) { cranial::detail::fail(reason); }
[[noreturn]] inline void scene_fail(std::string_view reason) {
    throw std::invalid_argument("SIMULATION_SCENE_INVALID: " + std::string(reason));
}
inline bool fields(const Value& value, std::initializer_list<std::string_view> required) {
    if (!value.is_object() || value.as_object().size() != required.size()) return false;
    for (auto key : required) if (!value.contains(key)) return false;
    return true;
}
inline bool hex(std::string_view value, bool prefix = false) {
    if (prefix) {
        if (value.size() != 71 || value.substr(0, 7) != "sha256:") return false;
        value.remove_prefix(7);
    } else if (value.size() != 64) return false;
    for (char c : value) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}
inline bool unicode_space(std::uint32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
        (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
inline bool nonblank(const std::string& text) {
    std::size_t index = 0;
    while (index != text.size()) if (!unicode_space(hash_detail::next_codepoint(text, index))) return true;
    return false;
}
inline Value affine_value(const Scene& scene) {
    Value result = Value::array();
    for (int row = 0; row < 4; ++row) {
        Value values = Value::array();
        for (int column = 0; column < 4; ++column) values.append(scene.affine[4 * row + column]);
        result.append(std::move(values));
    }
    return result;
}
inline std::string array_header(const Scene& scene, std::string_view dtype) {
    return "{\"dtype\": \"" + std::string(dtype) + "\", \"shape\": [" +
        std::to_string(scene.shape[0]) + ", " + std::to_string(scene.shape[1]) + ", " +
        std::to_string(scene.shape[2]) + "]}";
}
inline std::string sha_text(const std::array<unsigned char, 32>& bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output("sha256:"); output.reserve(71);
    for (auto byte : bytes) { output.push_back(digits[byte >> 4]); output.push_back(digits[byte & 15]); }
    return output;
}
inline std::string array_digest_bytes(const Scene& scene, std::string_view dtype,
                                       const void* raw, std::size_t count, const StopToken& stop) {
    stop.check();
    const auto header = array_header(scene, dtype);
#if defined(__APPLE__)
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    CC_SHA256_Update(&context, header.data(), static_cast<CC_LONG>(header.size()));
    const auto* bytes = static_cast<const unsigned char*>(raw);
    for (std::size_t offset = 0; offset != count;) {
        stop.check();
        const auto size = std::min<std::size_t>(32768, count - offset);
        CC_SHA256_Update(&context, bytes + offset, static_cast<CC_LONG>(size)); offset += size;
    }
    std::array<unsigned char, 32> result{};
    CC_SHA256_Final(result.data(), &context);
    stop.check();
    return sha_text(result);
#else
    std::string payload(header);
    payload.append(static_cast<const char*>(raw), count);
    const auto result = sha_text(hash_detail::sha256(payload));
    stop.check();
    return result;
#endif
}
inline std::string labels_digest(const Scene& scene, const StopToken& stop) {
    const std::uint16_t native_order = 1;
    const bool little = *reinterpret_cast<const unsigned char*>(&native_order) == 1;
    return array_digest_bytes(scene, little ? "<u2" : ">u2", scene.labels.data(), scene.labels.size() * sizeof(std::uint16_t), stop);
}
inline std::string mask_digest(const Scene& scene, const std::vector<unsigned char>& mask, const StopToken& stop) {
    return array_digest_bytes(scene, "|b1", mask.data(), mask.size(), stop);
}
inline Value without(Value value, std::string_view field) { value.erase(field); return value; }
inline std::vector<std::int64_t> source_cells(const Value& value, std::size_t size, std::string_view name,
                                              bool empty, bool sorted, bool stage, const StopToken& stop) {
    const auto fail = [&]() {
        if (stage) stage_fail("INVALID_SOURCE_CELLS");
        invalid(std::string(name) + " requires sorted distinct source integer cells");
    };
    if (!value.is_array() || (value.as_array().empty() && !empty)) fail();
    std::vector<std::int64_t> result;
    result.reserve(value.as_array().size());
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : value.as_array()) {
        pulse();
        if (!cell.is_int() || cell.as_int() < 0 || static_cast<std::uint64_t>(cell.as_int()) >= size) fail();
        result.push_back(cell.as_int());
    }
    if (sorted && !std::is_sorted(result.begin(), result.end())) fail();
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) fail();
    stop.check();
    return result;
}
inline Value cell_values(const std::vector<std::int64_t>& cells, const StopToken& stop) {
    Value result = Value::array(); result.as_array().reserve(cells.size());
    queries_detail::Pulse pulse{stop};
    for (auto cell : cells) { pulse(); result.append(cell); }
    stop.check(); return result;
}
inline std::vector<std::int64_t> property_cells(const Value& value, std::size_t size, const StopToken& stop) {
    if (!value.is_array()) scene_fail("property cells must be an array");
    std::vector<std::int64_t> result;
    queries_detail::Pulse pulse{stop};
    for (const auto& cell : value.as_array()) {
        pulse();
        if (!cell.is_int() || cell.as_int() < 0 || static_cast<std::uint64_t>(cell.as_int()) >= size)
            scene_fail("property cell must index this source grid");
        result.push_back(cell.as_int());
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    stop.check(); return result;
}
inline bool contains(const std::vector<std::int64_t>& cells, std::int64_t cell) {
    return std::binary_search(cells.begin(), cells.end(), cell);
}
inline std::vector<std::int64_t> set_union(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
    std::vector<std::int64_t> result;
    result.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}
inline bool subset(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
    return std::includes(b.begin(), b.end(), a.begin(), a.end());
}
inline bool intersects(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
    auto first = a.begin(), second = b.begin();
    while (first != a.end() && second != b.end()) {
        if (*first == *second) return true;
        if (*first < *second) ++first; else ++second;
    }
    return false;
}
template <class Visit> inline void neighbors(const Scene& scene, std::int64_t cell, Visit visit) {
    const auto index = geometry_detail::coordinates(scene, cell);
    for (int axis = 0; axis < 3; ++axis) for (std::int64_t direction : {-1, 1}) {
        auto other = index; other[axis] += direction;
        if (other[axis] >= 0 && static_cast<std::uint64_t>(other[axis]) < scene.shape[axis])
            visit((other[0] * static_cast<std::int64_t>(scene.shape[1]) + other[1]) *
                static_cast<std::int64_t>(scene.shape[2]) + other[2]);
    }
}
inline std::size_t component_count(const Scene& scene, const std::vector<std::int64_t>& cells, const StopToken& stop) {
    std::unordered_set<std::int64_t> remaining(cells.begin(), cells.end());
    std::size_t count = 0;
    queries_detail::Pulse pulse{stop};
    while (!remaining.empty()) {
        ++count;
        std::vector<std::int64_t> pending{*remaining.begin()}; remaining.erase(pending.back());
        while (!pending.empty()) {
            pulse(); const auto cell = pending.back(); pending.pop_back();
            neighbors(scene, cell, [&](auto other) { if (remaining.erase(other)) pending.push_back(other); });
        }
    }
    stop.check(); return count;
}
inline void finite_vector(const Value& value, std::size_t count, std::string_view name) {
    if (!value.is_array() || value.as_array().size() != count) invalid(std::string(name) + " requires finite explicit coordinates");
    for (const auto& number : value.as_array())
        if (!number.is_number() || !std::isfinite(number.as_number())) invalid(std::string(name) + " requires finite explicit coordinates");
}
inline double number(const Value& value, std::string_view name, double low, double high,
                      bool strict_low = false) {
    if (!value.is_number() || !std::isfinite(value.as_number()) || value.as_number() > high ||
        (strict_low ? value.as_number() <= low : value.as_number() < low)) invalid(std::string(name) + " is outside its declared bounds");
    return value.as_number();
}
inline void entry_geometry(const Value& entry) {
    if (!fields(entry, {"centerRasMm", "normalRasUnit", "uRasUnit", "halfExtentsMm", "maximumInwardMm", "maximumAngleDegrees"}))
        invalid("exact entry geometry required");
    finite_vector(entry.at("centerRasMm"), 3, "entry center");
    finite_vector(entry.at("normalRasUnit"), 3, "entry inward normal");
    finite_vector(entry.at("uRasUnit"), 3, "entry tangent");
    const auto normal = queries_detail::point(entry.at("normalRasUnit"));
    const auto u = queries_detail::point(entry.at("uRasUnit"));
    if (std::abs(point_norm(normal) - 1.) > 1e-6 || std::abs(point_norm(u) - 1.) > 1e-6 ||
        std::abs(geometry_detail::dot(normal, u)) > 1e-6) invalid("entry axes must be orthonormal");
    finite_vector(entry.at("halfExtentsMm"), 2, "entry half extents");
    if (entry.at("halfExtentsMm").at(0).as_number() <= 0. || entry.at("halfExtentsMm").at(1).as_number() <= 0.)
        invalid("positive entry half extents required");
    number(entry.at("maximumInwardMm"), "entry inward depth", 0., std::numeric_limits<double>::infinity(), true);
    number(entry.at("maximumAngleDegrees"), "entry maximum angle", 0., 80.);
}
inline void exit_geometry(const Scene& scene, const Value& exit) {
    if (!fields(exit, {"axis", "side", "planeVoxelBoundary", "minimumVoxelBoundary", "maximumVoxelBoundary"}))
        invalid("exact exit/workspace geometry required");
    const auto& axis = exit.at("axis"); const auto& side = exit.at("side");
    if (!axis.is_int() || axis.as_int() < 0 || axis.as_int() > 2 || !side.is_int() ||
        (side.as_int() != -1 && side.as_int() != 1)) invalid("exit axis/side must be explicit integers");
    finite_vector(exit.at("minimumVoxelBoundary"), 3, "workspace minimum");
    finite_vector(exit.at("maximumVoxelBoundary"), 3, "workspace maximum");
    for (std::size_t i = 0; i < 3; ++i) {
        const double low = exit.at("minimumVoxelBoundary").at(i).as_number();
        const double high = exit.at("maximumVoxelBoundary").at(i).as_number();
        if (low < -.5 || high > static_cast<double>(scene.shape[i]) - .5 || low >= high)
            invalid("workspace must lie inside full native voxel-face extents");
    }
    const auto index = static_cast<std::size_t>(axis.as_int());
    number(exit.at("planeVoxelBoundary"), "exit plane", exit.at("minimumVoxelBoundary").at(index).as_number(),
        exit.at("maximumVoxelBoundary").at(index).as_number());
}
inline void coverage(const Value& evidence, bool milling) {
    if (!(milling ? fields(evidence, {"scope", "window", "boneScope", "innerBoundary", "exit"})
                  : fields(evidence, {"scope", "window", "support", "attachments", "exit"})) ||
        !get(evidence, "scope").is_string() || !cranial::detail::in_names(evidence.at("scope").as_string(),
            {"idealized-engineering", "patient-source-review"}))
        invalid(milling ? "explicit evidence scope and four milling proofs required" : "explicit evidence scope and four coverage proofs required", milling);
    for (auto name : (milling ? std::vector<std::string>{"window", "boneScope", "innerBoundary", "exit"}
                             : std::vector<std::string>{"window", "support", "attachments", "exit"})) {
        const auto& proof = evidence.at(name);
        if (!fields(proof, {"status", "provenance", "evidenceSha256"}) || !get(proof, "status").is_string() ||
            !cranial::detail::in_names(proof.at("status").as_string(), {"unknown", "unreviewed", "reviewed"}) ||
            !get(proof, "provenance").is_string() || !nonblank(proof.at("provenance").as_string()))
            invalid("invalid " + name + (milling ? " proof" : " evidence declaration"), milling);
        const auto& seal = proof.at("evidenceSha256");
        if ((!seal.is_null() && (!seal.is_string() || !hex(seal.as_string()))) ||
            (same(proof.at("status"), Value("reviewed")) && seal.is_null()))
            invalid("reviewed " + name + " evidence requires a SHA256", milling);
    }
}
inline void stage_evidence(const Value& evidence) {
    if (!fields(evidence, {"exposure", "membrane", "support", "attachments", "work", "exit"})) stage_fail("EXACT_PROOFS_REQUIRED");
    for (const auto& item : evidence.as_object()) {
        const auto& proof = item.second;
        if (!fields(proof, {"status", "provenance", "evidenceSha256"}) || !get(proof, "status").is_string() ||
            !cranial::detail::in_names(proof.at("status").as_string(), {"unknown", "unreviewed", "reviewed"}) ||
            !get(proof, "provenance").is_string() || proof.at("provenance").as_string().empty() ||
            (!proof.at("evidenceSha256").is_null() && (!proof.at("evidenceSha256").is_string() || !hex(proof.at("evidenceSha256").as_string()))))
            stage_fail("MALFORMED_PROOF");
    }
}
inline void window_geometry(const Scene& scene, const std::vector<std::int64_t>& cells,
                              const Value& entry, const Value& exit, const StopToken& stop) {
    const auto center = queries_detail::point(entry.at("centerRasMm"));
    const auto normal = queries_detail::point(entry.at("normalRasUnit"));
    const auto u = queries_detail::point(entry.at("uRasUnit"));
    const Point v{normal[1] * u[2] - normal[2] * u[1], normal[2] * u[0] - normal[0] * u[2], normal[0] * u[1] - normal[1] * u[0]};
    const auto low = queries_detail::point(exit.at("minimumVoxelBoundary"));
    const auto high = queries_detail::point(exit.at("maximumVoxelBoundary"));
    queries_detail::Pulse pulse{stop};
    for (auto cell : cells) {
        const auto index = geometry_detail::coordinates(scene, cell);
        for (double dx : {-.5, .5}) for (double dy : {-.5, .5}) for (double dz : {-.5, .5}) {
            pulse();
            const Point corner{static_cast<double>(index[0]) + dx, static_cast<double>(index[1]) + dy, static_cast<double>(index[2]) + dz};
            for (int axis = 0; axis < 3; ++axis) if (corner[axis] < low[axis] - 1e-9 || corner[axis] > high[axis] + 1e-9)
                invalid("window cells extend outside handling workspace");
            const auto local = geometry_detail::subtract(geometry_detail::transform(scene.affine, corner), center);
            if (std::abs(geometry_detail::dot(local, u)) > entry.at("halfExtentsMm").at(0).as_number() + 1e-9 ||
                std::abs(geometry_detail::dot(local, v)) > entry.at("halfExtentsMm").at(1).as_number() + 1e-9 ||
                geometry_detail::dot(local, normal) < -1e-9 || geometry_detail::dot(local, normal) > entry.at("maximumInwardMm").as_number() + 1e-9)
                invalid("window full affine cells exceed the entry rectangle/depth");
        }
    }
    stop.check();
}

struct SourceFacts {
    std::string labels;
    Value affine;
    std::string grid;
    Value targets = Value::array();
    Value target_hashes = Value::object();
    std::vector<std::int64_t> owned_targets;
    std::vector<std::int64_t> bone;
    std::vector<std::int64_t> membrane;
};

inline SourceFacts source_facts(const Scene& scene, const StopToken& stop) {
    SourceFacts facts;
    facts.labels = labels_digest(scene, stop); facts.affine = affine_value(scene);
    facts.grid = digest(Value::object({{"labels", facts.labels}, {"affine", facts.affine}, {"source", scene.source}}));
    std::vector<unsigned char> owned(scene.labels.size(), 0);
    for (const auto& region : scene.regions.as_array()) if (get(region, "target").truthy()) {
        facts.targets.append(region);
        if (!region.at("name").is_string()) scene_fail("target names require strings");
        facts.target_hashes.set(region.at("name").as_string(), get(get(scene.source, "annotationHashes"), region.at("name").as_string()));
    }
    queries_detail::Pulse pulse{stop};
    for (std::size_t cell = 0; cell < scene.labels.size(); ++cell) {
        pulse(); if (!scene.labels[cell]) continue;
        const auto& region = scene.region(scene.labels[cell]);
        if (get(region, "target").truthy()) { owned[cell] = 1; facts.owned_targets.push_back(static_cast<std::int64_t>(cell)); }
        const auto& material = region.at("material").as_string();
        if (material == "bone") facts.bone.push_back(static_cast<std::int64_t>(cell));
        if (material == "membrane") facts.membrane.push_back(static_cast<std::int64_t>(cell));
    }
    facts.target_hashes = Value::object({{"ownedTarget", mask_digest(scene, owned, stop)},
        {"targetRegions", facts.targets}, {"originalAnnotationHashes", facts.target_hashes}});
    stop.check(); return facts;
}

inline void validate_source_records(const Scene& scene, const StopToken& stop) {
    for (auto field : {"attachments", "strands", "bleedingSources", "functionalEvidence"}) {
        const auto& records = cranial::detail::array(scene.properties, field);
        if (!records.is_array()) scene_fail(std::string(field) + " must be a list");
        std::set<std::string> ids;
        for (const auto& record : records.as_array()) {
            stop.check();
            if (!record.is_object() || !get(record, "id").is_string() || !ids.insert(record.at("id").as_string()).second)
                scene_fail(std::string(field) + " requires distinct string IDs");
            if (record.contains("cell")) {
                const auto& cell = record.at("cell");
                if (!cell.is_int() || cell.as_int() < 0 || static_cast<std::uint64_t>(cell.as_int()) >= scene.labels.size())
                    scene_fail("property cell must index this source grid");
            }
            const auto cells = property_cells(cranial::detail::array(record, "cells"), scene.labels.size(), stop);
            if (cranial::detail::in_names(field, {"attachments", "strands"}) && cells.empty())
                scene_fail(std::string(field) + " requires declared source cells");
            if (cranial::detail::in_names(field, {"bleedingSources", "functionalEvidence"}) && !record.contains("cell"))
                scene_fail(std::string(field) + " requires a declared source cell");
            for (auto name : {"rateMm3PerSecond", "coagulationExposure", "reservoirMm3", "exposure"})
                if (const auto* value = record.find(name); value && (!value->is_number() || !std::isfinite(value->as_number()) || value->as_number() < 0.))
                    scene_fail(std::string(name) + " must be finite and nonnegative");
            if (std::string_view(field) == "functionalEvidence" && (!get(record, "probe").is_string() ||
                !cranial::detail::in_names(record.at("probe").as_string(), {"cortical", "subcortical"})))
                scene_fail("functional evidence requires declared cortical/subcortical coverage");
        }
    }
    property_cells(cranial::detail::array(scene.properties, "boneAnchors"), scene.labels.size(), stop);
    stop.check();
}

inline Value access_bindings(const Scene& scene, const SourceFacts& facts, bool milling) {
    Value source = scene.source;
    const std::string field = milling ? "cranialMillingBaseCaseHash" : "cranialAccessBaseCaseHash";
    if (const auto* base = source.find(field)) {
        const auto saved = *base;
        if (!get(source, "caseHash").is_string() || !saved.is_string() || !hex(saved.as_string(), true))
            invalid("computed base case binding is invalid", milling);
        source.erase(field); source.set("caseHash", saved);
    }
    const auto properties = without(scene.properties, milling ? "cranialMillingV1" : "cranialAccessV1");
    return Value::object({{"sourceBinding", digest(Value::object({{"labels", facts.labels}, {"affine", facts.affine},
        {"regions", scene.regions}, {"source", source}, {"properties", properties}}))},
        {"targetBinding", digest(facts.target_hashes)},
        {"attachmentsBinding", digest(Value::object({{"present", scene.properties.contains("attachments")},
            {"records", get(scene.properties, "attachments")}}))}});
}

inline void validate_geometry(const Scene& scene, const StopToken& stop) {
    stop.check();
    if (static_cast<std::uint64_t>(geometry_detail::grid_size(scene)) != scene.labels.size()) scene_fail("labels do not match the native grid");
    if (!scene.regions.is_array() || !scene.source.is_object() || !scene.properties.is_object()) scene_fail("metadata containers differ");
    std::set<std::int64_t> declared;
    for (const auto& region : scene.regions.as_array()) {
        const auto& id = get(region, "id");
        if (!id.is_int() || id.as_int() <= 0 || id.as_int() >= 65536 || !declared.insert(id.as_int()).second) scene_fail("invalid region identity");
        if (!get(region, "material").is_string() || !cranial::detail::in_names(region.at("material").as_string(),
            {"unknown", "bone", "soft_tissue", "tumor", "membrane", "vessel", "fluid", "functional_region"})) scene_fail("invalid declared material");
        for (auto flag : {"protected", "critical"}) if (const auto* value = region.find(flag); value && !value->is_bool())
            scene_fail("explicit anatomical protection flags must be boolean");
        const auto index = scene.region_index[static_cast<std::size_t>(id.as_int())];
        if (index < 0 || &scene.regions.at(static_cast<std::size_t>(index)) != &region) scene_fail("region lookup differs from declarations");
    }
    queries_detail::Pulse pulse{stop};
    for (auto label : scene.labels) { pulse(); if (label && !declared.count(label)) scene_fail("occupied label has no material declaration"); }
    for (double x : scene.affine) if (!std::isfinite(x)) scene_fail("affine requires finite entries");
    if (scene.affine[12] != 0. || scene.affine[13] != 0. || scene.affine[14] != 0. || scene.affine[15] != 1.)
        scene_fail("native affine requires a homogeneous final row");
    // Signed-permutation admission lets us independently derive every operative transform.
    if (geometry_scene_supported(scene)) {
        std::array<double, 16> inverse{0.,0.,0.,0., 0.,0.,0.,0., 0.,0.,0.,0., 0.,0.,0.,1.};
        std::array<double, 3> spacing{};
        for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
            const double scale = scene.affine[4 * row + column];
            if (scale != 0.) {
                inverse[4 * column + row] = 1. / scale;
                inverse[4 * column + 3] = -scene.affine[4 * row + 3] / scale;
                spacing[column] = std::abs(scale);
            }
        }
        for (std::size_t i = 0; i < 16; ++i) if (scene.inverse[i] != inverse[i]) scene_fail("inverse differs from the bound affine");
        for (std::size_t i = 0; i < 3; ++i) if (scene.spacing[i] != spacing[i]) scene_fail("spacing differs from the bound affine");
        const auto& a = scene.affine;
        const double determinant = (a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8])) + a[2] * (a[4] * a[9] - a[5] * a[8]);
        // NumPy det converts the LU diagonal through its accumulated log magnitude.
        const double volume = std::exp(((0. + std::log(spacing[0])) + std::log(spacing[1])) + std::log(spacing[2]));
        if (std::abs(determinant) < 1e-9 || scene.voxel_mm3 != volume) scene_fail("voxel volume differs from the bound affine");
    } else scene_fail("native affine exactness is not qualified");
    if (scene.source_inspection_only != scene.properties.contains("cranialPlanningCandidateV1"))
        scene_fail("source inspection policy differs from its source declaration");
    stop.check();
}

inline void validate_original_targets(const Scene& scene, const SourceFacts& facts, const StopToken& stop) {
    auto expected = facts.owned_targets;
    const auto& memberships = scene.original_target_memberships;
    if (memberships.is_null()) {
        for (const auto& region : facts.targets.as_array())
            if (get(scene.source, "annotationHashes").contains(region.at("name").as_string()))
                throw std::invalid_argument("CRANIOTOMY_ORIGINAL_TARGET_EVIDENCE_REQUIRED");
    } else {
        if (!memberships.is_object()) throw std::invalid_argument("CRANIOTOMY_ORIGINAL_TARGET_BINDING_INVALID");
        std::set<std::string> names;
        for (const auto& region : facts.targets.as_array()) names.insert(region.at("name").as_string());
        if (memberships.as_object().size() != names.size()) throw std::invalid_argument("CRANIOTOMY_ORIGINAL_TARGET_BINDING_INVALID");
        for (const auto& name : names) {
            stop.check();
            const auto* member = memberships.find(name);
            if (!member) throw std::invalid_argument("CRANIOTOMY_ORIGINAL_TARGET_BINDING_INVALID");
            const auto ids = source_cells(*member, scene.labels.size(), "original target", false, true, false, stop);
            std::vector<unsigned char> mask(scene.labels.size(), 0);
            queries_detail::Pulse pulse{stop};
            for (auto cell : ids) { pulse(); mask[static_cast<std::size_t>(cell)] = 1; }
            if (!same(get(get(scene.source, "annotationHashes"), name), Value(mask_digest(scene, mask, stop))))
                throw std::invalid_argument("CRANIOTOMY_ORIGINAL_TARGET_BINDING_INVALID");
            expected = set_union(expected, ids);
        }
    }
    if (scene.original_target_cells != expected) throw std::invalid_argument("CRANIOTOMY_ORIGINAL_TARGET_BINDING_INVALID");
    stop.check();
}

inline void validate_protection(const Scene& scene, const SourceFacts& facts, const StopToken& stop) {
    const auto& record = scene.protection_contract;
    if (scene.source_inspection_only && (!record.is_null() || !scene.guard_config.is_null() || !scene.cranial_stages.is_null()))
        throw std::invalid_argument("SOURCE_INSPECTION_ONLY: executable protection/stage contracts cannot be attached");
    if (record.is_null()) {
        if (!scene.protected_cells.empty() || !scene.guard_config.is_null()) scene_fail("protection metadata requires its content-bound contract");
        if (!scene.source_inspection_only) {
            if (get(scene.source, "protectedAnatomyV1").truthy()) scene_fail("declared source protection is missing");
            for (const auto& region : scene.regions.as_array()) if (get(region, "protected").truthy() || get(region, "critical").truthy())
                scene_fail("declared source protection is missing");
        }
        return;
    }
    if (!fields(record, {"schema", "sceneId", "sourceBinding", "config", "clearanceMm", "required", "annotations", "contentHash"}) ||
        !same(get(record, "schema"), Value("resectionlab.surgical-protection-contract.v1")) ||
        !same(get(record, "contentHash"), Value(digest(without(record, "contentHash")))))
        throw std::invalid_argument("Protection contract integrity mismatch");
    if (!same(record.at("sceneId"), Value(scene.identity)) || !same(record.at("sourceBinding"), Value(facts.grid)))
        throw std::invalid_argument("Protection contract source scene mismatch");
    const auto& config = record.at("config");
    if (!fields(config, {"require_critical_coverage", "critical_max_displacement_mm"}) ||
        !get(config, "require_critical_coverage").is_bool() || !get(config, "critical_max_displacement_mm").is_number() ||
        !std::isfinite(config.at("critical_max_displacement_mm").as_number()) || config.at("critical_max_displacement_mm").as_number() < 0. ||
        !same(config, scene.guard_config)) scene_fail("protection configuration differs from its contract");
    if (!record.at("clearanceMm").is_number() || !std::isfinite(record.at("clearanceMm").as_number()) || record.at("clearanceMm").as_number() < 0.)
        scene_fail("invalid protection clearance");
    if (!record.at("required").is_array() || !record.at("annotations").is_array()) scene_fail("invalid protection annotations");
    std::set<std::string> names, present, missing;
    std::vector<std::int64_t> cells;
    for (const auto& annotation : record.at("annotations").as_array()) {
        stop.check();
        if (!fields(annotation, {"structureId", "cells", "provenance", "reviewStatus", "displayName"}) ||
            !get(annotation, "structureId").is_string() || annotation.at("structureId").as_string().empty() ||
            !get(annotation, "provenance").truthy() || !names.insert(annotation.at("structureId").as_string()).second)
            scene_fail("invalid protection annotation identity");
        const auto ids = source_cells(annotation.at("cells"), scene.labels.size(), "protection", true, true, false, stop);
        if (!ids.empty()) present.insert(annotation.at("structureId").as_string());
        cells = set_union(cells, ids);
    }
    if (cells != scene.protected_cells) scene_fail("protected memberships differ from their contract");
    std::set<std::string> declared_ids;
    const auto check_declared = [&](const std::string& identity, const std::vector<std::int64_t>& required) {
        for (const auto& annotation : record.at("annotations").as_array())
            if (same(annotation.at("structureId"), Value(identity))) {
                const auto actual = source_cells(annotation.at("cells"), scene.labels.size(), "protection", true, true, false, stop);
                if (subset(required, actual)) return;
                scene_fail("declared source protection was reduced");
            }
        scene_fail("declared source protection is missing");
    };
    for (const auto& row : cranial::detail::array(scene.source, "protectedAnatomyV1").as_array()) {
        stop.check();
        if (!get(row, "structureId").is_string()) scene_fail("invalid source protection identity");
        const auto ids = property_cells(row.at("cells"), scene.labels.size(), stop);
        declared_ids.insert(row.at("structureId").as_string()); check_declared(row.at("structureId").as_string(), ids);
    }
    for (const auto& region : scene.regions.as_array()) if (get(region, "protected").truthy() || get(region, "critical").truthy()) {
        stop.check();
        const auto identity = region.contains("structureId") ? region.at("structureId") : region.at("name");
        if (!identity.is_string()) scene_fail("invalid source protection identity");
        if (!declared_ids.insert(identity.as_string()).second) continue;
        std::vector<std::int64_t> ids;
        queries_detail::Pulse pulse{stop};
        for (std::size_t i = 0; i < scene.labels.size(); ++i) { pulse(); if (scene.labels[i] == region.at("id").as_int()) ids.push_back(static_cast<std::int64_t>(i)); }
        check_declared(identity.as_string(), ids);
    }
    for (const auto& name : record.at("required").as_array()) {
        if (!name.is_string()) scene_fail("protection required identities must be strings");
        if (!present.count(name.as_string())) missing.insert(name.as_string());
    }
    if (!present.count("brainstem")) missing.insert("brainstem");
    if (config.at("require_critical_coverage").as_bool() && !missing.empty()) {
        std::string message("MISSING_CRITICAL_ANNOTATIONS: ");
        for (const auto& name : missing) { if (message.back() != ' ') message += ", "; message += name; }
        throw std::invalid_argument(message);
    }
    stop.check();
}

inline void validate_access(State& engine, const SourceFacts& facts, const StopToken& stop) {
    const auto& scene = *engine.scene;
    const bool flap_present = scene.properties.contains("cranialAccessV1");
    const bool milling_present = scene.properties.contains("cranialMillingV1");
    if (flap_present && milling_present) invalid("milling and flap declarations are exclusive", true);
    const auto& raw = scene.cranial_access;
    if (raw.is_null()) {
        if (flap_present || milling_present) invalid("present declaration cannot be null", milling_present);
        return;
    }
    const bool milling = cranial::is_milling(raw);
    if (!(milling ? fields(raw, {"schema", "method", "sourceBinding", "targetBinding", "attachmentsBinding", "apertureCells", "retainedSkullCells", "boneAnchorCells", "entry", "exit", "evidence", "contentHash"})
                  : fields(raw, {"schema", "sourceBinding", "targetBinding", "attachmentsBinding", "windowCells", "flapCells", "kerfCells", "retainedSkullCells", "boneAnchorCells", "entry", "exit", "evidence", "contentHash"})) ||
        !same(get(raw, "schema"), Value(milling ? "resectionlab.cranial-milling.v1" : "resectionlab.cranial-access.v1")) ||
        (milling && !same(get(raw, "method"), Value("milling")))) invalid(milling ? "exact versioned milling declaration required" : "exact versioned declaration required", milling);
    const auto* declaration = scene.properties.find(milling ? "cranialMillingV1" : "cranialAccessV1");
    if (!declaration || !same(*declaration, raw)) invalid("native access differs from its source declaration", milling);
    if (!same(raw.at("contentHash"), Value(digest(without(raw, "contentHash"))))) invalid("content or source/target/attachment binding differs", milling);
    const auto bindings = access_bindings(scene, facts, milling);
    for (const auto& binding : bindings.as_object())
        if (!same(raw.at(binding.first), binding.second)) invalid("content or source/target/attachment binding differs", milling);
    std::vector<std::int64_t> window, flap, kerf, retained, anchors;
    try {
        window = source_cells(raw.at(milling ? "apertureCells" : "windowCells"), scene.labels.size(), milling ? "aperture" : "windowCells", false, true, false, stop);
        retained = source_cells(raw.at("retainedSkullCells"), scene.labels.size(), milling ? "retained skull" : "retainedSkullCells", false, true, false, stop);
        anchors = source_cells(raw.at("boneAnchorCells"), scene.labels.size(), milling ? "bone anchors" : "boneAnchorCells", true, true, false, stop);
        if (!milling) {
            flap = source_cells(raw.at("flapCells"), scene.labels.size(), "flapCells", false, true, false, stop);
            kerf = source_cells(raw.at("kerfCells"), scene.labels.size(), "kerfCells", false, true, false, stop);
        }
        entry_geometry(raw.at("entry")); exit_geometry(scene, raw.at("exit"));
        if (milling) window_geometry(scene, window, raw.at("entry"), raw.at("exit"), stop);
    } catch (const std::invalid_argument& error) { if (milling) invalid(error.what(), true); throw; }
    const auto declared_anchors = property_cells(cranial::detail::array(scene.properties, "boneAnchors"), scene.labels.size(), stop);
    if (milling) {
        if (intersects(window, retained) || set_union(window, retained) != facts.bone) invalid("A/R must partition all categorical source bone", true);
        if (!subset(anchors, retained) || !subset(declared_anchors, anchors)) invalid("explicit source bone anchors must remain in R and cannot be omitted", true);
        const auto axis = static_cast<std::size_t>(raw.at("exit").at("axis").as_int());
        std::int64_t low = std::numeric_limits<std::int64_t>::max(), high = -1;
        queries_detail::Pulse pulse{stop};
        for (std::size_t cell = 0; cell < scene.labels.size(); ++cell) if (scene.labels[cell]) {
            pulse(); const auto position = geometry_detail::coordinates(scene, static_cast<std::int64_t>(cell))[axis];
            low = std::min(low, position); high = std::max(high, position);
        }
        const auto side = raw.at("exit").at("side").as_int();
        const double plane = raw.at("exit").at("planeVoxelBoundary").as_number();
        if (high < 0 || (side == -1 && plane > static_cast<double>(low) - .5 + 1e-9) ||
            (side == 1 && plane < static_cast<double>(high) + .5 - 1e-9)) invalid("exit plane must bound the original source exterior", true);
    } else {
        const auto removed_bone = set_union(flap, kerf);
        if (intersects(flap, kerf) || intersects(retained, removed_bone) || set_union(removed_bone, retained) != facts.bone)
            invalid("F/K/R must partition all categorical source bone");
        if (!subset(removed_bone, window) || !subset(window, facts.bone) || !subset(anchors, retained))
            invalid("window must contain F union K; anchors must belong to retained skull");
        if (!subset(declared_anchors, anchors)) invalid("declared source bone anchors cannot be omitted or reassigned");
    }
    coverage(raw.at("evidence"), milling);
    if (!milling) window_geometry(scene, window, raw.at("entry"), raw.at("exit"), stop);
    stop.check();
}

inline void validate_stage(State& engine, const SourceFacts& facts, const StopToken& stop) {
    const auto& scene = *engine.scene;
    const auto& raw = scene.cranial_stages;
    if (raw.is_null()) return;
    if (!raw.is_object() || !same(get(raw, "schema"), Value("resectionlab.cranial-stages.v1"))) stage_fail("CONTRACT_SCHEMA");
    try {
        if (scene.cranial_access.is_null()) stage_fail("BONE_CONTRACT_BINDING");
        cranial::require_access_evidence(engine, stop);
        const auto patch = source_cells(raw.at("membraneCells"), scene.labels.size(), "", false, false, true, stop);
        const auto rim = source_cells(raw.at("retainedMembraneCells"), scene.labels.size(), "", false, false, true, stop);
        const auto anchors = source_cells(raw.at("membraneAnchorCells"), scene.labels.size(), "", true, false, true, stop);
        const auto work = source_cells(raw.at("workCells"), scene.labels.size(), "", false, false, true, stop);
        if (intersects(patch, rim) || set_union(patch, rim) != facts.membrane || !subset(anchors, rim)) stage_fail("MEMBRANE_PARTITION_REQUIRED");
        if (component_count(scene, patch, stop) != 1) stage_fail("CONNECTED_PATCH_REQUIRED");
        for (auto cell : work) if (!cranial::detail::in_names(scene.material(cell), {"soft_tissue", "tumor"})) stage_fail("WORK_MATERIAL_SCOPE");
        if (scene.original_target_cells.empty()) throw std::invalid_argument("CRANIOTOMY_NONEMPTY_TARGET_EVIDENCE_REQUIRED");
        if (intersects(patch, scene.original_target_cells) || intersects(patch, scene.protected_cells)) stage_fail("PATCH_TARGET_OR_PROTECTED_MEMBERSHIP");
        const auto& bone = scene.cranial_access;
        const auto window = source_cells(bone.at(cranial::is_milling(bone) ? "apertureCells" : "windowCells"), scene.labels.size(), "window", false, true, false, stop);
        if (intersects(work, patch) || intersects(work, rim) || intersects(work, window)) stage_fail("DISJOINT_WORK_SCOPE_REQUIRED");
        if (!scene.properties.contains("attachments")) stage_fail("EXPLICIT_ATTACHMENTS_REQUIRED");
        const auto& attachments = scene.properties.at("attachments");
        if (!attachments.is_array()) stage_fail("ATTACHMENT_IDENTITY");
        std::set<std::string> attachment_ids;
        for (const auto& attachment : attachments.as_array()) {
            stop.check();
            const auto ids = property_cells(attachment.at("cells"), scene.labels.size(), stop);
            if (intersects(patch, ids)) {
                if (!attachment.at("id").is_string() || !attachment_ids.insert(attachment.at("id").as_string()).second) stage_fail("ATTACHMENT_IDENTITY");
            }
        }
        Value adapter = get(scene.source, "adapter");
        if (!adapter.truthy()) adapter = Value::object();
        if (!adapter.is_object()) stage_fail("SOURCE_ADAPTER_DECLARATION");
        Value geometry = get(adapter, "sourceGeometry");
        if (!geometry.truthy()) geometry = get(scene.source, "sourceGeometry");
        if (!geometry.truthy()) geometry = scene.source;
        if (!geometry.is_object()) stage_fail("SOURCE_GEOMETRY_DECLARATION");
        if ((get(geometry, "topology").is_string() && cranial::detail::in_names(geometry.at("topology").as_string(), {"sheet", "strand"})) ||
            get(geometry, "faces").truthy()) stage_fail("MESH_PATCH_CORRESPONDENCE_UNSUPPORTED");
        if (!raw.at("startingExposure").is_string() || !cranial::detail::in_names(raw.at("startingExposure").as_string(), {"conditional_post_scalp_exposure", "intact_head"}))
            stage_fail("EXPOSURE_DECLARATION");
        std::set<std::pair<std::int64_t, std::int64_t>> edges;
        for (auto cell : patch) {
            stop.check();
            neighbors(scene, cell, [&](auto other) { if (contains(rim, other)) edges.emplace(std::min(cell, other), std::max(cell, other)); });
        }
        if (edges.empty()) stage_fail("RETAINED_RIM_BOUNDARY_REQUIRED");
        const auto normal = queries_detail::point(bone.at("entry").at("normalRasUnit"));
        Point mapped;
        for (int axis = 0; axis < 3; ++axis) mapped[axis] = (scene.inverse[4 * axis] * normal[0] + scene.inverse[4 * axis + 1] * normal[1]) + scene.inverse[4 * axis + 2] * normal[2];
        int axis = 0;
        for (int other = 1; other < 3; ++other) if (std::abs(mapped[other]) > std::abs(mapped[axis])) axis = other;
        const int sign = mapped[axis] > 0. ? -1 : 1;
        double low = std::numeric_limits<double>::infinity(), high = -std::numeric_limits<double>::infinity();
        for (auto cell : patch) {
            stop.check(); const double projection = geometry_detail::dot(world_point(scene, cell), normal);
            low = std::min(low, projection); high = std::max(high, projection);
        }
        const double depth = std::max(2., std::ceil((high - low) + 2. - 1e-9));
        if (depth > 5.) stage_fail("SOURCE_SPAN_EXCEEDS_SCISSOR_FAMILY");
        stage_evidence(raw.at("evidence"));
        Value boundary = Value::array(), ids = Value::array();
        for (const auto& edge : edges) boundary.append(Value::array({edge.first, edge.second}));
        for (const auto& id : attachment_ids) ids.append(id);
        Value expected = Value::object({{"schema", "resectionlab.cranial-stages.v1"}, {"sourceSceneId", scene.identity},
            {"sourceBinding", facts.grid}, {"targetBinding", digest(Value::object({{"sourceSceneId", scene.identity},
                {"originalTargetCells", cell_values(scene.original_target_cells, stop)}}))},
            {"protectionBinding", scene.protection_contract.is_null() ? Value() : scene.protection_contract.at("contentHash")},
            {"boneContractHash", bone.at("contentHash")}, {"boneCells", cell_values(window, stop)},
            {"membraneCells", cell_values(patch, stop)}, {"retainedMembraneCells", cell_values(rim, stop)},
            {"membraneAnchorCells", cell_values(anchors, stop)}, {"boundaryEdges", boundary}, {"attachmentIds", ids},
            {"workCells", cell_values(work, stop)}, {"originalTargetCells", cell_values(scene.original_target_cells, stop)},
            {"startingExposure", raw.at("startingExposure")}, {"membraneLiftAxis", axis}, {"membraneOutwardSign", sign},
            {"scissorsDepthMm", depth}, {"evidence", raw.at("evidence")}});
        expected.set("contentHash", digest(expected));
        if (!same(raw, expected)) stage_fail("CONTRACT_BINDING");
    } catch (const std::out_of_range&) { stage_fail("CONTRACT_BINDING"); }
      catch (const std::bad_variant_access&) { stage_fail("CONTRACT_BINDING"); }
    stop.check();
}

}  // namespace admission_detail

inline void validate_scene_contracts(State& engine, const StopToken& stop) {
    stop.check();
    if (!engine.scene) admission_detail::scene_fail("native state has no source scene");
    try {
        const auto& scene = *engine.scene;
        admission_detail::validate_geometry(scene, stop);
        admission_detail::validate_source_records(scene, stop);
        const auto facts = admission_detail::source_facts(scene, stop);
        if (digest(Value::object({{"labels", facts.labels}, {"affine", facts.affine}, {"regions", scene.regions},
            {"source", scene.source}, {"properties", scene.properties}})) != scene.identity)
            admission_detail::scene_fail("source identity differs from its owned buffers and metadata");
        admission_detail::validate_original_targets(scene, facts, stop);
        admission_detail::validate_protection(scene, facts, stop);
        admission_detail::validate_access(engine, facts, stop);
        admission_detail::validate_stage(engine, facts, stop);
    } catch (const std::out_of_range&) { admission_detail::scene_fail("incomplete source contract"); }
      catch (const std::bad_variant_access&) { admission_detail::scene_fail("source contract field has the wrong type"); }
    stop.check();
}

}  // namespace resectionlab::sim
