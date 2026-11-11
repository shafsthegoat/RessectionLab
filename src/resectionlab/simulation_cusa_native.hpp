#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_dyadic_native.hpp"
#include "simulation_hash_native.hpp"
#include "simulation_queries_native.hpp"

namespace resectionlab::sim::cusa_v3 {

inline constexpr std::string_view schema = "resectionlab.cusa-contact.v3";
inline constexpr double tip_radius_mm = 1.5;
inline constexpr double shaft_radius_mm = .5;

inline Value default_profile() {
    return Value::object({{"schema", std::string(schema)},
        {"provenance", Value::object({{"kind", "assumed-research"}, {"description",
            "Uncalibrated contact-capacity surrogate; rate is an engineering assumption, not device or patient calibration."}})},
        {"tipRadiusMm", tip_radius_mm}, {"shaftRadiusMm", shaft_radius_mm},
        {"capacityMm3PerSecond", 2.}, {"maxActivation", 1.}});
}

namespace detail {
inline bool keys(const Value& value, std::initializer_list<std::string_view> expected) {
    if (!value.is_object() || value.as_object().size() != expected.size()) return false;
    for (auto key : expected) if (!value.contains(key)) return false;
    return true;
}
inline bool text(const Value& value, std::string_view expected) {
    return value.is_string() && value.as_string() == expected;
}
inline bool nonblank(std::string_view value) {
    std::size_t position = 0;
    while (position < value.size()) {
        const auto codepoint = hash_detail::next_codepoint(value, position);
        if (!((codepoint >= 9 && codepoint <= 13) || (codepoint >= 0x1c && codepoint <= 0x20) ||
                codepoint == 0x85 || codepoint == 0xa0 || codepoint == 0x1680 ||
                (codepoint >= 0x2000 && codepoint <= 0x200a) || codepoint == 0x2028 ||
                codepoint == 0x2029 || codepoint == 0x202f || codepoint == 0x205f || codepoint == 0x3000))
            return true;
    }
    return false;
}
inline double number(const Value& owner, std::string_view key, double fallback = 0.) {
    const auto* value = owner.find(key); return value ? value->as_number() : fallback;
}
inline void add(Value& owner, std::string key, double amount) {
    const double before = number(owner, key);
    owner.set(std::move(key), before + amount);
}
inline std::set<std::int64_t> removed(const State& engine) {
    std::set<std::int64_t> result;
    for (const auto& cell : engine.state.at("removed").as_array()) result.insert(cell.as_int());
    return result;
}
inline std::int64_t source_key(const std::string& key) {
    std::int64_t result;
    const auto parsed = std::from_chars(key.data(), key.data() + key.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != key.data() + key.size() || std::to_string(result) != key)
        throw std::invalid_argument("NATIVE_CUSA_UNSUPPORTED_PROGRESS_KEY");
    return result;
}
inline Value cells_value(const std::vector<std::int64_t>& cells) {
    Value result = Value::array();
    for (auto cell : cells) result.append(cell);
    return result;
}
inline void finite(const Point& position) {
    for (double coordinate : position) if (!std::isfinite(coordinate))
        throw std::invalid_argument("CUSA_V3_FINITE_RAS_MM_POSITION_REQUIRED");
}
}  // namespace detail

inline const Value& validate(const Value& value) {
    if (!detail::keys(value, {"schema", "provenance", "tipRadiusMm", "shaftRadiusMm",
            "capacityMm3PerSecond", "maxActivation"}) || !detail::text(value.at("schema"), schema))
        throw std::invalid_argument("COMPLETE_EXPLICIT_CUSA_V3_PROFILE_REQUIRED");
    const auto& provenance = value.at("provenance");
    if (!detail::keys(provenance, {"kind", "description"}) ||
            !detail::text(provenance.at("kind"), "assumed-research") ||
            !provenance.at("description").is_string() || !detail::nonblank(provenance.at("description").as_string()))
        throw std::invalid_argument("EXPLICIT_ASSUMED_RESEARCH_CUSA_PROVENANCE_REQUIRED");
    for (auto key : {"tipRadiusMm", "shaftRadiusMm", "capacityMm3PerSecond", "maxActivation"}) {
        const auto& number = value.at(key);
        if (!(number.is_int() || number.is_double()) || !std::isfinite(number.as_number()) || number.as_number() <= 0.)
            throw std::invalid_argument(std::string("INVALID_CUSA_V3_PROFILE:") + key);
    }
    if (value.at("tipRadiusMm").as_number() != tip_radius_mm ||
            value.at("shaftRadiusMm").as_number() != shaft_radius_mm)
        throw std::invalid_argument("CUSA_V3_FIXED_CATALOG_GEOMETRY_REQUIRED");
    if (value.at("maxActivation").as_number() != 1.)
        throw std::invalid_argument("CUSA_V3_NORMALIZED_ACTIVATION_REQUIRED");
    return value;
}

inline const Value* profile(const State& engine) {
    const auto* explicit_profile = engine.scene->properties.find("cusaV3");
    if (explicit_profile && explicit_profile->truthy()) return explicit_profile;
    const auto* revision = engine.state.find("toolMechanicsRevision");
    if (revision && revision->is_int() && (revision->as_int() == 1 || revision->as_int() == 2)) {
        static const Value assumed = default_profile();
        return &assumed;
    }
    return nullptr;
}

inline void initialize(State& engine) {
    const auto* current = profile(engine);
    if (!current) return;
    validate(*current);
    if (!engine.state.contains("cusaV3")) engine.state.set("cusaV3", Value::object({
        {"schema", std::string(schema)}, {"progressMm3", Value::object()},
        {"progressExact", Value::object()}, {"contactSeconds", Value::object()},
        {"inputCapacityMm3", 0.}, {"aspiratedSolidMm3", 0.}, {"unusedCapacityMm3", 0.},
        {"discardedCapacityMm3", 0.}, {"effectiveActivationSeconds", 0.}}));
}

inline double nonnegative(const Value& value, std::string_view label) {
    if (!(value.is_int() || value.is_double()) || !std::isfinite(value.as_number()) || value.as_number() < 0.)
        throw std::invalid_argument("INVALID_CUSA_V3_INPUT:" + std::string(label));
    return value.as_number();
}

inline std::pair<double, double> validate_parameters(const Value& params) {
    if (!params.is_object()) throw std::invalid_argument("INVALID_CUSA_V3_PARAMETERS");
    const auto get = [&](std::string_view key, double fallback) {
        const auto* value = params.find(key); return value ? nonnegative(*value, key) : fallback;
    };
    const double radius = get("radiusMm", tip_radius_mm), shaft = get("shaftRadiusMm", shaft_radius_mm);
    if (radius != tip_radius_mm || shaft != shaft_radius_mm)
        throw std::invalid_argument("CUSA_V3_FIXED_CATALOG_GEOMETRY_REQUIRED");
    const double activation = get("power", 1.), duty = get("duty", 1.);
    if (activation > 1. || duty > 1.) throw std::invalid_argument("CUSA_V3_ACTIVATION_AND_DUTY_MUST_BE_IN_0_1");
    return {activation, duty};
}

inline bool progress_supported(const State& engine) {
    const auto* ledger = engine.state.find("cusaV3");
    if (!ledger) return true;
    if (!ledger->is_object()) return false;
    const auto* exact = ledger->find("progressExact");
    if (!exact || !exact->is_object()) return false;
    for (const auto& item : exact->as_object()) if (!dyadic_pair_supported(item.second)) return false;
    for (auto field : {"progressMm3", "contactSeconds"}) {
        const auto* rows = ledger->find(field);
        if (!rows || !rows->is_object()) return false;
        for (const auto& item : rows->as_object()) {
            try { (void)detail::source_key(item.first); } catch (const std::invalid_argument&) { return false; }
            if (!(item.second.is_int() || item.second.is_double()) || !std::isfinite(item.second.as_number())) return false;
        }
    }
    return true;
}

inline double reconcile_removed(State& engine) {
    if (!profile(engine)) return 0.;
    initialize(engine);
    auto& state = engine.state.at("cusaV3");
    const auto removed = detail::removed(engine);
    std::vector<std::string> stale;
    std::vector<double> discarded;
    for (const auto& item : state.at("progressMm3").as_object()) {
        if (removed.count(detail::source_key(item.first))) {
            stale.push_back(item.first); discarded.push_back(item.second.as_number());
        }
    }
    for (const auto& key : stale) state.at("progressMm3").erase(key);
    const double total = dyadic_fsum(discarded);
    for (const auto& key : stale) {
        state.at("contactSeconds").erase(key); state.at("progressExact").erase(key);
    }
    detail::add(state, "discardedCapacityMm3", total);
    return total;
}

inline std::vector<std::int64_t> ordered_contacts(const State& engine,
        const std::vector<std::int64_t>& cells, const Point& position, const StopToken& stop) {
    std::set<std::int64_t> candidates;
    std::size_t visited = 0;
    for (auto cell : cells) {
        if ((visited++ % 32768) == 0) stop.check();
        if (cell < 0 || static_cast<std::uint64_t>(cell) >= engine.scene->labels.size())
            throw std::invalid_argument("INVALID_CUSA_V3_SOURCE_CELL");
        candidates.insert(cell);
    }
    const auto touched_ids = touched(engine, position, position, tip_radius_mm, stop);
    const std::set<std::int64_t> reachable(touched_ids.begin(), touched_ids.end());
    const auto removed = detail::removed(engine);
    std::vector<std::int64_t> ids;
    for (auto cell : candidates) {
        if ((visited++ % 32768) == 0) stop.check();
        if (!reachable.count(cell) || removed.count(cell)) continue;
        const auto label = engine.scene->label_at(cell);
        if (label == 0) continue;
        const auto& material = engine.scene->region(label).at("material").as_string();
        if (material == "soft_tissue" || material == "tumor") ids.push_back(cell);
    }
    const auto positions = current_points(engine, ids, stop, false);
    std::vector<std::pair<double, std::int64_t>> ordered;
    ordered.reserve(ids.size());
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if ((visited++ % 32768) == 0) stop.check();
        Point delta;
        for (unsigned axis = 0; axis < 3; ++axis) delta[axis] = positions[index][axis] - position[axis];
        ordered.emplace_back(point_squared_norm(delta), ids[index]);
    }
    std::sort(ordered.begin(), ordered.end(), [&](const auto& first, const auto& second) {
        if ((visited++ % 32768) == 0) stop.check();
        return first < second;
    });
    ids.clear();
    for (const auto& item : ordered) ids.push_back(item.second);
    return ids;
}

inline Value apply(State& engine, const std::vector<std::int64_t>& cells, double dt,
        const Value& params, const Point& position, bool active, const StopToken& stop) {
    const auto* candidate_profile = profile(engine);
    if (!candidate_profile) throw std::invalid_argument("COMPLETE_EXPLICIT_CUSA_V3_PROFILE_REQUIRED");
    const auto& current = validate(*candidate_profile);
    dt = nonnegative(Value(dt), "dt");
    const auto activation_duty = validate_parameters(params);
    const double activation = activation_duty.first, duty = activation_duty.second;
    detail::finite(position);
    const auto ids = ordered_contacts(engine, cells, position, stop);
    const std::set<std::int64_t> protected_ids(engine.scene->protected_cells.begin(), engine.scene->protected_cells.end());
    for (auto cell : ids) if (protected_ids.count(cell))
        throw std::invalid_argument("PROTECTED_STRUCTURE: CUSA contact includes protected source membership");
    const double budget = active ? ((current.at("capacityMm3PerSecond").as_number() * activation) * duty) * dt : 0.;
    if (!std::isfinite(budget)) throw std::invalid_argument("CUSA_V3_CAPACITY_OVERFLOW");
    initialize(engine);
    const double discarded = reconcile_removed(engine);
    auto& state = engine.state.at("cusaV3");
    auto remaining = Dyadic::from_double(budget);
    const auto volume = Dyadic::from_double(engine.scene->voxel_mm3);
    std::vector<std::int64_t> cleared;
    std::vector<double> spent;
    if (budget > 0.) for (auto cell : ids) {
        stop.check();
        const auto key = std::to_string(cell);
        const auto* previous = state.at("progressExact").find(key);
        const auto old = previous ? Dyadic::from_pair(*previous) : Dyadic();
        const auto required = volume - old;
        const auto allocated = required < remaining ? required : remaining;
        if (!allocated.positive()) continue;
        detail::add(state.at("contactSeconds"), key, dt);
        remaining = remaining - allocated;
        spent.push_back(allocated.to_double());
        if (allocated >= required) {
            cleared.push_back(cell);
            state.at("progressMm3").erase(key); state.at("progressExact").erase(key);
            state.at("contactSeconds").erase(key);
        } else {
            const auto progress = old + allocated;
            state.at("progressMm3").set(key, progress.to_double());
            state.at("progressExact").set(key, progress.pair());
        }
    }
    const double aspirated = static_cast<double>(cleared.size()) * engine.scene->voxel_mm3;
    detail::add(state, "inputCapacityMm3", budget); detail::add(state, "aspiratedSolidMm3", aspirated);
    detail::add(state, "unusedCapacityMm3", remaining.to_double());
    detail::add(state, "effectiveActivationSeconds", active ? (activation * duty) * dt : 0.);
    if (!cleared.empty()) {
        auto removed = detail::removed(engine);
        removed.insert(cleared.begin(), cleared.end());
        engine.state.set("removed", detail::cells_value(std::vector<std::int64_t>(removed.begin(), removed.end())));
    }
    if (budget == 0. && discarded == 0.) return Value::array();
    return Value::array({Value::object({{"kind", "cusa_contact"}, {"inputCapacityMm3", budget},
        {"allocatedCapacityMm3", dyadic_fsum(spent)}, {"unusedCapacityMm3", remaining.to_double()},
        {"aspiratedSolidMm3", aspirated}, {"discardedCapacityMm3", discarded},
        {"removedCells", detail::cells_value(cleared)},
        {"pendingCells", state.at("progressMm3").as_object().size()}})});
}

inline std::pair<Point, std::vector<std::int64_t>> contact_limited_pose(const State& engine,
        const Point& start, const Point& requested, double radius, const StopToken& stop) {
    const auto* current = profile(engine);
    if (!current) throw std::invalid_argument("COMPLETE_EXPLICIT_CUSA_V3_PROFILE_REQUIRED");
    validate(*current); validate_parameters(Value::object({{"radiusMm", radius}}));
    detail::finite(start); detail::finite(requested);
    const auto contacts = [&](const Point& position) {
        return ordered_contacts(engine, touched(engine, position, position, tip_radius_mm, stop), position, stop);
    };
    auto initial = contacts(start);
    if (!initial.empty()) return {start, std::move(initial)};
    Point delta;
    for (unsigned axis = 0; axis < 3; ++axis) delta[axis] = requested[axis] - start[axis];
    const double distance = point_norm(delta);
    const double spacing = *std::min_element(engine.scene->spacing.begin(), engine.scene->spacing.end());
    const double count = std::max(1., std::ceil(distance / (spacing / 4.)));
    if (!std::isfinite(count) || count >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("CUSA contact search exceeds native step limit");
    const auto steps = static_cast<std::int64_t>(count);
    Point low = start;
    for (std::int64_t index = 1; index <= steps; ++index) {
        stop.check();
        Point high;
        for (unsigned axis = 0; axis < 3; ++axis)
            high[axis] = start[axis] + delta[axis] * (static_cast<double>(index) / static_cast<double>(steps));
        auto found = contacts(high);
        if (!found.empty()) {
            for (unsigned iteration = 0; iteration < 32; ++iteration) {
                const auto mid = midpoint_position(low, high);
                if (!contacts(mid).empty()) high = mid; else low = mid;
            }
            return {high, contacts(high)};
        }
        low = high;
    }
    return {requested, {}};
}

inline Value metrics(const State& engine) {
    const auto* state = engine.state.find("cusaV3");
    std::vector<double> values;
    std::size_t pending_cells = 0;
    if (state) if (const auto* progress = state->find("progressMm3")) {
        pending_cells = progress->as_object().size();
        for (const auto& item : progress->as_object()) values.push_back(item.second.as_number());
    }
    const auto number = [&](std::string_view key) { return state ? detail::number(*state, key) : 0.; };
    return Value::object({{"cusaInputCapacityMm3", number("inputCapacityMm3")},
        {"cusaAspiratedSolidMm3", number("aspiratedSolidMm3")}, {"cusaPendingCapacityMm3", dyadic_fsum(values)},
        {"cusaUnusedCapacityMm3", number("unusedCapacityMm3")},
        {"cusaDiscardedCapacityMm3", number("discardedCapacityMm3")}, {"cusaPendingCells", pending_cells}});
}

inline Value descriptor(const State& engine) {
    const auto* current = profile(engine);
    return Value::object({{"enabled", current != nullptr}, {"profile", current ? *current : Value()},
        {"calibration", "uncalibrated assumed research surrogate"},
        {"activation", "power and duty are dimensionless fractions in [0,1]"},
        {"limits", Value::array({"Fixed generic catalog geometry is not a manufacturer tip specification.",
            "Rate, constant capture, and material response are uncalibrated engineering assumptions.",
            "Whole source voxels remain until their complete capacity threshold is reached.",
            "No acoustic, suction-pressure, irrigation, cavitation, heating, or tissue selectivity physics."})}});
}

}  // namespace resectionlab::sim::cusa_v3
