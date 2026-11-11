#pragma once

#include "simulation_hash_native.hpp"
#include "simulation_state_native.hpp"
#include "simulation_stats_native.hpp"

namespace resectionlab::sim {

inline std::string state_id(const State& engine) {
    ++execution_stats.state_hashes;
    Value body = Value::object({{"scene", engine.scene->identity}, {"state", engine.state}});
    if (engine.scene->protection_contract.truthy())
        body.set("protectionContract", engine.scene->protection_contract.at("contentHash"));
    if (!engine.scene->cranial_stages.is_null())
        body.set("cranialStagesContract", engine.scene->cranial_stages.at("contentHash"));
    return digest(body);
}

inline Value snapshot(const State& engine) {
    const auto identity = state_id(engine);
    Value result = Value::object({{"schema", "resectionlab.surgical-toolbox.v1"},
        {"sceneId", engine.scene->identity}, {"actions", engine.actions}, {"stateId", identity}});
    if (!engine.scene->protected_cells.empty()) result.set("hardConstraintPolicy", "atomic-protection-v1");
    if (engine.scene->protection_contract.truthy())
        result.set("protectionContract", engine.scene->protection_contract);
    if (!engine.scene->cranial_stages.is_null())
        result.set("cranialStagesContract", engine.scene->cranial_stages);
    result.set("checkpoint", Value::object({{"position", engine.actions.as_array().size()},
        {"stateId", identity}, {"state", engine.state}}));
    return result;
}

}  // namespace resectionlab::sim
