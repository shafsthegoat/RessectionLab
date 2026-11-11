#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include "simulation_state_native.hpp"

namespace resectionlab::sim {

struct Tool {
    std::string_view id, geometry, effect;
    std::array<std::string_view, 4> materials;
    double radius = 1.;
    double shaft_radius = .5;

    bool accepts(std::string_view material) const {
        for (auto allowed : materials) if (!allowed.empty() && allowed == material) return true;
        return false;
    }
};

inline constexpr std::array<Tool, 17> tools{{
    {"cranial_drill", "burr", "remove", {"bone"}, 2.},
    {"cusa", "aspirator", "remove", {"soft_tissue", "tumor"}, 1.5},
    {"brain_retractor", "blade", "retract", {"soft_tissue", "tumor"}, 3.},
    {"scalpel", "blade", "incise", {"soft_tissue", "tumor"}},
    {"periosteal_elevator", "elevator", "elevate", {"bone", "soft_tissue"}, 2.},
    {"craniotome", "kerf", "kerf", {"bone"}},
    {"dural_scissors", "scissors", "sheet_cut", {"membrane"}},
    {"dissector", "dissector", "dissect", {"soft_tissue", "tumor", "membrane"}},
    {"frazier_suction", "suction", "suction", {}, 1.5},
    {"bipolar_forceps", "forceps", "coagulate", {"soft_tissue", "tumor", "vessel", "membrane"}},
    {"micro_scissors", "scissors", "transect", {"vessel", "soft_tissue", "tumor"}},
    {"tumor_forceps", "forceps", "grasp", {"tumor"}, 1.5},
    {"bone_flap_holder", "forceps", "grasp", {"bone"}, 1.5},
    {"irrigation", "irrigator", "irrigate", {}, 1.5},
    {"cortical_probe", "probe", "probe", {"soft_tissue", "functional_region"}},
    {"subcortical_probe", "probe", "probe", {"soft_tissue", "tumor", "functional_region"}},
    {"membrane_forceps", "forceps", "grasp", {"membrane"}, 1.5},
}};

inline const Tool& tool(std::string_view id) {
    for (const auto& value : tools) if (value.id == id) return value;
    throw std::out_of_range(std::string(id));
}

inline std::string_view tool_name(std::string_view id) {
    constexpr std::array<std::string_view, 17> names{
        "Cranial drill", "CUSA / ultrasonic aspirator", "Brain retractor", "Scalpel",
        "Periosteal elevator", "Craniotome", "Dural scissors", "Microsurgical dissector",
        "Frazier suction", "Bipolar forceps", "Micro scissors", "Tumor forceps",
        "Bone flap holder", "Irrigation", "Cortical stimulation probe",
        "Subcortical stimulation probe", "Membrane forceps"};
    for (std::size_t i = 0; i < tools.size(); ++i) if (tools[i].id == id) return names[i];
    throw std::out_of_range(std::string(id));
}

inline Value parameters(const State& engine, const Value& slot) {
    const auto& instrument = tool(slot.at("instrument").as_string());
    Value result = Value::object();
    const auto* v2 = engine.scene->properties.find("tissuePhysicsV2");
    if (v2 && v2->truthy()) {
        result.set("rpm", 40000.); result.set("forceN", 1.); result.set("duty", 1.);
    }
    result.set("radiusMm", instrument.radius); result.set("depthMm", 2.);
    result.set("flowMm3PerSecond", 20.); result.set("power", 1.); result.set("jawOpeningMm", 2.);
    for (const auto& item : slot.at("parameters").as_object()) result.set(item.first, item.second);
    const auto* revision = engine.state.find("toolMechanicsRevision");
    if (revision && revision->is_int() && revision->as_int() == 2)
        result.set("contactGeometryRevision", 2);
    if (instrument.id == "brain_retractor" && engine.scene->properties.contains("retractorSurfaceContactV1"))
        result.set("retractorFaceContactRevision", 1);
    return result;
}

}  // namespace resectionlab::sim
