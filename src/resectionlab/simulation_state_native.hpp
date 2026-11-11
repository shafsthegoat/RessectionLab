#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "simulation_value_native.hpp"

namespace resectionlab::sim {

struct DeformationSystem;
struct PositionPatchSignature;

struct Scene {
    std::vector<std::uint16_t> labels;
    std::array<std::size_t, 3> shape{};
    // Row-major transforms, matching the source scene's affine layout.
    std::array<double, 16> affine{1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.};
    std::array<double, 16> inverse{1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.};
    std::array<double, 3> spacing{1.,1.,1.};
    double voxel_mm3 = 1.;
    Value regions = Value::array();
    Value properties = Value::object();
    Value source = Value::object();
    std::string identity;
    std::array<std::int32_t, 65536> region_index;

    Value protection_contract;
    Value guard_config;
    Value cranial_access;
    Value cranial_stages;
    std::vector<std::int64_t> protected_cells;
    std::vector<std::int64_t> original_target_cells;
    Value original_target_memberships;
    std::string initial_state_id;
    bool source_inspection_only = false;

    Scene() { region_index.fill(-1); }

    void rebuild_region_index() {
        region_index.fill(-1);
        const auto& declarations = regions.as_array();
        for (std::size_t index = 0; index < declarations.size(); ++index) {
            const auto* id = declarations[index].find("id");
            if (id == nullptr || !id->is_int() || id->as_int() <= 0 || id->as_int() >= 65536 ||
                    region_index[static_cast<std::size_t>(id->as_int())] != -1) {
                throw std::invalid_argument("Region IDs must be distinct positive uint16 values");
            }
            region_index[static_cast<std::size_t>(id->as_int())] = static_cast<std::int32_t>(index);
        }
    }

    const Value& region(std::uint16_t label) const {
        const auto index = region_index[label];
        if (index < 0) throw std::out_of_range("Region ID is not declared");
        return regions.at(static_cast<std::size_t>(index));
    }
    std::uint16_t label_at(std::int64_t cell) const {
        if (cell < 0 || static_cast<std::uint64_t>(cell) >= labels.size()) {
            throw std::out_of_range("Cell outside source grid");
        }
        return labels[static_cast<std::size_t>(cell)];
    }
    std::string_view material(std::int64_t cell) const {
        const auto label = label_at(cell);
        return label == 0 ? std::string_view{} : std::string_view(region(label).at("material").as_string());
    }
    bool target(std::int64_t cell) const {
        const auto label = label_at(cell);
        if (label == 0) return false;
        const auto* flag = region(label).find("target");
        return flag != nullptr && flag->truthy();
    }
};

struct State {
    std::shared_ptr<const Scene> scene;
    Value state = Value::object();
    Value actions = Value::array();
    Value history = Value::array();
    std::uint64_t version = 0;
    bool in_transition = false;
    Value bone_prefix;
    Value stage_permission;
    std::shared_ptr<const DeformationSystem> deformation_system;
    std::shared_ptr<const PositionPatchSignature> position_patch_signature;

    State() = default;
    explicit State(std::shared_ptr<const Scene> source) : scene(std::move(source)) {}
};

}  // namespace resectionlab::sim
