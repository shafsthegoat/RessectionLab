// Ordered physical action execution behind the existing Python simulator interface.
#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace resectionlab_executor {
namespace px = pybind11;

inline px::object owned(PyObject* value) {
    if (value == nullptr) throw px::error_already_set();
    return px::reinterpret_steal<px::object>(value);
}
inline px::object item(px::handle value, px::handle key) {
    return owned(PyObject_GetItem(value.ptr(), key.ptr()));
}
inline px::object item(px::handle value, const char* key) { return item(value, px::str(key)); }
inline bool truth(px::handle value) {
    const int result = PyObject_IsTrue(value.ptr());
    if (result < 0) throw px::error_already_set();
    return result != 0;
}
inline bool equal(px::handle left, px::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) throw px::error_already_set();
    return result != 0;
}
inline bool equal(px::handle left, const char* right) { return equal(left, px::str(right)); }
inline bool includes(px::handle values, px::handle value) {
    const int result = PySequence_Contains(values.ptr(), value.ptr());
    if (result < 0) throw px::error_already_set();
    return result != 0;
}
inline bool one_of(px::handle value, std::initializer_list<const char*> choices) {
    for (const char* choice : choices) if (equal(value, choice)) return true;
    return false;
}
inline px::object difference(px::handle left, px::handle right) {
    return owned(PyNumber_Subtract(left.ptr(), right.ptr()));
}
inline px::object addition(px::handle left, px::handle right) {
    return owned(PyNumber_Add(left.ptr(), right.ptr()));
}
inline px::object as_set(px::handle value) { return owned(PySet_New(value.ptr())); }
inline px::object sorted(px::handle value) {
    auto result = owned(PySequence_List(value.ptr()));
    if (PyList_Sort(result.ptr()) < 0) throw px::error_already_set();
    return result;
}
inline bool greater(px::handle value, double limit) {
    const int result = PyObject_RichCompareBool(value.ptr(), px::float_(limit).ptr(), Py_GT);
    if (result < 0) throw px::error_already_set();
    return result != 0;
}
inline void refuse(const char* reason) { throw px::value_error(reason); }
inline void refuse(const char* prefix, px::handle suffix) {
    throw px::value_error(px::str(addition(px::str(prefix), suffix)).cast<std::string>());
}

// Source labels are owned, immutable uint16 arrays. Reacquire after physics callbacks.
class MaterialLookup {
    px::object engine_, labels_, regions_;
    const unsigned char* labels_data_ = nullptr;
    px::ssize_t count_ = 0;
    bool initialized_ = false;
    std::unordered_map<px::ssize_t, px::object> materials_;
public:
    explicit MaterialLookup(px::object engine) : engine_(std::move(engine)) {}
    px::object operator()(px::handle cell) {
        if (!initialized_) {
            labels_ = engine_.attr("scene").attr("labels");
            regions_ = engine_.attr("by_label");
            if (px::isinstance<px::array>(labels_)) {
                auto array = px::reinterpret_borrow<px::array>(labels_);
                if (array.dtype().is(px::dtype::of<std::uint16_t>()) &&
                        (array.flags() & px::array::c_style) && PyDict_CheckExact(regions_.ptr())) {
                    labels_data_ = static_cast<const unsigned char*>(array.data());
                    count_ = array.size();
                }
            }
            initialized_ = true;
        }
        if (labels_data_ != nullptr && PyLong_CheckExact(cell.ptr())) {
            px::ssize_t index = PyLong_AsSsize_t(cell.ptr());
            if (index == -1 && PyErr_Occurred()) throw px::error_already_set();
            if (index < 0) index += count_;
            if (index >= 0 && index < count_) {
                auto existing = materials_.find(index);
                if (existing != materials_.end()) return existing->second;
                std::uint16_t label_id;
                std::memcpy(&label_id, labels_data_ + index * sizeof(label_id), sizeof(label_id));
                auto label = px::int_(label_id);
                auto region = item(regions_, label);
                auto material = item(region, "material");
                if (PyDict_CheckExact(region.ptr()) && PyUnicode_CheckExact(material.ptr())) {
                    materials_.emplace(index, material);
                } else {
                    // Custom rows must retain every lookup and its callbacks.
                    labels_data_ = nullptr;
                    materials_.clear();
                }
                return material;
            }
        }
        auto labels = engine_.attr("scene").attr("labels").attr("ravel")();
        auto label = owned(PyNumber_Long(item(labels, cell).ptr()));
        return item(item(engine_.attr("by_label"), label), "material");
    }
};

inline px::object physical_substep(px::object engine, px::object key, px::object op,
                                   px::object p, px::object q, px::object active, px::object dt) {
    auto import = [](const char* name) { return px::module_::import(name); };
    if (equal(op, "bladeManeuver"))
        return import("resectionlab.surgical_planar_maneuver").attr("advance")(engine, key, p, q, dt);
    auto globals = import("resectionlab.surgical_engine");
    auto motion = import("resectionlab.surgical_motion");
    px::object envelopes = motion.attr("envelopes");
    px::object swept_envelopes = motion.attr("swept_envelopes");
    px::object pair_conflict = motion.attr("pair_conflict");
    auto state = [&]() -> px::object { return engine.attr("state"); };
    auto holds = [&]() { return item(state(), "holds"); };
    auto revision = [&]() { return state().attr("get")("toolMechanicsRevision"); };
    auto revised = [&]() { auto value = revision(); return equal(value, px::int_(1)) || equal(value, px::int_(2)); };
    auto numpy = [&]() -> px::object { return globals.attr("np"); };
    auto slot = item(item(state(), "slots"), key);
    auto tool = item(globals.attr("TOOLS"), item(slot, "instrument"));
    auto params = engine.attr("parameters")(slot);
    auto hold_kind = [&]() { return holds().attr("get")(key, px::dict()).attr("get")("kind"); };
    auto changed_rotation = [&]() {
        auto first = globals.attr("rotation")(q);
        auto second = globals.attr("rotation")(item(slot, "quaternion"));
        auto inverse = second.attr("inv")();
        return owned(PyNumber_Multiply(first.ptr(), inverse.ptr())).attr("magnitude")();
    };
    if (equal(hold_kind(), "membrane-patch-v1") && greater(changed_rotation(), 1e-8))
        refuse("MEMBRANE_PATCH_ROTATION_UNSUPPORTED");
    if (equal(hold_kind(), "bone-flap-v1") && greater(changed_rotation(), 1e-8))
        refuse("BONE_FLAP_ROTATION_UNSUPPORTED: translate the source-bound fragment");
    px::list unit_axis; unit_axis.append(0); unit_axis.append(0); unit_axis.append(1);
    auto axis = globals.attr("rotation")(q).attr("apply")(unit_axis);
    auto previous = globals.attr("finite")(item(slot, "position"), 3);
    const bool v2 = truth(engine.attr("scene").attr("properties").attr("get")("tissuePhysicsV2"));
    px::object drill, bipolar_heat;
    if (v2) {
        drill = import("resectionlab.surgical_drilling_v2");
        drill.attr("begin_step")(engine);
        bipolar_heat = import("resectionlab.surgical_bipolar_thermal");
        bipolar_heat.attr("begin_step")(engine);
        if (truth(active) && equal(item(slot, "instrument"), "cranial_drill"))
            p = item(drill.attr("contact_limited_pose")(engine, previous, p, item(params, "radiusMm")), px::int_(0));
    }
    const bool cusa_v3 = equal(item(slot, "instrument"), "cusa") &&
        !globals.attr("cusa_profile")(engine).is_none();
    if (cusa_v3 && truth(active))
        p = item(import("resectionlab.surgical_cusa_v3").attr("contact_limited_pose")(
            engine, previous, p, item(params, "radiusMm")), px::int_(0));
    px::object tip_cells = engine.attr("touched")(previous, p, item(params, "radiusMm"));
    px::dict end_pose; end_pose["position"] = p; end_pose["quaternion"] = q;
    auto swept = swept_envelopes(tool, params, slot, end_pose);
    if (!engine.attr("_cranial_access").is_none())
        import("resectionlab.surgical_bone_transport").attr("check_tool_sweep")(engine, key, end_pose, op);
    if (!engine.attr("_cranial_stages").is_none())
        import("resectionlab.surgical_membrane_transport").attr("check_tool_sweep")(engine, key, end_pose, op);
    px::set shaft_cells;
    const bool revised_contact = revised() && !one_of(item(tool, "effect"), {"remove", "suction", "irrigate", "probe"});
    px::object working_cells = revised_contact ? px::object(px::set()) : as_set(tip_cells);
    px::list queries;
    for (auto raw : swept) {
        auto row = px::reinterpret_borrow<px::object>(raw);
        queries.append(px::make_tuple(item(row, px::int_(0)), item(row, px::int_(1)), item(row, px::int_(2))));
    }
    auto covers = engine.attr("touched_many")(owned(PyList_AsTuple(queries.ptr())));
    auto cover = px::iter(covers);
    for (auto raw : swept) {
        if (cover == px::iterator::sentinel()) break;
        auto row = px::reinterpret_borrow<px::object>(raw);
        px::object target = equal(item(row, px::int_(3)), "shaft") ? px::object(shaft_cells) : working_cells;
        target.attr("update")(*cover);
        ++cover;
    }
    if (revised_contact) tip_cells = sorted(working_cells);
    if (!engine.attr("_cranial_access").is_none())
        import("resectionlab.surgical_cranial_guards").attr("enforce")(engine, key, op, p, q, working_cells);
    auto held = as_set(holds().attr("get")(key, px::dict()).attr("get")("cells", px::list()));
    auto shaft_copy = as_set(shaft_cells);
    auto shaft_held = engine.attr("_shaft_held_cells")(key);
    if (truth(difference(shaft_copy, shaft_held))) refuse("SHAFT_COLLISION");
    if (cusa_v3 && truth(active) && !equal(revision(), px::int_(2))) {
        auto not_held = difference(working_cells, held);
        auto current = as_set(engine.attr("touched")(p, p, item(params, "radiusMm")));
        if (truth(difference(not_held, current)))
            refuse("RETAINED_TISSUE_COLLISION: CUSA body requires a cleared corridor");
    }
    for (auto raw : item(state(), "slots").attr("items")()) {
        auto row = px::reinterpret_borrow<px::object>(raw);
        auto other_key = item(row, px::int_(0)), other = item(row, px::int_(1));
        if (equal(other_key, key)) continue;
        auto other_tool = item(globals.attr("TOOLS"), item(other, "instrument"));
        auto other_params = engine.attr("parameters")(other);
        auto other_envelopes = envelopes(other_tool, other_params, item(other, "position"), item(other, "quaternion"));
        if (truth(pair_conflict(swept, other_envelopes))) refuse("TOOL_COLLISION:", other_key);
    }
    px::list eligible;
    {
        MaterialLookup material(engine);
        if (!one_of(item(tool, "effect"), {"suction", "irrigate"}) && truth(difference(working_cells, held)) &&
                !one_of(op, {"withdraw", "release", "deactivate", "inspect", "exchange"})) {
            for (auto cell : difference(working_cells, held))
                if (!includes(item(tool, "materials"), material(cell))) refuse("UNSUPPORTED_MATERIAL");
            if (!truth(active) && !includes(holds(), key)) refuse("RETAINED_TISSUE_COLLISION");
        }
        auto effect_cells = revised() && !one_of(item(tool, "effect"), {"remove", "suction", "irrigate", "probe"})
            ? sorted(working_cells) : tip_cells;
        for (auto cell : effect_cells)
            if (includes(item(tool, "materials"), material(cell))) eligible.append(cell);
    }
    const bool field_action = truth(active) && one_of(item(tool, "effect"), {"suction", "irrigate"});
    if (field_action && greater(numpy().attr("linalg").attr("norm")(difference(p, previous)), 1e-8)) {
        auto physical = as_set(engine.attr("touched")(p, p, item(tool, "shaftRadiusMm")));
        auto previous_physical = as_set(engine.attr("touched")(previous, previous, item(tool, "shaftRadiusMm")));
        if (truth(difference(physical, previous_physical))) refuse("RETAINED_TISSUE_COLLISION");
    }
    if (equal(op, "withdraw")) {
        auto previous_contacts = as_set(engine.attr("touched")(previous, previous, item(params, "radiusMm")));
        auto current_contacts = as_set(tip_cells);
        if (revised_contact) {
            previous_contacts = px::set(); current_contacts = px::set();
            for (auto raw : envelopes(tool, params, previous, item(slot, "quaternion"))) {
                auto row = px::reinterpret_borrow<px::object>(raw);
                if (!equal(item(row, px::int_(3)), "shaft")) previous_contacts.attr("update")(
                    engine.attr("touched")(item(row, px::int_(0)), item(row, px::int_(1)), item(row, px::int_(2))));
            }
            for (auto raw : envelopes(tool, params, p, q)) {
                auto row = px::reinterpret_borrow<px::object>(raw);
                if (!equal(item(row, px::int_(3)), "shaft")) current_contacts.attr("update")(
                    engine.attr("touched")(item(row, px::int_(0)), item(row, px::int_(1)), item(row, px::int_(2))));
            }
        }
        if (truth(difference(current_contacts, previous_contacts))) refuse("WITHDRAWAL_PATH_BLOCKED");
    }
    if (truth(tip_cells) && !field_action) {
        bool unsupported = !truth(active) && !includes(holds(), key);
        if (!unsupported) {
            MaterialLookup material(engine);
            for (auto cell : tip_cells) if (!includes(item(tool, "materials"), material(cell))) { unsupported = true; break; }
        }
        if (unsupported && !one_of(op, {"withdraw", "release", "deactivate", "inspect", "exchange"}))
            refuse(truth(active) ? "UNSUPPORTED_MATERIAL" : "RETAINED_TISSUE_COLLISION");
    }
    px::object events = px::list();
    const bool shared_clock = truth(globals.attr("surgical_clock").attr("enabled")(engine));
    if (shared_clock)
        events = globals.attr("surgical_clock").attr("advance_instruments")(
            engine, key, op, p, q, active, dt, working_cells, eligible);
    else if (truth(active)) events = engine.attr("_effect")(key, op, eligible, p, q, dt);
    if (includes(holds(), key)) engine.attr("occupancy")();
    if (includes(holds(), key) && (equal(op, "move") ||
            one_of(item(holds(), key).attr("get")("kind"), {"bone-flap-v1", "membrane-patch-v1"}))) {
        import("resectionlab.surgical_effects").attr("validate_hold_motion")(engine, key);
        auto hold = item(holds(), key);
        if (v2 && includes(hold, px::str("target"))) {
            import("resectionlab.surgical_deformation_v2").attr("set_target")(engine, key, p);
        } else {
            if (revised() && greater(changed_rotation(), 1e-8))
                refuse("FRAGMENT_ROTATION_UNSUPPORTED: translate the source-bound fragment");
            px::tuple slices(2);
            slices[0] = px::slice(px::none(), px::int_(3), px::none());
            slices[1] = px::slice(px::none(), px::int_(3), px::none());
            auto inverse = item(engine.attr("scene").attr("inverse"), slices);
            auto origin = globals.attr("finite")(item(hold, "origin"), 3);
            auto relative = difference(p, origin);
            auto delta = owned(PyNumber_MatrixMultiply(inverse.ptr(), relative.ptr()));
            auto rounded = numpy().attr("rint")(delta);
            auto snapped = rounded.attr("astype")(import("builtins").attr("int"));
            px::list zero_offset; zero_offset.append(0); zero_offset.append(0); zero_offset.append(0);
            auto base = numpy().attr("asarray")(hold.attr("get")("baseOffset", zero_offset));
            auto offset = addition(snapped, base).attr("tolist")();
            if (equal(item(slot, "instrument"), "bone_flap_holder"))
                import("resectionlab.surgical_bone_transport").attr("check_sweep")(engine, key, offset, end_pose);
            if (equal(item(slot, "instrument"), "membrane_forceps"))
                import("resectionlab.surgical_membrane_transport").attr("check_sweep")(engine, key, offset, end_pose);
            hold[px::str("offset")] = offset;
        }
        engine.attr("occupancy")();
    }
    for (auto raw : item(state(), "slots").attr("items")()) {
        auto row = px::reinterpret_borrow<px::object>(raw);
        auto other_key = item(row, px::int_(0)), other = item(row, px::int_(1));
        if (shared_clock || equal(other_key, key) || !truth(item(other, "active"))) continue;
        auto other_tool = item(globals.attr("TOOLS"), item(other, "instrument"));
        auto other_position = item(other, "position");
        auto other_radius = item(engine.attr("parameters")(other), "radiusMm");
        px::object other_cells = engine.attr("touched")(other_position, other_position, other_radius);
        if (revised() && !one_of(item(other_tool, "effect"), {"remove", "suction", "irrigate", "probe"})) {
            px::set contacts;
            auto other_params = engine.attr("parameters")(other);
            for (auto raw_envelope : envelopes(other_tool, other_params, item(other, "position"), item(other, "quaternion"))) {
                auto envelope = px::reinterpret_borrow<px::object>(raw_envelope);
                if (!equal(item(envelope, px::int_(3)), "shaft")) contacts.attr("update")(
                    engine.attr("touched")(item(envelope, px::int_(0)), item(envelope, px::int_(1)), item(envelope, px::int_(2))));
            }
            other_cells = sorted(contacts);
        }
        px::list other_eligible;
        {
            MaterialLookup material(engine);
            for (auto cell : other_cells)
                if (includes(item(other_tool, "materials"), material(cell))) other_eligible.append(cell);
        }
        if (truth(item(other_tool, "materials")) && px::len(other_eligible) != px::len(other_cells))
            refuse("OTHER_ACTIVE_TOOL_UNSUPPORTED_MATERIAL:", other_key);
        auto other_p = numpy().attr("asarray")(item(other, "position"));
        auto other_q = numpy().attr("asarray")(item(other, "quaternion"));
        events.attr("extend")(engine.attr("_effect")(other_key, "apply", other_eligible, other_p, other_q, dt));
        if (v2) {
            px::object contact = import("resectionlab.surgical_vessels_v2").attr("contact");
            auto force = item(engine.attr("parameters")(other), "forceN");
            contact(engine, other_cells, dt, force);
        }
    }
    px::list vessels;
    {
        MaterialLookup material(engine);
        for (auto cell : working_cells) {
            auto cell_material = material(cell);
            if (equal(cell_material, "vessel")) vessels.append(cell);
        }
        if (!shared_clock) {
            auto previous_vessels = as_set(item(state(), "vesselEncounters"));
            auto current_vessels = as_set(vessels);
            auto combined = owned(PyNumber_Or(previous_vessels.ptr(), current_vessels.ptr()));
            state()[px::str("vesselEncounters")] = sorted(combined);
            for (auto cell : working_cells) {
                auto cell_material = material(cell);
                auto exposure = item(state(), "materialExposure");
                auto previous_exposure = exposure.attr("get")(cell_material, 0.);
                auto amount = addition(previous_exposure, dt);
                item(state(), "materialExposure")[cell_material] = amount;
            }
        }
    }
    px::object contact_cells = one_of(item(tool, "effect"), {"elevate", "dissect"})
        ? as_set(working_cells) : px::object(px::set());
    px::object is_surface_slot = import("resectionlab.surgical_retractor_runtime").attr("is_surface_slot");
    for (auto raw : holds().attr("items")()) {
        auto row = px::reinterpret_borrow<px::object>(raw);
        auto hold_key = item(row, px::int_(0)), hold = item(row, px::int_(1));
        if (!truth(is_surface_slot(engine, hold_key))) contact_cells.attr("update")(item(hold, "cells"));
    }
    contact_cells.attr("difference_update")(item(state(), "removed"));
    if (!shared_clock) for (auto cell : contact_cells) {
        auto contact_key = px::str(cell);
        auto contacts = item(state(), "contact");
        auto previous_contact = contacts.attr("get")(contact_key, 0.);
        auto amount = addition(previous_contact, dt);
        item(state(), "contact")[contact_key] = amount;
    }
    auto position_list = p.attr("tolist")();
    auto quaternion_list = q.attr("tolist")();
    slot.attr("update")(px::arg("position") = position_list, px::arg("quaternion") = quaternion_list);
    if (!one_of(op, {"place", "exchange", "release", "inspect", "deactivate"})) slot[px::str("inserted")] = px::bool_(true);
    auto elapsed_state = state();
    auto elapsed = item(elapsed_state, "elapsedSeconds");
    elapsed_state[px::str("elapsedSeconds")] = owned(PyNumber_InPlaceAdd(elapsed.ptr(), dt.ptr()));
    auto travel_state = state();
    auto travel = item(travel_state, "travelMm");
    auto travel_distance = numpy().attr("linalg").attr("norm")(difference(p, previous));
    auto travel_float = owned(PyNumber_Float(travel_distance.ptr()));
    travel_state[px::str("travelMm")] = owned(PyNumber_InPlaceAdd(travel.ptr(), travel_float.ptr()));
    px::object tick_fields = import("resectionlab.surgical_effects").attr("tick_fields");
    if (v2) {
        auto brain = import("resectionlab.surgical_deformation_v2");
        auto vessel = import("resectionlab.surgical_vessels_v2");
        if (!shared_clock) {
            auto force = truth(active) ? item(params, "forceN") : px::object(px::float_(0.));
            vessel.attr("contact")(engine, vessels, dt, force);
        }
        brain.attr("tick")(engine, dt);
        engine.attr("occupancy")();
        if (truth(state().attr("get")("deformation"))) {
            for (auto raw : item(state(), "slots").attr("items")()) {
                auto row = px::reinterpret_borrow<px::object>(raw);
                auto shaft_key = item(row, px::int_(0)), shaft_slot = item(row, px::int_(1));
                auto own_held = engine.attr("_shaft_held_cells")(shaft_key);
                auto shaft_tool = item(globals.attr("TOOLS"), item(shaft_slot, "instrument"));
                auto shaft_params = engine.attr("parameters")(shaft_slot);
                for (auto raw_envelope : envelopes(shaft_tool, shaft_params, item(shaft_slot, "position"), item(shaft_slot, "quaternion"))) {
                    auto envelope = px::reinterpret_borrow<px::object>(raw_envelope);
                    if (equal(item(envelope, px::int_(3)), "shaft")) {
                        auto touched = as_set(engine.attr("touched")(item(envelope, px::int_(0)), item(envelope, px::int_(1)), item(envelope, px::int_(2))));
                        if (truth(difference(touched, own_held))) refuse("POST_DEFORMATION_SHAFT_COLLISION:", shaft_key);
                    }
                }
            }
        }
        drill.attr("end_step")(engine, dt);
        bipolar_heat.attr("end_step")(engine, dt);
        vessel.attr("tick")(engine, dt);
    }
    for (auto raw : holds().attr("items")()) {
        auto row = px::reinterpret_borrow<px::object>(raw);
        auto hold_key = item(row, px::int_(0)), hold = item(row, px::int_(1));
        if (equal(hold.attr("get")("kind"), "bone-flap-v1"))
            import("resectionlab.surgical_bone_transport").attr("check_sweep")(
                engine, hold_key, item(hold, "offset"), item(item(state(), "slots"), hold_key));
    }
    if (!engine.attr("_cranial_access").is_none())
        import("resectionlab.surgical_bone_transport").attr("check_detached")(engine);
    tick_fields(engine, dt);
    if (!engine.attr("_cranial_stages").is_none()) {
        auto membrane = import("resectionlab.surgical_membrane_transport");
        px::object check_sweep = membrane.attr("check_sweep");
        px::object check_detached = membrane.attr("check_detached");
        px::object validate_effects = import("resectionlab.surgical_cranial_stages").attr("validate_effects");
        for (auto raw : holds().attr("items")()) {
            auto row = px::reinterpret_borrow<px::object>(raw);
            auto hold_key = item(row, px::int_(0)), hold = item(row, px::int_(1));
            if (equal(hold.attr("get")("kind"), "membrane-patch-v1"))
                check_sweep(engine, hold_key, item(hold, "offset"), item(item(state(), "slots"), hold_key));
        }
        check_detached(engine);
        validate_effects(engine);
    }
    return events;
}

}  // namespace resectionlab_executor

inline void bind_executor_physical(pybind11::module_& module) {
    module.def("physical_substep", &resectionlab_executor::physical_substep,
        pybind11::arg("engine"), pybind11::arg("key"), pybind11::arg("op"),
        pybind11::arg("p"), pybind11::arg("q"), pybind11::arg("active"), pybind11::arg("dt"));
}

#pragma once

#include <pybind11/pybind11.h>

namespace resectionlab_executor_trajectory {
namespace py = pybind11;

inline py::object owned(PyObject* value) {
    if (value == nullptr) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(value);
}

inline bool truth(py::handle value) {
    const int result = PyObject_IsTrue(value.ptr());
    if (result < 0) throw py::error_already_set();
    return result != 0;
}

inline bool compare(py::handle first, py::handle second, int operation) {
    const int result = PyObject_RichCompareBool(first.ptr(), second.ptr(), operation);
    if (result < 0) throw py::error_already_set();
    return result != 0;
}

inline py::object add(py::handle first, py::handle second) {
    return owned(PyNumber_Add(first.ptr(), second.ptr()));
}

inline py::object subtract(py::handle first, py::handle second) {
    return owned(PyNumber_Subtract(first.ptr(), second.ptr()));
}

inline py::object multiply(py::handle first, py::handle second) {
    return owned(PyNumber_Multiply(first.ptr(), second.ptr()));
}

inline py::object divide(py::handle first, py::handle second) {
    return owned(PyNumber_TrueDivide(first.ptr(), second.ptr()));
}

inline py::object distance(py::handle engine_module, py::handle first, py::handle second) {
    py::object norm = engine_module.attr("np").attr("linalg").attr("norm");
    py::object difference = engine_module.attr("np").attr("subtract")(first, second);
    return owned(PyNumber_Float(norm(difference).ptr()));
}

inline py::object path_length(py::handle engine_module, py::handle builtins, const py::list& segments) {
    py::object sum = builtins.attr("sum");
    py::list lengths;
    for (py::handle row : segments) {
        py::tuple segment = py::reinterpret_borrow<py::tuple>(row);
        py::object start = segment[0], end = segment[1];
        lengths.append(distance(engine_module, end["position"], start["position"]));
    }
    // Python 3.12's float sum must retain its established accumulation order.
    return sum(lengths);
}

inline py::tuple unpack_quaternion(py::object value) {
    PyObject* raw_iterator = PyObject_GetIter(value.ptr());
    if (raw_iterator == nullptr) {
        if (PyErr_ExceptionMatches(PyExc_TypeError) && Py_TYPE(value.ptr())->tp_iter == nullptr
                && !PySequence_Check(value.ptr()))
            PyErr_Format(PyExc_TypeError, "cannot unpack non-iterable %.200s object", Py_TYPE(value.ptr())->tp_name);
        throw py::error_already_set();
    }
    py::object iterator = owned(raw_iterator);
    py::tuple result(2);
    for (int index = 0; index < 2; ++index) {
        PyObject* item = PyIter_Next(iterator.ptr());
        if (item == nullptr) {
            if (PyErr_Occurred()) throw py::error_already_set();
            PyErr_Format(PyExc_ValueError, "not enough values to unpack (expected 2, got %d)", index);
            throw py::error_already_set();
        }
        result[index] = owned(item);
    }
    PyObject* extra = PyIter_Next(iterator.ptr());
    if (extra != nullptr) {
        Py_DECREF(extra);
        throw py::value_error("too many values to unpack (expected 2)");
    }
    if (PyErr_Occurred()) throw py::error_already_set();
    return result;
}

class HandledException {
    PyObject* type_;
    PyObject* value_;
    PyObject* traceback_;
public:
    explicit HandledException(const py::error_already_set& error) {
        PyErr_GetExcInfo(&type_, &value_, &traceback_);
        PyErr_SetExcInfo(Py_XNewRef(error.type().ptr()), Py_XNewRef(error.value().ptr()),
                        Py_XNewRef(error.trace().ptr()));
    }
    ~HandledException() { PyErr_SetExcInfo(type_, value_, traceback_); }
    HandledException(const HandledException&) = delete;
    HandledException& operator=(const HandledException&) = delete;
};

inline py::tuple run_trajectory(py::object engine, py::object key, py::object operation,
        py::list segments, bool active, py::object duration, int total, bool articulate,
        py::object jaw_target, py::object previous_jaw, py::object progress_reason,
        bool cusa_v3, py::object check, py::list events) {
    if (total < 1 || total > 20000 || segments.empty() || segments.size() > 512)
        throw py::value_error("Native trajectory requires 1..20000 substeps and 1..512 segments");
    int scheduled = 0;
    for (py::handle row : segments) {
        if (!PyTuple_CheckExact(row.ptr()) || PyTuple_GET_SIZE(row.ptr()) != 3)
            throw py::value_error("Native trajectory segments must be (start, end, count) tuples");
        PyObject* count = PyTuple_GET_ITEM(row.ptr(), 2);
        if (!PyLong_CheckExact(count))
            throw py::value_error("Native trajectory counts must be integers");
        const long n = PyLong_AsLong(count);
        if (PyErr_Occurred()) throw py::error_already_set();
        if (n < 1 || n > total - scheduled)
            throw py::value_error("Native trajectory segment count exceeds its substep bound");
        scheduled += static_cast<int>(n);
    }
    if (scheduled != total)
        throw py::value_error("Native trajectory total differs from its segment counts");

    py::module_ engine_module = py::module_::import("resectionlab.surgical_engine");
    py::module_ motion_module = py::module_::import("resectionlab.surgical_motion");
    py::module_ builtins = py::module_::import("builtins");
    py::object trajectory_quaternion = motion_module.attr("_trajectory_quaternion");
    int done = 0;
    py::object reason = py::none(), contact_position = py::none(), path_fraction = py::none();
    py::object completed_length = py::float_(0.);
    int visited_segments = 0;
    for (py::handle row : segments) {
        if (++visited_segments > 512 || !PyTuple_CheckExact(row.ptr()) || PyTuple_GET_SIZE(row.ptr()) != 3)
            throw py::value_error("Native trajectory segment inventory changed");
        py::tuple segment = py::reinterpret_borrow<py::tuple>(row);
        py::object start = segment[0], end = segment[1];
        if (!PyLong_CheckExact(segment[2].ptr()))
            throw py::value_error("Native trajectory segment count changed");
        const int n = py::cast<int>(segment[2]);
        if (n < 1 || n > total - done)
            throw py::value_error("Native trajectory segment exceeds its remaining substep bound");
        py::object slerp = engine_module.attr("Slerp");
        py::list times(2), quaternions(2);
        times[0] = 0; times[1] = 1;
        py::object from_quat = engine_module.attr("Rotation").attr("from_quat");
        quaternions[0] = start["quaternion"]; quaternions[1] = end["quaternion"];
        py::object interpolation = slerp(times, from_quat(quaternions));
        py::object quaternion_chunk = py::none();
        for (int j = 0; j < n; ++j) {
            if (!check.is_none()) engine_module.attr("_checked_operation")(check);
            py::object t = divide(py::int_(j + 1), py::int_(n));
            py::object weight = subtract(py::int_(1), t);
            py::object first = engine_module.attr("finite")(start["position"], 3);
            py::object first_term = multiply(weight, first);
            py::object last = engine_module.attr("finite")(end["position"], 3);
            py::object last_term = multiply(t, last);
            py::object position = add(first_term, last_term);
            py::object rotation_type = engine_module.attr("Rotation");
            py::object slerp_type = engine_module.attr("Slerp");
            py::tuple quaternion_result = unpack_quaternion(trajectory_quaternion(interpolation, j, n,
                quaternion_chunk, rotation_type, slerp_type));
            py::object quaternion = quaternion_result[py::int_(0)];
            quaternion_chunk = quaternion_result[py::int_(1)];
            if (quaternion.is_none()) quaternion = interpolation(t).attr("as_quat")();
            py::object checkpoint = engine_module.attr("_microstep_checkpoint");
            py::object old = checkpoint(engine.attr("state"));
            py::object delta_events;
            try {
                if (articulate) {
                    py::object jaw = jaw_target;
                    if (done + 1 != total) {
                        py::object delta = subtract(jaw_target, previous_jaw);
                        py::object scaled = multiply(delta, py::int_(done + 1));
                        jaw = add(previous_jaw, divide(scaled, py::int_(total)));
                    }
                    py::object slots = engine.attr("state")["slots"];
                    py::object slot = slots[key];
                    py::object parameters = slot["parameters"];
                    parameters["jawOpeningMm"] = jaw;
                }
                py::object substep = engine.attr("_substep");
                py::object dt = divide(duration, py::int_(total));
                delta_events = substep(key, operation, position, quaternion, active, dt);
            } catch (py::error_already_set& error) {
                if (!error.matches(PyExc_ValueError)) throw;
                HandledException handled(error);
                if (!check.is_none()) engine_module.attr("_checked_operation")(check);
                py::object code = py::str(error.value()).attr("split")(":", 1)[py::int_(0)];
                const bool protected_stop = compare(code, py::str("PROTECTED_STRUCTURE"), Py_EQ)
                    && !truth(py::getattr(engine, "_restoring_legacy_guard_partial", py::bool_(false)));
                if (protected_stop || !engine.attr("_cranial_access").is_none()
                        || truth(code.attr("startswith")(py::make_tuple(
                            "CRANIAL_ACCESS_", "CRANIOTOMY_", "BONE_FLAP_"))))
                    throw;
                engine.attr("state") = old;
                reason = py::str(error.value());
                contact_position = position.attr("tolist")();
                break;
            }
            events.attr("extend")(delta_events);
            ++done;
        }
        if (truth(reason)) break;
        py::object segment_length = distance(engine_module, end["position"], start["position"]);
        if (segments.size() > 1 && active && truth(progress_reason)) {
            py::object slots = engine.attr("state")["slots"];
            py::object slot = slots[key];
            py::object remaining = distance(engine_module, slot["position"], end["position"]);
            if (compare(remaining, py::float_(1e-5), Py_GT)) {
                py::object length = path_length(engine_module, builtins, segments);
                py::object progress = builtins.attr("max")(py::float_(0.), subtract(segment_length, remaining));
                py::object denominator = builtins.attr("max")(length, py::float_(1e-15));
                path_fraction = divide(add(completed_length, progress), denominator);
                reason = progress_reason;
                slots = engine.attr("state")["slots"];
                slot = slots[key];
                contact_position = slot["position"][py::slice(py::none(), py::none(), py::none())];
                py::dict event;
                event["kind"] = cusa_v3 ? "cusa_contact_dwell" : "drill_contact_dwell";
                event["position"] = contact_position;
                event["remainingPathMm"] = multiply(length, subtract(py::float_(1.), path_fraction));
                events.attr("append")(event);
                break;
            }
        }
        completed_length = add(completed_length, segment_length);
    }
    return py::make_tuple(done, reason, contact_position, path_fraction);
}
}  // namespace resectionlab_executor_trajectory

inline void bind_executor_trajectory(pybind11::module_& module) {
    namespace py = pybind11;
    module.def("run_trajectory", &resectionlab_executor_trajectory::run_trajectory,
        py::arg("engine"), py::arg("key"), py::arg("op"), py::arg("segments"),
        py::arg("active"), py::arg("duration"), py::arg("total"), py::arg("articulate"),
        py::arg("jaw_target"), py::arg("previous_jaw"), py::arg("progress_reason"),
        py::arg("cusa_v3"), py::arg("check"), py::arg("events"));
}

#pragma once

#include <unordered_set>
#include <vector>
#include <pybind11/pybind11.h>

namespace resectionlab_executor_state {

inline bool container(PyObject* value) {
    return PyDict_CheckExact(value) || PyList_CheckExact(value) || PyTuple_CheckExact(value);
}

inline bool atom(PyObject* value) {
    return value == Py_None || PyBool_Check(value) || PyLong_CheckExact(value) ||
        PyUnicode_CheckExact(value) || PyFloat_CheckExact(value);
}

inline bool owned_graph(PyObject* value) {
    std::vector<PyObject*> pending{value};
    std::unordered_set<PyObject*> seen;
    // No Python allocations/calls or GIL release occur while borrowing graph entries.
    while (!pending.empty()) {
        PyObject* current = pending.back();
        pending.pop_back();
        if (!seen.insert(current).second) {
            continue;
        }
        if (!container(current)) {
            return false;
        }
        if (PyDict_CheckExact(current)) {
            Py_ssize_t position = 0;
            PyObject* key;
            PyObject* item;
            while (PyDict_Next(current, &position, &key, &item)) {
                if (!PyUnicode_CheckExact(key)) {
                    return false;
                }
            }
            position = 0;
            while (PyDict_Next(current, &position, &key, &item)) {
                if (container(item)) {
                    pending.push_back(item);
                } else if (!atom(item)) {
                    return false;
                }
            }
        } else {
            const bool list = PyList_CheckExact(current);
            const Py_ssize_t size = list ? PyList_GET_SIZE(current) : PyTuple_GET_SIZE(current);
            for (Py_ssize_t index = 0; index < size; ++index) {
                PyObject* item = list ? PyList_GET_ITEM(current, index) : PyTuple_GET_ITEM(current, index);
                if (container(item)) {
                    pending.push_back(item);
                } else if (!atom(item)) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline pybind11::object owned_digest(pybind11::handle value, pybind11::handle json_dumps,
                                    pybind11::handle sha256, pybind11::handle fallback_marker) {
    if (!owned_graph(value.ptr())) {
        return pybind11::reinterpret_borrow<pybind11::object>(fallback_marker);
    }
    pybind11::object payload;
    try {
        payload = json_dumps(value, pybind11::arg("sort_keys") = true,
            pybind11::arg("separators") = pybind11::make_tuple(",", ":"),
            pybind11::arg("allow_nan") = false);
    } catch (pybind11::error_already_set& error) {
        if (!error.matches(PyExc_TypeError) && !error.matches(PyExc_ValueError)) {
            throw;
        }
        return pybind11::reinterpret_borrow<pybind11::object>(fallback_marker);
    }
    auto hexadecimal = sha256(payload.attr("encode")("utf-8")).attr("hexdigest")();
    pybind11::str prefix("sha256:");
    PyObject* result = PyNumber_Add(prefix.ptr(), hexadecimal.ptr());
    if (result == nullptr) {
        throw pybind11::error_already_set();
    }
    return pybind11::reinterpret_steal<pybind11::object>(result);
}

}  // namespace resectionlab_executor_state

inline void bind_executor_state(pybind11::module_& module) {
    module.def("owned_digest", &resectionlab_executor_state::owned_digest,
        pybind11::arg("value"), pybind11::arg("json_dumps"), pybind11::arg("sha256"),
        pybind11::arg("fallback_marker"));
}

#pragma once

#include <pybind11/pybind11.h>

namespace resectionlab_executor_cache {
namespace py = pybind11;

inline py::object owned(PyObject* value) {
    if (value == nullptr) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(value);
}

inline py::object identity(py::handle value) {
    py::object result = owned(PyLong_FromVoidPtr(value.ptr()));
    if (PySys_Audit("builtins.id", "O", result.ptr()) < 0)
        throw py::error_already_set();
    return result;
}

inline py::object marker() {
    return owned(PyObject_CallNoArgs(reinterpret_cast<PyObject*>(&PyBaseObject_Type)));
}

inline bool scalar_default(PyObject* value, const py::tuple& atoms) {
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(value));
    for (py::ssize_t index = 0; index < PyTuple_GET_SIZE(atoms.ptr()); ++index)
        if (type == PyTuple_GET_ITEM(atoms.ptr(), index)) return true;
    return false;
}

inline py::tuple function_identity(py::object function, const py::tuple& atoms) {
    if (!PyFunction_Check(function.ptr())) {
        py::object function_id = identity(function);
        py::object fresh = marker();
        return py::make_tuple(function_id, function, fresh);
    }

    // Attribute access preserves the reference's object.__getattr__ audit events.
    py::object code = owned(PyObject_GetAttrString(function.ptr(), "__code__"));
    py::object defaults = owned(PyObject_GetAttrString(function.ptr(), "__defaults__"));
    py::object keywords = owned(PyObject_GetAttrString(function.ptr(), "__kwdefaults__"));
    if (defaults.is_none() && keywords.is_none()) {
        py::object function_id = identity(function);
        py::object code_id = identity(code);
        return py::make_tuple(function_id, function, code_id, code, py::none());
    }

    // Own the snapshot before allocations or id audit hooks can mutate defaults.
    py::object items = py::none();
    if (PyDict_CheckExact(keywords.ptr())) items = owned(PyDict_Items(keywords.ptr()));
    const py::ssize_t item_count = items.is_none() ? 0 : PyList_GET_SIZE(items.ptr());
    bool scalar_defaults = (defaults.is_none() || PyTuple_CheckExact(defaults.ptr()))
        && (keywords.is_none() || PyDict_CheckExact(keywords.ptr()));
    if (scalar_defaults && !defaults.is_none()) {
        for (py::ssize_t index = 0; index < PyTuple_GET_SIZE(defaults.ptr()); ++index) {
            if (!scalar_default(PyTuple_GET_ITEM(defaults.ptr(), index), atoms)) {
                scalar_defaults = false;
                break;
            }
        }
    }
    if (scalar_defaults) {
        for (py::ssize_t index = 0; index < item_count; ++index) {
            PyObject* pair = PyList_GET_ITEM(items.ptr(), index);
            if (!scalar_default(PyTuple_GET_ITEM(pair, 1), atoms)) {
                scalar_defaults = false;
                break;
            }
        }
    }

    py::object function_id = identity(function);
    py::object code_id = identity(code);
    py::object defaults_id = identity(defaults);
    py::object keywords_id = identity(keywords);
    py::tuple keyword_identities(item_count);
    for (py::ssize_t index = 0; index < item_count; ++index) {
        PyObject* pair = PyList_GET_ITEM(items.ptr(), index);
        py::object key = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(pair, 0));
        py::object value = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(pair, 1));
        py::object key_id = identity(key);
        py::object value_id = identity(value);
        keyword_identities[index] = py::make_tuple(key_id, key, value_id, value);
    }
    py::object reuse = scalar_defaults ? py::object(py::none()) : marker();
    return py::make_tuple(function_id, function, code_id, code, defaults_id, defaults,
        keywords_id, keywords, keyword_identities, reuse);
}

inline py::tuple function_identities(py::tuple functions, py::tuple default_atoms) {
    if (!PyTuple_CheckExact(functions.ptr()) || !PyTuple_CheckExact(default_atoms.ptr()))
        throw py::type_error("Function identity batches require exact tuples");
    py::tuple result(functions.size());
    for (py::ssize_t index = 0; index < PyTuple_GET_SIZE(functions.ptr()); ++index) {
        py::object function = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(functions.ptr(), index));
        result[index] = function_identity(function, default_atoms);
    }
    return result;
}
}  // namespace resectionlab_executor_cache

inline void bind_executor_cache(pybind11::module_& module) {
    module.def("function_identities", &resectionlab_executor_cache::function_identities,
        pybind11::arg("functions"), pybind11::arg("default_atoms"));
}

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace resectionlab_executor_occupancy {
namespace py = pybind11;

inline py::object owned(PyObject* value) {
    if (value == nullptr) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(value);
}

inline bool integer64(PyObject* value) {
    if (!PyLong_CheckExact(value)) return false;
    int overflow = 0;
    PyLong_AsLongLongAndOverflow(value, &overflow);
    if (PyErr_Occurred()) throw py::error_already_set();
    return overflow == 0;
}

inline py::object occupancy_values(py::object labels, py::object ids, py::object cleared,
                                   py::object moved, py::object fallback, py::handle array_type) {
    if (Py_TYPE(labels.ptr()) != reinterpret_cast<PyTypeObject*>(array_type.ptr()) ||
            Py_TYPE(ids.ptr()) != reinterpret_cast<PyTypeObject*>(array_type.ptr()) ||
            !PyFrozenSet_CheckExact(cleared.ptr()) || !PyDict_CheckExact(moved.ptr())) return fallback;
    auto source = py::reinterpret_borrow<py::array>(labels);
    auto queries = py::reinterpret_borrow<py::array>(ids);
    if (source.ndim() != 1 || !source.dtype().is(py::dtype::of<std::uint16_t>()) ||
            !queries.dtype().is(py::dtype::of<std::int64_t>()) ||
            !(source.flags() & py::array::c_style) || !(queries.flags() & py::array::c_style)) return fallback;

    // The caller admits cleared is _integer_cleared; never scan that growing ledger here.
    auto owners = owned(PyDict_Copy(moved.ptr()));
    Py_ssize_t position = 0;
    PyObject* key;
    PyObject* value;
    while (PyDict_Next(owners.ptr(), &position, &key, &value))
        if (!integer64(key) || !integer64(value)) return fallback;

    const auto count = source.size();
    const auto size = queries.size();
    const auto* query_data = static_cast<const unsigned char*>(queries.data());
    auto read_id = [&](py::ssize_t offset) {
        std::int64_t cell;
        std::memcpy(&cell, query_data + offset * sizeof(cell), sizeof(cell));
        return cell;
    };
    // Match advanced indexing: every index is checked before any occupancy result is allocated.
    for (py::ssize_t offset = 0; offset < size; ++offset) {
        const auto cell = read_id(offset);
        if (cell < -count || cell >= count)
            throw py::index_error("index " + std::to_string(cell) +
                " is out of bounds for axis 0 with size " + std::to_string(count));
    }
    std::vector<py::ssize_t> shape;
    for (py::ssize_t axis = 0; axis < queries.ndim(); ++axis) shape.push_back(queries.shape(axis));
    py::array_t<std::int64_t> result(shape);
    auto* output = result.mutable_data();
    const auto* source_data = static_cast<const unsigned char*>(source.data());
    const bool has_cleared = PySet_Size(cleared.ptr()) != 0;
    const bool has_moved = PyDict_Size(owners.ptr()) != 0;
    for (py::ssize_t offset = 0; offset < size; ++offset) {
        const auto cell = read_id(offset);
        const auto wrapped = cell < 0 ? cell + count : cell;
        std::uint16_t label;
        std::memcpy(&label, source_data + wrapped * sizeof(label), sizeof(label));
        std::int64_t identity = label != 0 ? cell + 1 : 0;
        if (has_cleared || has_moved) {
            auto cell_key = owned(PyLong_FromLongLong(cell));
            if (has_cleared) {
                const int removed = PySet_Contains(cleared.ptr(), cell_key.ptr());
                if (removed < 0) throw py::error_already_set();
                if (removed) identity = 0;
            }
            if (has_moved) {
                PyObject* owner = PyDict_GetItemWithError(owners.ptr(), cell_key.ptr());
                if (owner != nullptr) {
                    identity = PyLong_AsLongLong(owner);
                    if (PyErr_Occurred()) throw py::error_already_set();
                } else if (PyErr_Occurred()) throw py::error_already_set();
            }
        }
        output[offset] = identity;
    }
    return result;
}

}  // namespace resectionlab_executor_occupancy

inline void bind_executor_occupancy(pybind11::module_& module) {
    namespace py = pybind11;
    // Obtain the genuine ndarray type through the existing native NumPy interface.
    py::type array_type = py::type::of(py::array_t<std::int64_t>(std::vector<py::ssize_t>{0}));
    module.def("occupancy_values", [array_type](py::object labels, py::object ids, py::object cleared,
                                               py::object moved, py::object fallback) {
        return resectionlab_executor_occupancy::occupancy_values(
            std::move(labels), std::move(ids), std::move(cleared), std::move(moved), std::move(fallback), array_type);
    }, py::arg("labels"), py::arg("ids"), py::arg("cleared"), py::arg("moved"), py::arg("fallback"));
}
