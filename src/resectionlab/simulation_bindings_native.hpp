#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <memory>
#include <unordered_set>
#include "simulation_transport_native.hpp"
#include "simulation_guards_native.hpp"
#include "simulation_physical_native.hpp"
#include "simulation_engine_native.hpp"
#include "simulation_admission_native.hpp"
#include "simulation_deformation_native.hpp"

namespace resectionlab::sim::boundary {
namespace py = pybind11;

struct UnsupportedBoundary : std::runtime_error { using std::runtime_error::runtime_error; };

inline std::string string(PyObject* value) {
    py::object bytes = py::reinterpret_steal<py::object>(PyUnicode_AsEncodedString(value,"utf-8","surrogatepass"));
    if (!bytes) throw py::error_already_set();
    return {PyBytes_AS_STRING(bytes.ptr()),static_cast<std::size_t>(PyBytes_GET_SIZE(bytes.ptr()))};
}
struct Parser {
    std::unordered_set<PyObject*> active;
    std::size_t count = 0;
    Value operator()(PyObject* value,std::size_t depth=0) {
        if (++count > 1048576 || depth > 64) throw UnsupportedBoundary("Native simulation boundary exceeds its value limit");
        if (value == Py_None) return {};
        if (PyBool_Check(value)) return value == Py_True;
        if (PyLong_CheckExact(value)) {
            int overflow = 0; const auto integer = PyLong_AsLongLongAndOverflow(value,&overflow);
            if (PyErr_Occurred()) throw py::error_already_set();
            if (!overflow) return integer;
            auto decimal = py::reinterpret_steal<py::object>(PyObject_Str(value));
            if (!decimal) throw py::error_already_set();
            return Value::big_integer(string(decimal.ptr()));
        }
        if (PyFloat_CheckExact(value)) return PyFloat_AS_DOUBLE(value);
        if (PyUnicode_CheckExact(value)) return string(value);
        if (!PyList_CheckExact(value) && !PyTuple_CheckExact(value) && !PyDict_CheckExact(value))
            throw UnsupportedBoundary("Native simulation requires builtin JSON values");
        if (!active.insert(value).second) throw UnsupportedBoundary("Native simulation requires acyclic JSON values");
        Value result;
        if (PyDict_CheckExact(value)) {
            result = Value::object();
            Py_ssize_t position = 0; PyObject *key,*item;
            while (PyDict_Next(value,&position,&key,&item)) {
                if (!PyUnicode_CheckExact(key)) throw UnsupportedBoundary("Native simulation requires exact string keys");
                result.set(string(key),(*this)(item,depth+1));
            }
        } else {
            result = Value::array();
            const auto size = PyList_CheckExact(value) ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value);
            result.as_array().reserve(static_cast<std::size_t>(size));
            for (Py_ssize_t i = 0; i < size; ++i)
                result.append((*this)(PyList_CheckExact(value) ? PyList_GET_ITEM(value,i) : PyTuple_GET_ITEM(value,i),depth+1));
        }
        active.erase(value); return result;
    }
};
inline Value from_python(py::handle value) { return Parser{}(value.ptr()); }

inline bool exact(const Value& a,const Value& b) {
    if(a.kind()!=b.kind()) return false;
    switch(a.kind()) {
        case Value::Kind::Null: return true;
        case Value::Kind::Bool: return a.as_bool()==b.as_bool();
        case Value::Kind::Int: return a.as_int()==b.as_int();
        case Value::Kind::Double: {
            const double first=a.as_double(),second=b.as_double();
            return std::memcmp(&first,&second,sizeof(double))==0;
        }
        case Value::Kind::String: return a.as_string()==b.as_string();
        case Value::Kind::BigInt: return a.as_bigint()==b.as_bigint();
        case Value::Kind::Array:
            if(a.as_array().size()!=b.as_array().size()) return false;
            for(std::size_t i=0;i<a.as_array().size();++i) if(!exact(a.at(i),b.at(i))) return false;
            return true;
        case Value::Kind::Object:
            if(a.as_object().size()!=b.as_object().size()) return false;
            for(std::size_t i=0;i<a.as_object().size();++i)
                if(a.as_object()[i].first!=b.as_object()[i].first || !exact(a.as_object()[i].second,b.as_object()[i].second)) return false;
            return true;
    }
    return false;
}
inline py::object to_python(const Value& value) {
    switch(value.kind()) {
        case Value::Kind::Null: return py::none();
        case Value::Kind::Bool: return py::bool_(value.as_bool());
        case Value::Kind::Int: return py::int_(value.as_int());
        case Value::Kind::Double: return py::float_(value.as_double());
        case Value::Kind::BigInt: {
            auto result = py::reinterpret_steal<py::object>(PyLong_FromString(value.as_bigint().c_str(),nullptr,10));
            if (!result) throw py::error_already_set(); return result;
        }
        case Value::Kind::String: {
            const auto& text = value.as_string();
            auto result = py::reinterpret_steal<py::object>(PyUnicode_DecodeUTF8(text.data(),text.size(),"surrogatepass"));
            if (!result) throw py::error_already_set(); return result;
        }
        case Value::Kind::Array: {
            py::list result;
            for (const auto& child:value.as_array()) result.append(to_python(child));
            return result;
        }
        case Value::Kind::Object: {
            py::dict result;
            for (const auto& item : value.as_object()) result[to_python(Value(item.first))]=to_python(item.second);
            return result;
        }
    }
    throw py::value_error("Invalid native JSON tag");
}
template<class T> inline std::vector<T> array(const py::array& input,std::initializer_list<py::ssize_t> shape) {
    if (!input.dtype().is(py::dtype::of<T>()) || !(input.flags() & py::array::c_style) ||
        input.ndim()!=static_cast<py::ssize_t>(shape.size()))
        throw py::value_error("Native simulation buffers require exact dtype, shape and contiguous storage");
    std::size_t axis = 0;
    for (auto length : shape) {
        if ((length >= 0 && input.shape(axis)!=length) || input.shape(axis)<=0)
            throw py::value_error("Invalid native simulation buffer shape");
        ++axis;
    }
    std::vector<T> result(static_cast<std::size_t>(input.size()));
    std::memcpy(result.data(),input.data(),result.size()*sizeof(T)); return result;
}
inline std::shared_ptr<Scene> scene(const py::array& labels,const py::array& affine,
                                  const py::array& inverse,const py::array& spacing,
                                  double voxel_mm3,const Value& metadata) {
    auto result=std::make_shared<Scene>();
    result->labels=array<std::uint16_t>(labels,{-1,-1,-1});
    if (result->labels.size()>16777216) throw py::value_error("Native simulation source exceeds its cell limit");
    for (int axis=0;axis<3;++axis) result->shape[axis]=static_cast<std::size_t>(labels.shape(axis));
    const auto a=array<double>(affine,{4,4}), inv=array<double>(inverse,{4,4}), sp=array<double>(spacing,{3});
    std::copy(a.begin(),a.end(),result->affine.begin()); std::copy(inv.begin(),inv.end(),result->inverse.begin());
    std::copy(sp.begin(),sp.end(),result->spacing.begin()); result->voxel_mm3=voxel_mm3;
    if (!(voxel_mm3>0.) || !std::isfinite(voxel_mm3)) throw py::value_error("Invalid native simulation voxel volume");
    for(double x:a) if(!std::isfinite(x)) throw py::value_error("Invalid native simulation affine");
    for(double x:inv) if(!std::isfinite(x)) throw py::value_error("Invalid native simulation inverse");
    for(double x:sp) if(!(x>0.) || !std::isfinite(x)) throw py::value_error("Invalid native simulation spacing");
    result->regions=metadata.at("regions"); result->properties=metadata.at("properties");
    result->source=metadata.at("source"); result->identity=metadata.at("identity").as_string();
    for(const auto* key:{"protectionContract","guardConfig","cranialAccess","cranialStages"}) {
        const auto* item=metadata.find(key); if(!item) continue;
        if(std::string_view(key)=="protectionContract") result->protection_contract=*item;
        else if(std::string_view(key)=="guardConfig") result->guard_config=*item;
        else if(std::string_view(key)=="cranialAccess") result->cranial_access=*item;
        else result->cranial_stages=*item;
    }
    if(const auto* cells=metadata.find("protectedCells")) for(const auto& cell:cells->as_array()) {
        result->label_at(cell.as_int()); result->protected_cells.push_back(cell.as_int());
    }
    if(const auto* cells=metadata.find("originalTargetCells")) for(const auto& cell:cells->as_array()) {
        result->label_at(cell.as_int()); result->original_target_cells.push_back(cell.as_int());
    }
    if(const auto* memberships=metadata.find("originalTargetMemberships")) result->original_target_memberships=*memberships;
    std::sort(result->protected_cells.begin(),result->protected_cells.end());
    result->protected_cells.erase(std::unique(result->protected_cells.begin(),result->protected_cells.end()),result->protected_cells.end());
    result->source_inspection_only=engine_detail::flag(metadata,"inspectionOnly");
    result->rebuild_region_index();
    if(!metadata.contains("originalTargetCells")) {
        for(std::size_t cell=0;cell<result->labels.size();++cell) if(result->target(cell)) result->original_target_cells.push_back(cell);
        if(result->original_target_memberships.is_object())
            for(const auto& row:result->original_target_memberships.as_object())
                for(const auto& cell:row.second.as_array()) result->original_target_cells.push_back(cell.as_int());
        auto& targets=result->original_target_cells;
        std::sort(targets.begin(),targets.end()); targets.erase(std::unique(targets.begin(),targets.end()),targets.end());
    }
    return result;
}
inline std::string unsupported(const State& engine) {
    if (!native_hash_supported()) return "native-number-formatting-unavailable";
    if (!geometry_scene_supported(*engine.scene)) return "affine-exactness-not-qualified";
    if (const auto reason=legacy_effects_unsupported(engine);!reason.empty()) return reason;
    for(const auto* property:{"planarBladeManeuverV1","retractorSurfaceContactV1"})
        if(engine.scene->properties.contains(property)) return std::string("unsupported-profile:")+property;
    if (!engine.scene->cranial_access.is_null() && engine_detail::is_text(engine.scene->cranial_access,"method","milling"))
        return "unsupported-profile:milling";
    if (!cusa_v3::progress_supported(engine)) return "unsupported-cusa-progress";
    return {};
}

inline void bind(py::module_& module) {
    py::register_exception<UnsupportedBoundary>(module,"UnsupportedSimulationBoundary",PyExc_TypeError);
    module.def("simulation_json_guard",[](py::object guard) { return resectionlab_json::guard_matches(guard.ptr()); });
    py::class_<Engine,std::shared_ptr<Engine>>(module,"ResidentEngine");
    py::class_<StopToken,std::shared_ptr<StopToken>>(module,"SimulationStopToken")
        .def(py::init<>()).def("cancel",&StopToken::cancel)
        .def("check",&StopToken::check)
        .def("set_timeout",[](StopToken& token,py::object timeout) {
            if(timeout.is_none()) { token.deadline(StopToken::Clock::time_point::max()); return; }
            const auto seconds=timeout.cast<double>();
            if (!std::isfinite(seconds) || seconds<0.) throw py::value_error("Native deadline requires finite nonnegative seconds");
            token.deadline(StopToken::Clock::now()+std::chrono::duration_cast<StopToken::Clock::duration>(std::chrono::duration<double>(seconds)));
        });
    module.def("simulation_create",[](const py::array& labels,const py::array& affine,const py::array& inverse,
                    const py::array& spacing,double voxel_mm3,py::object metadata,py::object data) -> py::tuple {
        const auto meta=from_python(metadata);
        auto source=scene(labels,affine,inverse,spacing,voxel_mm3,meta);
        auto owned=from_python(data); State value(source); value.state=owned.at("state");
        value.actions=owned.at("actions"); value.history=owned.at("history");
        if(const auto* initial=meta.find("initialStateId")) source->initial_state_id=initial->as_string();
        else if(value.actions.as_array().empty()) source->initial_state_id=state_id(value);
        const auto reason=unsupported(value);
        if (!reason.empty()) return py::make_tuple(py::none(),py::str(reason));
        StopToken stop;
        validate_scene_contracts(value,stop);
        return py::make_tuple(py::cast(std::make_shared<Engine>(std::move(value))),py::none());
    });
    module.def("simulation_identity",[](const Engine& engine) { py::gil_scoped_release release; return engine.identity(); });
    module.def("simulation_clone",[](const Engine& engine) { py::gil_scoped_release release; return std::make_shared<Engine>(engine.clone()); });
    module.def("simulation_data",[](const Engine& engine,std::string field) {
        Value result;
        { py::gil_scoped_release release;
          auto held=engine.lock();
          if(field=="state") result=engine.state().state;
          else if(field=="actions") result=engine.state().actions;
          else if(field=="history") result=engine.state().history;
          else if(field=="snapshot") result=engine.snapshot();
          else if(field=="metrics") { StopToken stop; result=engine.metrics(stop); }
          else throw std::invalid_argument("Unknown native simulation field"); }
        return to_python(result);
    });
    module.def("simulation_import",[](Engine& engine,py::object data) {
        auto incoming=from_python(data);
        std::string reason;
        { py::gil_scoped_release release;
          auto held=engine.lock();
          auto candidate=engine.state();
          if(auto* state=incoming.find("state")) candidate.state=std::move(*state);
          if(auto* actions=incoming.find("actions")) candidate.actions=std::move(*actions);
          if(auto* history=incoming.find("history")) candidate.history=std::move(*history);
          reason=unsupported(candidate);
          if(reason.empty() && (!exact(candidate.state,engine.state().state) || !exact(candidate.actions,engine.state().actions) || !exact(candidate.history,engine.state().history))) {
              if(!candidate.bone_prefix.is_null() && candidate.bone_prefix.at("position").as_int()>static_cast<std::int64_t>(candidate.actions.as_array().size())) candidate.bone_prefix={};
              candidate.stage_permission={};++candidate.version; engine.boundary_state()=std::move(candidate);
          } }
        return reason.empty() ? py::object(py::none()) : py::cast(reason);
    });
    module.def("simulation_step",[](Engine& engine,py::object action,const std::shared_ptr<StopToken>& token) {
        const auto owned=from_python(action); Value result;
        {py::gil_scoped_release release;result=engine.step(owned,*token);} return to_python(result);
    });
    module.def("simulation_prepare",[](const Engine& engine,py::object action,const std::shared_ptr<StopToken>& token) -> py::tuple {
        Value owned;
        try { owned=from_python(action); }
        catch (const UnsupportedBoundary&) { return py::make_tuple(py::none(),py::none()); }
        // Coercible/custom duration inputs retain the established Python boundary.
        if (owned.is_object()) if(const auto* duration=owned.find("duration"))
            if(!duration->is_number() && !duration->is_bool()) return py::make_tuple(py::none(),py::none());
        std::pair<std::unique_ptr<Engine>,Value> prepared;
        {py::gil_scoped_release release;prepared=engine.prepare(owned,*token);}
        return py::make_tuple(prepared.first ? py::cast(std::shared_ptr<Engine>(std::move(prepared.first))) : py::none(),to_python(prepared.second));
    });
    module.def("simulation_query",[](Engine& engine,std::string field) {
        Value result;
        { py::gil_scoped_release release; auto held=engine.lock(); StopToken stop; auto& state=engine.boundary_state();
          if(field=="cranialStages") result=cranial::stage_readiness(state,stop);
          else if(field=="craniotomy") result=bone_transport::describe(state,stop);
          else if(field=="membrane") result=membrane_transport::clearance(state,stop);
          else if(field=="bonePrefix") {
              if(!state.bone_prefix.is_null()) {cranial::verified_bone_prefix(state,true,stop);result=state.bone_prefix;}
          } else throw std::invalid_argument("Unknown native simulation query"); }
        return to_python(result);
    });
    module.def("simulation_json",[](py::object input) { const auto owned=from_python(input);py::gil_scoped_release release;return canonical_json(owned); });
    module.def("simulation_stats",[](bool reset) {
        Value result=Value::object();
        const std::pair<const char*,std::atomic<std::uint64_t>*> counters[]={
            {"actions",&execution_stats.actions},{"trajectories",&execution_stats.trajectories},
            {"physicalSubsteps",&execution_stats.physical_substeps},{"clones",&execution_stats.clones},
            {"rollbacks",&execution_stats.rollbacks},{"stateHashes",&execution_stats.state_hashes},
            {"geometryBatches",&execution_stats.geometry_batches},{"geometryBatchItems",&execution_stats.geometry_batch_items},
            {"geometryBatchMaxItems",&execution_stats.geometry_batch_max_items}};
        for(const auto& counter:counters) result.set(counter.first,
            reset ? counter.second->exchange(0) : counter.second->load());
        return to_python(result);
    },py::arg("reset")=false);
}

}  // namespace resectionlab::sim::boundary
