#pragma once

#include <cmath>
#include <utility>
#include <pybind11/pybind11.h>

namespace resectionlab_state {

// A null object means unsupported input, with no Python exception pending.
inline pybind11::object copy_builtin_json(pybind11::object value, int depth) {
    if (depth < 0 || depth >= 32) {
        return {};
    }
    if (value.ptr() == Py_None || PyBool_Check(value.ptr()) || PyLong_CheckExact(value.ptr()) ||
        PyUnicode_CheckExact(value.ptr())) {
        return value;
    }
    if (PyFloat_CheckExact(value.ptr())) {
        if (!std::isfinite(PyFloat_AS_DOUBLE(value.ptr()))) {
            return {};
        }
        return value;
    }
    if (PyList_CheckExact(value.ptr())) {
        // Allocations can run unrelated finalizers that mutate caller containers.
        PyObject* captured = PyList_AsTuple(value.ptr());
        if (captured == nullptr) {
            throw pybind11::error_already_set();
        }
        auto snapshot = pybind11::reinterpret_steal<pybind11::object>(captured);
        const Py_ssize_t size = PyTuple_GET_SIZE(snapshot.ptr());
        PyObject* allocated = PyList_New(size);
        if (allocated == nullptr) {
            throw pybind11::error_already_set();
        }
        auto result = pybind11::reinterpret_steal<pybind11::object>(allocated);
        for (Py_ssize_t index = 0; index < size; ++index) {
            auto copied = copy_builtin_json(pybind11::reinterpret_borrow<pybind11::object>(
                PyTuple_GET_ITEM(snapshot.ptr(), index)), depth + 1);
            if (!copied) {
                return {};
            }
            PyList_SET_ITEM(result.ptr(), index, copied.release().ptr());
        }
        return result;
    }
    if (PyDict_CheckExact(value.ptr())) {
        PyObject* captured = PyDict_Items(value.ptr());
        if (captured == nullptr) {
            throw pybind11::error_already_set();
        }
        auto snapshot = pybind11::reinterpret_steal<pybind11::object>(captured);
        PyObject* allocated = PyDict_New();
        if (allocated == nullptr) {
            throw pybind11::error_already_set();
        }
        auto result = pybind11::reinterpret_steal<pybind11::object>(allocated);
        for (Py_ssize_t index = 0; index < PyList_GET_SIZE(snapshot.ptr()); ++index) {
            PyObject* pair = PyList_GET_ITEM(snapshot.ptr(), index);
            PyObject* key = PyTuple_GET_ITEM(pair, 0);
            if (!PyUnicode_CheckExact(key)) {
                return {};
            }
            auto copied = copy_builtin_json(pybind11::reinterpret_borrow<pybind11::object>(
                PyTuple_GET_ITEM(pair, 1)), depth + 1);
            if (!copied) {
                return {};
            }
            if (PyDict_SetItem(result.ptr(), key, copied.ptr()) < 0) {
                throw pybind11::error_already_set();
            }
        }
        return result;
    }
    return {};
}

}  // namespace resectionlab_state

inline void bind_state_native(pybind11::module_& module) {
    module.def(
        "copy_builtin_json",
        [](pybind11::object value, pybind11::object fallback, int depth) {
            auto copied = resectionlab_state::copy_builtin_json(std::move(value), depth);
            return copied ? std::move(copied) : std::move(fallback);
        },
        pybind11::arg("value"), pybind11::arg("fallback"), pybind11::arg("depth") = 0);
}
