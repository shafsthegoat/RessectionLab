#pragma once

#include <cmath>
#include <utility>
#include <vector>
#include <pybind11/pybind11.h>

namespace resectionlab_state {

constexpr int max_engine_copy_items = 262144;
constexpr int max_engine_copy_depth = 64;

inline bool engine_atom(PyObject* value) {
    return value == Py_None || PyBool_Check(value) || PyLong_CheckExact(value) ||
        PyUnicode_CheckExact(value) || PyFloat_CheckExact(value);
}

// Work only in a private memo until the complete supported graph succeeds.
inline pybind11::object copy_engine_graph(
    pybind11::object value, pybind11::dict& memo,
    std::vector<pybind11::object>& originals, int& remaining, int depth) {
    if (remaining-- <= 0) {
        return {};
    }
    if (engine_atom(value.ptr())) {
        return value;
    }
    auto key = pybind11::reinterpret_steal<pybind11::object>(PyLong_FromVoidPtr(value.ptr()));
    if (!key) {
        throw pybind11::error_already_set();
    }
    PyObject* existing = PyDict_GetItemWithError(memo.ptr(), key.ptr());
    if (existing != nullptr) {
        return pybind11::reinterpret_borrow<pybind11::object>(existing);
    }
    if (PyErr_Occurred()) {
        throw pybind11::error_already_set();
    }
    if (depth >= max_engine_copy_depth) {
        return {};
    }
    const bool list = PyList_CheckExact(value.ptr());
    const bool dict = PyDict_CheckExact(value.ptr());
    if (!list && !dict) {
        return {};
    }
    const Py_ssize_t size = list ? PyList_GET_SIZE(value.ptr()) : PyDict_Size(value.ptr());
    if (size > remaining / (dict ? 2 : 1)) {
        return {};
    }
    PyObject* captured = list ? PyList_AsTuple(value.ptr()) : PyDict_Items(value.ptr());
    if (captured == nullptr) {
        throw pybind11::error_already_set();
    }
    auto snapshot = pybind11::reinterpret_steal<pybind11::object>(captured);
    const Py_ssize_t captured_size = list ? PyTuple_GET_SIZE(snapshot.ptr()) : PyList_GET_SIZE(snapshot.ptr());
    if (captured_size > remaining / (dict ? 2 : 1)) {
        return {};
    }
    // Retain every original through publication, including when a finalizer clears its parent.
    originals.push_back(value);
    PyObject* allocated = list ? PyList_New(0) : PyDict_New();
    if (allocated == nullptr) {
        throw pybind11::error_already_set();
    }
    auto result = pybind11::reinterpret_steal<pybind11::object>(allocated);
    if (PyDict_SetItem(memo.ptr(), key.ptr(), result.ptr()) < 0) {
        throw pybind11::error_already_set();
    }
    for (Py_ssize_t index = 0; index < captured_size; ++index) {
        PyObject* item = list ? PyTuple_GET_ITEM(snapshot.ptr(), index) :
            PyTuple_GET_ITEM(PyList_GET_ITEM(snapshot.ptr(), index), 1);
        PyObject* dict_key = list ? nullptr : PyTuple_GET_ITEM(PyList_GET_ITEM(snapshot.ptr(), index), 0);
        if (dict && (!engine_atom(dict_key) || remaining-- <= 0)) {
            return {};
        }
        auto copied = copy_engine_graph(pybind11::reinterpret_borrow<pybind11::object>(item),
            memo, originals, remaining, depth + 1);
        if (!copied) {
            return {};
        }
        if (list) {
            if (PyList_Append(result.ptr(), copied.ptr()) < 0) {
                throw pybind11::error_already_set();
            }
        } else if (PyDict_SetItem(result.ptr(), dict_key, copied.ptr()) < 0) {
            throw pybind11::error_already_set();
        }
    }
    return result;
}

inline pybind11::object copy_engine_graph(
    pybind11::object value, pybind11::object memo, pybind11::object fallback) {
    if (!PyDict_CheckExact(memo.ptr()) || PyDict_Size(memo.ptr()) > max_engine_copy_items) {
        return fallback;
    }
    Py_ssize_t position = 0;
    PyObject* key;
    PyObject* item;
    while (PyDict_Next(memo.ptr(), &position, &key, &item)) {
        if (!PyLong_CheckExact(key)) {
            return fallback;
        }
    }
    PyObject* allocated = PyDict_Copy(memo.ptr());
    if (allocated == nullptr) {
        throw pybind11::error_already_set();
    }
    auto working = pybind11::reinterpret_steal<pybind11::dict>(allocated);
    std::vector<pybind11::object> originals;
    int remaining = max_engine_copy_items;
    auto result = copy_engine_graph(std::move(value), working, originals, remaining, 0);
    if (!result) {
        return fallback;
    }
    if (PyDict_Update(memo.ptr(), working.ptr()) < 0) {
        throw pybind11::error_already_set();
    }
    return result;
}

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
        PyObject* allocated = PyList_New(0);
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
            if (PyList_Append(result.ptr(), copied.ptr()) < 0) {
                throw pybind11::error_already_set();
            }
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
    module.attr("MAX_ENGINE_COPY_ITEMS") = resectionlab_state::max_engine_copy_items;
    module.attr("MAX_ENGINE_COPY_DEPTH") = resectionlab_state::max_engine_copy_depth;
    module.def("copy_engine_graph", static_cast<pybind11::object (*)(
        pybind11::object, pybind11::object, pybind11::object)>(resectionlab_state::copy_engine_graph),
        pybind11::arg("value"), pybind11::arg("memo"), pybind11::arg("fallback"));
    module.def(
        "copy_builtin_json",
        [](pybind11::object value, pybind11::object fallback, int depth) {
            auto copied = resectionlab_state::copy_builtin_json(std::move(value), depth);
            return copied ? std::move(copied) : std::move(fallback);
        },
        pybind11::arg("value"), pybind11::arg("fallback"), pybind11::arg("depth") = 0);
}
