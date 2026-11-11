// Complete stock cranial checks; custom containers retain the literal Python path.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace resectionlab_validation {
namespace py = pybind11;

inline bool string_dict(PyObject* value) {
    if (!value || !PyDict_CheckExact(value)) return false;
    Py_ssize_t position = 0;
    PyObject *key, *item;
    while (PyDict_Next(value, &position, &key, &item))
        if (!PyUnicode_CheckExact(key)) return false;
    return true;
}
inline PyObject* get(PyObject* value, const char* key) {
    return PyDict_GetItemString(value, key);  // All keys were admitted as exact strings.
}
inline bool sequence(PyObject* value) {
    return value && (PyList_CheckExact(value) || PyTuple_CheckExact(value));
}
inline Py_ssize_t size(PyObject* value) {
    return PyList_CheckExact(value) ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value);
}
inline PyObject* at(PyObject* value, Py_ssize_t index) {
    return PyList_CheckExact(value) ? PyList_GET_ITEM(value, index) : PyTuple_GET_ITEM(value, index);
}
inline bool integers(PyObject* value) {
    if (!sequence(value)) return false;
    for (Py_ssize_t i = 0; i < size(value); ++i)
        if (!PyLong_CheckExact(at(value, i))) return false;
    return true;
}
inline bool is_string(PyObject* value, const char* expected) {
    return PyUnicode_CompareWithASCIIString(value, expected) == 0;
}
inline bool current_bindings(py::handle space, py::handle builtins, py::handle bindings) {
    if (!string_dict(space.ptr()) || !string_dict(builtins.ptr()) || !PyTuple_CheckExact(bindings.ptr()))
        return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(bindings.ptr()); ++i) {
        PyObject* pair = PyTuple_GET_ITEM(bindings.ptr(), i);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2) return false;
        PyObject* name = PyTuple_GET_ITEM(pair, 0);
        if (!PyUnicode_CheckExact(name)) return false;
        PyObject* original = PyTuple_GET_ITEM(pair, 1);
        PyObject* local = PyDict_GetItemWithError(space.ptr(), name);
        PyObject* builtin = PyDict_GetItemWithError(builtins.ptr(), name);
        if (PyErr_Occurred()) throw py::error_already_set();
        if ((local && local != original) || builtin != original) return false;
    }
    return true;
}

// Generic attribute reads are safe only for ordinary registered classes without descriptors.
inline py::object ordinary_dict(py::handle value, PyObject* expected,
                                std::initializer_list<const char*> fields) {
    if (!PyType_Check(expected) || Py_TYPE(value.ptr()) != reinterpret_cast<PyTypeObject*>(expected) ||
            Py_TYPE(value.ptr())->tp_getattro != PyObject_GenericGetAttr) return {};
    PyObject* mro = Py_TYPE(value.ptr())->tp_mro;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); ++i) {
        auto* type = reinterpret_cast<PyTypeObject*>(PyTuple_GET_ITEM(mro, i));
        auto attributes = py::reinterpret_steal<py::object>(PyType_GetDict(type));
        if (!attributes) throw py::error_already_set();
        for (const char* field : fields)
            if (PyDict_GetItemString(attributes.ptr(), field)) return {};
    }
    auto result = py::reinterpret_steal<py::object>(PyObject_GenericGetDict(value.ptr(), nullptr));
    if (!result) throw py::error_already_set();
    return string_dict(result.ptr()) ? std::move(result) : py::object();
}

inline void contract_error(const char* reason) {
    throw py::value_error(std::string("CRANIOTOMY_CONTRACT_INVALID: ") + reason);
}
inline py::object cranial_cells_partition(py::handle raw, py::handle scene, py::tuple types,
        py::handle space, py::handle builtins, py::handle bindings, py::object fallback) {
    if (types.size() != 2 || !current_bindings(space, builtins, bindings) || !string_dict(raw.ptr())) return fallback;
    auto source = ordinary_dict(scene, types[0].ptr(), {"labels", "regions", "properties"});
    if (!source) return fallback;
    PyObject* labels = get(source.ptr(), "labels");
    if (!labels || reinterpret_cast<PyObject*>(Py_TYPE(labels)) != types[1].ptr()) return fallback;
    auto label_owner = py::reinterpret_borrow<py::object>(labels);
    if (!py::array::check_(label_owner)) return fallback;
    auto array = py::reinterpret_borrow<py::array>(label_owner);
    // shortcut: cap one no-callback source scan; larger grids retain compiled NumPy validation.
    if (!array.dtype().is(py::dtype::of<std::uint16_t>()) || !(array.flags() & py::array::c_style) ||
            array.writeable() || array.size() > 1048576) return fallback;
    const Py_ssize_t count = array.size();
    PyObject* regions = get(source.ptr(), "regions"), *properties = get(source.ptr(), "properties");
    if (!sequence(regions) || !string_dict(properties)) return fallback;
    constexpr std::array<const char*, 5> names = {"windowCells", "flapCells", "kerfCells", "retainedSkullCells", "boneAnchorCells"};
    std::array<PyObject*, 5> cells;
    for (int i = 0; i < 5; ++i) {
        cells[i] = get(raw.ptr(), names[i]);
        if (!integers(cells[i])) return fallback;
    }
    PyObject* anchors = get(properties, "boneAnchors");
    if (anchors && !integers(anchors)) return fallback;
    for (Py_ssize_t i = 0; i < size(regions); ++i) {
        PyObject* row = at(regions, i);
        if (!string_dict(row) || !get(row, "material") || !PyUnicode_CheckExact(get(row, "material")) ||
                !get(row, "id") || !PyLong_CheckExact(get(row, "id"))) return fallback;
    }
    std::vector<unsigned char> membership(static_cast<std::size_t>(count), 0);
    for (int i = 0; i < 5; ++i) {
        bool invalid = i != 4 && size(cells[i]) == 0;
        Py_ssize_t previous = -1;
        for (Py_ssize_t j = 0; j < size(cells[i]) && !invalid; ++j) {
            int overflow = 0;
            const auto cell = PyLong_AsLongLongAndOverflow(at(cells[i], j), &overflow);
            if (overflow) invalid = true;
            else if (cell < 0 || cell >= count || cell <= previous) invalid = true;
            else { membership[cell] |= static_cast<unsigned char>(1 << i); previous = cell; }
        }
        if (invalid) contract_error((std::string(names[i]) + " requires sorted distinct source integer cells").c_str());
    }
    std::array<bool, 65536> bone_labels{};
    for (Py_ssize_t i = 0; i < size(regions); ++i) {
        PyObject* row = at(regions, i);
        if (!is_string(get(row, "material"), "bone")) continue;
        int overflow = 0;
        const auto id = PyLong_AsLongLongAndOverflow(get(row, "id"), &overflow);
        if (overflow || id < 0 || id > 65535) return fallback;
        bone_labels[static_cast<std::size_t>(id)] = true;
    }
    const auto* data = static_cast<const unsigned char*>(array.data());
    bool partition = true, window = true;
    for (Py_ssize_t i = 0; i < count; ++i) {
        std::uint16_t label;
        std::memcpy(&label, data + sizeof(label) * i, sizeof(label));
        const unsigned char flags = membership[i];
        const bool bone = bone_labels[label], fragment = flags & 6, retained = flags & 8;
        partition &= (flags & 6) != 6 && !(retained && fragment) && (fragment || retained) == bone;
        window &= (!fragment || (flags & 1)) && (!(flags & 1) || bone) && (!(flags & 16) || retained);
    }
    if (!partition) contract_error("F/K/R must partition all categorical source bone");
    if (!window) contract_error("window must contain F union K; anchors must belong to retained skull");
    if (anchors) for (Py_ssize_t i = 0; i < size(anchors); ++i) {
        int overflow = 0;
        const auto cell = PyLong_AsLongLongAndOverflow(at(anchors, i), &overflow);
        if (overflow || cell < 0 || cell >= count || !(membership[cell] & 16))
            contract_error("declared source bone anchors cannot be omitted or reassigned");
    }
    // Snapshot every group before tracked output allocations can run finalizers.
    std::array<py::object, 5> original_tuples;
    std::array<std::vector<py::object>, 5> snapshots;
    for (int i = 0; i < 5; ++i) {
        if (PyTuple_CheckExact(cells[i])) original_tuples[i] = py::reinterpret_borrow<py::object>(cells[i]);
        else for (Py_ssize_t j = 0; j < size(cells[i]); ++j)
            snapshots[i].push_back(py::reinterpret_borrow<py::object>(at(cells[i], j)));
    }
    py::list result;
    for (int i = 0; i < 5; ++i) {
        if (original_tuples[i]) result.append(original_tuples[i]);
        else {
            py::tuple values(snapshots[i].size());
            for (std::size_t j = 0; j < snapshots[i].size(); ++j) values[j] = snapshots[i][j];
            result.append(values);
        }
    }
    return std::move(result);
}
}  // namespace resectionlab_validation

inline void bind_validation_native(pybind11::module_& module) {
    namespace py = pybind11;
    module.def("cranial_cells_partition", &resectionlab_validation::cranial_cells_partition,
        py::arg("raw"), py::arg("scene"), py::arg("types"), py::arg("namespace"),
        py::arg("builtins"), py::arg("bindings"), py::arg("fallback"));
}
