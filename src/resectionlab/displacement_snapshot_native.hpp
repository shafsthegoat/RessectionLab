#pragma once

#include <cmath>
#include <cstring>
#include <limits>
#include <pybind11/pybind11.h>

namespace resectionlab::displacement_snapshot {
namespace py = pybind11;

inline py::object owned(PyObject* value) {
    if (!value) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(value);
}

inline py::object exact_snapshot(py::handle field) {
    constexpr Py_ssize_t row_bytes = 3 * sizeof(double);
    constexpr Py_ssize_t max_rows = 65536;
    if (!PyDict_CheckExact(field.ptr()) || PyDict_GET_SIZE(field.ptr()) > max_rows ||
            sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559)
        return py::none();

    // Retain the GIL while borrowing from publicly mutable dictionaries and
    // lists. Sort exact Unicode using Python's own ordering, with no coercions.
    const auto keys = owned(PyDict_Keys(field.ptr()));
    const auto count = PyList_GET_SIZE(keys.ptr());
    if (count > max_rows) return py::none();
    for (Py_ssize_t index = 0; index < count; ++index) {
        if (!PyUnicode_CheckExact(PyList_GET_ITEM(keys.ptr(), index))) return py::none();
    }
    if (PyList_Sort(keys.ptr()) < 0) throw py::error_already_set();

    const auto bytes = owned(PyBytes_FromStringAndSize(nullptr, count * row_bytes));
    char* output = PyBytes_AS_STRING(bytes.ptr());
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject* key = PyList_GET_ITEM(keys.ptr(), index);
        PyObject* row = PyDict_GetItemWithError(field.ptr(), key);
        if (!row && PyErr_Occurred()) throw py::error_already_set();
        if (!row || !PyList_CheckExact(row) || PyList_GET_SIZE(row) != 3) return py::none();
        for (Py_ssize_t axis = 0; axis < 3; ++axis) {
            PyObject* item = PyList_GET_ITEM(row, axis);
            if (!PyFloat_CheckExact(item)) return py::none();
            const double number = PyFloat_AS_DOUBLE(item);
            if (!std::isfinite(number)) return py::none();
            std::memcpy(output, &number, sizeof(number));
            output += sizeof(number);
        }
    }
    return py::make_tuple(owned(PyList_AsTuple(keys.ptr())), bytes);
}
}  // namespace resectionlab::displacement_snapshot

inline void bind_displacement_snapshot(pybind11::module_& module) {
    module.def("try_displacement_snapshot", &resectionlab::displacement_snapshot::exact_snapshot);
}
