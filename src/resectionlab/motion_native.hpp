#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace resectionlab_motion {
namespace py = pybind11;

inline bool ordinary_dict(py::handle value) {
    if (!PyDict_CheckExact(value.ptr())) return false;
    Py_ssize_t offset = 0;
    PyObject *key, *item;
    while (PyDict_Next(value.ptr(), &offset, &key, &item))
        if (!PyUnicode_CheckExact(key)) return false;
    return true;
}

inline bool number(py::handle value, double& result) {
    if (PyFloat_CheckExact(value.ptr())) result = PyFloat_AS_DOUBLE(value.ptr());
    else if (PyLong_CheckExact(value.ptr())) {
        result = PyLong_AsDouble(value.ptr());
        if (PyErr_Occurred()) { PyErr_Clear(); return false; }
    } else return false;
    // Keep NumPy's exceptional overflow/underflow paths in the reference helper.
    return std::isfinite(result) && std::abs(result) <= 1e100;
}

inline bool optional_number(py::handle mapping, const char* name, double initial, double& result) {
    auto* value = PyDict_GetItemString(mapping.ptr(), name);
    if (!value) { result = initial; return true; }
    return number(py::handle(value), result);
}

template <std::size_t N>
inline bool vector_values(py::handle position, std::array<double, N>& result) {
    if (PyList_CheckExact(position.ptr()) || PyTuple_CheckExact(position.ptr())) {
        if (PySequence_Size(position.ptr()) != N) return false;
        for (Py_ssize_t axis = 0; axis != N; ++axis) {
            auto* component = PyList_CheckExact(position.ptr())
                ? PyList_GET_ITEM(position.ptr(), axis) : PyTuple_GET_ITEM(position.ptr(), axis);
            if (!number(py::handle(component), result[axis])) return false;
        }
        return true;
    }
    if (!py::isinstance<py::array>(position)) return false;
    auto values = py::reinterpret_borrow<py::array>(position);
    if (Py_TYPE(position.ptr()) != py::detail::npy_api::get().PyArray_Type_
            || values.ndim() != 1 || values.shape(0) != N
            || !values.dtype().is(py::dtype::of<double>())) return false;
    for (Py_ssize_t axis = 0; axis != N; ++axis) {
        std::memcpy(&result[axis], static_cast<const char*>(values.data()) + axis * values.strides(0), sizeof(double));
        if (!std::isfinite(result[axis]) || std::abs(result[axis]) > 1e100) return false;
    }
    return true;
}

inline py::object envelopes(py::handle tool, py::handle parameters, py::handle position,
                           py::handle quaternion, py::handle inflation, py::handle rotation_type,
                           py::object fallback) {
    if (!ordinary_dict(tool) || !ordinary_dict(parameters)) return fallback;
    auto* geometry = PyDict_GetItemString(tool.ptr(), "geometry");
    auto* raw_radius = PyDict_GetItemString(parameters.ptr(), "radiusMm");
    auto* raw_shaft = PyDict_GetItemString(tool.ptr(), "shaftRadiusMm");
    if (!geometry || !PyUnicode_CheckExact(geometry) || !raw_radius || !raw_shaft) return fallback;
    double radius, shaft, depth, jaw, contact_revision, face_revision;
    if (!number(py::handle(raw_radius), radius) || !number(py::handle(raw_shaft), shaft)
            || !optional_number(parameters, "depthMm", 2., depth)
            || !optional_number(parameters, "jawOpeningMm", 2., jaw)
            || !optional_number(parameters, "contactGeometryRevision", 0., contact_revision)
            || !optional_number(parameters, "retractorFaceContactRevision", 0., face_revision)) return fallback;
    std::array<double, 3> origin;
    std::array<double, 4> angular;
    if (!vector_values(position, origin) || !vector_values(quaternion, angular)) return fallback;
    const auto numpy_scalar = py::dtype::of<double>().attr("type");
    const bool numpy_inflation = Py_TYPE(inflation.ptr()) == reinterpret_cast<PyTypeObject*>(numpy_scalar.ptr());
    double extra;
    if (numpy_inflation) extra = PyFloat_AsDouble(inflation.ptr());
    else if (!number(inflation, extra)) return fallback;
    if (!std::isfinite(extra) || std::abs(extra) > 1e100) return fallback;
    const auto geometry_owner = py::reinterpret_borrow<py::object>(geometry);
    const auto shape = [&](const char* name) { return PyUnicode_CompareWithASCIIString(geometry_owner.ptr(), name) == 0; };
    // shortcut: preserve NumPy's exact linspace/hypot for the finite retractor pad.
    if (shape("blade") && face_revision == 1.) return fallback;
    py::array_t<double> owned_quaternion(4);
    std::memcpy(owned_quaternion.mutable_data(), angular.data(), sizeof(angular));
    const auto matrix = rotation_type.attr("from_quat")(owned_quaternion).attr("as_matrix")().cast<py::array>();
    if (matrix.ndim() != 2 || matrix.shape(0) != 3 || matrix.shape(1) != 3
            || !matrix.dtype().is(py::dtype::of<double>()) || !(matrix.flags() & py::array::c_style))
        throw py::value_error("stock rotation returned invalid matrix");
    std::array<double, 9> rotation;
    std::memcpy(rotation.data(), matrix.data(), sizeof(rotation));
    py::list result;
    const auto point = [&](double x, double z) {
        py::array_t<double> output(3);
        auto* data = output.mutable_data();
        for (int axis = 0; axis != 3; ++axis) {
            const auto offset = axis * 3;
            const double rotated = ((0. + rotation[offset] * x) + rotation[offset + 1] * 0.)
                + rotation[offset + 2] * z;
            data[axis] = origin[axis] + rotated;
        }
        return output;
    };
    const auto capsule = [&](double ax, double az, double bx, double bz, double width, const char* part) {
        auto a = point(ax, az), b = point(bx, bz);
        py::object output_width = py::float_(width + extra);
        if (numpy_inflation) output_width = numpy_scalar(output_width);
        result.append(py::make_tuple(a, b, output_width, part));
    };
    if (shape("aspirator") && contact_revision == 2.) {
        capsule(0., -40., 0., 0., shaft, "shaft");
        capsule(0., 0., 0., 0., radius, "working");
        return std::move(result);
    }
    capsule(0., -40., 0., -std::max(1., std::min(depth, 5.)), shaft, "shaft");
    if (shape("scissors") || shape("forceps")) {
        const double half = jaw / 2.;
        capsule(-half, -depth, -half, 0., .35, "working");
        capsule(half, -depth, half, 0., .35, "working");
    } else if (shape("blade") || shape("elevator") || shape("kerf")) {
        capsule(-radius, -0., radius, -0., .4, "working");
        capsule(-radius, -std::min(depth, 5.), radius, -std::min(depth, 5.), .4, "working");
    } else capsule(0., -std::min(depth, 3.), 0., 0., radius, "working");
    return std::move(result);
}
}  // namespace resectionlab_motion

inline void bind_motion_native(pybind11::module_& module) {
    module.def("motion_envelopes", &resectionlab_motion::envelopes,
        pybind11::arg("tool"), pybind11::arg("parameters"), pybind11::arg("position"),
        pybind11::arg("quaternion"), pybind11::arg("inflation"), pybind11::arg("rotation_type"),
        pybind11::arg("fallback"));
}
