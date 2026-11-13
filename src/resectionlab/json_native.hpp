#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <pybind11/pybind11.h>

namespace resectionlab_json {

// Larger/unusual metadata keeps the reference path; native work stays bounded.
constexpr std::size_t max_json_bytes = 4 * 1024 * 1024;
constexpr std::size_t max_json_items = 65536;

inline bool unicode_keys(PyObject* mapping) {
    if (!PyDict_CheckExact(mapping) || PyDict_GET_SIZE(mapping) > 65536) return false;
    Py_ssize_t position = 0;
    PyObject* key;
    PyObject* item;
    while (PyDict_Next(mapping, &position, &key, &item)) {
        if (!PyUnicode_CheckExact(key)) return false;
    }
    return true;
}

inline bool stock_encoder_type(PyObject* owner) {
    if (!PyType_CheckExact(owner)) return false;
    auto* type = reinterpret_cast<PyTypeObject*>(owner);
    if (PyType_GetSlot(type, Py_tp_base) != &PyBaseObject_Type ||
        PyType_GetSlot(type, Py_tp_getattro) != reinterpret_cast<void*>(PyObject_GenericGetAttr) ||
        PyType_GetSlot(type, Py_tp_setattro) != reinterpret_cast<void*>(PyObject_GenericSetAttr)) return false;
    if (PyType_GetSlot(type, Py_tp_new) == PyType_GetSlot(&PyBaseObject_Type, Py_tp_new)) return true;
    // The stdlib C encoder has its own constructor and native call descriptor.
    const auto attributes = pybind11::reinterpret_steal<pybind11::object>(PyType_GetDict(type));
    if (!attributes) throw pybind11::error_already_set();
    if (!unicode_keys(attributes.ptr())) return false;
    PyObject* call = PyDict_GetItemString(attributes.ptr(), "__call__");
    return call != nullptr && Py_TYPE(call) == &PyWrapperDescr_Type;
}

inline bool guard_matches(PyObject* guard) {
    if (guard == Py_None) return true;
    if (!PyTuple_CheckExact(guard) || PyTuple_GET_SIZE(guard) != 2) return false;
    PyObject* bindings = PyTuple_GET_ITEM(guard, 0);
    PyObject* functions = PyTuple_GET_ITEM(guard, 1);
    if (!PyTuple_CheckExact(bindings) || !PyTuple_CheckExact(functions) ||
            PyTuple_GET_SIZE(bindings) > 256 || PyTuple_GET_SIZE(functions) > 64) return false;
    std::vector<std::pair<PyObject*, pybind11::object>> namespaces;
    for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(bindings); ++index) {
        PyObject* row = PyTuple_GET_ITEM(bindings, index);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 4) return false;
        PyObject* owner = PyTuple_GET_ITEM(row, 0);
        PyObject* key = PyTuple_GET_ITEM(row, 1);
        PyObject* expected = PyTuple_GET_ITEM(row, 2);
        PyObject* present = PyTuple_GET_ITEM(row, 3);
        if (!PyUnicode_CheckExact(key) || (present != Py_True && present != Py_False)) return false;
        PyObject* mapping = nullptr;
        for (const auto& item : namespaces) {
            if (item.first == owner) { mapping = item.second.ptr(); break; }
        }
        if (mapping == nullptr) {
            pybind11::object retained;
            if (PyDict_CheckExact(owner)) {
                retained = pybind11::reinterpret_borrow<pybind11::object>(owner);
            } else if (stock_encoder_type(owner)) {
                // Public 3.12 API returns a new reference to the read-only namespace.
                retained = pybind11::reinterpret_steal<pybind11::object>(
                    PyType_GetDict(reinterpret_cast<PyTypeObject*>(owner)));
                if (!retained) throw pybind11::error_already_set();
            } else return false;
            mapping = retained.ptr();
            if (!unicode_keys(mapping)) return false;
            namespaces.emplace_back(owner, std::move(retained));
        }
        PyObject* actual = PyDict_GetItemWithError(mapping, key);
        if (PyErr_Occurred()) throw pybind11::error_already_set();
        if ((present == Py_True && actual != expected) || (present == Py_False && actual != nullptr))
            return false;
    }
    for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(functions); ++index) {
        PyObject* row = PyTuple_GET_ITEM(functions, index);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 5) return false;
        PyObject* function = PyTuple_GET_ITEM(row, 0);
        if (!PyFunction_Check(function)) return false;
        PyObject* defaults = PyFunction_GetDefaults(function);
        PyObject* kwdefaults = PyFunction_GetKwDefaults(function);
        PyObject* kwitems = PyTuple_GET_ITEM(row, 4);
        if (PyFunction_GetCode(function) != PyTuple_GET_ITEM(row, 1) ||
                (defaults == nullptr ? Py_None : defaults) != PyTuple_GET_ITEM(row, 2) ||
                (kwdefaults == nullptr ? Py_None : kwdefaults) != PyTuple_GET_ITEM(row, 3) ||
                !PyTuple_CheckExact(kwitems) || PyTuple_GET_SIZE(kwitems) > 256) return false;
        if (kwdefaults == nullptr) {
            if (PyTuple_GET_SIZE(kwitems) != 0) return false;
            continue;
        }
        if (!unicode_keys(kwdefaults) || PyDict_GET_SIZE(kwdefaults) != PyTuple_GET_SIZE(kwitems))
            return false;
        for (Py_ssize_t item = 0; item < PyTuple_GET_SIZE(kwitems); ++item) {
            PyObject* pair = PyTuple_GET_ITEM(kwitems, item);
            if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2 ||
                    !PyUnicode_CheckExact(PyTuple_GET_ITEM(pair, 0))) return false;
            PyObject* actual = PyDict_GetItemWithError(kwdefaults, PyTuple_GET_ITEM(pair, 0));
            if (PyErr_Occurred()) throw pybind11::error_already_set();
            if (actual != PyTuple_GET_ITEM(pair, 1)) return false;
        }
    }
    return true;
}

class CanonicalWriter {
public:
    CanonicalWriter(bool allow_tuple, int max_depth)
        : allow_tuple_(allow_tuple), max_depth_(max_depth) {
        output_.reserve(512);
    }

    bool encode(PyObject* value, int depth = 0) {
        if (++items_ > max_json_items || depth >= max_depth_) return false;
        if (value == Py_None) return append("null", 4);
        if (value == Py_True) return append("true", 4);
        if (value == Py_False) return append("false", 5);
        if (PyUnicode_CheckExact(value)) return string(value);
        if (PyLong_CheckExact(value)) return integer(value);
        if (PyFloat_CheckExact(value)) {
            const double number = PyFloat_AS_DOUBLE(value);
            if (!std::isfinite(number)) return false;
            std::unique_ptr<char, decltype(&PyMem_Free)> text(
                PyOS_double_to_string(number, 'r', 0, Py_DTSF_ADD_DOT_0, nullptr), &PyMem_Free);
            if (!text) throw pybind11::error_already_set();
            return append(text.get(), std::strlen(text.get()));
        }
        if (PyDict_CheckExact(value)) return dictionary(value, depth);
        const bool list = PyList_CheckExact(value);
        if (!list && !(allow_tuple_ && PyTuple_CheckExact(value))) return false;
        const Py_ssize_t count = list ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value);
        if (static_cast<std::size_t>(count) > max_json_items - items_ || !append("[", 1))
            return false;
        for (Py_ssize_t index = 0; index < count; ++index) {
            if ((index != 0 && !append(",", 1)) ||
                    !encode(list ? PyList_GET_ITEM(value, index) : PyTuple_GET_ITEM(value, index),
                            depth + 1)) return false;
        }
        return append("]", 1);
    }

    const std::string& output() const { return output_; }

private:
    bool append(const char* text, std::size_t size) {
        if (size > max_json_bytes - output_.size()) return false;
        output_.append(text, size);
        return true;
    }

    bool escaped_codepoint(Py_UCS4 codepoint) {
        constexpr char hex[] = "0123456789abcdef";
        const char escaped[] = {'\\', 'u', hex[(codepoint >> 12) & 15],
            hex[(codepoint >> 8) & 15], hex[(codepoint >> 4) & 15], hex[codepoint & 15]};
        return append(escaped, sizeof(escaped));
    }

    bool string(PyObject* value) {
        const Py_ssize_t count = PyUnicode_GET_LENGTH(value);
        if (static_cast<std::size_t>(count) > max_json_bytes - output_.size() || !append("\"", 1))
            return false;
        const int kind = PyUnicode_KIND(value);
        const void* data = PyUnicode_DATA(value);
        for (Py_ssize_t index = 0; index < count; ++index) {
            Py_UCS4 codepoint = PyUnicode_READ(kind, data, index);
            switch (codepoint) {
                case '"': if (!append("\\\"", 2)) return false; break;
                case '\\': if (!append("\\\\", 2)) return false; break;
                case '\b': if (!append("\\b", 2)) return false; break;
                case '\f': if (!append("\\f", 2)) return false; break;
                case '\n': if (!append("\\n", 2)) return false; break;
                case '\r': if (!append("\\r", 2)) return false; break;
                case '\t': if (!append("\\t", 2)) return false; break;
                default:
                    if (codepoint >= 0x20 && codepoint <= 0x7e) {
                        if (kind == PyUnicode_1BYTE_KIND) {
                            const auto* ascii = static_cast<const char*>(data);
                            Py_ssize_t end = index + 1;
                            while (end < count && ascii[end] >= 0x20 && ascii[end] <= 0x7e &&
                                    ascii[end] != '"' && ascii[end] != '\\') ++end;
                            if (!append(ascii + index, static_cast<std::size_t>(end - index))) return false;
                            index = end - 1;
                        } else {
                            const char ascii = static_cast<char>(codepoint);
                            if (!append(&ascii, 1)) return false;
                        }
                    } else if (codepoint <= 0xffff) {
                        if (!escaped_codepoint(codepoint)) return false;
                    } else {
                        codepoint -= 0x10000;
                        if (!escaped_codepoint(0xd800 + (codepoint >> 10)) ||
                                !escaped_codepoint(0xdc00 + (codepoint & 0x3ff))) return false;
                    }
            }
        }
        return append("\"", 1);
    }

    bool integer(PyObject* value) {
        int overflow = 0;
        const long long number = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (PyErr_Occurred()) throw pybind11::error_already_set();
        if (!overflow) {
            char digits[32];
            const auto converted = std::to_chars(digits, digits + sizeof(digits), number);
            return append(digits, static_cast<std::size_t>(converted.ptr - digits));
        }
        // Bound decimal formatting work without using private integer layouts.
        (void)PyLong_AsDouble(value);
        if (PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
                PyErr_Clear();
                return false;
            }
            throw pybind11::error_already_set();
        }
        // The magnitude check never supplies encoded digits; retain the exact integer.
        PyObject* allocated = PyObject_Str(value);
        if (allocated == nullptr) {
            if (PyErr_ExceptionMatches(PyExc_ValueError)) {
                PyErr_Clear();
                return false;
            }
            throw pybind11::error_already_set();
        }
        const auto text = pybind11::reinterpret_steal<pybind11::object>(allocated);
        return append(static_cast<const char*>(PyUnicode_DATA(text.ptr())),
            static_cast<std::size_t>(PyUnicode_GET_LENGTH(text.ptr())));
    }

    bool dictionary(PyObject* value, int depth) {
        const auto count = static_cast<std::size_t>(PyDict_GET_SIZE(value));
        if (count > (max_json_items - items_) / 2) return false;
        std::vector<std::pair<PyObject*, PyObject*>> rows;
        rows.reserve(count);
        Py_ssize_t position = 0;
        PyObject* key;
        PyObject* item;
        while (PyDict_Next(value, &position, &key, &item)) {
            if (!PyUnicode_CheckExact(key)) return false;
            rows.emplace_back(key, item);
        }
        std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
            return PyUnicode_Compare(left.first, right.first) < 0;
        });
        if (!append("{", 1)) return false;
        bool first = true;
        for (const auto& row : rows) {
            ++items_;
            if ((!first && !append(",", 1)) || !string(row.first) || !append(":", 1) ||
                    !encode(row.second, depth + 1)) return false;
            first = false;
        }
        return append("}", 1);
    }

    bool allow_tuple_;
    int max_depth_;
    std::size_t items_ = 0;
    std::string output_;
};

inline pybind11::object canonical_json_digest(pybind11::handle value, pybind11::handle sha256,
        pybind11::object fallback_marker, bool allow_tuple = false, int max_depth = 32,
        pybind11::object guard = pybind11::none()) {
    if (max_depth < 1 || max_depth > 256 || !guard_matches(guard.ptr())) return fallback_marker;
    CanonicalWriter writer(allow_tuple, max_depth);
    // Borrow entries only while the GIL is held, without Python callbacks or
    // GC-tracked allocations. The finished bytes snapshot precedes hashlib calls.
    if (!writer.encode(value.ptr())) return fallback_marker;
    const auto& payload = writer.output();
    const auto hexadecimal = sha256(pybind11::bytes(payload.data(), payload.size())).attr("hexdigest")();
    pybind11::str prefix("sha256:");
    PyObject* result = PyNumber_Add(prefix.ptr(), hexadecimal.ptr());
    if (result == nullptr) throw pybind11::error_already_set();
    return pybind11::reinterpret_steal<pybind11::object>(result);
}

}  // namespace resectionlab_json

inline void bind_canonical_json(pybind11::module_& module) {
    module.def("canonical_json_digest", &resectionlab_json::canonical_json_digest,
        pybind11::arg("value"), pybind11::arg("sha256"), pybind11::arg("fallback_marker"),
        pybind11::arg("allow_tuple") = false, pybind11::arg("max_depth") = 32,
        pybind11::arg("guard") = pybind11::none());
}
