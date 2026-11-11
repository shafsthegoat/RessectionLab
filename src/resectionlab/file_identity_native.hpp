#pragma once

#include <pybind11/pybind11.h>
#include <cerrno>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

namespace resectionlab::file_identity {
namespace py = pybind11;

inline bool bindings_unchanged(py::handle module, py::tuple expected) {
    if (!PyModule_CheckExact(module.ptr())) return false;
    PyObject* values = PyModule_GetDict(module.ptr());
    for (py::handle pair : expected) {
        if (!PyTuple_CheckExact(pair.ptr()) || PyTuple_GET_SIZE(pair.ptr()) != 2) return false;
        PyObject* key = PyTuple_GET_ITEM(pair.ptr(), 0);
        if (!PyUnicode_CheckExact(key) || PyDict_GetItemWithError(values,key) != PyTuple_GET_ITEM(pair.ptr(),1)) {
            if (PyErr_Occurred()) throw py::error_already_set();
            return false;
        }
    }
    return true;
}

class FileIdentities {
    struct Time {
        std::int64_t seconds;
        long nanoseconds;
        bool operator!=(const Time& other) const noexcept {
            return seconds != other.seconds || nanoseconds != other.nanoseconds;
        }
    };
    struct Entry {
        std::string path;
        std::int64_t size;
        Time modified, changed;
        std::uint64_t inode;
    };
    std::vector<Entry> entries_;

    static Time timestamp(PyObject* value, PyObject* billion) {
        // Keep seconds and remainder separate: Python's signed ns can exceed int64.
        py::object whole = py::reinterpret_steal<py::object>(PyNumber_FloorDivide(value, billion));
        if (!whole) throw py::error_already_set();
        const auto seconds = PyLong_AsLongLong(whole.ptr());
        if (seconds == -1 && PyErr_Occurred()) throw py::error_already_set();
        py::object remainder = py::reinterpret_steal<py::object>(PyNumber_Remainder(value, billion));
        if (!remainder) throw py::error_already_set();
        const auto nanos = PyLong_AsLong(remainder.ptr());
        if (nanos == -1 && PyErr_Occurred()) throw py::error_already_set();
        return {seconds, nanos};
    }

public:
    FileIdentities(py::handle paths, py::handle stamps) {
        if (!PyList_CheckExact(paths.ptr()) || !PyList_CheckExact(stamps.ptr()) ||
                PyList_GET_SIZE(paths.ptr()) != PyList_GET_SIZE(stamps.ptr()))
            throw py::type_error("file identities require equal exact lists");
        py::int_ billion(1000000000);
        const auto count = PyList_GET_SIZE(paths.ptr());
        entries_.reserve(static_cast<std::size_t>(count));
        for (py::ssize_t index = 0; index < count; ++index) {
            PyObject* path = PyList_GET_ITEM(paths.ptr(), index);
            PyObject* stamp = PyList_GET_ITEM(stamps.ptr(), index);
            if (!PyUnicode_CheckExact(path) || !PyTuple_CheckExact(stamp) || PyTuple_GET_SIZE(stamp) != 4)
                throw py::type_error("file identities require exact strings and four-item tuples");
            for (int field = 0; field < 4; ++field)
                if (!PyLong_CheckExact(PyTuple_GET_ITEM(stamp, field)))
                    throw py::type_error("file identity fields must be exact integers");
            py::object encoded = py::reinterpret_steal<py::object>(PyUnicode_EncodeFSDefault(path));
            if (!encoded) throw py::error_already_set();
            std::string native_path(PyBytes_AS_STRING(encoded.ptr()),
                static_cast<std::size_t>(PyBytes_GET_SIZE(encoded.ptr())));
            if (native_path.find('\0') != std::string::npos) throw py::value_error("embedded null byte");
            const auto size = PyLong_AsLongLong(PyTuple_GET_ITEM(stamp, 0));
            if (size == -1 && PyErr_Occurred()) throw py::error_already_set();
            const auto inode = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(stamp, 3));
            if (inode == static_cast<unsigned long long>(-1) && PyErr_Occurred()) throw py::error_already_set();
            entries_.push_back({std::move(native_path), size,
                timestamp(PyTuple_GET_ITEM(stamp, 1), billion.ptr()),
                timestamp(PyTuple_GET_ITEM(stamp, 2), billion.ptr()), inode});
        }
    }

    py::list changed() const {
        std::vector<std::size_t> changed;
        changed.reserve(entries_.size());
        std::size_t failed = 0;
        int error = 0;
        {
            py::gil_scoped_release release;
            for (std::size_t index = 0; index < entries_.size(); ++index) {
                const auto& entry = entries_[index];
                struct stat current;
                if (::stat(entry.path.c_str(), &current) != 0) {
                    error = errno; failed = index; break;
                }
#if defined(__APPLE__)
                const Time modified{current.st_mtimespec.tv_sec, current.st_mtimespec.tv_nsec};
                const Time status_changed{current.st_ctimespec.tv_sec, current.st_ctimespec.tv_nsec};
#else
                const Time modified{current.st_mtim.tv_sec, current.st_mtim.tv_nsec};
                const Time status_changed{current.st_ctim.tv_sec, current.st_ctim.tv_nsec};
#endif
                if (current.st_size != entry.size || modified != entry.modified ||
                        status_changed != entry.changed || current.st_ino != entry.inode)
                    changed.push_back(index);
            }
        }
        if (error) {
            errno = error;
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, entries_[failed].path.c_str());
            throw py::error_already_set();
        }
        py::list result;
        for (auto index : changed) result.append(py::int_(index));
        return result;
    }
};
}  // namespace resectionlab::file_identity
