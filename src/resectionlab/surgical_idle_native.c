#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef IDLE_THERMAL_BUILD_IDENTITY
#error "Use the explicit optional builder to bind this kernel to its build inputs"
#endif

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 12
#error "This isolated prototype is pinned to the reviewed CPython3.12 dict layout"
#endif
#if !defined(__APPLE__) || !defined(__aarch64__)
#error "This optional kernel is qualified only for CPython3.12 on macOS arm64"
#endif

/* Optional idle-cooling implementation of the unchanged scalar thermal law. */
#define MAX_ROWS 32768
#define MAX_STATE_KEYS 64
#define MAX_SCRATCH_BYTES (8u * 1024u * 1024u)
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53,
               "This prototype requires IEEE binary64 doubles");

typedef struct {
    PyObject *key, *old_temp, *old_dose, *new_temp, *new_dose;
    uint64_t cell;
} Row;

typedef struct { PyObject *key, *value; } DoseRef;
typedef struct { PyObject *temperatures, *doses, *cooling; } ModuleKeys;

static int unwatched(PyObject *mapping) {
    /* CPython3.12 pycore_dict_state.h/pycore_dict.h: the low8 version bits
       select native watchers. Build receipt verifies and hashes those headers.
       A watched dictionary could call arbitrary code inside PyDict_SetItem. */
    return PyDict_CheckExact(mapping) && ((((PyDictObject *)mapping)->ma_version_tag & 255u) == 0u);
}

static int state_keys_are_native(PyObject *state) {
    Py_ssize_t position = 0;
    PyObject *key, *value;
    if (!unwatched(state) || PyDict_Size(state) > MAX_STATE_KEYS) return 0;
    while (PyDict_Next(state, &position, &key, &value)) {
        if (!PyUnicode_CheckExact(key)) return 0;
    }
    return 1;
}

static int finite_float(PyObject *value) {
    return value && PyFloat_CheckExact(value) && isfinite(PyFloat_AS_DOUBLE(value));
}

static int cell_number(PyObject *key, uint64_t *result) {
    if (!PyUnicode_CheckExact(key) || !PyUnicode_IS_ASCII(key)) return 0;
    Py_ssize_t length = PyUnicode_GET_LENGTH(key);
    if (length < 1 || length > 19) return 0;
    const Py_UCS1 *text = PyUnicode_1BYTE_DATA(key);
    if (length > 1 && text[0] == '0') return 0;
    uint64_t cell = 0;
    for (Py_ssize_t index = 0; index < length; ++index) {
        if (text[index] < '0' || text[index] > '9') return 0;
        uint64_t digit = (uint64_t)(text[index] - '0');
        if (cell > ((uint64_t)INT64_MAX - digit) / 10u) return 0;
        cell = cell * 10u + digit;
    }
    *result = cell;
    return 1;
}

static int compare_rows(const void *left, const void *right) {
    const Row *a = *(Row * const *)left, *b = *(Row * const *)right;
    return (a->cell > b->cell) - (a->cell < b->cell);
}

static void release_rows(Row *rows, Py_ssize_t initialized, Row **ordered,
                         DoseRef *dose_refs, Py_ssize_t doses_initialized) {
    for (Py_ssize_t index = 0; index < initialized; ++index) {
        Py_XDECREF(rows[index].key);
        Py_XDECREF(rows[index].old_temp);
        Py_XDECREF(rows[index].old_dose);
        Py_XDECREF(rows[index].new_temp);
        Py_XDECREF(rows[index].new_dose);
    }
    for (Py_ssize_t index = 0; index < doses_initialized; ++index) {
        Py_XDECREF(dose_refs[index].key);
        Py_XDECREF(dose_refs[index].value);
    }
    PyMem_Free(rows); PyMem_Free(ordered); PyMem_Free(dose_refs);
}

static int same_inputs(PyObject *state, ModuleKeys *keys, PyObject *temperatures,
                        PyObject *doses, PyObject *powers, PyObject *old_cooling,
                        Row *rows, DoseRef *dose_refs, Py_ssize_t count) {
    /* No lookups against a dictionary with arbitrary custom keys. */
    if (!state_keys_are_native(state) || !unwatched(temperatures) || !unwatched(doses) ||
        !unwatched(powers) || PyDict_Size(powers) != 0) return 0;
    if (PyDict_GetItemWithError(state, keys->temperatures) != temperatures ||
        PyDict_GetItemWithError(state, keys->doses) != doses ||
        PyDict_GetItemWithError(state, keys->cooling) != old_cooling ||
        PyDict_Size(temperatures) != count || PyDict_Size(doses) != count) return 0;
    Py_ssize_t position = 0, index = 0;
    PyObject *key, *value;
    while (PyDict_Next(temperatures, &position, &key, &value)) {
        if (index >= count || key != rows[index].key || value != rows[index].old_temp) return 0;
        ++index;
    }
    if (index != count) return 0;
    position = 0; index = 0;
    while (PyDict_Next(doses, &position, &key, &value)) {
        if (index >= count || key != dose_refs[index].key || value != dose_refs[index].value) return 0;
        ++index;
    }
    return index == count;
}

static void restore_first_error(PyObject *state, ModuleKeys *keys, PyObject *temperatures,
                                PyObject *doses, PyObject *old_cooling, Row *rows,
                                Py_ssize_t count) {
    /* Every key still exists; exact-string replacements allocate no table.
       Keep the first exception even if an impossible recovery error occurs. */
    PyObject *type = NULL, *value = NULL, *traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    for (Py_ssize_t index = 0; index < count; ++index) {
        if (PyDict_SetItem(temperatures, rows[index].key, rows[index].old_temp) < 0) PyErr_Clear();
        if (PyDict_SetItem(doses, rows[index].key, rows[index].old_dose) < 0) PyErr_Clear();
    }
    if (PyDict_SetItem(state, keys->cooling, old_cooling) < 0) PyErr_Clear();
    PyErr_Restore(type, value, traceback);
}

static PyObject *try_idle(PyObject *module, PyObject *args) {
    PyObject *state, *powers, *scalars[7];
    if (!PyArg_ParseTuple(args, "OOOOOOOOO:try_idle", &state, &powers, &scalars[0],
                          &scalars[1], &scalars[2], &scalars[3], &scalars[4],
                          &scalars[5], &scalars[6])) return NULL;
    ModuleKeys *keys = PyModule_GetState(module);
    if (!keys || !keys->temperatures || !keys->doses || !keys->cooling) {
        PyErr_SetString(PyExc_RuntimeError, "Prototype module state is unavailable");
        return NULL;
    }
    if (!state_keys_are_native(state) || !unwatched(powers) || PyDict_Size(powers)) Py_RETURN_FALSE;
    for (int index = 0; index < 7; ++index) if (!finite_float(scalars[index])) Py_RETURN_FALSE;
    const double dt = PyFloat_AS_DOUBLE(scalars[0]), capacity = PyFloat_AS_DOUBLE(scalars[1]);
    const double tau = PyFloat_AS_DOUBLE(scalars[2]), ambient = PyFloat_AS_DOUBLE(scalars[3]);
    const double threshold = PyFloat_AS_DOUBLE(scalars[4]), decay = PyFloat_AS_DOUBLE(scalars[5]);
    const double one_minus = PyFloat_AS_DOUBLE(scalars[6]);
    if (dt < 0.0 || capacity <= 0.0 || tau <= 0.0 || threshold < 0.0) Py_RETURN_FALSE;
    PyObject *temperatures = PyDict_GetItemWithError(state, keys->temperatures);
    PyObject *doses = PyDict_GetItemWithError(state, keys->doses);
    PyObject *old_cooling = PyDict_GetItemWithError(state, keys->cooling);
    if (PyErr_Occurred()) return NULL;
    if (!temperatures || !doses || !unwatched(temperatures) || !unwatched(doses) ||
        temperatures == doses || temperatures == powers || doses == powers || !finite_float(old_cooling)) Py_RETURN_FALSE;
    Py_ssize_t count = PyDict_Size(temperatures);
    if (count != PyDict_Size(doses) || count > MAX_ROWS) Py_RETURN_FALSE;
    if (!count) Py_RETURN_TRUE; /* Preserve exact old cooling object and -0.0. */
    size_t owned_bound = (size_t)count * (sizeof(Row) + sizeof(Row *) + sizeof(DoseRef) + 128u) + 64u;
    if (owned_bound > MAX_SCRATCH_BYTES) Py_RETURN_FALSE;

    Row *rows = NULL; Row **ordered = NULL; DoseRef *dose_refs = NULL;
    Py_ssize_t initialized = 0, doses_initialized = 0;
    PyObject *new_cooling = NULL, *answer = NULL;
    Py_INCREF(temperatures); Py_INCREF(doses); Py_INCREF(old_cooling);
    rows = PyMem_Calloc((size_t)count, sizeof(Row));
    ordered = PyMem_Malloc((size_t)count * sizeof(Row *));
    dose_refs = PyMem_Calloc((size_t)count, sizeof(DoseRef));
    if (!rows || !ordered || !dose_refs) { PyErr_NoMemory(); goto error; }

    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (PyDict_Next(doses, &position, &key, &value)) {
        uint64_t unused;
        if (doses_initialized >= count || !cell_number(key, &unused) || !finite_float(value)) goto decline;
        dose_refs[doses_initialized].key = Py_NewRef(key);
        dose_refs[doses_initialized].value = Py_NewRef(value);
        ++doses_initialized;
    }
    position = 0;
    while (PyDict_Next(temperatures, &position, &key, &value)) {
        uint64_t cell;
        if (initialized >= count || !cell_number(key, &cell) || !finite_float(value)) goto decline;
        PyObject *old_dose = PyDict_GetItemWithError(doses, key);
        if (!old_dose) { if (PyErr_Occurred()) goto error; goto decline; }
        Row *row = &rows[initialized];
        row->key = Py_NewRef(key); row->old_temp = Py_NewRef(value); row->old_dose = Py_NewRef(old_dose);
        row->cell = cell; ordered[initialized] = row; ++initialized;
    }
    if (initialized != count || doses_initialized != count) goto decline;
    qsort(ordered, (size_t)count, sizeof(Row *), compare_rows);
    volatile double cooling = PyFloat_AS_DOUBLE(old_cooling);
    for (Py_ssize_t index = 0; index < count; ++index) {
        Row *row = ordered[index];
        volatile double old = PyFloat_AS_DOUBLE(row->old_temp) - ambient;
        if (!isfinite(old) || !(old < threshold)) goto decline;
        volatile double numerator = 0.0 * tau;
        volatile double equilibrium = numerator / capacity;
        volatile double weighted_old = old * decay;
        volatile double weighted_zero = equilibrium * one_minus;
        volatile double new_excess = weighted_old + weighted_zero;
        volatile double next_temperature = ambient + new_excess;
        volatile double next_dose = PyFloat_AS_DOUBLE(row->old_dose) + 0.0;
        volatile double difference = new_excess - old;
        volatile double work = capacity * difference;
        volatile double zero_dt = 0.0 * dt;
        volatile double raw_loss = zero_dt - work;
        double loss = raw_loss > 0.0 ? raw_loss : 0.0;
        if (!isfinite(equilibrium) || !isfinite(weighted_old) || !isfinite(weighted_zero) ||
            !isfinite(new_excess) || !isfinite(next_temperature) || !isfinite(next_dose) ||
            !isfinite(difference) || !isfinite(work) || !isfinite(raw_loss)) goto decline;
        cooling = cooling + loss; /* Exact sorted-key scalar leftfold. */
        if (!isfinite(cooling)) goto decline;
        row->new_temp = PyFloat_FromDouble(next_temperature);
        if (!row->new_temp) goto error;
        row->new_dose = PyFloat_FromDouble(next_dose);
        if (!row->new_dose) goto error;
    }
    new_cooling = PyFloat_FromDouble(cooling);
    if (!new_cooling) goto error;
    /* Signal handlers may execute Python. Revalidate every borrowed input and
       insertion order after this final callback boundary, then retain the GIL. */
    if (PyErr_CheckSignals() < 0) goto error;
    if (!same_inputs(state, keys, temperatures, doses, powers, old_cooling, rows, dose_refs, count)) {
        if (PyErr_Occurred()) goto error;
        goto decline;
    }
    /* No allocations, callbacks, signal checks or GIL release during commit.
       All keys already exist and all replacement values were preallocated. */
    for (Py_ssize_t index = 0; index < count; ++index) {
        Row *row = ordered[index];
        if (PyDict_SetItem(temperatures, row->key, row->new_temp) < 0 ||
            PyDict_SetItem(doses, row->key, row->new_dose) < 0) goto commit_error;
    }
    if (PyDict_SetItem(state, keys->cooling, new_cooling) < 0) goto commit_error;
    answer = Py_NewRef(Py_True); goto cleanup;

commit_error:
    restore_first_error(state, keys, temperatures, doses, old_cooling, rows, count);
    goto error;
decline:
    answer = Py_NewRef(Py_False); goto cleanup;
error:
    answer = NULL;
cleanup:
    release_rows(rows, initialized, ordered, dose_refs, doses_initialized);
    Py_XDECREF(new_cooling);
    Py_DECREF(old_cooling); Py_DECREF(doses); Py_DECREF(temperatures);
    return answer;
}

#define MAX_EXACT_INT_LIST_ITEMS 1048576
/* Pure admission only. Every accepted call scans the whole exact list while
   holding the GIL; no allocation, coercion, callback, mutation or cached result.
   None preserves the caller's original predicate for unsupported values. */
static PyObject *try_exact_int_list(PyObject *module, PyObject *value) {
    (void)module;
    if (!PyList_CheckExact(value)) Py_RETURN_NONE;
    const Py_ssize_t count = PyList_GET_SIZE(value);
    if (count > MAX_EXACT_INT_LIST_ITEMS) Py_RETURN_NONE;
    for (Py_ssize_t index = 0; index < count; ++index) {
        if (Py_TYPE(PyList_GET_ITEM(value, index)) != &PyLong_Type) Py_RETURN_NONE;
    }
    Py_RETURN_TRUE;
}

static PyMethodDef methods[] = {
    {"try_idle", try_idle, METH_VARARGS, "Try bounded exact idle cooling; False leaves inputs untouched."},
    {"try_exact_int_list", try_exact_int_list, METH_O, "Try a bounded exact list/int predicate; None requests legacy fallback."},
    {NULL, NULL, 0, NULL}
};
static int traverse(PyObject *module, visitproc visit, void *arg) {
    ModuleKeys *keys = PyModule_GetState(module);
    if (keys) { Py_VISIT(keys->temperatures); Py_VISIT(keys->doses); Py_VISIT(keys->cooling); }
    return 0;
}
static int clear(PyObject *module) {
    ModuleKeys *keys = PyModule_GetState(module);
    if (keys) { Py_CLEAR(keys->temperatures); Py_CLEAR(keys->doses); Py_CLEAR(keys->cooling); }
    return 0;
}
static void free_module(void *module) { (void)clear((PyObject *)module); }
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_surgical_idle_native_kernel", "Optional exact idle cooling.",
    sizeof(ModuleKeys), methods, NULL, traverse, clear, free_module
};
PyMODINIT_FUNC PyInit__surgical_idle_native_kernel(void) {
    PyObject *module = PyModule_Create(&definition);
    if (!module) return NULL;
    ModuleKeys *keys = PyModule_GetState(module);
    keys->temperatures = PyUnicode_FromString("temperatureC");
    keys->doses = PyUnicode_FromString("thermalDoseCSeconds");
    keys->cooling = PyUnicode_FromString("coolingLossJ");
    if (!keys->temperatures || !keys->doses || !keys->cooling ||
        PyModule_AddIntConstant(module, "MAX_ROWS", MAX_ROWS) < 0 ||
        PyModule_AddIntConstant(module, "MAX_SCRATCH_BYTES", MAX_SCRATCH_BYTES) < 0 ||
        PyModule_AddIntConstant(module, "MAX_EXACT_INT_LIST_ITEMS", MAX_EXACT_INT_LIST_ITEMS) < 0 ||
        PyModule_AddStringConstant(module, "BUILD_IDENTITY", IDLE_THERMAL_BUILD_IDENTITY) < 0) {
        Py_DECREF(module); return NULL;
    }
    return module;
}
