#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#ifndef PREFIX_JSON_BUILD_IDENTITY
#error "Use the explicit prefix JSON builder to bind the kernel to its build inputs"
#endif

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 12 || PY_MICRO_VERSION != 12 || !defined(__APPLE__) || !defined(__aarch64__)
#error "This optional writer is qualified for CPython3.12.12/macOS arm64"
#endif
#define RAW_CAP (16u * 1024u * 1024u)
#define DEPTH_CAP 64
#define NODE_CAP 1000000

typedef struct {
    size_t bytes, limit, nodes;
    PyObject *active[DEPTH_CAP];
} Bound;

typedef struct {
    PyObject *json_module, *ascii_function;
    PyCFunction ascii_pointer;
    reprfunc float_pointer;
    int busy;
} ModuleState;

typedef struct {char *data; size_t used, capacity;} Writer;

static int add(Bound *bound, size_t amount) {
    if (amount > bound->limit - bound->bytes) return 0;
    bound->bytes += amount;
    return 1;
}

static int walk(PyObject *value, Bound *bound, size_t depth) {
    if (++bound->nodes > NODE_CAP) return 0;
    if (value == Py_None) return add(bound, 4);
    if (value == Py_True || value == Py_False) return add(bound, 5);
    if (PyUnicode_CheckExact(value)) {
        Py_ssize_t length = PyUnicode_GET_LENGTH(value);
        size_t remaining = bound->limit - bound->bytes;
        if (remaining < 2 || (size_t)length > (remaining - 2) / 12u) return 0;
        return add(bound, (size_t)length * 12u + 2u);
    }
    if (PyLong_CheckExact(value)) {
        int overflow = 0;
        (void)PyLong_AsLongLongAndOverflow(value, &overflow);
        if (PyErr_Occurred()) return -1;
        return !overflow && add(bound, 21);
    }
    if (PyFloat_CheckExact(value)) return isfinite(PyFloat_AS_DOUBLE(value)) && add(bound, 32);
    int dictionary = PyDict_CheckExact(value);
    int list = PyList_CheckExact(value), tuple = PyTuple_CheckExact(value);
    if (!dictionary && !list && !tuple) return 0;
    if (depth >= DEPTH_CAP) return 0;
    for (size_t index = 0; index < depth; ++index) {
        if (bound->active[index] == value) return 0;
    }
    bound->active[depth] = value;
    Py_ssize_t count = dictionary ? PyDict_GET_SIZE(value) : list ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value);
    if (!add(bound, 2u) || (count && !add(bound, (size_t)count - 1u))) return 0;
    if (dictionary) {
        Py_ssize_t position = 0;
        PyObject *key, *child;
        while (PyDict_Next(value, &position, &key, &child)) {
            if (!PyUnicode_CheckExact(key)) return 0;
            int answer = walk(key, bound, depth + 1u);
            if (answer <= 0) return answer;
            if (!add(bound, 1u)) return 0;
            answer = walk(child, bound, depth + 1u);
            if (answer <= 0) return answer;
        }
    } else {
        for (Py_ssize_t index = 0; index < count; ++index) {
            PyObject *child = list ? PyList_GET_ITEM(value, index) : PyTuple_GET_ITEM(value, index);
            int answer = walk(child, bound, depth + 1u);
            if (answer <= 0) return answer;
        }
    }
    return 1;
}

static int append(Writer *out, const char *data, size_t length) {
    if (length > out->capacity - out->used) {
        PyErr_SetString(PyExc_ValueError, "Native JSON bound unexpectedly exceeded");
        return -1;
    }
    memcpy(out->data + out->used, data, length);
    out->used += length;
    return 0;
}

static int append_unicode(Writer *out, PyObject *text) {
    if (!text) return -1;
    if (!PyUnicode_CheckExact(text) || !PyUnicode_IS_ASCII(text)) {
        Py_DECREF(text);
        PyErr_SetString(PyExc_ValueError, "Native JSON helper did not return exact ASCII");
        return -1;
    }
    int answer = append(out, (const char *)PyUnicode_1BYTE_DATA(text), (size_t)PyUnicode_GET_LENGTH(text));
    Py_DECREF(text);
    return answer;
}

static int emit(ModuleState *state, Writer *out, PyObject *value) {
    if (value == Py_None) return append(out, "null", 4);
    if (value == Py_True) return append(out, "true", 4);
    if (value == Py_False) return append(out, "false", 5);
    if (PyUnicode_CheckExact(value)) return append_unicode(out, state->ascii_pointer(NULL, value));
    if (PyFloat_CheckExact(value)) return append_unicode(out, state->float_pointer(value));
    if (PyLong_CheckExact(value)) {
        int overflow = 0;
        long long integer = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (PyErr_Occurred()) return -1;
        if (overflow) {PyErr_SetString(PyExc_ValueError, "Validated integer changed"); return -1;}
        uint64_t magnitude = integer < 0 ? (uint64_t)(-(integer + 1)) + 1u : (uint64_t)integer;
        char digits[32]; size_t position = sizeof(digits);
        do {digits[--position] = (char)('0' + magnitude % 10u); magnitude /= 10u;} while (magnitude);
        if (integer < 0) digits[--position] = '-';
        return append(out, digits + position, sizeof(digits) - position);
    }
    if (PyDict_CheckExact(value)) {
        if (append(out, "{", 1)) return -1;
        Py_ssize_t position = 0; PyObject *key, *child; int first = 1;
        while (PyDict_Next(value, &position, &key, &child)) {
            if (!first && append(out, ",", 1)) return -1;
            first = 0;
            if (append_unicode(out, state->ascii_pointer(NULL, key)) || append(out, ":", 1) || emit(state, out, child)) return -1;
        }
        return append(out, "}", 1);
    }
    if (PyList_CheckExact(value) || PyTuple_CheckExact(value)) {
        int list = PyList_CheckExact(value);
        Py_ssize_t count = list ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value);
        if (append(out, "[", 1)) return -1;
        for (Py_ssize_t index = 0; index < count; ++index) {
            if (index && append(out, ",", 1)) return -1;
            PyObject *child = list ? PyList_GET_ITEM(value, index) : PyTuple_GET_ITEM(value, index);
            if (emit(state, out, child)) return -1;
        }
        return append(out, "]", 1);
    }
    PyErr_SetString(PyExc_ValueError, "Validated type changed");
    return -1;
}

static int live_helpers(ModuleState *state) {
    if (!state || !state->json_module || !state->ascii_function ||
        Py_TYPE(state->json_module) != &PyModule_Type ||
        Py_TYPE(state->ascii_function) != &PyCFunction_Type ||
        PyCFunction_GetFlags(state->ascii_function) != METH_O ||
        PyCFunction_GetSelf(state->ascii_function) != state->json_module ||
        PyCFunction_GetFunction(state->ascii_function) != state->ascii_pointer ||
        PyFloat_Type.tp_repr != state->float_pointer) {
        PyErr_SetString(PyExc_ValueError, "Native JSON helper identity changed");
        return -1;
    }
    PyObject *current = PyObject_GetAttrString(state->json_module, "encode_basestring_ascii");
    if (!current) return -1;
    int same = current == state->ascii_function;
    Py_DECREF(current);
    if (!same) {PyErr_SetString(PyExc_ValueError, "Native JSON ASCII binding changed"); return -1;}
    return 0;
}

static PyObject *try_encode(PyObject *module, PyObject *args) {
    PyObject *value, *limit_object;
    if (!PyArg_ParseTuple(args, "OO:try_encode", &value, &limit_object)) return NULL;
    if (!PyLong_CheckExact(limit_object)) Py_RETURN_NONE;
    int overflow = 0;
    long long limit = PyLong_AsLongLongAndOverflow(limit_object, &overflow);
    if (PyErr_Occurred()) return NULL;
    if (overflow || limit < 0 || limit > RAW_CAP) Py_RETURN_NONE;
    /* Pinned CPython3.12.12 header layout. Near recursive Python callers keep
       the legacy encoder path instead of acquiring different error behavior. */
    if (PyThreadState_Get()->py_recursion_remaining < 256) Py_RETURN_NONE;
    ModuleState *state = PyModule_GetState(module);
    if (live_helpers(state) < 0) return NULL;
    if (state->busy) {PyErr_SetString(PyExc_RuntimeError, "Reentrant writer"); return NULL;}
    Bound first = {.bytes=0, .limit=(size_t)limit, .nodes=0};
    int admitted = walk(value, &first, 0);
    if (admitted < 0) return NULL;
    if (!admitted) Py_RETURN_NONE;
    PyObject *result = PyUnicode_New((Py_ssize_t)first.bytes, 127);
    if (!result) return NULL;
    /* Every potential Python callback boundary precedes final validation. */
    if (PyErr_CheckSignals() < 0 || live_helpers(state) < 0) {Py_DECREF(result); return NULL;}
    if (PyThreadState_Get()->py_recursion_remaining < 256) {Py_DECREF(result); Py_RETURN_NONE;}
    Bound final = {.bytes=0, .limit=first.bytes, .nodes=0};
    state->busy = 1;
    admitted = walk(value, &final, 0);
    if (admitted <= 0) {
        state->busy = 0; Py_DECREF(result);
        if (admitted < 0) return NULL;
        Py_RETURN_NONE;
    }
    Writer out = {.data=(char *)PyUnicode_1BYTE_DATA(result), .used=0, .capacity=first.bytes};
    if (emit(state, &out, value) < 0) {state->busy=0; Py_DECREF(result); return NULL;}
    /* No input is read beyond this point. The only owner of result is this C
       frame; exact ASCII resizing cannot invoke a user formatter/finalizer. */
    if (PyUnicode_Resize(&result, (Py_ssize_t)out.used) < 0) {state->busy=0; Py_XDECREF(result); return NULL;}
    state->busy = 0;
    return result;
}

static PyObject *identity(PyObject *module, PyObject *unused) {
    (void)unused;
    ModuleState *state = PyModule_GetState(module);
    if (live_helpers(state) < 0) return NULL;
    Dl_info ascii, floating, conversion;
    if (!dladdr((void *)state->ascii_pointer,&ascii) || !dladdr((void *)state->float_pointer,&floating) ||
        !dladdr((void *)PyOS_double_to_string,&conversion)) {
        PyErr_SetString(PyExc_ValueError,"Loaded helper library origin unavailable");return NULL;
    }
    return Py_BuildValue("{s:s,s:K,s:s,s:K,s:s,s:K}",
        "asciiLibrary",ascii.dli_fname,"asciiOffset",(unsigned long long)((uintptr_t)state->ascii_pointer-(uintptr_t)ascii.dli_fbase),
        "floatLibrary",floating.dli_fname,"floatOffset",(unsigned long long)((uintptr_t)state->float_pointer-(uintptr_t)floating.dli_fbase),
        "conversionLibrary",conversion.dli_fname,"conversionOffset",(unsigned long long)((uintptr_t)PyOS_double_to_string-(uintptr_t)conversion.dli_fbase));
}

static PyMethodDef methods[] = {
    {"try_encode", try_encode, METH_VARARGS, "Atomic bounded ASCII snapshot, or None for unsupported inputs."},
    {"raw_identity", identity, METH_NOARGS, "Actual loaded raw helper image identities."},
    {NULL,NULL,0,NULL}
};
static int traverse(PyObject *module, visitproc visit, void *arg) {
    ModuleState *state=PyModule_GetState(module);
    if (state) {Py_VISIT(state->json_module);Py_VISIT(state->ascii_function);}
    return 0;
}
static int clear(PyObject *module) {
    ModuleState *state=PyModule_GetState(module);
    if (state) {Py_CLEAR(state->json_module);Py_CLEAR(state->ascii_function);}
    return 0;
}
static void free_module(void *module) {(void)clear((PyObject *)module);}
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT,"_surgical_prefix_json_kernel","Optional bounded prefix JSON snapshot.",
    sizeof(ModuleState),methods,NULL,traverse,clear,free_module
};
PyMODINIT_FUNC PyInit__surgical_prefix_json_kernel(void) {
    PyObject *module=PyModule_Create(&definition);
    if (!module) return NULL;
    ModuleState *state=PyModule_GetState(module);
    state->json_module=PyImport_ImportModule("_json");
    if (!state->json_module) goto error;
    if (Py_TYPE(state->json_module)!=&PyModule_Type) {PyErr_SetString(PyExc_ValueError,"Stock JSON module required");goto error;}
    state->ascii_function=PyObject_GetAttrString(state->json_module,"encode_basestring_ascii");
    if (!state->ascii_function) goto error;
    if (Py_TYPE(state->ascii_function)!=&PyCFunction_Type || PyCFunction_GetFlags(state->ascii_function)!=METH_O ||
        PyCFunction_GetSelf(state->ascii_function)!=state->json_module) {
        PyErr_SetString(PyExc_ValueError,"Stock ASCII C function required");goto error;
    }
    PyObject *name=PyObject_GetAttrString(state->ascii_function,"__name__");
    if (!name) goto error;
    int correct=PyUnicode_CheckExact(name) && PyUnicode_CompareWithASCIIString(name,"encode_basestring_ascii")==0;
    Py_DECREF(name);
    if (!correct) {PyErr_SetString(PyExc_ValueError,"Wrong ASCII C helper");goto error;}
    state->ascii_pointer=PyCFunction_GetFunction(state->ascii_function);
    state->float_pointer=PyFloat_Type.tp_repr;
    if (PyModule_AddStringConstant(module,"BUILD_IDENTITY",PREFIX_JSON_BUILD_IDENTITY)<0) goto error;
    return module;
error:
    Py_DECREF(module);return NULL;
}
