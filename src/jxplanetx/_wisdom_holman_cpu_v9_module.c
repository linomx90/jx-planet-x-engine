/*
 * Python binding for the v9 bit-next Wisdom--Holman prototype.
 *
 * Reuse the v8 binding's checked buffer boundary unchanged, while mapping its
 * native entry points and public identity onto the v9 implementation.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#if defined(__linux__)
#include <sys/utsname.h>
#define JX_WH_V9_COMPILED_LINUX 1
#else
#define JX_WH_V9_COMPILED_LINUX 0
#endif

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#define JX_WH_V9_COMPILED_X86_64 1
#else
#define JX_WH_V9_COMPILED_X86_64 0
#endif

#if JX_WH_V9_COMPILED_LINUX && JX_WH_V9_COMPILED_X86_64
#include <xmmintrin.h>
#define JX_WH_V9_HAS_MXCSR 1
#else
#define JX_WH_V9_HAS_MXCSR 0
#endif

#if defined(__clang__)
#define JX_WH_V9_COMPILER_FAMILY "CLANG"
#define JX_WH_V9_COMPILER_IS_GCC 0
#elif defined(_MSC_VER)
#define JX_WH_V9_COMPILER_FAMILY "MSVC"
#define JX_WH_V9_COMPILER_IS_GCC 0
#elif defined(__GNUC__) \
    && !defined(__INTEL_COMPILER) \
    && !defined(__INTEL_LLVM_COMPILER) \
    && !defined(__NVCOMPILER) \
    && !defined(__PGI) \
    && !defined(__TINYC__)
#define JX_WH_V9_COMPILER_FAMILY "GCC"
#define JX_WH_V9_COMPILER_IS_GCC 1
#else
#define JX_WH_V9_COMPILER_FAMILY "OTHER"
#define JX_WH_V9_COMPILER_IS_GCC 0
#endif

#if defined(__BYTE_ORDER__) \
    && defined(__ORDER_LITTLE_ENDIAN__) \
    && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define JX_WH_V9_COMPILED_LITTLE_ENDIAN 1
#elif defined(_WIN32) || JX_WH_V9_COMPILED_X86_64
#define JX_WH_V9_COMPILED_LITTLE_ENDIAN 1
#else
#define JX_WH_V9_COMPILED_LITTLE_ENDIAN 0
#endif


#define JX_WH_V9_KERNEL_ID \
    "jx.wisdom-holman.native-g-bundle.bundled-series-bit-next.prototype.v9"
#define JX_WH_V9_MAP_KERNEL_ID \
    "jx.wisdom-holman.native-complete-map.bundled-series-bit-next.prototype.v9"


double jx_wh_loop_v9_nextafter_bits(double from, double toward);


static int
jx_wh_v9_dict_set_bool(PyObject *dictionary, const char *key, int value)
{
    PyObject *item = PyBool_FromLong(value != 0);
    int status;

    if (item == NULL) {
        return -1;
    }
    status = PyDict_SetItemString(dictionary, key, item);
    Py_DECREF(item);
    return status;
}


static int
jx_wh_v9_dict_set_long(PyObject *dictionary, const char *key, long value)
{
    PyObject *item = PyLong_FromLong(value);
    int status;

    if (item == NULL) {
        return -1;
    }
    status = PyDict_SetItemString(dictionary, key, item);
    Py_DECREF(item);
    return status;
}


static int
jx_wh_v9_dict_set_unsigned_long(
    PyObject *dictionary,
    const char *key,
    unsigned long value)
{
    PyObject *item = PyLong_FromUnsignedLong(value);
    int status;

    if (item == NULL) {
        return -1;
    }
    status = PyDict_SetItemString(dictionary, key, item);
    Py_DECREF(item);
    return status;
}


static int
jx_wh_v9_dict_set_string(
    PyObject *dictionary,
    const char *key,
    const char *value)
{
    PyObject *item = PyUnicode_FromString(value);
    int status;

    if (item == NULL) {
        return -1;
    }
    status = PyDict_SetItemString(dictionary, key, item);
    Py_DECREF(item);
    return status;
}


static int
jx_wh_v9_ieee_binary64_layout(void)
{
    double value;
    uint64_t bits;

    if (CHAR_BIT != 8
        || sizeof(double) != sizeof(uint64_t)
        || FLT_RADIX != 2
        || DBL_MANT_DIG != 53
        || DBL_MAX_EXP != 1024
        || DBL_MIN_EXP != -1021) {
        return 0;
    }
#ifdef DBL_HAS_SUBNORM
    if (DBL_HAS_SUBNORM != 1) {
        return 0;
    }
#endif

    value = 1.0;
    memcpy(&bits, &value, sizeof(bits));
    if (bits != UINT64_C(0x3ff0000000000000)) {
        return 0;
    }
    value = -0.0;
    memcpy(&bits, &value, sizeof(bits));
    if (bits != UINT64_C(0x8000000000000000)) {
        return 0;
    }
    value = DBL_MIN;
    memcpy(&bits, &value, sizeof(bits));
    if (bits != UINT64_C(0x0010000000000000)) {
        return 0;
    }
    value = DBL_MAX;
    memcpy(&bits, &value, sizeof(bits));
    if (bits != UINT64_C(0x7fefffffffffffff)) {
        return 0;
    }
#ifdef DBL_TRUE_MIN
    value = DBL_TRUE_MIN;
    memcpy(&bits, &value, sizeof(bits));
    if (bits != UINT64_C(0x0000000000000001)) {
        return 0;
    }
#else
    return 0;
#endif
    return 1;
}


static PyObject *
jx_wh_v9_runtime_fp_environment(PyObject *self, PyObject *ignored)
{
    PyObject *result;
    uint16_t endian_probe = UINT16_C(1);
    unsigned char endian_bytes[sizeof(endian_probe)];
    int runtime_linux = 0;
    int runtime_x86_64 = 0;
    int runtime_little_endian;
    int ieee_binary64_layout;
    int rounding_mode = -1;
    int fe_tonearest = -1;
    int round_to_nearest = 0;
    int mxcsr_available = 0;
    unsigned int mxcsr = 0;
    int mxcsr_all_exceptions_masked = 0;
    int mxcsr_rounding_control = -1;
    int mxcsr_round_to_nearest = 0;
    int mxcsr_ftz = 0;
    int mxcsr_daz = 0;
    int supported;

    (void)self;
    (void)ignored;

#if JX_WH_V9_COMPILED_LINUX
    {
        struct utsname runtime_name;
        if (uname(&runtime_name) == 0) {
            runtime_linux = strcmp(runtime_name.sysname, "Linux") == 0;
            runtime_x86_64 = (
                strcmp(runtime_name.machine, "x86_64") == 0
                || strcmp(runtime_name.machine, "amd64") == 0);
        }
    }
#endif

    memcpy(endian_bytes, &endian_probe, sizeof(endian_probe));
    runtime_little_endian = endian_bytes[0] == 1;
    ieee_binary64_layout = jx_wh_v9_ieee_binary64_layout();

#ifdef FE_TONEAREST
    rounding_mode = fegetround();
    fe_tonearest = FE_TONEAREST;
    round_to_nearest = rounding_mode == fe_tonearest;
#endif

#if JX_WH_V9_HAS_MXCSR
    mxcsr_available = 1;
    mxcsr = _mm_getcsr();
    mxcsr_all_exceptions_masked = (mxcsr & 0x1f80U) == 0x1f80U;
    mxcsr_rounding_control = (int)((mxcsr >> 13) & 3U);
    mxcsr_round_to_nearest = mxcsr_rounding_control == 0;
    mxcsr_ftz = (mxcsr & (1U << 15)) != 0;
    mxcsr_daz = (mxcsr & (1U << 6)) != 0;
#endif

    supported = (
        JX_WH_V9_COMPILER_IS_GCC
        && JX_WH_V9_COMPILED_LINUX
        && JX_WH_V9_COMPILED_X86_64
        && JX_WH_V9_COMPILED_LITTLE_ENDIAN
        && runtime_linux
        && runtime_x86_64
        && runtime_little_endian
        && ieee_binary64_layout
        && round_to_nearest
        && mxcsr_available
        && mxcsr_all_exceptions_masked
        && mxcsr_round_to_nearest
        && !mxcsr_ftz
        && !mxcsr_daz);

    result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }
    if (jx_wh_v9_dict_set_string(
            result, "compiler_family", JX_WH_V9_COMPILER_FAMILY) < 0
        || jx_wh_v9_dict_set_bool(
            result, "compiled_linux", JX_WH_V9_COMPILED_LINUX) < 0
        || jx_wh_v9_dict_set_bool(
            result, "compiled_x86_64", JX_WH_V9_COMPILED_X86_64) < 0
        || jx_wh_v9_dict_set_bool(
            result,
            "compiled_little_endian",
            JX_WH_V9_COMPILED_LITTLE_ENDIAN) < 0
        || jx_wh_v9_dict_set_bool(
            result, "runtime_linux", runtime_linux) < 0
        || jx_wh_v9_dict_set_bool(
            result, "runtime_x86_64", runtime_x86_64) < 0
        || jx_wh_v9_dict_set_bool(
            result, "runtime_little_endian", runtime_little_endian) < 0
        || jx_wh_v9_dict_set_bool(
            result, "ieee_binary64_layout", ieee_binary64_layout) < 0
        || jx_wh_v9_dict_set_long(
            result, "rounding_mode", (long)rounding_mode) < 0
        || jx_wh_v9_dict_set_long(
            result, "fe_tonearest", (long)fe_tonearest) < 0
        || jx_wh_v9_dict_set_bool(
            result, "round_to_nearest", round_to_nearest) < 0
        || jx_wh_v9_dict_set_bool(
            result, "mxcsr_available", mxcsr_available) < 0
        || jx_wh_v9_dict_set_unsigned_long(
            result, "mxcsr", (unsigned long)mxcsr) < 0
        || jx_wh_v9_dict_set_bool(
            result,
            "mxcsr_all_exceptions_masked",
            mxcsr_all_exceptions_masked) < 0
        || jx_wh_v9_dict_set_long(
            result,
            "mxcsr_rounding_control",
            (long)mxcsr_rounding_control) < 0
        || jx_wh_v9_dict_set_bool(
            result,
            "mxcsr_round_to_nearest",
            mxcsr_round_to_nearest) < 0
        || jx_wh_v9_dict_set_bool(result, "mxcsr_ftz", mxcsr_ftz) < 0
        || jx_wh_v9_dict_set_bool(result, "mxcsr_daz", mxcsr_daz) < 0
        || jx_wh_v9_dict_set_bool(result, "supported", supported) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}


static PyObject *
jx_wh_v9_nextafter_bits(PyObject *self, PyObject *args)
{
    double from;
    double toward;

    (void)self;
    if (!PyArg_ParseTuple(
            args, "dd:_nextafter_bits", &from, &toward)) {
        return NULL;
    }
    return PyFloat_FromDouble(
        jx_wh_loop_v9_nextafter_bits(from, toward));
}


static PyMethodDef jx_wh_v9_screening_methods[] = {
    {
        "_runtime_fp_environment",
        jx_wh_v9_runtime_fp_environment,
        METH_NOARGS,
        PyDoc_STR(
            "_runtime_fp_environment() -> dict\n"
            "\n"
            "Return the private closed V9 floating-point support report.")
    },
    {
        "_nextafter_bits",
        jx_wh_v9_nextafter_bits,
        METH_VARARGS,
        PyDoc_STR(
            "_nextafter_bits(from_float, toward_float) -> float\n"
            "\n"
            "Return the private v9 binary64 bit-step screening primitive.")
    },
    {NULL, NULL, 0, NULL}
};


static PyObject *
jx_wh_v9_module_create(struct PyModuleDef *definition)
{
    PyObject *module;

    definition->m_name = "_wisdom_holman_cpu_v9";
    definition->m_doc =
        "Private v9 bit-next Wisdom--Holman prototype; screening only.";
    module = PyModule_Create2(definition, PYTHON_API_VERSION);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddFunctions(module, jx_wh_v9_screening_methods) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}


static int
jx_wh_v9_add_string_constant(
    PyObject *module,
    const char *name,
    const char *value)
{
    if (strcmp(name, "KERNEL_ID") == 0) {
        value = JX_WH_V9_KERNEL_ID;
    } else if (strcmp(name, "MAP_KERNEL_ID") == 0) {
        value = JX_WH_V9_MAP_KERNEL_ID;
    }
    return PyModule_AddStringConstant(module, name, value);
}


#define jx_wh_loop_integrate_v8 jx_wh_loop_integrate_v9
#define jx_wh_loop_v8_last_fallback_count jx_wh_loop_v9_last_fallback_count
#define jx_wh_loop_v8_last_cache_entries jx_wh_loop_v9_last_cache_entries
#define jx_wh_loop_v8_last_cache_hits jx_wh_loop_v9_last_cache_hits
#define jx_wh_loop_v8_last_cache_misses jx_wh_loop_v9_last_cache_misses
#define jx_wh_loop_v8_g_bundle jx_wh_loop_v9_g_bundle
#define PyInit__wisdom_holman_cpu_v8 PyInit__wisdom_holman_cpu_v9
#undef PyModule_Create
#define PyModule_Create(definition) jx_wh_v9_module_create(definition)
#define PyModule_AddStringConstant jx_wh_v9_add_string_constant

#include "_wisdom_holman_cpu_v8_module.c"

#undef PyModule_AddStringConstant
#undef PyModule_Create
#undef PyInit__wisdom_holman_cpu_v8
#undef jx_wh_loop_v8_g_bundle
#undef jx_wh_loop_v8_last_cache_misses
#undef jx_wh_loop_v8_last_cache_hits
#undef jx_wh_loop_v8_last_cache_entries
#undef jx_wh_loop_v8_last_fallback_count
#undef jx_wh_loop_integrate_v8
