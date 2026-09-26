#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>

#include "_gr15_eih_1pn_core.h"

#define JX_GR15_EIH_1PN_CORE_SOURCE_SHA256 \
    "bef3a607c4c3674d342745570c68b462a6178e789089c84419d6f23dd810fad1"
#define JX_EIH_1PN_FORCE_CORE_SOURCE_SHA256 \
    "de4b173ff26aab1a5d012aca79b2687c81ce8ef97db290e48b00ca3d21f40746"
#define JX_GR15_COMPILER_FLAGS \
    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"

static int
jx_checked_product(Py_ssize_t left, Py_ssize_t right, Py_ssize_t *result)
{
    if (left < 0 || right < 0 || (right != 0 && left > PY_SSIZE_T_MAX / right)) {
        PyErr_SetString(PyExc_OverflowError, "native buffer size overflow");
        return -1;
    }
    *result = left * right;
    return 0;
}

static int
jx_acquire_buffer(
    PyObject *object,
    Py_buffer *view,
    Py_ssize_t item_count,
    Py_ssize_t item_size,
    int writable,
    const char *label)
{
    Py_ssize_t expected_bytes;
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) {
        flags |= PyBUF_WRITABLE;
    }
    if (jx_checked_product(item_count, item_size, &expected_bytes) < 0) {
        return -1;
    }
    if (PyObject_GetBuffer(object, view, flags) < 0) {
        return -1;
    }
    if (view->itemsize != item_size || view->len != expected_bytes) {
        PyErr_Format(
            PyExc_BufferError,
            "%s must be a C-contiguous buffer of exactly %zd %zd-byte items",
            label,
            item_count,
            item_size);
        PyBuffer_Release(view);
        view->obj = NULL;
        return -1;
    }
    return 0;
}

static void
jx_release_buffers(Py_buffer *views, Py_ssize_t count)
{
    Py_ssize_t index;
    for (index = 0; index < count; ++index) {
        if (views[index].obj != NULL) {
            PyBuffer_Release(&views[index]);
        }
    }
}

static PyObject *
jx_integrate(PyObject *self, PyObject *args)
{
    int body_count;
    int checkpoint_count;
    PyObject *objects[11];
    double speed_of_light;
    double maximum_compactness;
    double maximum_speed_fraction_squared;
    double initial_step;
    double minimum_step;
    double maximum_step;
    double epsilon;
    double safety_factor;
    double minimum_scale_factor;
    double maximum_scale_factor;
    double convergence_factor;
    int maximum_iterations;
    long long maximum_steps;
    long long maximum_rejections;
    Py_buffer views[11] = {{0}};
    Py_ssize_t component_count;
    Py_ssize_t checkpoint_component_count;
    int status;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOOdddddddddddiLLOOOOOOO:integrate",
            &body_count,
            &checkpoint_count,
            &objects[0],
            &objects[1],
            &objects[2],
            &objects[3],
            &speed_of_light,
            &maximum_compactness,
            &maximum_speed_fraction_squared,
            &initial_step,
            &minimum_step,
            &maximum_step,
            &epsilon,
            &safety_factor,
            &minimum_scale_factor,
            &maximum_scale_factor,
            &convergence_factor,
            &maximum_iterations,
            &maximum_steps,
            &maximum_rejections,
            &objects[4],
            &objects[5],
            &objects[6],
            &objects[7],
            &objects[8],
            &objects[9],
            &objects[10])) {
        return NULL;
    }
    if (body_count < 2 || body_count > JX_GR15_MAX_BODIES) {
        PyErr_SetString(PyExc_ValueError, "body_count must be between 2 and 32");
        return NULL;
    }
    if (checkpoint_count < 2) {
        PyErr_SetString(PyExc_ValueError, "checkpoint_count must be at least two");
        return NULL;
    }
    component_count = (Py_ssize_t)body_count * 3;
    if (jx_checked_product(
            (Py_ssize_t)checkpoint_count,
            component_count,
            &checkpoint_component_count) < 0) {
        return NULL;
    }
    if (jx_acquire_buffer(
            objects[0], &views[0], component_count, 8, 0, "positions") < 0
        || jx_acquire_buffer(
            objects[1], &views[1], component_count, 8, 0, "velocities") < 0
        || jx_acquire_buffer(
            objects[2], &views[2], body_count, 8, 0, "gravitational_parameters") < 0
        || jx_acquire_buffer(
            objects[3], &views[3], checkpoint_count, 8, 0, "checkpoints") < 0
        || jx_acquire_buffer(
            objects[4], &views[4], checkpoint_component_count, 8, 1,
            "checkpoint_positions") < 0
        || jx_acquire_buffer(
            objects[5], &views[5], checkpoint_component_count, 8, 1,
            "checkpoint_velocities") < 0
        || jx_acquire_buffer(
            objects[6], &views[6], checkpoint_count, 8, 1,
            "checkpoint_accepted") < 0
        || jx_acquire_buffer(
            objects[7], &views[7], checkpoint_count, 8, 1,
            "checkpoint_rejected") < 0
        || jx_acquire_buffer(
            objects[8], &views[8], 11, 8, 1, "counters") < 0
        || jx_acquire_buffer(
            objects[9], &views[9], 5, 8, 1, "metrics") < 0
        || jx_acquire_buffer(
            objects[10], &views[10], 2, 8, 1, "force_diagnostics") < 0) {
        jx_release_buffers(views, 11);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    status = jx_gr15_eih_1pn_integrate(
        body_count,
        checkpoint_count,
        (const double *)views[0].buf,
        (const double *)views[1].buf,
        (const double *)views[2].buf,
        (const double *)views[3].buf,
        speed_of_light,
        maximum_compactness,
        maximum_speed_fraction_squared,
        initial_step,
        minimum_step,
        maximum_step,
        epsilon,
        safety_factor,
        minimum_scale_factor,
        maximum_scale_factor,
        convergence_factor,
        maximum_iterations,
        (int64_t)maximum_steps,
        (int64_t)maximum_rejections,
        (double *)views[4].buf,
        (double *)views[5].buf,
        (int64_t *)views[6].buf,
        (int64_t *)views[7].buf,
        (int64_t *)views[8].buf,
        (double *)views[9].buf,
        (double *)views[10].buf);
    Py_END_ALLOW_THREADS

    jx_release_buffers(views, 11);
    return PyLong_FromLong((long)status);
}

static PyObject *
jx_tableau(PyObject *self, PyObject *args)
{
    PyObject *objects[3];
    Py_buffer views[3] = {{0}};
    int status;

    (void)self;
    if (!PyArg_ParseTuple(
            args, "OOO:tableau", &objects[0], &objects[1], &objects[2])) {
        return NULL;
    }
    if (jx_acquire_buffer(
            objects[0], &views[0], JX_GR15_STAGE_COUNT, 8, 1, "nodes") < 0
        || jx_acquire_buffer(
            objects[1], &views[1], JX_GR15_STAGE_COUNT, 8, 1, "weights") < 0
        || jx_acquire_buffer(
            objects[2], &views[2],
            JX_GR15_STAGE_COUNT * JX_GR15_STAGE_COUNT,
            8,
            1,
            "matrix") < 0) {
        jx_release_buffers(views, 3);
        return NULL;
    }
    status = jx_gr15_eih_1pn_tableau(
        (double *)views[0].buf,
        (double *)views[1].buf,
        (double *)views[2].buf);
    jx_release_buffers(views, 3);
    return PyLong_FromLong((long)status);
}

static PyMethodDef jx_methods[] = {
    {
        "integrate",
        jx_integrate,
        METH_VARARGS,
        PyDoc_STR("Run packaged GR15 with mutual point-mass EIH 1PN gravity.")
    },
    {
        "tableau",
        jx_tableau,
        METH_VARARGS,
        PyDoc_STR("Copy the embedded GR15 nodes, weights, and matrix.")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef jx_module = {
    PyModuleDef_HEAD_INIT,
    "_gr15_eih_1pn_cpu",
    "Private compiled GR15-EIH1PN screening component.",
    -1,
    jx_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__gr15_eih_1pn_cpu(void)
{
#ifdef __VERSION__
    const char *compiler_version = __VERSION__;
#else
    const char *compiler_version = "UNKNOWN_C_COMPILER";
#endif
    PyObject *module = PyModule_Create(&jx_module);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "MAX_BODIES", JX_GR15_MAX_BODIES) < 0
        || PyModule_AddIntConstant(
            module, "STAGE_COUNT", JX_GR15_STAGE_COUNT) < 0
        || PyModule_AddStringConstant(
            module, "METHOD_ID", "JX_GR15_EIH1PN_V1") < 0
        || PyModule_AddStringConstant(
            module, "MODEL_ID", "jx.gr15.eih-1pn-mutual-point-mass.v1") < 0
        || PyModule_AddStringConstant(
            module, "SCIENTIFIC_CLAIM_STATE", "SCREENING_ONLY") < 0
        || PyModule_AddStringConstant(
            module, "CORE_SOURCE_SHA256",
            JX_GR15_EIH_1PN_CORE_SOURCE_SHA256) < 0
        || PyModule_AddStringConstant(
            module, "FORCE_CORE_SOURCE_SHA256",
            JX_EIH_1PN_FORCE_CORE_SOURCE_SHA256) < 0
        || PyModule_AddStringConstant(
            module, "COMPILER_VERSION", compiler_version) < 0
        || PyModule_AddStringConstant(
            module, "COMPILER_FLAGS", JX_GR15_COMPILER_FLAGS) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
