#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>

#include "_eih_1pn_core.h"

#define JX_EIH_1PN_CORE_SOURCE_SHA256 \
    "de4b173ff26aab1a5d012aca79b2687c81ce8ef97db290e48b00ca3d21f40746"
#define JX_EIH_1PN_COMPILER_FLAGS \
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
    int writable,
    const char *label)
{
    Py_ssize_t expected_bytes;
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) {
        flags |= PyBUF_WRITABLE;
    }
    if (jx_checked_product(item_count, 8, &expected_bytes) < 0) {
        return -1;
    }
    if (PyObject_GetBuffer(object, view, flags) < 0) {
        return -1;
    }
    if (view->itemsize != 8 || view->len != expected_bytes) {
        PyErr_Format(
            PyExc_BufferError,
            "%s must contain exactly %zd binary64 values",
            label,
            item_count);
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
jx_evaluate_batch(PyObject *self, PyObject *args)
{
    int batch_count;
    int body_count;
    PyObject *objects[5];
    double speed_of_light;
    double maximum_compactness;
    double maximum_speed_fraction_squared;
    Py_buffer views[5] = {{0}};
    Py_ssize_t component_count;
    Py_ssize_t batch_component_count;
    double observed_compactness = 0.0;
    double observed_speed_fraction_squared = 0.0;
    int status = JX_EIH_1PN_SUCCESS;
    int batch;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdddOO:evaluate_batch",
            &batch_count,
            &body_count,
            &objects[0],
            &objects[1],
            &objects[2],
            &speed_of_light,
            &maximum_compactness,
            &maximum_speed_fraction_squared,
            &objects[3],
            &objects[4])) {
        return NULL;
    }
    if (batch_count < 1 || batch_count > 8
        || body_count < 2 || body_count > JX_EIH_1PN_MAX_BODIES) {
        PyErr_SetString(
            PyExc_ValueError,
            "batch_count must be 1--8 and body_count must be 2--32");
        return NULL;
    }
    component_count = (Py_ssize_t)body_count * 3;
    if (jx_checked_product(
            (Py_ssize_t)batch_count,
            component_count,
            &batch_component_count) < 0) {
        return NULL;
    }
    if (jx_acquire_buffer(
            objects[0], &views[0], batch_component_count, 0, "positions") < 0
        || jx_acquire_buffer(
            objects[1], &views[1], batch_component_count, 0, "velocities") < 0
        || jx_acquire_buffer(
            objects[2], &views[2], body_count, 0,
            "gravitational_parameters") < 0
        || jx_acquire_buffer(
            objects[3], &views[3], batch_component_count, 1,
            "accelerations") < 0
        || jx_acquire_buffer(
            objects[4], &views[4], 2, 1, "diagnostics") < 0) {
        jx_release_buffers(views, 5);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    for (batch = 0; batch < batch_count; ++batch) {
        double current_compactness = 0.0;
        double current_speed_fraction_squared = 0.0;
        status = jx_eih_1pn_total_acceleration(
            body_count,
            (const double *)views[0].buf + batch * component_count,
            (const double *)views[1].buf + batch * component_count,
            (const double *)views[2].buf,
            speed_of_light,
            maximum_compactness,
            maximum_speed_fraction_squared,
            (double *)views[3].buf + batch * component_count,
            &current_compactness,
            &current_speed_fraction_squared);
        if (status != JX_EIH_1PN_SUCCESS) {
            break;
        }
        if (current_compactness > observed_compactness) {
            observed_compactness = current_compactness;
        }
        if (current_speed_fraction_squared > observed_speed_fraction_squared) {
            observed_speed_fraction_squared = current_speed_fraction_squared;
        }
    }
    Py_END_ALLOW_THREADS

    if (status == JX_EIH_1PN_SUCCESS) {
        ((double *)views[4].buf)[0] = observed_compactness;
        ((double *)views[4].buf)[1] = observed_speed_fraction_squared;
    }
    jx_release_buffers(views, 5);
    return PyLong_FromLong((long)status);
}

static PyMethodDef jx_methods[] = {
    {
        "evaluate_batch",
        jx_evaluate_batch,
        METH_VARARGS,
        PyDoc_STR("Evaluate one to eight EIH 1PN total-acceleration states.")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef jx_module = {
    PyModuleDef_HEAD_INIT,
    "_eih_1pn_cpu",
    "Private compiled EIH 1PN acceleration kernel.",
    -1,
    jx_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__eih_1pn_cpu(void)
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
    if (PyModule_AddIntConstant(
            module, "MAX_BODIES", JX_EIH_1PN_MAX_BODIES) < 0
        || PyModule_AddStringConstant(
            module, "MODEL_ID",
            "jx.eih-1pn.mutual-point-mass-total-acceleration.v1") < 0
        || PyModule_AddStringConstant(
            module, "SCIENTIFIC_CLAIM_STATE", "SCREENING_ONLY") < 0
        || PyModule_AddStringConstant(
            module, "CORE_SOURCE_SHA256", JX_EIH_1PN_CORE_SOURCE_SHA256) < 0
        || PyModule_AddStringConstant(
            module, "COMPILER_VERSION", compiler_version) < 0
        || PyModule_AddStringConstant(
            module, "COMPILER_FLAGS", JX_EIH_1PN_COMPILER_FLAGS) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
