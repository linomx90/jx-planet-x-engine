#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>


#define JX_STAGE_COUNT 13
#define JX_MAX_BODIES 32
#define JX_CORE_SOURCE_SHA256 \
    "3952fb198482261d2ac210f3a9ef3f941af60c56c9e97bcf403b3b937c73397c"
#define JX_COMPILER_FLAGS \
    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"


int jx_smalln_cpu_rkf78_integrate(
    int body_count,
    int checkpoint_count,
    const double *input_position,
    const double *input_velocity,
    const double *gm,
    const double *checkpoints,
    const double *tableau_a,
    const double *weight_eighth,
    const double *weight_defect,
    double initial_step,
    double minimum_step,
    double maximum_step,
    double position_atol,
    double position_rtol,
    double velocity_atol,
    double velocity_rtol,
    double safety_factor,
    double minimum_scale_factor,
    double maximum_scale_factor,
    int64_t maximum_steps,
    int64_t maximum_rejections,
    int64_t ledger_capacity,
    double *checkpoint_positions,
    double *checkpoint_velocities,
    int64_t *checkpoint_accepted,
    int64_t *checkpoint_rejected,
    double *accepted_epoch_ledger,
    double *accepted_magnitude_ledger,
    int64_t *attempted_out,
    int64_t *accepted_out,
    int64_t *rejected_out,
    double *maximum_error_out);


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


static int
jx_valid_controller(
    double initial_step,
    double minimum_step,
    double maximum_step,
    double position_atol,
    double position_rtol,
    double velocity_atol,
    double velocity_rtol,
    double safety_factor,
    double minimum_scale_factor,
    double maximum_scale_factor)
{
    return isfinite(initial_step)
        && isfinite(minimum_step)
        && isfinite(maximum_step)
        && isfinite(position_atol)
        && isfinite(position_rtol)
        && isfinite(velocity_atol)
        && isfinite(velocity_rtol)
        && isfinite(safety_factor)
        && isfinite(minimum_scale_factor)
        && isfinite(maximum_scale_factor)
        && minimum_step > 0.0
        && initial_step >= minimum_step
        && maximum_step >= initial_step
        && position_atol > 0.0
        && position_rtol > 0.0
        && velocity_atol > 0.0
        && velocity_rtol > 0.0
        && safety_factor > 0.0
        && safety_factor < 1.0
        && minimum_scale_factor > 0.0
        && minimum_scale_factor <= 1.0
        && maximum_scale_factor >= 1.0
        && minimum_scale_factor <= maximum_scale_factor;
}


static PyObject *
jx_integrate(PyObject *self, PyObject *args)
{
    int body_count;
    int checkpoint_count;
    PyObject *objects[13];
    double initial_step;
    double minimum_step;
    double maximum_step;
    double position_atol;
    double position_rtol;
    double velocity_atol;
    double velocity_rtol;
    double safety_factor;
    double minimum_scale_factor;
    double maximum_scale_factor;
    long long maximum_steps;
    long long maximum_rejections;
    long long ledger_capacity;
    Py_buffer views[13] = {{0}};
    Py_ssize_t component_count;
    Py_ssize_t checkpoint_component_count;
    int64_t attempted = 0;
    int64_t accepted = 0;
    int64_t rejected = 0;
    double maximum_error = 0.0;
    int status;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOOOOOddddddddddLLLOOOOOO:integrate",
            &body_count,
            &checkpoint_count,
            &objects[0],
            &objects[1],
            &objects[2],
            &objects[3],
            &objects[4],
            &objects[5],
            &objects[6],
            &initial_step,
            &minimum_step,
            &maximum_step,
            &position_atol,
            &position_rtol,
            &velocity_atol,
            &velocity_rtol,
            &safety_factor,
            &minimum_scale_factor,
            &maximum_scale_factor,
            &maximum_steps,
            &maximum_rejections,
            &ledger_capacity,
            &objects[7],
            &objects[8],
            &objects[9],
            &objects[10],
            &objects[11],
            &objects[12])) {
        return NULL;
    }
    if (body_count < 2 || body_count > JX_MAX_BODIES) {
        PyErr_SetString(PyExc_ValueError, "body_count must be between 2 and 32");
        return NULL;
    }
    if (checkpoint_count < 2) {
        PyErr_SetString(PyExc_ValueError, "checkpoint_count must be at least two");
        return NULL;
    }
    if (maximum_steps <= 0 || maximum_rejections < 0 || ledger_capacity <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "step limits and ledger capacity are outside the supported domain");
        return NULL;
    }
    if (!jx_valid_controller(
            initial_step,
            minimum_step,
            maximum_step,
            position_atol,
            position_rtol,
            velocity_atol,
            velocity_rtol,
            safety_factor,
            minimum_scale_factor,
            maximum_scale_factor)) {
        PyErr_SetString(PyExc_ValueError, "adaptive controller values are invalid");
        return NULL;
    }

    component_count = (Py_ssize_t)body_count * 3;
    if (jx_checked_product(
            (Py_ssize_t)checkpoint_count,
            component_count,
            &checkpoint_component_count) < 0) {
        return NULL;
    }
    if (jx_acquire_buffer(objects[0], &views[0], component_count, 8, 0, "position") < 0
        || jx_acquire_buffer(objects[1], &views[1], component_count, 8, 0, "velocity") < 0
        || jx_acquire_buffer(objects[2], &views[2], body_count, 8, 0, "gm") < 0
        || jx_acquire_buffer(objects[3], &views[3], checkpoint_count, 8, 0, "checkpoints") < 0
        || jx_acquire_buffer(
            objects[4], &views[4], JX_STAGE_COUNT * JX_STAGE_COUNT, 8, 0, "tableau_a") < 0
        || jx_acquire_buffer(
            objects[5], &views[5], JX_STAGE_COUNT, 8, 0, "weight_eighth") < 0
        || jx_acquire_buffer(
            objects[6], &views[6], JX_STAGE_COUNT, 8, 0, "weight_defect") < 0
        || jx_acquire_buffer(
            objects[7], &views[7], checkpoint_component_count, 8, 1, "checkpoint_positions") < 0
        || jx_acquire_buffer(
            objects[8], &views[8], checkpoint_component_count, 8, 1, "checkpoint_velocities") < 0
        || jx_acquire_buffer(
            objects[9], &views[9], checkpoint_count, 8, 1, "checkpoint_accepted") < 0
        || jx_acquire_buffer(
            objects[10], &views[10], checkpoint_count, 8, 1, "checkpoint_rejected") < 0
        || jx_acquire_buffer(
            objects[11], &views[11], (Py_ssize_t)ledger_capacity, 8, 1, "accepted_epochs") < 0
        || jx_acquire_buffer(
            objects[12], &views[12], (Py_ssize_t)ledger_capacity, 8, 1, "accepted_magnitudes") < 0) {
        jx_release_buffers(views, 13);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    status = jx_smalln_cpu_rkf78_integrate(
        body_count,
        checkpoint_count,
        (const double *)views[0].buf,
        (const double *)views[1].buf,
        (const double *)views[2].buf,
        (const double *)views[3].buf,
        (const double *)views[4].buf,
        (const double *)views[5].buf,
        (const double *)views[6].buf,
        initial_step,
        minimum_step,
        maximum_step,
        position_atol,
        position_rtol,
        velocity_atol,
        velocity_rtol,
        safety_factor,
        minimum_scale_factor,
        maximum_scale_factor,
        (int64_t)maximum_steps,
        (int64_t)maximum_rejections,
        (int64_t)ledger_capacity,
        (double *)views[7].buf,
        (double *)views[8].buf,
        (int64_t *)views[9].buf,
        (int64_t *)views[10].buf,
        (double *)views[11].buf,
        (double *)views[12].buf,
        &attempted,
        &accepted,
        &rejected,
        &maximum_error);
    Py_END_ALLOW_THREADS

    jx_release_buffers(views, 13);
    return Py_BuildValue(
        "(iLLLd)",
        status,
        (long long)attempted,
        (long long)accepted,
        (long long)rejected,
        maximum_error);
}


static PyMethodDef jx_methods[] = {
    {
        "integrate",
        jx_integrate,
        METH_VARARGS,
        PyDoc_STR(
            "integrate(...) -> (status, attempted, accepted, rejected, max_error)\n"
            "\n"
            "Run the private screening-only 2--32 body point-mass RKF78 core "
            "against exact-size C-contiguous buffers.")
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef jx_module = {
    PyModuleDef_HEAD_INIT,
    "_smalln_cpu",
    "Private compiled small-N point-mass RKF78 kernel; screening use only.",
    -1,
    jx_methods,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC
PyInit__smalln_cpu(void)
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
    if (PyModule_AddIntConstant(module, "MAX_BODIES", JX_MAX_BODIES) < 0
        || PyModule_AddIntConstant(module, "STAGE_COUNT", JX_STAGE_COUNT) < 0
        || PyModule_AddStringConstant(module, "SCIENTIFIC_CLAIM_STATE", "SCREENING_ONLY") < 0
        || PyModule_AddStringConstant(module, "CORE_SOURCE_SHA256", JX_CORE_SOURCE_SHA256) < 0
        || PyModule_AddStringConstant(module, "COMPILER_VERSION", compiler_version) < 0
        || PyModule_AddStringConstant(module, "COMPILER_FLAGS", JX_COMPILER_FLAGS) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
