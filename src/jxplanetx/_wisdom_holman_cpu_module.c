#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>
#include <string.h>


#define JX_WH_SERIES_TERM_COUNT 64
#define JX_WH_SERIES_SWITCH_ABS_ARGUMENT 0.5
#define JX_WH_KERNEL_ID "jx.wisdom-holman.native-g-bundle.prototype.v1"
#define JX_WH_MAP_KERNEL_ID "jx.wisdom-holman.native-complete-map.prototype.v1"
#define JX_WH_MAP_MAX_BODIES 32
#define JX_WH_MAP_METRIC_COUNT 17
#define JX_WH_MAP_COUNTER_COUNT 12
#define JX_WH_COMPILER_FLAGS \
    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"


int jx_wh_loop_integrate(
    int body_count,
    int checkpoint_count,
    const double *input_positions,
    const double *input_velocities,
    const double *gm,
    const double *radii,
    const int64_t *checkpoint_steps,
    double fixed_step,
    double minimum_encounter,
    double minimum_periapse,
    double maximum_barycenter_position,
    double maximum_barycenter_velocity,
    double *checkpoint_positions,
    double *checkpoint_velocities,
    double *metrics,
    int64_t *counters);


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
    int signed_integer,
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
    if (view->itemsize != item_size
        || view->len != expected_bytes
        || view->format == NULL
        || (signed_integer
            ? (strcmp(view->format, "l") != 0
               && strcmp(view->format, "q") != 0)
            : strcmp(view->format, "d") != 0)) {
        PyErr_Format(
            PyExc_BufferError,
            "%s must be a native C-contiguous %s buffer of exactly %zd items",
            label,
            signed_integer ? "signed int64" : "float64",
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


static int
jx_wh_g_series(
    int order,
    double beta,
    double anomaly,
    double *result)
{
    double term;
    double total;
    double rho;
    int series_index;

    switch (order) {
        case 0:
            term = 1.0;
            break;
        case 1:
            term = anomaly;
            break;
        case 2:
            term = (anomaly * anomaly) / 2.0;
            break;
        case 3:
            term = ((anomaly * anomaly) * anomaly) / 6.0;
            break;
        default:
            return -1;
    }
    if (!isfinite(term)) {
        return 1;
    }
    rho = ((-beta) * anomaly) * anomaly;
    total = term;
    for (series_index = 0;
         series_index < JX_WH_SERIES_TERM_COUNT - 1;
         ++series_index) {
        const int first = order + 2 * series_index + 1;
        const int second = first + 1;
        term = (term * rho) / (double)(first * second);
        total = total + term;
        if (!isfinite(term) || !isfinite(total)) {
            return series_index + 2;
        }
    }
    *result = total;
    return 0;
}


static PyObject *
jx_wh_series_g_bundle(PyObject *self, PyObject *args)
{
    double beta;
    double anomaly;
    double root_beta;
    double argument;
    double values[4];
    int order;

    (void)self;
    if (!PyArg_ParseTuple(args, "dd:series_g_bundle", &beta, &anomaly)) {
        return NULL;
    }
    if (!isfinite(beta) || beta <= 0.0 || !isfinite(anomaly)) {
        PyErr_SetString(
            PyExc_ValueError,
            "series_g_bundle requires finite anomaly and positive finite beta");
        return NULL;
    }
    root_beta = sqrt(beta);
    argument = root_beta * anomaly;
    if (!isfinite(argument)
        || fabs(argument) > JX_WH_SERIES_SWITCH_ABS_ARGUMENT) {
        PyErr_SetString(
            PyExc_ValueError,
            "series_g_bundle is restricted to the fixed series branch");
        return NULL;
    }
    for (order = 0; order < 4; ++order) {
        const int status = jx_wh_g_series(order, beta, anomaly, &values[order]);
        if (status != 0) {
            PyErr_Format(
                PyExc_FloatingPointError,
                "universal G series became nonfinite at order %d term %d",
                order,
                status);
            return NULL;
        }
    }
    return Py_BuildValue("(dddd)", values[0], values[1], values[2], values[3]);
}


static PyObject *
jx_wh_integrate_map(PyObject *self, PyObject *args)
{
    int body_count;
    int checkpoint_count;
    PyObject *objects[9];
    double fixed_step;
    double minimum_encounter;
    double minimum_periapse;
    double maximum_barycenter_position;
    double maximum_barycenter_velocity;
    Py_buffer views[9] = {{0}};
    Py_ssize_t component_count;
    Py_ssize_t checkpoint_component_count;
    int status;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOOOdddddOOOO:integrate_map",
            &body_count,
            &checkpoint_count,
            &objects[0],
            &objects[1],
            &objects[2],
            &objects[3],
            &objects[4],
            &fixed_step,
            &minimum_encounter,
            &minimum_periapse,
            &maximum_barycenter_position,
            &maximum_barycenter_velocity,
            &objects[5],
            &objects[6],
            &objects[7],
            &objects[8])) {
        return NULL;
    }
    if (body_count < 2 || body_count > JX_WH_MAP_MAX_BODIES) {
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
            objects[0], &views[0], component_count, 8, 0, 0, "positions") < 0
        || jx_acquire_buffer(
            objects[1], &views[1], component_count, 8, 0, 0, "velocities") < 0
        || jx_acquire_buffer(
            objects[2], &views[2], body_count, 8, 0, 0, "gm") < 0
        || jx_acquire_buffer(
            objects[3], &views[3], body_count, 8, 0, 0, "radii") < 0
        || jx_acquire_buffer(
            objects[4], &views[4], checkpoint_count, 8, 0, 1, "checkpoint_steps") < 0
        || jx_acquire_buffer(
            objects[5],
            &views[5],
            checkpoint_component_count,
            8,
            1,
            0,
            "checkpoint_positions") < 0
        || jx_acquire_buffer(
            objects[6],
            &views[6],
            checkpoint_component_count,
            8,
            1,
            0,
            "checkpoint_velocities") < 0
        || jx_acquire_buffer(
            objects[7], &views[7], JX_WH_MAP_METRIC_COUNT, 8, 1, 0, "metrics") < 0
        || jx_acquire_buffer(
            objects[8], &views[8], JX_WH_MAP_COUNTER_COUNT, 8, 1, 1, "counters") < 0) {
        jx_release_buffers(views, 9);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    status = jx_wh_loop_integrate(
        body_count,
        checkpoint_count,
        (const double *)views[0].buf,
        (const double *)views[1].buf,
        (const double *)views[2].buf,
        (const double *)views[3].buf,
        (const int64_t *)views[4].buf,
        fixed_step,
        minimum_encounter,
        minimum_periapse,
        maximum_barycenter_position,
        maximum_barycenter_velocity,
        (double *)views[5].buf,
        (double *)views[6].buf,
        (double *)views[7].buf,
        (int64_t *)views[8].buf);
    Py_END_ALLOW_THREADS

    jx_release_buffers(views, 9);
    return PyLong_FromLong((long)status);
}


static PyMethodDef jx_wh_methods[] = {
    {
        "integrate_map",
        jx_wh_integrate_map,
        METH_VARARGS,
        PyDoc_STR(
            "integrate_map(...) -> status\n"
            "\n"
            "Run the private screening-only complete fixed-step "
            "Wisdom--Holman numerical loop against exact-size buffers.")
    },
    {
        "series_g_bundle",
        jx_wh_series_g_bundle,
        METH_VARARGS,
        PyDoc_STR(
            "series_g_bundle(beta, anomaly) -> (g0, g1, g2, g3)\n"
            "\n"
            "Evaluate the fixed 64-term universal-G series bundle using the "
            "same operation order as the Python screening implementation.")
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef jx_wh_module = {
    PyModuleDef_HEAD_INIT,
    "_wisdom_holman_cpu",
    "Private native Wisdom--Holman kernel prototype; screening use only.",
    -1,
    jx_wh_methods,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC
PyInit__wisdom_holman_cpu(void)
{
#ifdef __VERSION__
    const char *compiler_version = __VERSION__;
#else
    const char *compiler_version = "UNKNOWN_C_COMPILER";
#endif
    PyObject *module = PyModule_Create(&jx_wh_module);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(
            module, "SERIES_TERM_COUNT", JX_WH_SERIES_TERM_COUNT) < 0
        || PyModule_AddStringConstant(module, "KERNEL_ID", JX_WH_KERNEL_ID) < 0
        || PyModule_AddStringConstant(
            module, "MAP_KERNEL_ID", JX_WH_MAP_KERNEL_ID) < 0
        || PyModule_AddIntConstant(
            module, "MAP_MAX_BODIES", JX_WH_MAP_MAX_BODIES) < 0
        || PyModule_AddIntConstant(
            module, "MAP_METRIC_COUNT", JX_WH_MAP_METRIC_COUNT) < 0
        || PyModule_AddIntConstant(
            module, "MAP_COUNTER_COUNT", JX_WH_MAP_COUNTER_COUNT) < 0
        || PyModule_AddStringConstant(
            module, "SCIENTIFIC_CLAIM_STATE", "SCREENING_ONLY") < 0
        || PyModule_AddStringConstant(
            module, "COMPILER_VERSION", compiler_version) < 0
        || PyModule_AddStringConstant(
            module, "COMPILER_FLAGS", JX_WH_COMPILER_FLAGS) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
