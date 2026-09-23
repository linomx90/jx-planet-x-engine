#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>


#define JX_WH_KERNEL_ID \
    "jx.wisdom-holman.native-g-bundle.endpoint-seed.prototype.v4"
#define JX_WH_EXACT_FALLBACK_KERNEL_ID \
    "jx.wisdom-holman.native-g-bundle.early-stop.prototype.v2"
#define JX_WH_MAP_KERNEL_ID \
    "jx.wisdom-holman.native-complete-map.endpoint-seed.prototype.v4"
#define JX_WH_MAP_MAX_BODIES 32
#define JX_WH_MAP_METRIC_COUNT 17
#define JX_WH_MAP_COUNTER_COUNT 12
#define JX_WH_COMPILER_FLAGS \
    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"


int jx_wh_loop_integrate_v4(
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

int64_t jx_wh_loop_v4_last_fallback_count(void);


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
    status = jx_wh_loop_integrate_v4(
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


static PyObject *
jx_wh_last_fallback_count(PyObject *self, PyObject *args)
{
    (void)self;
    if (!PyArg_ParseTuple(args, ":last_fallback_count")) {
        return NULL;
    }
    return PyLong_FromLongLong(
        (long long)jx_wh_loop_v4_last_fallback_count());
}


static PyMethodDef jx_wh_methods[] = {
    {
        "integrate_map",
        jx_wh_integrate_map,
        METH_VARARGS,
        PyDoc_STR(
            "integrate_map(...) -> status\n"
            "\n"
            "Run the private v4 endpoint-seeded Wisdom--Holman prototype.")
    },
    {
        "last_fallback_count",
        jx_wh_last_fallback_count,
        METH_VARARGS,
        PyDoc_STR(
            "last_fallback_count() -> int\n"
            "\n"
            "Return this thread's fallback count from its latest v4 map call.")
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef jx_wh_module = {
    PyModuleDef_HEAD_INIT,
    "_wisdom_holman_cpu_v4",
    "Private v4 endpoint-seeded Wisdom--Holman prototype; screening only.",
    -1,
    jx_wh_methods,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC
PyInit__wisdom_holman_cpu_v4(void)
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
    if (PyModule_AddStringConstant(module, "KERNEL_ID", JX_WH_KERNEL_ID) < 0
        || PyModule_AddStringConstant(
            module,
            "EXACT_FALLBACK_KERNEL_ID",
            JX_WH_EXACT_FALLBACK_KERNEL_ID) < 0
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
