#ifndef JX_GR15_CORE_H
#define JX_GR15_CORE_H

#include <stdint.h>

#define JX_GR15_STAGE_COUNT 8
#define JX_GR15_MAX_BODIES 32

enum {
    JX_GR15_SUCCESS = 0,
    JX_GR15_INPUT_DOMAIN = 1,
    JX_GR15_NONFINITE = 2,
    JX_GR15_FORCE_SINGULARITY = 3,
    JX_GR15_STEP_LIMIT = 4,
    JX_GR15_REJECTION_LIMIT = 5,
    JX_GR15_MINIMUM_STEP = 6,
    JX_GR15_CHECKPOINT_SCHEDULE = 7
};

int jx_gr15_tableau(double *nodes, double *weights, double *matrix);

int jx_gr15_integrate(
    int body_count,
    int checkpoint_count,
    const double *input_position,
    const double *input_velocity,
    const double *gm,
    const double *checkpoints,
    double initial_step,
    double minimum_step,
    double maximum_step,
    double epsilon,
    double safety_factor,
    double minimum_scale_factor,
    double maximum_scale_factor,
    double convergence_factor,
    int maximum_iterations,
    int64_t maximum_steps,
    int64_t maximum_rejections,
    double *checkpoint_positions,
    double *checkpoint_velocities,
    int64_t *checkpoint_accepted,
    int64_t *checkpoint_rejected,
    int64_t *counters,
    double *metrics);

#endif
