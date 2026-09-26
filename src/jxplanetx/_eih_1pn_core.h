#ifndef JX_EIH_1PN_CORE_H
#define JX_EIH_1PN_CORE_H

#define JX_EIH_1PN_MAX_BODIES 32

enum {
    JX_EIH_1PN_SUCCESS = 0,
    JX_EIH_1PN_INPUT_DOMAIN = 1,
    JX_EIH_1PN_NONFINITE = 2,
    JX_EIH_1PN_FORCE_SINGULARITY = 3,
    JX_EIH_1PN_WEAK_FIELD_DOMAIN = 4
};

int jx_eih_1pn_total_acceleration(
    int body_count,
    const double *positions,
    const double *velocities,
    const double *gravitational_parameters,
    double speed_of_light,
    double maximum_compactness,
    double maximum_speed_fraction_squared,
    double *accelerations,
    double *observed_maximum_compactness,
    double *observed_maximum_speed_fraction_squared);

#endif
