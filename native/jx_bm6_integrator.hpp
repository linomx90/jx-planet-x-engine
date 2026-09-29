#pragma once

#include "jx_bm6_types.hpp"

namespace jx {

inline void acceleration(const State& state, std::vector<Vec3>& output) {
    const std::size_t count = state.q.size();
    if (output.size() != count) {
        output.resize(count);
    }
    std::fill(output.begin(), output.end(), Vec3{});

    for (std::size_t i = 0; i + 1 < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            const Vec3 separation = state.q[j] - state.q[i];
            const double radius_squared = dot(separation, separation);
            if (!(radius_squared > 0.0) || !std::isfinite(radius_squared)) {
                throw std::runtime_error("invalid pair separation");
            }
            const double inverse_radius_cubed =
                1.0 / (radius_squared * std::sqrt(radius_squared));
            const Vec3 direction = inverse_radius_cubed * separation;
            output[i] += state.mu[j] * direction;
            output[j] -= state.mu[i] * direction;
        }
    }
}

inline Invariants invariants(const State& state) {
    std::vector<double> energy_terms;
    energy_terms.reserve(state.q.size() * state.q.size());
    for (std::size_t i = 0; i < state.q.size(); ++i) {
        energy_terms.push_back(0.5 * state.mu[i] * dot(state.v[i], state.v[i]));
    }
    for (std::size_t i = 0; i + 1 < state.q.size(); ++i) {
        for (std::size_t j = i + 1; j < state.q.size(); ++j) {
            energy_terms.push_back(
                -state.mu[i] * state.mu[j] / norm(state.q[j] - state.q[i]));
        }
    }

    Vec3 angular_momentum{};
    for (std::size_t i = 0; i < state.q.size(); ++i) {
        angular_momentum += state.mu[i] * cross(state.q[i], state.v[i]);
    }
    return {stable_sum(energy_terms), angular_momentum};
}

inline void bm6_step(State& state, double dt, const Coefficients& coefficients,
                     std::vector<Vec3>& acceleration_buffer,
                     std::uint64_t& force_evaluations) {
    for (std::size_t stage = 0; stage < coefficients.kick.size(); ++stage) {
        const double drift = dt * coefficients.drift[stage];
        for (std::size_t i = 0; i < state.q.size(); ++i) {
            state.q[i].x += drift * state.v[i].x;
            state.q[i].y += drift * state.v[i].y;
            state.q[i].z += drift * state.v[i].z;
        }

        acceleration(state, acceleration_buffer);
        ++force_evaluations;

        const double kick = dt * coefficients.kick[stage];
        for (std::size_t i = 0; i < state.v.size(); ++i) {
            state.v[i].x += kick * acceleration_buffer[i].x;
            state.v[i].y += kick * acceleration_buffer[i].y;
            state.v[i].z += kick * acceleration_buffer[i].z;
        }
    }

    const double final_drift = dt * coefficients.drift.back();
    for (std::size_t i = 0; i < state.q.size(); ++i) {
        state.q[i].x += final_drift * state.v[i].x;
        state.q[i].y += final_drift * state.v[i].y;
        state.q[i].z += final_drift * state.v[i].z;
    }
    state.elapsed_days += dt;
}

inline double state_checksum(const State& state) {
    std::vector<double> terms;
    terms.reserve(state.q.size() * 6);
    for (std::size_t i = 0; i < state.q.size(); ++i) {
        const double weight = static_cast<double>(i + 1);
        terms.push_back(weight * state.q[i].x);
        terms.push_back(weight * state.q[i].y);
        terms.push_back(weight * state.q[i].z);
        terms.push_back(weight * state.v[i].x);
        terms.push_back(weight * state.v[i].y);
        terms.push_back(weight * state.v[i].z);
    }
    return stable_sum(terms);
}

inline Snapshot capture(const State& state, std::size_t step,
                        double exact_elapsed_days,
                        const Invariants& initial) {
    const Invariants current = invariants(state);
    const double relative_energy =
        (current.energy - initial.energy) / std::fabs(initial.energy);
    const double relative_angular =
        norm(current.angular_momentum - initial.angular_momentum) /
        norm(initial.angular_momentum);
    if (!std::isfinite(relative_energy) || !std::isfinite(relative_angular)) {
        throw std::runtime_error("non-finite invariant diagnostic");
    }
    return {step, exact_elapsed_days, state.q, state.v,
            relative_energy, relative_angular};
}

inline RunResult run_authoritative(const State& initial_state,
                                   const std::string& contest, double dt,
                                   std::size_t steps,
                                   std::size_t output_every_steps,
                                   std::size_t timing_repeats) {
    if (!std::isfinite(dt) || dt == 0.0 || steps == 0 ||
        output_every_steps == 0 || steps % output_every_steps != 0 ||
        timing_repeats == 0) {
        throw std::runtime_error("invalid integration schedule");
    }

    const Coefficients coefficients = bm6_coefficients();
    validate_coefficients(coefficients);
    const Invariants initial_invariants = invariants(initial_state);

    State state = initial_state;
    std::vector<Vec3> acceleration_buffer(state.q.size());
    std::uint64_t force_evaluations = 0;
    std::vector<Snapshot> snapshots;
    snapshots.reserve(steps / output_every_steps + 1);
    snapshots.push_back(capture(state, 0, 0.0, initial_invariants));

    const auto trajectory_start = std::chrono::steady_clock::now();
    for (std::size_t step = 1; step <= steps; ++step) {
        bm6_step(state, dt, coefficients, acceleration_buffer, force_evaluations);
        if (step % output_every_steps == 0) {
            snapshots.push_back(capture(
                state, step, static_cast<double>(step) * dt, initial_invariants));
        }
    }
    const double trajectory_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - trajectory_start).count();

    std::vector<double> timings;
    timings.reserve(timing_repeats);
    double timing_checksum = 0.0;
    for (std::size_t repeat = 0; repeat < timing_repeats; ++repeat) {
        State timed_state = initial_state;
        std::vector<Vec3> timed_acceleration(timed_state.q.size());
        std::uint64_t timed_force_evaluations = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < steps; ++step) {
            bm6_step(timed_state, dt, coefficients, timed_acceleration,
                     timed_force_evaluations);
        }
        timings.push_back(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count());
        if (timed_force_evaluations != 10ULL * steps) {
            throw std::runtime_error("force-evaluation accounting failure");
        }
        timing_checksum +=
            static_cast<double>(repeat + 1) * state_checksum(timed_state);
    }
    std::sort(timings.begin(), timings.end());

    double maximum_energy_error = 0.0;
    double minimum_energy = std::numeric_limits<double>::infinity();
    double maximum_energy = -std::numeric_limits<double>::infinity();
    double maximum_angular_error = 0.0;
    for (const Snapshot& snapshot : snapshots) {
        maximum_energy_error = std::max(
            maximum_energy_error, std::fabs(snapshot.relative_energy_error));
        minimum_energy = std::min(minimum_energy, snapshot.relative_energy_error);
        maximum_energy = std::max(maximum_energy, snapshot.relative_energy_error);
        maximum_angular_error = std::max(
            maximum_angular_error, snapshot.relative_angular_momentum_error);
    }

    return {
        "bm6_native_cpp",
        contest,
        dt,
        steps,
        output_every_steps,
        force_evaluations,
        trajectory_wall_seconds,
        timing_repeats,
        timings[timings.size() / 2],
        timings.front(),
        timings.back(),
        maximum_energy_error,
        snapshots.back().relative_energy_error,
        0.5 * (maximum_energy - minimum_energy),
        maximum_angular_error,
        state_checksum(state),
        timing_checksum,
        std::move(snapshots),
    };
}

}  // namespace jx
