#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __FAST_MATH__
#error "JX BM6 native benchmark forbids fast-math"
#endif

static_assert(sizeof(double) == 8, "JX BM6 requires 64-bit double");
static_assert(std::numeric_limits<double>::is_iec559,
              "JX BM6 requires IEEE-754 double");

namespace jx {

constexpr double kAuKm = 149597870.700;
constexpr double kDaySeconds = 86400.0;
constexpr int kExpectedBodies = 10;
constexpr int kSunBodyId = 10;

struct Vec3 {
    double x{};
    double y{};
    double z{};
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(double scale, const Vec3& value) noexcept {
    return {scale * value.x, scale * value.y, scale * value.z};
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) noexcept {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline Vec3& operator-=(Vec3& a, const Vec3& b) noexcept {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline double dot(const Vec3& a, const Vec3& b) noexcept {
    return (a.x * b.x + a.y * b.y) + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline double norm(const Vec3& value) noexcept {
    return std::sqrt(dot(value, value));
}

struct State {
    double epoch_jd_tdb{};
    double elapsed_days{};
    std::vector<int> ids;
    std::vector<std::string> names;
    std::vector<double> mu;
    std::vector<Vec3> q;
    std::vector<Vec3> v;
};

struct Snapshot {
    std::size_t step{};
    double elapsed_days{};
    std::vector<Vec3> q;
    std::vector<Vec3> v;
    double relative_energy_error{};
    double relative_angular_momentum_error{};
};

struct Invariants {
    double energy{};
    Vec3 angular_momentum{};
};

struct RunResult {
    std::string lane{"bm6_native_cpp"};
    std::string contest;
    double dt_days{};
    std::size_t steps{};
    std::size_t output_every_steps{};
    std::uint64_t force_evaluations{};
    double trajectory_wall_seconds{};
    std::size_t timing_repeats{};
    double timing_median_seconds{};
    double timing_min_seconds{};
    double timing_max_seconds{};
    double max_abs_relative_energy_error{};
    double final_signed_relative_energy_error{};
    double bounded_energy_half_range{};
    double max_relative_angular_momentum_vector_error{};
    double terminal_checksum{};
    double timing_checksum{};
    std::vector<Snapshot> snapshots;
};

struct Coefficients {
    std::array<double, 11> drift{};
    std::array<double, 10> kick{};
};

inline Coefficients bm6_coefficients() {
    const double a1 = 0.0502627644003922;
    const double a2 = 0.413514300428344;
    const double a3 = 0.0450798897943977;
    const double a4 = -0.188054853819569;
    const double a5 = 0.541960678450780;
    const double a6 = 1.0 - 2.0 * ((((a1 + a2) + a3) + a4) + a5);

    const double b1 = 0.148816447901042;
    const double b2 = -0.132385865767784;
    const double b3 = 0.067307604692185;
    const double b4 = 0.432666402578175;
    const double b5 = 0.5 - (((b1 + b2) + b3) + b4);

    return {
        {a1, a2, a3, a4, a5, a6, a5, a4, a3, a2, a1},
        {b1, b2, b3, b4, b5, b5, b4, b3, b2, b1},
    };
}

inline void validate_coefficients(const Coefficients& coefficients) {
    for (std::size_t i = 0; i < coefficients.drift.size(); ++i) {
        if (coefficients.drift[i] !=
            coefficients.drift[coefficients.drift.size() - 1 - i]) {
            throw std::runtime_error("BM6 drift coefficients are not symmetric");
        }
    }
    for (std::size_t i = 0; i < coefficients.kick.size(); ++i) {
        if (coefficients.kick[i] !=
            coefficients.kick[coefficients.kick.size() - 1 - i]) {
            throw std::runtime_error("BM6 kick coefficients are not symmetric");
        }
    }
    double drift_sum = 0.0;
    for (double value : coefficients.drift) {
        drift_sum += value;
    }
    double kick_sum = 0.0;
    for (double value : coefficients.kick) {
        kick_sum += value;
    }
    if (std::fabs(drift_sum - 1.0) > 2e-15 ||
        std::fabs(kick_sum - 1.0) > 2e-15) {
        throw std::runtime_error("BM6 coefficient closure failure");
    }
}

inline double stable_sum(const std::vector<double>& values) {
    // Shewchuk-style partial summation: the same purpose as Python math.fsum.
    std::vector<double> partials;
    partials.reserve(values.size());
    for (double x : values) {
        std::size_t used = 0;
        for (double y : partials) {
            if (std::fabs(x) < std::fabs(y)) {
                std::swap(x, y);
            }
            const double high = x + y;
            const double low = y - (high - x);
            if (low != 0.0) {
                partials[used++] = low;
            }
            x = high;
        }
        partials.resize(used);
        if (x != 0.0) {
            partials.push_back(x);
        }
    }
    double total = 0.0;
    for (auto it = partials.rbegin(); it != partials.rend(); ++it) {
        total += *it;
    }
    return total;
}

}  // namespace jx
