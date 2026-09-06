#include "jx_bm6_integrator.hpp"
#include "jx_bm6_io.hpp"

namespace jx {

struct Options {
    bool self_test{};
    std::string state_path;
    std::string gm_path;
    std::string trajectory_path;
    std::string result_path;
    std::string contest{"equal_force_budget"};
    double dt_days{365.25 / 294.0};
    std::size_t steps{2940};
    std::size_t output_every_steps{294};
    std::size_t timing_repeats{21};
};

inline void print_usage(std::ostream& output, const char* program) {
    output << "Usage:\n"
           << "  " << program << " --self-test\n"
           << "  " << program
           << " --state FILE --gm FILE --trajectory FILE --result FILE"
              " [--contest NAME] [--dt-days X] [--steps N]"
              " [--output-every-steps N] [--timing-repeats N]\n";
}

inline Options parse_options(int argc, char** argv) {
    Options options;
    auto next_value = [&](int& index, const std::string& option) {
        if (++index >= argc) {
            throw std::runtime_error("missing value for " + option);
        }
        return std::string(argv[index]);
    };

    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--self-test") {
            options.self_test = true;
        } else if (option == "--state") {
            options.state_path = next_value(index, option);
        } else if (option == "--gm") {
            options.gm_path = next_value(index, option);
        } else if (option == "--trajectory") {
            options.trajectory_path = next_value(index, option);
        } else if (option == "--result") {
            options.result_path = next_value(index, option);
        } else if (option == "--contest") {
            options.contest = next_value(index, option);
        } else if (option == "--dt-days") {
            options.dt_days = parse_finite_double(next_value(index, option), option);
        } else if (option == "--steps") {
            options.steps = std::stoull(next_value(index, option));
        } else if (option == "--output-every-steps") {
            options.output_every_steps = std::stoull(next_value(index, option));
        } else if (option == "--timing-repeats") {
            options.timing_repeats = std::stoull(next_value(index, option));
        } else if (option == "--help" || option == "-h") {
            print_usage(std::cout, argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + option);
        }
    }

    if (!options.self_test &&
        (options.state_path.empty() || options.gm_path.empty() ||
         options.trajectory_path.empty() || options.result_path.empty())) {
        throw std::runtime_error("state, GM, trajectory, and result paths are required");
    }
    return options;
}

inline int self_test() {
    const Coefficients coefficients = bm6_coefficients();
    validate_coefficients(coefficients);

    State initial;
    initial.ids = {1, 2};
    initial.names = {"center", "tracer"};
    initial.mu = {1.0, 0.0};
    initial.q = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    initial.v = {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};

    State state = initial;
    std::vector<Vec3> acceleration_buffer(2);
    std::uint64_t force_evaluations = 0;
    constexpr double dt = 0.001;
    for (int step = 0; step < 1000; ++step) {
        bm6_step(state, dt, coefficients, acceleration_buffer, force_evaluations);
    }
    for (int step = 0; step < 1000; ++step) {
        bm6_step(state, -dt, coefficients, acceleration_buffer, force_evaluations);
    }

    double maximum_error = 0.0;
    for (std::size_t i = 0; i < state.q.size(); ++i) {
        const std::array<double, 6> errors = {
            std::fabs(state.q[i].x - initial.q[i].x),
            std::fabs(state.q[i].y - initial.q[i].y),
            std::fabs(state.q[i].z - initial.q[i].z),
            std::fabs(state.v[i].x - initial.v[i].x),
            std::fabs(state.v[i].y - initial.v[i].y),
            std::fabs(state.v[i].z - initial.v[i].z),
        };
        for (double error : errors) {
            if (!std::isfinite(error)) {
                throw std::runtime_error("self-test produced a non-finite state");
            }
            maximum_error = std::max(maximum_error, error);
        }
    }

    if (force_evaluations != 20000 || maximum_error > 5e-12) {
        throw std::runtime_error("BM6 forward/backward self-test failed");
    }
    std::cout << std::setprecision(17)
              << "SELF_TEST_PASS force_calls=" << force_evaluations
              << " return_max_component_error=" << maximum_error << '\n';
    return 0;
}

}  // namespace jx

int main(int argc, char** argv) {
    try {
        const jx::Options options = jx::parse_options(argc, argv);
        if (options.self_test) {
            return jx::self_test();
        }
        const jx::State state =
            jx::load_state(options.state_path, options.gm_path);
        const jx::RunResult run = jx::run_authoritative(
            state, options.contest, options.dt_days, options.steps,
            options.output_every_steps, options.timing_repeats);
        jx::write_trajectory(options.trajectory_path, state, run);
        jx::write_result(options.result_path, state, run, options.state_path,
                         options.gm_path, options.trajectory_path);
        std::cout << std::setprecision(17)
                  << "JX_BM6_NATIVE_CPP_OK contest=" << run.contest
                  << " steps=" << run.steps
                  << " force_evaluations=" << run.force_evaluations
                  << " median_seconds=" << run.timing_median_seconds
                  << " max_rel_energy=" << run.max_abs_relative_energy_error
                  << " max_rel_L="
                  << run.max_relative_angular_momentum_vector_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "BLOCKED: " << error.what() << '\n';
        jx::print_usage(std::cerr, argc > 0 ? argv[0] : "jx_bm6_native");
        return 2;
    }
}
