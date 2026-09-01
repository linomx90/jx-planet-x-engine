#pragma once

#include "jx_bm6_types.hpp"

namespace jx {

inline std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];
        if (quoted) {
            if (character == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                current.push_back(character);
            }
        } else if (character == '"') {
            quoted = true;
        } else if (character == ',') {
            fields.push_back(current);
            current.clear();
        } else if (character != '\r') {
            current.push_back(character);
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated quoted CSV field");
    }
    fields.push_back(current);
    return fields;
}

inline std::map<std::string, std::size_t> header_index(
    const std::vector<std::string>& header) {
    std::map<std::string, std::size_t> result;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (!result.emplace(header[i], i).second) {
            throw std::runtime_error("duplicate CSV header: " + header[i]);
        }
    }
    return result;
}

inline std::size_t require_column(
    const std::map<std::string, std::size_t>& index,
    const std::string& name) {
    const auto iterator = index.find(name);
    if (iterator == index.end()) {
        throw std::runtime_error("missing CSV column: " + name);
    }
    return iterator->second;
}

inline int parse_integer(const std::string& text, const std::string& label) {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size()) {
        throw std::runtime_error("invalid integer for " + label);
    }
    return value;
}

inline double parse_finite_double(const std::string& text,
                                  const std::string& label) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("invalid finite double for " + label);
    }
    return value;
}

struct StateRow {
    int id{};
    std::string name;
    double epoch{};
    Vec3 position;
    Vec3 velocity;
};

inline State load_state(const std::string& state_path,
                        const std::string& gm_path) {
    std::ifstream state_input(state_path);
    if (!state_input) {
        throw std::runtime_error("cannot open state CSV: " + state_path);
    }
    std::string line;
    if (!std::getline(state_input, line)) {
        throw std::runtime_error("empty state CSV");
    }
    const auto state_columns = header_index(parse_csv_line(line));
    const auto body_id = require_column(state_columns, "body_id");
    const auto body_name = require_column(state_columns, "body_name");
    const auto epoch_column = require_column(state_columns, "jd_tdb");
    const auto x = require_column(state_columns, "x_au");
    const auto y = require_column(state_columns, "y_au");
    const auto z = require_column(state_columns, "z_au");
    const auto vx = require_column(state_columns, "vx_au_per_day");
    const auto vy = require_column(state_columns, "vy_au_per_day");
    const auto vz = require_column(state_columns, "vz_au_per_day");
    const std::size_t final_state_column =
        std::max({body_id, body_name, epoch_column, x, y, z, vx, vy, vz});

    std::vector<StateRow> rows;
    double earliest_epoch = std::numeric_limits<double>::infinity();
    while (std::getline(state_input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = parse_csv_line(line);
        if (fields.size() <= final_state_column) {
            throw std::runtime_error("short state CSV row");
        }
        StateRow row{
            parse_integer(fields[body_id], "body_id"),
            fields[body_name],
            parse_finite_double(fields[epoch_column], "jd_tdb"),
            {parse_finite_double(fields[x], "x_au"),
             parse_finite_double(fields[y], "y_au"),
             parse_finite_double(fields[z], "z_au")},
            {parse_finite_double(fields[vx], "vx_au_per_day"),
             parse_finite_double(fields[vy], "vy_au_per_day"),
             parse_finite_double(fields[vz], "vz_au_per_day")},
        };
        earliest_epoch = std::min(earliest_epoch, row.epoch);
        rows.push_back(std::move(row));
    }

    std::ifstream gm_input(gm_path);
    if (!gm_input || !std::getline(gm_input, line)) {
        throw std::runtime_error("cannot read GM CSV: " + gm_path);
    }
    const auto gm_columns = header_index(parse_csv_line(line));
    const auto gm_body_id = require_column(gm_columns, "body_id");
    const auto gm_body_name = require_column(gm_columns, "body_name");
    const auto gm_value = require_column(gm_columns, "gm_km3_s2");
    const std::size_t final_gm_column =
        std::max({gm_body_id, gm_body_name, gm_value});
    const double day_squared = kDaySeconds * kDaySeconds;
    const double au_cubed = (kAuKm * kAuKm) * kAuKm;
    std::map<int, std::pair<std::string, double>> gm;
    while (std::getline(gm_input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = parse_csv_line(line);
        if (fields.size() <= final_gm_column) {
            throw std::runtime_error("short GM CSV row");
        }
        const int id = parse_integer(fields[gm_body_id], "body_id");
        const double converted =
            parse_finite_double(fields[gm_value], "gm_km3_s2") *
            day_squared / au_cubed;
        if (!gm.emplace(id, std::make_pair(fields[gm_body_name], converted)).second) {
            throw std::runtime_error("duplicate GM body ID");
        }
    }

    std::map<int, StateRow> initial;
    for (const StateRow& row : rows) {
        if (row.epoch == earliest_epoch && !initial.emplace(row.id, row).second) {
            throw std::runtime_error("duplicate body at initial epoch");
        }
    }
    if (initial.size() != kExpectedBodies || gm.size() != kExpectedBodies) {
        throw std::runtime_error("locked benchmark requires exactly ten bodies");
    }

    State state;
    state.epoch_jd_tdb = earliest_epoch;
    for (int id = 1; id <= kExpectedBodies; ++id) {
        const auto state_iterator = initial.find(id);
        const auto gm_iterator = gm.find(id);
        if (state_iterator == initial.end() || gm_iterator == gm.end() ||
            state_iterator->second.name != gm_iterator->second.first) {
            throw std::runtime_error("locked body roster/name mismatch");
        }
        state.ids.push_back(id);
        state.names.push_back(state_iterator->second.name);
        state.mu.push_back(gm_iterator->second.second);
        state.q.push_back(state_iterator->second.position);
        state.v.push_back(state_iterator->second.velocity);
    }
    if (state.ids.back() != kSunBodyId) {
        throw std::runtime_error("Sun must be body ID 10");
    }
    return state;
}

inline std::string json_escape(std::string_view text) {
    std::ostringstream output;
    for (const unsigned char character : text) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << static_cast<char>(character); break;
        }
    }
    return output.str();
}

inline void write_trajectory(const std::string& path, const State& initial,
                             const RunResult& run) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create trajectory CSV");
    }
    output << "lane,contest,step,time_days,jd_tdb,body_id,body_name,"
              "x,y,z,vx,vy,vz,signed_relative_energy_error,"
              "relative_angular_momentum_vector_error\n";
    output << std::scientific << std::setprecision(17);
    for (const Snapshot& snapshot : run.snapshots) {
        for (std::size_t i = 0; i < initial.ids.size(); ++i) {
            output << run.lane << ',' << run.contest << ',' << snapshot.step << ','
                   << snapshot.elapsed_days << ','
                   << std::fixed << std::setprecision(9)
                   << initial.epoch_jd_tdb + snapshot.elapsed_days << ','
                   << std::scientific << std::setprecision(17)
                   << initial.ids[i] << ',' << initial.names[i] << ','
                   << snapshot.q[i].x << ',' << snapshot.q[i].y << ','
                   << snapshot.q[i].z << ',' << snapshot.v[i].x << ','
                   << snapshot.v[i].y << ',' << snapshot.v[i].z << ','
                   << snapshot.relative_energy_error << ','
                   << snapshot.relative_angular_momentum_error << '\n';
        }
    }
    if (!output) {
        throw std::runtime_error("failed while writing trajectory CSV");
    }
}

inline void write_result(const std::string& path, const State& initial,
                         const RunResult& run, const std::string& state_path,
                         const std::string& gm_path,
                         const std::string& trajectory_path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create result JSON");
    }
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"jx-bm6-native-cpp-run/v1\",\n"
           << "  \"classification\": \"MODEL_OUTPUT_NUMERICAL_ENGINEERING_ONLY\",\n"
           << "  \"lane\": \"" << json_escape(run.lane) << "\",\n"
           << "  \"contest\": \"" << json_escape(run.contest) << "\",\n"
           << "  \"arithmetic\": \"IEEE-754 binary64; no fast-math; FP contraction disabled\",\n"
           << "  \"force_model\": \"direct mutual unsoftened Newtonian point masses\",\n"
           << "  \"integrator\": \"Blanes-Moan symmetric sixth-order S10/BM6\",\n"
           << "  \"initial_epoch_jd_tdb\": " << initial.epoch_jd_tdb << ",\n"
           << "  \"body_count\": " << initial.ids.size() << ",\n"
           << "  \"dt_days\": " << run.dt_days << ",\n"
           << "  \"steps\": " << run.steps << ",\n"
           << "  \"output_every_steps\": " << run.output_every_steps << ",\n"
           << "  \"duration_days\": "
           << run.dt_days * static_cast<double>(run.steps) << ",\n"
           << "  \"force_evaluations\": " << run.force_evaluations << ",\n"
           << "  \"force_count_semantics\": \"ten measured direct force solves per macro-step\",\n"
           << "  \"trajectory_wall_seconds\": "
           << run.trajectory_wall_seconds << ",\n"
           << "  \"timing_repeats\": " << run.timing_repeats << ",\n"
           << "  \"timing_median_seconds\": "
           << run.timing_median_seconds << ",\n"
           << "  \"timing_min_seconds\": " << run.timing_min_seconds << ",\n"
           << "  \"timing_max_seconds\": " << run.timing_max_seconds << ",\n"
           << "  \"max_abs_relative_energy_error\": "
           << run.max_abs_relative_energy_error << ",\n"
           << "  \"final_signed_relative_energy_error\": "
           << run.final_signed_relative_energy_error << ",\n"
           << "  \"bounded_energy_half_range\": "
           << run.bounded_energy_half_range << ",\n"
           << "  \"max_relative_angular_momentum_vector_error\": "
           << run.max_relative_angular_momentum_vector_error << ",\n"
           << "  \"terminal_checksum\": " << run.terminal_checksum << ",\n"
           << "  \"timing_checksum\": " << run.timing_checksum << ",\n"
           << "  \"inputs\": {\"state_csv\": \""
           << json_escape(state_path) << "\", \"gm_csv\": \""
           << json_escape(gm_path) << "\"},\n"
           << "  \"trajectory_csv\": \""
           << json_escape(trajectory_path) << "\",\n"
           << "  \"coefficient_gate\": {\"exact_symmetry\": true, "
              "\"closure_tolerance\": 2e-15, \"passed\": true},\n"
           << "  \"nonclaim\": \"Native replay only; not a universal, close-encounter, arbitrary-precision, full-ephemeris, or Planet-X claim.\"\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error("failed while writing result JSON");
    }
}

}  // namespace jx
