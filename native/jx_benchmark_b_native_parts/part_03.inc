
static std::string esc(std::string_view s){std::string o;for(char c:s){if(c=='\\'||c=='"')o.push_back('\\');o.push_back(c);}return o;}
static void write_json(const std::string&path,const State&initial,const Result&r,int primary_id,int tracer_id){std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("cannot write result");o<<std::setprecision(17)<<"{\n  \"schema\": \"jx-benchmark-b-native-run/v1\",\n  \"mode\": \""<<esc(r.mode)<<"\",\n  \"duration\": "<<r.duration<<",\n  \"dt\": "<<r.dt<<",\n  \"steps\": "<<r.steps<<",\n  \"accepted_steps\": "<<r.accepted<<",\n  \"rejected_steps\": "<<r.rejected<<",\n  \"force_evaluations\": "<<r.force_evaluations<<",\n  \"wall_seconds\": "<<r.wall_seconds<<",\n  \"timing_median_seconds\": "<<r.timing_median<<",\n  \"timing_min_seconds\": "<<r.timing_min<<",\n  \"timing_max_seconds\": "<<r.timing_max<<",\n  \"primary_id\": "<<primary_id<<",\n  \"tracer_id\": "<<tracer_id<<",\n  \"min_separation\": "<<r.encounter.min_sep<<",\n  \"time_of_min_separation\": "<<r.encounter.t_min<<",\n  \"relative_speed_at_min\": "<<r.encounter.rel_speed_at_min<<",\n  \"reversibility_max_scaled\": "<<r.reversibility_max_scaled<<",\n  \"final_state\": [\n";for(std::size_t i=0;i<r.q.size();++i){o<<"    {\"body_id\": "<<initial.ids[i]<<", \"body_name\": \""<<esc(initial.names[i])<<"\", \"x\": "<<r.q[i].x<<", \"y\": "<<r.q[i].y<<", \"z\": "<<r.q[i].z<<", \"vx\": "<<r.v[i].x<<", \"vy\": "<<r.v[i].y<<", \"vz\": "<<r.v[i].z<<"}"<<(i+1<r.q.size()?",":"")<<"\n";}o<<"  ],\n  \"classification\": \"MODEL_OUTPUT_NUMERICAL_ENGINEERING_ONLY\",\n  \"nonclaim\": \"One Benchmark B encounter lane; not a universal or production qualification.\"\n}\n";if(!o)throw std::runtime_error("write failed");}

struct Opt{std::string mode,state,result;double duration{},dt{},rtol{1e-28},atol{1e-34},initial_step{},max_step{},qscale{1},vscale{1};std::size_t steps{},repeats{7};int primary_id{},tracer_id{};};
static Opt parse(int argc,char**argv){Opt o;auto nv=[&](int&i){if(++i>=argc)throw std::runtime_error("missing value");return std::string(argv[i]);};for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--mode")o.mode=nv(i);else if(a=="--state")o.state=nv(i);else if(a=="--result")o.result=nv(i);else if(a=="--duration")o.duration=std::stod(nv(i));else if(a=="--dt")o.dt=std::stod(nv(i));else if(a=="--steps")o.steps=std::stoull(nv(i));else if(a=="--primary-id")o.primary_id=std::stoi(nv(i));else if(a=="--tracer-id")o.tracer_id=std::stoi(nv(i));else if(a=="--rtol")o.rtol=std::stod(nv(i));else if(a=="--atol")o.atol=std::stod(nv(i));else if(a=="--initial-step")o.initial_step=std::stod(nv(i));else if(a=="--max-step")o.max_step=std::stod(nv(i));else if(a=="--qscale")o.qscale=std::stod(nv(i));else if(a=="--vscale")o.vscale=std::stod(nv(i));else if(a=="--timing-repeats")o.repeats=std::stoull(nv(i));else throw std::runtime_error("unknown arg "+a);}if(o.mode.empty()||o.state.empty()||o.result.empty()||!(o.duration>0)||o.primary_id==0||o.tracer_id==0)throw std::runtime_error("missing required args");if((o.mode=="bm6"||o.mode=="kdk")&&o.steps==0){if(!(o.dt>0))throw std::runtime_error("need dt/steps");o.steps=(std::size_t)std::llround(o.duration/o.dt);}if((o.mode=="bs160"||o.mode=="bs224")&&(!(o.initial_step>0)||!(o.max_step>0)))throw std::runtime_error("need BS steps");return o;}

 } // namespace jxb

int main(int argc, char** argv) {
    try {
        jxb::validate_coeff();
        const auto options = jxb::parse(argc, argv);
        const auto state = jxb::load_state(options.state);
        jxb::Result result;

        if (options.mode == "bm6" || options.mode == "kdk") {
            result = jxb::fixed_run(
                state, options.mode, options.duration, options.steps,
                options.primary_id, options.tracer_id, options.qscale,
                options.vscale, options.repeats);
#ifndef JXB_FIXED_ONLY
        } else if (options.mode == "bs160") {
            result = jxb::bs_run<160>(
                state, options.duration, options.primary_id, options.tracer_id,
                options.rtol, options.atol, options.initial_step,
                options.max_step);
        } else if (options.mode == "bs224") {
            result = jxb::bs_run<224>(
                state, options.duration, options.primary_id, options.tracer_id,
                options.rtol, options.atol, options.initial_step,
                options.max_step);
#endif
        } else {
            throw std::runtime_error("unknown or unavailable mode");
        }

        jxb::write_json(options.result, state, result, options.primary_id,
                        options.tracer_id);
        std::cout << std::setprecision(17)
                  << "JX_BENCHMARK_B_NATIVE_OK mode=" << result.mode
                  << " force_evaluations=" << result.force_evaluations
                  << " wall=" << result.wall_seconds
                  << " min_sep=" << result.encounter.min_sep << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "BLOCKED: " << error.what() << '\n';
        return 2;
    }
}
