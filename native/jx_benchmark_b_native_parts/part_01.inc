#include <algorithm>
#include <array>
#ifndef JXB_FIXED_ONLY
#include <boost/multiprecision/cpp_bin_float.hpp>
#endif
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __FAST_MATH__
#error "JX Benchmark B forbids fast-math"
#endif
static_assert(sizeof(double) == 8);
static_assert(std::numeric_limits<double>::is_iec559);

namespace jxb {

struct Vec3 { double x{}, y{}, z{}; };
inline Vec3 operator+(const Vec3& a,const Vec3& b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline Vec3 operator-(const Vec3& a,const Vec3& b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline Vec3 operator*(double s,const Vec3& a){return {s*a.x,s*a.y,s*a.z};}
inline Vec3& operator+=(Vec3& a,const Vec3& b){a.x+=b.x;a.y+=b.y;a.z+=b.z;return a;}
inline Vec3& operator-=(Vec3& a,const Vec3& b){a.x-=b.x;a.y-=b.y;a.z-=b.z;return a;}
inline double dot(const Vec3&a,const Vec3&b){return (a.x*b.x+a.y*b.y)+a.z*b.z;}
inline double norm(const Vec3&a){return std::sqrt(dot(a,a));}

struct State {
    std::vector<int> ids;
    std::vector<std::string> names;
    std::vector<double> mu;
    std::vector<Vec3> q,v;
};
struct Sample { double t{}, d2{}, rel_speed{}; };
struct Metrics {
    double min_sep{}, t_min{}, rel_speed_at_min{};
};
struct Result {
    std::string mode;
    double duration{}, dt{};
    std::size_t steps{}, accepted{}, rejected{};
    std::uint64_t force_evaluations{};
    double wall_seconds{}, timing_median{}, timing_min{}, timing_max{};
    Metrics encounter;
    double reversibility_max_scaled{};
    std::vector<Vec3> q,v;
};

static std::vector<std::string> split_csv(const std::string& line){
    std::vector<std::string> out; std::string cur; bool quote=false;
    for(std::size_t i=0;i<line.size();++i){char c=line[i];
        if(quote){ if(c=='"'){ if(i+1<line.size()&&line[i+1]=='"'){cur.push_back('"');++i;} else quote=false;} else cur.push_back(c); }
        else if(c=='"') quote=true; else if(c==','){out.push_back(cur);cur.clear();} else if(c!='\r') cur.push_back(c);
    }
    if (quote) { throw std::runtime_error("unterminated CSV quote"); }
    out.push_back(cur);
    return out;
}
static std::map<std::string,std::size_t> header_map(const std::vector<std::string>& h){std::map<std::string,std::size_t> m;for(std::size_t i=0;i<h.size();++i)if(!m.emplace(h[i],i).second)throw std::runtime_error("duplicate header");return m;}
static std::size_t col(const std::map<std::string,std::size_t>&m,const char* n){auto it=m.find(n);if(it==m.end())throw std::runtime_error(std::string("missing column ")+n);return it->second;}
static double finite_double(const std::string&s,const char*label){std::size_t u=0;double x=std::stod(s,&u);if(u!=s.size()||!std::isfinite(x))throw std::runtime_error(std::string("invalid ")+label);return x;}
static int integer(const std::string&s,const char*label){std::size_t u=0;int x=std::stoi(s,&u);if(u!=s.size())throw std::runtime_error(std::string("invalid ")+label);return x;}

static State load_state(const std::string& path){
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open state"); std::string line; if(!std::getline(f,line))throw std::runtime_error("empty state");
    auto hm=header_map(split_csv(line)); auto cid=col(hm,"body_id"), cn=col(hm,"body_name"), cm=col(hm,"mu"), cx=col(hm,"x"), cy=col(hm,"y"), cz=col(hm,"z"), cvx=col(hm,"vx"), cvy=col(hm,"vy"), cvz=col(hm,"vz");
    std::size_t mx=std::max({cid,cn,cm,cx,cy,cz,cvx,cvy,cvz}); State s; std::map<int,bool> seen;
    while(std::getline(f,line)){if(line.empty())continue;auto a=split_csv(line);if(a.size()<=mx)throw std::runtime_error("short row");int id=integer(a[cid],"body_id");if(seen[id])throw std::runtime_error("duplicate body_id");seen[id]=true;
        s.ids.push_back(id);s.names.push_back(a[cn]);s.mu.push_back(finite_double(a[cm],"mu"));s.q.push_back({finite_double(a[cx],"x"),finite_double(a[cy],"y"),finite_double(a[cz],"z")});s.v.push_back({finite_double(a[cvx],"vx"),finite_double(a[cvy],"vy"),finite_double(a[cvz],"vz")});}
    if (s.ids.size() < 2 || s.ids.size() != s.mu.size()) {
        throw std::runtime_error("invalid state size");
    }
    return s;
}
static std::size_t index_of(const State&s,int id){for(std::size_t i=0;i<s.ids.size();++i)if(s.ids[i]==id)return i;throw std::runtime_error("body id absent");}

static void acceleration(const State&s,std::vector<Vec3>&a,std::uint64_t&calls){a.assign(s.q.size(),{});for(std::size_t i=0;i+1<s.q.size();++i)for(std::size_t j=i+1;j<s.q.size();++j){Vec3 d=s.q[j]-s.q[i];double r2=dot(d,d);if(!(r2>0)||!std::isfinite(r2))throw std::runtime_error("invalid separation");double z=1.0/(r2*std::sqrt(r2));Vec3 u=z*d;a[i]+=s.mu[j]*u;a[j]-=s.mu[i]*u;}++calls;}

struct Coeff{std::array<double,11>a;std::array<double,10>b;};
static Coeff coeff(){double a1=.0502627644003922,a2=.413514300428344,a3=.0450798897943977,a4=-.188054853819569,a5=.541960678450780,a6=1-2*(a1+a2+a3+a4+a5);double b1=.148816447901042,b2=-.132385865767784,b3=.067307604692185,b4=.432666402578175,b5=.5-(b1+b2+b3+b4);return{{a1,a2,a3,a4,a5,a6,a5,a4,a3,a2,a1},{b1,b2,b3,b4,b5,b5,b4,b3,b2,b1}};}
static void validate_coeff(){auto c=coeff();for(std::size_t i=0;i<c.a.size();++i)if(c.a[i]!=c.a[c.a.size()-1-i])throw std::runtime_error("a symmetry");for(std::size_t i=0;i<c.b.size();++i)if(c.b[i]!=c.b[c.b.size()-1-i])throw std::runtime_error("b symmetry");double sa=std::accumulate(c.a.begin(),c.a.end(),0.0),sb=std::accumulate(c.b.begin(),c.b.end(),0.0);if(std::fabs(sa-1)>2e-15||std::fabs(sb-1)>2e-15)throw std::runtime_error("coefficient closure");}
static void bm6_step(State&s,double h,std::vector<Vec3>&a,std::uint64_t&calls){auto c=coeff();for(std::size_t k=0;k<c.b.size();++k){double d=h*c.a[k];for(std::size_t i=0;i<s.q.size();++i){s.q[i].x+=d*s.v[i].x;s.q[i].y+=d*s.v[i].y;s.q[i].z+=d*s.v[i].z;}acceleration(s,a,calls);double b=h*c.b[k];for(std::size_t i=0;i<s.v.size();++i){s.v[i].x+=b*a[i].x;s.v[i].y+=b*a[i].y;s.v[i].z+=b*a[i].z;}}double d=h*c.a.back();for(std::size_t i=0;i<s.q.size();++i){s.q[i].x+=d*s.v[i].x;s.q[i].y+=d*s.v[i].y;s.q[i].z+=d*s.v[i].z;}}
static Sample sample(const State&s,double t,std::size_t p,std::size_t tr){Vec3 r=s.q[tr]-s.q[p],v=s.v[tr]-s.v[p];return{t,dot(r,r),norm(v)};}
static Metrics encounter_metrics(const std::vector<Sample>&x){if(x.empty())throw std::runtime_error("empty samples");std::size_t k=0;for(std::size_t i=1;i<x.size();++i)if(x[i].d2<x[k].d2)k=i;double t=x[k].t,d2=x[k].d2,sp=x[k].rel_speed;
    if(k>0&&k+1<x.size()){
        double t0=x[k-1].t,t1=x[k].t,t2=x[k+1].t,y0=x[k-1].d2,y1=x[k].d2,y2=x[k+1].d2;
        double den=(t0-t1)*(t0-t2)*(t1-t2);
        if(den!=0){double A=(t2*(y1-y0)+t1*(y0-y2)+t0*(y2-y1))/den;double B=(t2*t2*(y0-y1)+t1*t1*(y2-y0)+t0*t0*(y1-y2))/den;if(A>0){double tv=-B/(2*A);if(tv>=t0&&tv<=t2){double C=(t1*t2*(t1-t2)*y0+t2*t0*(t2-t0)*y1+t0*t1*(t0-t1)*y2)/den;double yv=A*tv*tv+B*tv+C;if(yv>=0&&std::isfinite(yv)){t=tv;d2=yv;double w=(tv-t0)/(t2-t0);sp=(1-w)*x[k-1].rel_speed+w*x[k+1].rel_speed;}}}}
    }
    return{std::sqrt(std::max(0.0,d2)),t,sp};
}
static double max_scaled_delta(const State&a,const State&b,double qscale,double vscale){double m=0;for(std::size_t i=0;i<a.q.size();++i){std::array<double,6>d={std::fabs(a.q[i].x-b.q[i].x)/qscale,std::fabs(a.q[i].y-b.q[i].y)/qscale,std::fabs(a.q[i].z-b.q[i].z)/qscale,std::fabs(a.v[i].x-b.v[i].x)/vscale,std::fabs(a.v[i].y-b.v[i].y)/vscale,std::fabs(a.v[i].z-b.v[i].z)/vscale};for(double z:d)m=std::max(m,z);}return m;}
static double checksum(const State&s){long double z=0;for(std::size_t i=0;i<s.q.size();++i){long double w=i+1;z+=w*(s.q[i].x+s.q[i].y+s.q[i].z+s.v[i].x+s.v[i].y+s.v[i].z);}return(double)z;}

static void kdk_integrate(State& s, double h, std::size_t steps,
                          std::vector<Vec3>& a, std::uint64_t& calls,
                          std::vector<Sample>* samples,
                          std::size_t primary, std::size_t tracer) {
    acceleration(s, a, calls);
    if (samples != nullptr) {
        samples->push_back(sample(s, 0.0, primary, tracer));
    }
    for (std::size_t n = 0; n < steps; ++n) {
        for (std::size_t i = 0; i < s.v.size(); ++i) {
            s.v[i].x += 0.5 * h * a[i].x;
            s.v[i].y += 0.5 * h * a[i].y;
            s.v[i].z += 0.5 * h * a[i].z;
        }
        for (std::size_t i = 0; i < s.q.size(); ++i) {
            s.q[i].x += h * s.v[i].x;
            s.q[i].y += h * s.v[i].y;
            s.q[i].z += h * s.v[i].z;
        }
        acceleration(s, a, calls);
        for (std::size_t i = 0; i < s.v.size(); ++i) {
            s.v[i].x += 0.5 * h * a[i].x;
            s.v[i].y += 0.5 * h * a[i].y;
            s.v[i].z += 0.5 * h * a[i].z;
        }
        if (samples != nullptr) {
            samples->push_back(sample(s, static_cast<double>(n + 1) * h,
                                      primary, tracer));
        }
    }
}

static Result fixed_run(const State&initial,const std::string&mode,double duration,std::size_t steps,int primary_id,int tracer_id,double qscale,double vscale,std::size_t repeats){if(steps==0||repeats==0)throw std::runtime_error("invalid fixed schedule");double h=duration/(double)steps;std::size_t p=index_of(initial,primary_id),tr=index_of(initial,tracer_id);State s=initial;std::vector<Vec3>a;std::uint64_t calls=0;std::vector<Sample> samples;samples.reserve(steps+1);auto start=std::chrono::steady_clock::now();if(mode=="bm6"){samples.push_back(sample(s,0,p,tr));for(std::size_t n=0;n<steps;++n){bm6_step(s,h,a,calls);samples.push_back(sample(s,(n+1)*h,p,tr));}}else{kdk_integrate(s,h,steps,a,calls,&samples,p,tr);}double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    State back=s;std::uint64_t back_calls=0;if(mode=="bm6"){for(std::size_t n=0;n<steps;++n)bm6_step(back,-h,a,back_calls);}else{kdk_integrate(back,-h,steps,a,back_calls,nullptr,p,tr);}double rev=max_scaled_delta(initial,back,qscale,vscale);
    std::vector<double>ts;ts.reserve(repeats);volatile double sink=0;for(std::size_t r=0;r<repeats;++r){State z=initial;std::uint64_t c=0;auto st=std::chrono::steady_clock::now();if(mode=="bm6"){for(std::size_t n=0;n<steps;++n)bm6_step(z,h,a,c);}else{kdk_integrate(z,h,steps,a,c,nullptr,p,tr);}ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count());sink+=checksum(z);}if(!std::isfinite((double)sink))throw std::runtime_error("timing sink");std::sort(ts.begin(),ts.end());
    return{mode,duration,h,steps,steps,0,calls,wall,ts[ts.size()/2],ts.front(),ts.back(),encounter_metrics(samples),rev,s.q,s.v};}

