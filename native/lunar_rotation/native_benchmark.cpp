// Standalone, opt-in native rotation regression. No changes to orbital State or BM6.
#include "jx_lunar_rotation.hpp"
#include <filesystem>
#include <functional>
#include <sstream>
// Minimal declarations of the documented CSPICE C ABI, verified with N0067.
// Use the official SpiceUsr.h instead when building against the full toolkit.
extern "C" {
 void furnsh_c(const char*);void kclear_c();int failed_c();void reset_c();
 void getmsg_c(const char*,int,char*);void erract_c(const char*,int,char*);
 void spkpos_c(const char*,double,const char*,const char*,const char*,double*,double*);
 void sxform_c(const char*,const char*,double,double[6][6]);
 void xf2rav_c(const double[6][6],double[3][3],double[3]);
 void m2q_c(const double[3][3],double[4]);
}
namespace fs=std::filesystem;using namespace jx::lunar;
void spice_check(){if(failed_c()){char m[2048]{};getmsg_c("LONG",2048,m);reset_c();throw std::runtime_error(m);}}
struct Spice {
 std::uint64_t geometry_calls=0,reference_calls=0;
 explicit Spice(const fs::path& p) {
  char action[]="RETURN";erract_c("SET",0,action);spice_check();kclear_c();
  for(auto name:{"de440s.bsp","moon_pa_de440_200625.bpc","moon_de440_250416.tf","gm_de440.tpc"}) {
   auto file=p/name;if(!fs::is_regular_file(file))throw std::runtime_error("missing kernel "+file.string());
   furnsh_c(file.c_str());spice_check();
  }
 }
 ~Spice(){kclear_c();}
 auto geometry(double day) {
  ++geometry_calls;std::array<SourceGeometry,2> out{};int i=0;
  for(auto name:{"EARTH","SUN"}) {double p[3]{},lt{};spkpos_c(name,day*86400.,"J2000","NONE","MOON",p,&lt);spice_check();out[i]={{p[0],p[1],p[2]},constants::gm[i]};++i;}
  return out;
 }
 RotationState reference(double day) {
  ++reference_calls;double X[6][6]{},R[3][3]{},Q[3][3]{},wi[3]{},q[4]{};
  sxform_c("J2000","MOON_PA_DE440",day*86400.,X);spice_check();xf2rav_c(X,R,wi);spice_check();
  RotationState y{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){Q[i][j]=R[j][i];y[4+i]+=R[i][j]*wi[j]*86400.;}
  m2q_c(Q,q);spice_check();for(int i=0;i<4;++i)y[i]=q[i];return y;
 }
};
std::vector<double> csv_numbers(const std::string& line){std::istringstream in(line);std::string s;std::vector<double> v;while(std::getline(in,s,',')){const auto first=s.find_first_not_of(" \t\r"),last=s.find_last_not_of(" \t\r");if(first==std::string::npos)throw std::runtime_error("empty numeric fixture");s=s.substr(first,last-first+1);std::size_t n=0;double x=std::stod(s,&n);if(n!=s.size()||!std::isfinite(x))throw std::runtime_error("invalid numeric fixture");v.push_back(x);}return v;}
int selftest(){int count=0;const Quaternion q{1,0,0,0};std::array<SourceGeometry,2> g{{{{384400,12000,-9000},constants::gm[0]},{{1.5e8,2e7,-3e6},constants::gm[1]}}};
 auto expect_bad=[&](auto fun){bool bad=false;try{fun();}catch(const std::exception&){bad=true;}if(!bad)throw std::runtime_error("invalid input accepted");++count;};
 if(ModelOptions{}.include_degree3_to6)throw std::runtime_error("feature enabled by default");
 ++count;
 expect_bad([&]{torque_over_C({0,0,0,0},g);});auto invalid=g;invalid[0].relative_position_km={0,0,0};expect_bad([&]{torque_over_C(q,invalid);});invalid=g;invalid[0].gm_km3_s2=-1;expect_bad([&]{torque_over_C(q,invalid);});
 invalid=g;invalid[0].relative_position_km.x=std::numeric_limits<double>::quiet_NaN();expect_bad([&]{torque_over_C(q,invalid);});
 const auto a=torque_over_C(q,g,{true}),b=torque_over_C({-1,0,0,0},g,{true});if(jx::norm(a-b)!=0)throw std::runtime_error("quaternion sign failure");++count;
 for(auto n:{V{1,0,0},V{0,1,0},V{0,0,1},V{0,0,-1}})if(!finite(harmonic_gradient(6,n)))throw std::runtime_error("pole failure");
 ++count;
 RotationState y{1,0,0,0,0,0,0.2,0,0,0,0},original=y;expect_bad([&]{rk4_step(y,0,0,[&](double){return g;});});if(y!=original)throw std::runtime_error("invalid step mutated state");
 expect_bad([&]{rk4_step(y,0,0.01,[&](double t){if(t>0)throw std::runtime_error("provider stopped");return g;});});if(y!=original)throw std::runtime_error("failed stage mutated state");++count;
 jx::validate_coefficients(jx::bm6_coefficients());++count; // Existing native utility unchanged.
 std::cout<<"{\"selftest_checks\":"<<count<<",\"passed\":true}\n";return 0;}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--selftest")return selftest();
 if(argc==3&&std::string(argv[1])=="--fixtures") {
  std::ifstream f(argv[2]);if(!f)throw std::runtime_error("missing fixture");std::string line;std::getline(f,line);std::size_t count=0;double mr=0,ma=0;
  while(std::getline(f,line)){if(line.empty())continue;auto v=csv_numbers(line);if(v.size()!=16)throw std::runtime_error("fixture column count");Quaternion q{v[0],v[1],v[2],v[3]};std::array<SourceGeometry,2> g{{{{v[4],v[5],v[6]},constants::gm[0]},{{v[7],v[8],v[9]},constants::gm[1]}}};
   for(int k=0;k<2;++k){auto t=torque_over_C(q,g,{k!=0});V expected{v[10+3*k],v[11+3*k],v[12+3*k]};double d=jx::norm(t-expected),n=jx::norm(expected);ma=std::max(ma,d);if(n>0)mr=std::max(mr,d/n);if(d>1e-29+1e-11*n)throw std::runtime_error("torque fixture mismatch");++count;}}
  if(count==0)throw std::runtime_error("empty fixture");
  std::cout<<std::setprecision(17)<<"{\"torque_checks\":"<<count<<",\"max_absolute_s2\":"<<ma<<",\"max_relative\":"<<mr<<",\"passed\":true}\n";return 0;
 }
 if(argc!=4)throw std::invalid_argument("usage: native_benchmark KERNEL_DIR OUTPUT_DIR STEPS_PER_DAY; or --fixtures CSV; --selftest");
 const std::string ns=argv[3];std::size_t parsed=0;const int per=std::stoi(ns,&parsed);
 if(parsed!=ns.size()||per<4||per>1024||(per&(per-1)))throw std::invalid_argument("steps/day must be power of two in [4,1024]");
 fs::path out=argv[2];if(fs::exists(out))throw std::runtime_error("output exists; choose new directory");Spice sp(argv[1]);fs::create_directories(out);const double h=1./per;
 for(int start:{0,32,64,96}){
  const auto initial=sp.reference(start);const auto nr=sp.reference_calls;std::array<RotationState,2> lanes{initial,initial};
  std::ofstream f(out/("arc"+std::to_string(start)+".csv"));if(!f)throw std::runtime_error("cannot create output");
  f<<"elapsed_day,lane,q0,q1,q2,q3,w0_day,w1_day,w2_day,imp0,imp1,imp2,work\n"<<std::setprecision(17);
  auto emit=[&](int step){for(int lane=0;lane<2;++lane){f<<double(step)/per<<','<<lane;for(double x:lanes[lane])f<<','<<x;f<<'\n';}};emit(0);
  for(int step=0;step<32*per;++step){const double t=start+double(step)/per;
   // Each lane has its own attitude. Position queries shared by a stage cache only.
   double cached_t=std::numeric_limits<double>::quiet_NaN();std::array<SourceGeometry,2> cached{};
   auto geometry=[&](double day){if(day!=cached_t){cached=sp.geometry(day);cached_t=day;}return cached;};
   for(int lane=0;lane<2;++lane)rk4_step(lanes[lane],t,h,geometry,{lane!=0});
   if((step+1)%(per/4)==0)emit(step+1);
  }
  if(sp.reference_calls!=nr)throw std::runtime_error("reference attitude requested during propagation");
  if(!f)throw std::runtime_error("output write failed");
 }
 std::ofstream m(out/"execution.json");m<<"{\"reference_queries\":"<<sp.reference_calls<<",\"reference_queries_during_integration\":0,\"geometry_queries\":"<<sp.geometry_calls<<",\"steps_per_day\":"<<per<<",\"lane_trajectories\":8,\"precision\":\"CPU binary64; no fast-math\",\"coupled_orbit\":false}\n";
 if(!m)throw std::runtime_error("metadata write failed");
 return 0;
}catch(const std::exception& e){std::cerr<<"NATIVE_LUNAR_ERROR: "<<e.what()<<'\n';return 2;}}
