#pragma once
// Experimental externally forced lunar rotation. Does NOT modify jx::State or bm6_step.
#include "../jx_bm6_types.hpp"
#include "de440_constants.hpp"
#include <span>

namespace jx::lunar {
using V = ::jx::Vec3;
using Quaternion = std::array<double,4>;
using Matrix = std::array<std::array<double,3>,3>;
// q (body->inertial); Omega (body, rad/day); inertial torque impulse; work.
using RotationState = std::array<double,11>;
struct SourceGeometry { V relative_position_km; double gm_km3_s2; };
struct ModelOptions { bool include_degree3_to6 = false; }; // Opt in; no baseline promotion.
inline V add(V a,V b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline V scale(double s,V a) { return {s*a.x,s*a.y,s*a.z}; }
inline bool finite(V a) { return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z); }
inline V mul(const Matrix& q,V a) {
 return {q[0][0]*a.x+q[0][1]*a.y+q[0][2]*a.z,
         q[1][0]*a.x+q[1][1]*a.y+q[1][2]*a.z,
         q[2][0]*a.x+q[2][1]*a.y+q[2][2]*a.z};
}
inline V tmul(const Matrix& q,V a) {
 return {q[0][0]*a.x+q[1][0]*a.y+q[2][0]*a.z,
         q[0][1]*a.x+q[1][1]*a.y+q[2][1]*a.z,
         q[0][2]*a.x+q[1][2]*a.y+q[2][2]*a.z};
}
inline Matrix rotation_matrix(const Quaternion& q) {
 double n2=0;for(double v:q) { if(!std::isfinite(v)) throw std::invalid_argument("nonfinite quaternion"); n2+=v*v; }
 if(!(n2>=0.25 && n2<=4)) throw std::invalid_argument("invalid quaternion norm");
 const double n=std::sqrt(n2),s=q[0]/n,x=q[1]/n,y=q[2]/n,z=q[3]/n;
 return {{{1-2*(y*y+z*z),2*(x*y-z*s),2*(x*z+y*s)},
          {2*(x*y+z*s),1-2*(x*x+z*z),2*(y*z-x*s)},
          {2*(x*z-y*s),2*(y*z+x*s),1-2*(x*x+y*y)}}};
}
// Ordinary Legendre polynomial coefficients in ascending powers, exact binary rationals.
inline constexpr double legendre[7][7] = {
 {1,0,0,0,0,0,0}, {0,1,0,0,0,0,0}, {-0.5,0,1.5,0,0,0,0},
 {0,-1.5,0,2.5,0,0,0}, {0.375,0,-3.75,0,4.375,0,0},
 {0,1.875,0,-8.75,0,7.875,0}, {-0.3125,0,6.5625,0,-19.6875,0,14.4375}
};
inline double derivative_legendre(int l,int m,double z) {
 if(m>l) return 0;
 double out=0;
 for(int k=l;k>=m;--k) {
  double coefficient=legendre[l][k];
  for(int j=0;j<m;++j) coefficient*=k-j;
  out=out*z+coefficient;
 }
 return out;
}
// Gradient of H_l(n), using polynomial Re/Im(x+iy)^m. No pole division.
inline V harmonic_gradient(int l,V n) {
 std::array<double,7> re{},im{};re[0]=1;
 for(int m=1;m<=l;++m) { re[m]=re[m-1]*n.x-im[m-1]*n.y; im[m]=re[m-1]*n.y+im[m-1]*n.x; }
 V out{0,0,constants::cosine[l][0]*derivative_legendre(l,1,n.z)};
 for(int m=1;m<=l;++m) {
  const double c=constants::cosine[l][m],s=constants::sine[l][m];
  const double p=derivative_legendre(l,m,n.z);
  out.x+=p*m*(c*re[m-1]+s*im[m-1]);
  out.y+=p*m*(-c*im[m-1]+s*re[m-1]);
  out.z+=derivative_legendre(l,m+1,n.z)*(c*re[m]+s*im[m]);
 }
 return out;
}
// Returns torque/C in s^-2. All source positions are simultaneous geometric km.
inline V torque_over_C(const Quaternion& q,std::span<const SourceGeometry> sources,
                       ModelOptions options={}) {
 const auto Q=rotation_matrix(q);V out{};
 for(const auto& src:sources) {
  if(!finite(src.relative_position_km)||!std::isfinite(src.gm_km3_s2)||src.gm_km3_s2<0)
   throw std::invalid_argument("invalid source geometry or GM");
  const double d=::jx::norm(src.relative_position_km);
  if(!std::isfinite(d)||d<=constants::radius_km) throw std::invalid_argument("source not outside lunar field");
  const V n=tmul(Q,scale(1/d,src.relative_position_km));
  const V dn{(constants::inertia[0]-1)*n.x,(constants::inertia[1]-1)*n.y,0};
  const double f=src.gm_km3_s2/(d*d*d);
  out+=scale(3*f,::jx::cross(n,dn));
  if(options.include_degree3_to6) {
   const double rd=constants::radius_km/d;double power=rd;
   for(int l=3;l<=6;++l,power*=rd)
    out+=scale(-f*power/constants::polar_scale,::jx::cross(n,harmonic_gradient(l,n)));
  }
 }
 if(!finite(out)) throw std::runtime_error("nonfinite torque");
 return out;
}
inline RotationState derivative(const RotationState& y,std::span<const SourceGeometry> sources,
                                 ModelOptions options={}) {
 for(double x:y) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite rotation state");
 const Quaternion q{y[0],y[1],y[2],y[3]};const V w{y[4],y[5],y[6]},v{q[1],q[2],q[3]};
 const auto Q=rotation_matrix(q);const V tau=scale(kDaySeconds*kDaySeconds,torque_over_C(q,sources,options));
 const V dw{(constants::inertia[0]-1)*w.x,(constants::inertia[1]-1)*w.y,0};
 const V gyro=::jx::cross(w,dw),vrot=add(scale(q[0],w),::jx::cross(v,w)),ti=mul(Q,tau);
 return {-0.5*::jx::dot(v,w),0.5*vrot.x,0.5*vrot.y,0.5*vrot.z,
         (tau.x-gyro.x)/constants::inertia[0],(tau.y-gyro.y)/constants::inertia[1],tau.z-gyro.z,
         ti.x,ti.y,ti.z,::jx::dot(w,tau)};
}
// Geometry provider evaluated at every stage: no orientation reference belongs here.
// Transactional update: state is committed only after all stages and validity checks.
template<class GeometryProvider>
inline void rk4_step(RotationState& state,double absolute_day,double h,
                     GeometryProvider&& geometry,ModelOptions options={}) {
 if(!std::isfinite(absolute_day)||!std::isfinite(h)||h==0) throw std::invalid_argument("invalid step");
 const auto f=[&](double t,const RotationState& y){auto g=geometry(t);return derivative(y,g,options);};
 const auto shifted=[&](const RotationState& k,double s){auto y=state;for(std::size_t i=0;i<y.size();++i)y[i]+=s*k[i];return y;};
 auto k1=f(absolute_day,state),k2=f(absolute_day+h/2,shifted(k1,h/2));
 auto k3=f(absolute_day+h/2,shifted(k2,h/2)),k4=f(absolute_day+h,shifted(k3,h));
 auto next=state;for(std::size_t i=0;i<next.size();++i) {
  next[i]+=(h/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
  if(!std::isfinite(next[i])) throw std::runtime_error("nonfinite RK4 update");
 }
 rotation_matrix({next[0],next[1],next[2],next[3]});state=next;
}
} // namespace jx::lunar
