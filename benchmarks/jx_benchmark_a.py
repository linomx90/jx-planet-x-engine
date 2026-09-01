#!/usr/bin/env python3
"""Locked JX Benchmark A: KDK, cached Yoshida-6, BM6, REBOUND 5.1.1.

Scoped numerical engineering only. All lanes use the same ten-body Newtonian
DE441/Horizons initial state and are judged against an independent DOP853 force
implementation. Equal-force budget is the primary contest.
"""
from __future__ import annotations
import csv, hashlib, json, math, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

REQ_REBOUND="5.1.1"; AU_KM=149_597_870.700; DAY=86_400.0
IDS=tuple(range(1,11)); SUN=10; OUTER=(5,6,7,8); FIELDS=("x","y","z","vx","vy","vz")
Y6=np.array([.7845136104775572638,.2355732133593581337,-1.1776799841788710069,
             1.3151863206839112189,-1.1776799841788710069,.2355732133593581337,
             .7845136104775572638],float)
a1,a2,a3,a4,a5=.0502627644003922,.413514300428344,.0450798897943977,-.188054853819569,.541960678450780
a6=1-2*(a1+a2+a3+a4+a5)
b1,b2,b3,b4=.148816447901042,-.132385865767784,.067307604692185,.432666402578175
b5=.5-(b1+b2+b3+b4)
BA=np.array([a1,a2,a3,a4,a5,a6,a5,a4,a3,a2,a1],float)
BB=np.array([b1,b2,b3,b4,b5,b5,b4,b3,b2,b1],float)

def sh(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def canon(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load(refp,gmp):
 rows=list(csv.DictReader(Path(refp).open(newline="",encoding="utf-8"))); epoch=min(float(r["jd_tdb"]) for r in rows)
 init={int(r["body_id"]):r for r in rows if float(r["jd_tdb"])==epoch}
 gm={int(r["body_id"]):float(r["gm_km3_s2"])*DAY**2/AU_KM**3 for r in csv.DictReader(Path(gmp).open(newline="",encoding="utf-8"))}
 if tuple(sorted(init))!=IDS or tuple(sorted(gm))!=IDS: raise RuntimeError("locked body roster changed")
 names=tuple(init[i]["body_name"] for i in IDS)
 q=np.array([[float(init[i][k]) for k in ("x_au","y_au","z_au")] for i in IDS],float)
 v=np.array([[float(init[i][k]) for k in ("vx_au_per_day","vy_au_per_day","vz_au_per_day")] for i in IDS],float)
 mu=np.array([gm[i] for i in IDS],float)
 state_hash=canon({"epoch":format(epoch,".9f"),"ids":IDS,"names":names,
                   "mu":[format(x,".17e") for x in mu],
                   "q":[[format(x,".17e") for x in r] for r in q],
                   "v":[[format(x,".17e") for x in r] for r in v]})
 return dict(epoch=epoch,names=names,mu=mu,q=q,v=v,state_hash=state_hash)

def acc(q,mu):
 a=np.zeros_like(q)
 for i in range(len(mu)-1):
  for j in range(i+1,len(mu)):
   d=q[j]-q[i]; r2=float(d@d); z=1/(r2*math.sqrt(r2)); f=d*z
   a[i]+=mu[j]*f; a[j]-=mu[i]*f
 return a

def acc_ref(q,mu):
 a=np.zeros_like(q)
 for i in range(len(mu)):
  t=[[],[],[]]
  for j in range(len(mu)):
   if i==j: continue
   d=q[j]-q[i]; r2=float(d@d); z=mu[j]/(r2*math.sqrt(r2))
   for k in range(3): t[k].append(float(d[k])*z)
  a[i]=[math.fsum(x) for x in t]
 return a

def inv(q,v,mu):
 e=math.fsum(.5*mu[i]*float(v[i]@v[i]) for i in range(len(mu)))
 e+=math.fsum(-mu[i]*mu[j]/float(np.linalg.norm(q[j]-q[i])) for i in range(len(mu)-1) for j in range(i+1,len(mu)))
 L=np.sum(mu[:,None]*np.cross(q,v),axis=0); return e,L

def grid(T,out,h):
 n=round(T/h); m=round(out/h)
 if not math.isclose(n*h,T,abs_tol=2e-10,rel_tol=0) or not math.isclose(m*h,out,abs_tol=2e-12,rel_tol=0) or n%m: raise RuntimeError("noninteger grid")
 return n,m,np.arange(n//m+1,dtype=float)*out

def capture(q,v,e0,L0,mu):
 e,L=inv(q,v,mu); return np.c_[q,v].copy(),(e-e0)/abs(e0),float(np.linalg.norm(L-L0)/np.linalg.norm(L0))

def jx(lane,contest,S,T,out,h):
 n,m,t=grid(T,out,h); q=S["q"].copy(); v=S["v"].copy(); e0,L0=inv(q,v,S["mu"]); st=[]; es=[]; ls=[]
 def cap():
  x,e,l=capture(q,v,e0,L0,S["mu"]); st.append(x); es.append(e); ls.append(l)
 cap(); calls=0; start=time.perf_counter()
 if lane=="kdk":
  a=acc(q,S["mu"]); calls=1
  for k in range(1,n+1):
   v+=.5*h*a; q+=h*v; a=acc(q,S["mu"]); calls+=1; v+=.5*h*a
   if k%m==0: cap()
  sem="measured: one initial plus one force solve per step"
 elif lane=="y6_cached":
  a=acc(q,S["mu"]); calls=1
  for k in range(1,n+1):
   v+=.5*h*Y6[0]*a
   for s,c in enumerate(Y6):
    q+=h*c*v; a=acc(q,S["mu"]); calls+=1
    v+=h*(.5*c if s==6 else .5*(c+Y6[s+1]))*a
   if k%m==0: cap()
  sem="measured: one initial plus seven force solves per macro-step"
 elif lane=="bm6":
  for k in range(1,n+1):
   for s,b in enumerate(BB): q+=h*BA[s]*v; a=acc(q,S["mu"]); calls+=1; v+=h*b*a
   q+=h*BA[-1]*v
   if k%m==0: cap()
  sem="measured: ten force solves per macro-step"
 else: raise RuntimeError(lane)
 return dict(lane=lane,contest=contest,dt=h,steps=n,calls=calls,cost_semantics=sem,wall=time.perf_counter()-start,t=t,states=np.array(st),energy=np.array(es),ang=np.array(ls),extra={})

def reb(contest,S,T,out,h):
 import rebound
 if rebound.__version__!=REQ_REBOUND: raise RuntimeError(f"BLOCKED expected REBOUND {REQ_REBOUND}, got {rebound.__version__}")
 n,m,t=grid(T,out,h); sim=rebound.Simulation(); sim.G=1.; sim.integrator="leapfrog"; sim.dt=h
 for i in range(len(S["mu"])): sim.add(m=S["mu"][i],x=S["q"][i,0],y=S["q"][i,1],z=S["q"][i,2],vx=S["v"][i,0],vy=S["v"][i,1],vz=S["v"][i,2])
 q=np.array([[p.x,p.y,p.z] for p in sim.particles]); v=np.array([[p.vx,p.vy,p.vz] for p in sim.particles]); e0,L0=inv(q,v,S["mu"]); st=[]; es=[]; ls=[]
 def cap():
  q=np.array([[p.x,p.y,p.z] for p in sim.particles]); v=np.array([[p.vx,p.vy,p.vz] for p in sim.particles]); x,e,l=capture(q,v,e0,L0,S["mu"]); st.append(x); es.append(e); ls.append(l)
 cap(); start=time.perf_counter()
 for k in range(1,n+1):
  sim.step()
  if k%m==0: cap()
 return dict(lane="rebound_leapfrog_5.1.1",contest=contest,dt=h,steps=n,calls=n,
  cost_semantics="one-force-solve-per-step Leapfrog cost model; REBOUND internals not instrumented",
  wall=time.perf_counter()-start,t=t,states=np.array(st),energy=np.array(es),ang=np.array(ls),extra={"version":rebound.__version__})

def dop(label,S,T,out,rtol,atol,max_step):
 t=np.arange(round(T/out)+1,dtype=float)*out; N=len(S["mu"]); y0=np.r_[S["q"].ravel(),S["v"].ravel()]; calls=0
 def f(_,y):
  nonlocal calls; calls+=1; q=y[:3*N].reshape(N,3); v=y[3*N:].reshape(N,3); return np.r_[v.ravel(),acc_ref(q,S["mu"]).ravel()]
 start=time.perf_counter(); z=solve_ivp(f,(0,T),y0,method="DOP853",t_eval=t,rtol=rtol,atol=atol,max_step=max_step)
 if not z.success: raise RuntimeError(z.message)
 e0,L0=inv(S["q"],S["v"],S["mu"]); st=[]; es=[]; ls=[]
 for k in range(len(t)):
  q=z.y[:3*N,k].reshape(N,3); v=z.y[3*N:,k].reshape(N,3); x,e,l=capture(q,v,e0,L0,S["mu"]); st.append(x); es.append(e); ls.append(l)
 return dict(lane=label,contest="reference",dt=max_step,steps=z.nfev,calls=calls,cost_semantics="measured independent RHS calls",wall=time.perf_counter()-start,t=t,states=np.array(st),energy=np.array(es),ang=np.array(ls),extra={"rtol":rtol,"atol":atol,"max_step":max_step,"nfev":z.nfev})

def metric(run,ref,S):
 if run["states"].shape!=ref["states"].shape or not np.array_equal(run["t"],ref["t"]): raise RuntimeError("output grid mismatch")
 a=run["states"].copy(); b=ref["states"].copy(); si=IDS.index(SUN)
 for x in (a,b): x[:,:,:3]-=x[:,si:si+1,:3]; x[:,:,3:]-=x[:,si:si+1,3:]
 def one(indices):
  d=a[:,indices]-b[:,indices]; p=np.linalg.norm(d[:,:,:3],axis=2).ravel(); v=np.linalg.norm(d[:,:,3:],axis=2).ravel()
  return dict(max_pos=float(p.max()),rms_pos=float(np.sqrt(np.mean(p*p))),max_vel=float(v.max()),rms_vel=float(np.sqrt(np.mean(v*v))))
 return {"all":one([i for i,x in enumerate(IDS) if x!=SUN]),"outer":one([IDS.index(x) for x in OUTER])}

def summary(r,ref,S):
 return dict(lane=r["lane"],contest=r["contest"],dt_days=r["dt"],steps=r["steps"],force_evaluations=r["calls"],force_count_semantics=r["cost_semantics"],wall_seconds=r["wall"],
  max_abs_relative_energy_error=float(np.max(np.abs(r["energy"]))),final_signed_relative_energy_error=float(r["energy"][-1]),
  bounded_energy_half_range=float((r["energy"].max()-r["energy"].min())/2),max_relative_angular_momentum_vector_error=float(r["ang"].max()),
  trajectory_error_vs_tight_dop853=metric(r,ref,S),extra=r["extra"])

def write_traj(path,r,S):
 with Path(path).open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f); w.writerow(("lane","contest","time_days","jd_tdb","body_id","body_name",*FIELDS,"signed_relative_energy_error","relative_angular_momentum_vector_error"))
  for k,t in enumerate(r["t"]):
   for i,bid in enumerate(IDS): w.writerow((r["lane"],r["contest"],format(t,".17e"),format(S["epoch"]+t,".9f"),bid,S["names"][i],*[format(x,".17e") for x in r["states"][k,i]],format(r["energy"][k],".17e"),format(r["ang"][k],".17e")))

def ref_gate(loose,tight,sums,S):
 d=metric(loose,tight,S); bp=min(x["trajectory_error_vs_tight_dop853"]["all"]["rms_pos"] for x in sums); bv=min(x["trajectory_error_vs_tight_dop853"]["all"]["rms_vel"] for x in sums)
 pl=max(.01*bp,5e-15); vl=max(.01*bv,5e-17); return dict(disagreement=d,position_limit=pl,velocity_limit=vl,passed=d["all"]["rms_pos"]<=pl and d["all"]["rms_vel"]<=vl)

def beats(c,r):
 cm=c["trajectory_error_vs_tight_dop853"]["all"]; rm=r["trajectory_error_vs_tight_dop853"]["all"]
 return all(cm[k]<rm[k] for k in ("max_pos","rms_pos","max_vel","rms_vel")) and c["max_abs_relative_energy_error"]<=r["max_abs_relative_energy_error"] and c["max_relative_angular_momentum_vector_error"]<=max(r["max_relative_angular_momentum_vector_error"],5e-15)

def verdict(sums,rg):
 if not rg["passed"]: return "INVALID_REFERENCE"
 d={x["lane"]:x for x in sums}; r=d["rebound_leapfrog_5.1.1"]
 if beats(d["bm6"],r): return "JX_BM6_WIN"
 if beats(d["y6_cached"],r): return "JX_Y6_WIN"
 return "NO_CLEAR_WINNER"

def main():
 root=Path(__file__).resolve().parents[1]; contract=root/"benchmarks/jx_benchmark_a_contract.json"; refp=root/"runs/de441_horizons_10yr/reference/horizons_de441_vectors.csv"; gmp=root/"runs/de441_horizons_10yr/gm_de440_major_barycenters.csv"; out=root/"runs/jx_benchmark_a"; out.mkdir(parents=True,exist_ok=True)
 C=json.load(contract.open()); S=load(refp,gmp); T=float(C["workload"]["duration_years"])*365.25; oi=float(C["workload"]["output_interval_days"])
 gate={"y6_sum":float(Y6.sum()),"y6_symmetric":bool(np.array_equal(Y6,Y6[::-1])),"bm6_a_sum":float(BA.sum()),"bm6_b_sum":float(BB.sum()),"bm6_a_symmetric":bool(np.array_equal(BA,BA[::-1])),"bm6_b_symmetric":bool(np.array_equal(BB,BB[::-1]))}; gate["passed"]=all([gate["y6_symmetric"],gate["bm6_a_symmetric"],gate["bm6_b_symmetric"],abs(gate["y6_sum"]-1)<2e-15,abs(gate["bm6_a_sum"]-1)<2e-15,abs(gate["bm6_b_sum"]-1)<2e-15])
 if not gate["passed"]: raise RuntimeError(gate)
 loose=dop("dop853_loose",S,T,oi,2.5e-13,2.5e-16,.5); tight=dop("dop853_tight",S,T,oi,2.5e-14,2.5e-17,.25)
 write_traj(out/"trajectory_dop853_loose.csv",loose,S); write_traj(out/"trajectory_dop853_tight.csv",tight,S)
 runs={}; dt=float(C["contests"]["equal_timestep"]["dt_days"]); runs["equal_timestep"]=[jx(x,"equal_timestep",S,T,oi,dt) for x in ("kdk","y6_cached","bm6")]+[reb("equal_timestep",S,T,oi,dt)]
 B=C["contests"]["equal_force_budget"]; sp={"kdk":B["kdk_steps_per_year"],"y6_cached":B["y6_steps_per_year"],"bm6":B["bm6_steps_per_year"],"rebound_leapfrog_5.1.1":B["rebound_steps_per_year"]}
 runs["equal_force_budget"]=[jx(x,"equal_force_budget",S,T,oi,365.25/int(sp[x])) for x in ("kdk","y6_cached","bm6")]+[reb("equal_force_budget",S,T,oi,365.25/int(sp["rebound_leapfrog_5.1.1"]))]
 contests={}; rows=[]
 for name,rr in runs.items():
  ss=[summary(x,tight,S) for x in rr]; rg=ref_gate(loose,tight,ss,S); contests[name]={"verdict":verdict(ss,rg),"reference_gate":rg,"summaries":ss}
  for x in rr: write_traj(out/f"trajectory_{name}_{x['lane']}.csv",x,S)
  for x in ss:
   m=x["trajectory_error_vs_tight_dop853"]; rows.append({"contest":name,"lane":x["lane"],"dt_days":x["dt_days"],"steps":x["steps"],"force_evaluations":x["force_evaluations"],"wall_seconds":x["wall_seconds"],"max_abs_relative_energy_error":x["max_abs_relative_energy_error"],"max_relative_angular_momentum_vector_error":x["max_relative_angular_momentum_vector_error"],**{f"all_{k}":v for k,v in m["all"].items()},**{f"outer_{k}":v for k,v in m["outer"].items()}})
 result={"schema":"jx-general-dynamics-benchmark-a-result/v1","classification":"MODEL_OUTPUT_NUMERICAL_ENGINEERING_ONLY","primary_contest":"equal_force_budget","primary_verdict":contests["equal_force_budget"]["verdict"],"contract_sha256":sh(contract),"reference_vectors_sha256":sh(refp),"gm_table_sha256":sh(gmp),"normalized_initial_state_sha256":S["state_hash"],"initial_epoch_jd_tdb":S["epoch"],"coefficient_gate":gate,"reference":{"loose":summary(loose,tight,S),"tight":{"wall_seconds":tight["wall"],"force_evaluations":tight["calls"],"max_abs_relative_energy_error":float(np.max(np.abs(tight["energy"])))},"loose_vs_tight":metric(loose,tight,S)},"contests":contests,"environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":__import__("scipy").__version__,"rebound_required":REQ_REBOUND,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()},"nonclaim":C["claim_ceiling"]}
 (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 (out/"contract.lock.json").write_text(contract.read_text())
 (out/"README.md").write_text(f"# JX Benchmark A\n\nPrimary verdict: **{result['primary_verdict']}**\n\nEqual-timestep verdict: **{contests['equal_timestep']['verdict']}**\n\nScoped numerical engineering only; see result.json and summary.csv.\n")
 sums=[]
 for p in sorted(out.iterdir()):
  if p.is_file() and p.name!="SHA256SUMS.txt": sums.append(f"{sh(p)}  {p.name}")
 (out/"SHA256SUMS.txt").write_text("\n".join(sums)+"\n")
 print(json.dumps({"primary_verdict":result["primary_verdict"],"equal_timestep_verdict":contests["equal_timestep"]["verdict"],"result_sha256":sh(out/"result.json")},indent=2))
 return 2 if result["primary_verdict"]=="INVALID_REFERENCE" else 0
if __name__=="__main__": raise SystemExit(main())
