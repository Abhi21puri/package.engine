import sys, json, itertools
sys.path.insert(0,'/home/claude/v3')
import numpy as np
from scipy.integrate import solve_ivp
from model import *

out={}
# ---------- INDEPENDENT CHECK: scipy RK45 ODE of the baseline (nominal params) vs Euler model ----------
p=nominal_params()
R=p['r_dc'][0]*p['k_sus'][0]; C=CELL['C_th']
G=p['h_nat'][0]*CELL['area']*EXPO; gnb=p['g_nb'][0]
def I_of(t): return 15.0 if t<480 else 3.0
def f(t,T):
    I=I_of(t); dT=np.zeros(4)
    for i in range(4):
        dT[i]=I*I*R-G[i]*(T[i]-45.0)
        if i>0: dT[i]-=gnb*(T[i]-T[i-1])
        if i<3: dT[i]-=gnb*(T[i]-T[i+1])
    return dT/C
sol=solve_ivp(f,[0,1800],[45.0]*4,method='RK45',max_step=1.0,rtol=1e-8,atol=1e-8)
ode_peak=sol.y.max()
o=simulate(DESIGNS['Baseline'],p,STRESS,charge=False)
print(f"INDEPENDENT ODE CHECK: RK45 peak={ode_peak:.3f}  Euler model peak={o['Tpeak'][0]:.3f}")
out['ode_check']=dict(rk45=float(ode_peak),euler=float(o['Tpeak'][0]))

# ---------- Selection rule check (N=1000) ----------
rows=[]
for plate,fan,ds in itertools.product([False,True],[False,True],[None,53.0,55.0]):
    des=dict(series=4,par=1,plate=plate,fan=fan,derate_start=ds)
    rng=np.random.default_rng(21); pp=sample_params(1000,rng)
    o=simulate(des,pp,STRESS,charge=False)
    cost,mass=design_cost_mass(des)
    ph=float(np.mean((o['Tpeak']<55)&(o['Ah_del']/o['Ah_req']>=0.99)))
    rows.append(dict(plate=plate,fan=fan,derate=ds,p_health99=ph,p_safe=float(np.mean(o['Tpeak']<60)),deliv=float(np.median(o['Ah_del']/o['Ah_req'])),cost=cost,mass=mass))
for r in rows: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
ok=[r for r in rows if r['p_health99']>=0.9 and r['p_safe']>=0.99]
ok.sort(key=lambda r:(r['cost'],r['mass']))
print("RULE-PASSING (cheapest first):",ok[:3])
out['sweep']=rows

# ---------- Bench prediction: lone cell, 25 C air, 15 A for 8 min ----------
rng=np.random.default_rng(5); n=5000
r=rng.uniform(0.010,0.016,n); k=rng.uniform(1.0,1.5,n); h=rng.uniform(5.0,15.0,n)   # lone cell in still air (LIT 5-25; ASM 5-15)
T=np.full(n,25.0); dt=1.0
for _ in range(480):
    T+= dt*(15.0**2*r*k - h*CELL['area']*(T-25.0))/CELL['C_th']
dT=T-25.0
print("BENCH PREDICTION (avg cell body): dT median %.1f K, 5-95%% [%.1f, %.1f]"%(np.median(dT),np.percentile(dT,5),np.percentile(dT,95)))
out['bench_pred']=dict(med=float(np.median(dT)),p5=float(np.percentile(dT,5)),p95=float(np.percentile(dT,95)))

# ---------- Safe-operating maps (thermal-only red rule) ----------
amb=[35,38,41,44,47,50]; crates=[0.5,1.0,1.5,2.0,2.5,3.0]
REC=dict(series=4,par=1,plate=True,fan=True,derate_start=53.0)
maps={}
for name,d in [("Baseline",DESIGNS['Baseline']),("Recommended",REC)]:
    M=np.zeros((len(crates),len(amb)))
    for i,c in enumerate(crates):
        for j,a in enumerate(amb):
            prof=[(480,6.0*c),(1320,3.0)] if c>0.5 else [(1800,3.0)]
            rng=np.random.default_rng(3); pp=sample_params(120,rng)
            o=simulate(d,pp,prof,T_amb_duty=float(a),charge=False)
            hl=((o['Tpeak']<55)&(o['Ah_del']/o['Ah_req']>=0.95)).mean()
            sf=(o['Tpeak']<60).mean()
            M[i,j]=2 if hl>=0.9 else (1 if sf>=0.95 else 0)
    maps[name]=M.tolist()
out['maps']=maps
json.dump(out,open('results2.json','w'),indent=1)
for n_,M in maps.items():
    print(n_)
    for c,row in zip(crates,M): print(c,row)
