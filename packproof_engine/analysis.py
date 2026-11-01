import sys, json, time
sys.path.insert(0,'/home/claude/v3')
import numpy as np
from scipy.stats import spearmanr
from model import *
import model

out={}
# ---------- VERIFICATION TEST 1: adiabatic energy balance (no losses) ----------
p=nominal_params(); p['h_nat']=np.array([0.0]); p['g_nb']=np.array([0.0])
des=dict(series=4,par=1,plate=False,fan=False,derate_start=None)
o=simulate(des,p,[(480,15.0)],charge=False)
R=p['r_dc'][0]*p['k_sus'][0]
analytic=15.0**2*R*480/CELL['C_th']
sim=o['Tpeak'][0]-45.0
print(f"TEST1 adiabatic dT: sim={sim:.3f} K  analytic={analytic:.3f} K  err={abs(sim-analytic)/analytic*100:.2f}%")
out['test_adiabatic']=dict(sim=float(sim),analytic=float(analytic))
# ---------- TEST 2: steady state at 0.5C with no neighbour coupling ----------
p=nominal_params(); p['g_nb']=np.array([0.0])
o=simulate(des,p,[(30*3600,3.0)],charge=False)
G_mid=p['h_nat'][0]*CELL['area']*EXPO[1]; Q=3.0**2*R
ss=45+Q/G_mid
print(f"TEST2 steady-state mid cell: sim={o['Tpeak'][0]:.3f}  analytic={ss:.3f}")  # hottest is mid cells
out['test_steady']=dict(sim=float(o['Tpeak'][0]),analytic=float(ss))
# ---------- TEST 3: datasheet charge window ----------
chk={-1:0.0,5:0.2,12:0.5,25:1.0,44.9:1.0,45:0.5,59:0.5}
ok=all(abs(float(charge_limit_C(np.array([t]))[0])-v)<1e-9 for t,v in chk.items())
print("TEST3 charge window table matches datasheet 2.14:",ok)
# ---------- TEST 4: datasheet cycle-life anchors ----------
print("TEST4 cycle life @30C, @55C:",float(cycle_life(30)),float(cycle_life(55)))

# ---------- Final Monte Carlo, all designs + recommended ----------
REC=dict(series=4,par=1,plate=True,fan=True,derate_start=53.0)
ALL=dict(DESIGNS); ALL["Recommended"]=REC
N=1000
res={}
t0=time.time()
curves={}
for name,d in ALL.items():
    rng=np.random.default_rng(7); p=sample_params(N,rng)
    o=simulate(d,p,STRESS)
    cost,mass=design_cost_mass(d)
    Tavg=(o['Tavg_dis']*1800+o['Tavg_chg']*o['t_charge_s'])/(1800+o['t_charge_s'])
    res[name]=dict(p_safe=float(np.mean(o['Tpeak']<60)),p_health=float(np.mean((o['Tpeak']<55)&(o['Ah_del']/o['Ah_req']>=0.95))),
        Tpeak_med=float(np.median(o['Tpeak'])),Tpeak_p5=float(np.percentile(o['Tpeak'],5)),Tpeak_p95=float(np.percentile(o['Tpeak'],95)),
        deliv=float(np.median(o['Ah_del']/o['Ah_req'])),wh=float(np.median(o['Wh_net'])),
        chg=float(np.median(o['t_charge_s'])/60),cost=float(cost),mass=float(mass),life=float(np.median(cycle_life(Tavg))))
    h=o['hist']                       # (steps/10, N) hottest cell
    curves[name]=dict(med=np.median(h,axis=1).tolist(),p5=np.percentile(h,5,axis=1).tolist(),p95=np.percentile(h,95,axis=1).tolist())
    if name=="Baseline":
        # sensitivity (Spearman) of peak T to each uncertain parameter
        sens={}
        for k,v in p.items():
            sens[k]=float(spearmanr(v,o['Tpeak'])[0])
        out['sensitivity_baseline']=sens
print("MC runtime for 5 designs x",N,"samples:",round(time.time()-t0,1),"s")
out['mc_runtime_s']=time.time()-t0; out['N']=N
out['designs']=res
for k,v in res.items(): print(k,{a:round(b,3) for a,b in v.items()})
print("sensitivity",out['sensitivity_baseline'])

# ---------- Typical e-rickshaw shift (reuse test): 3 h @0.3C at 45C ----------
TYP=[(3*3600,1.8)]
typ={}
for name in ["Baseline","Recommended"]:
    d=ALL[name]; rng=np.random.default_rng(11); p=sample_params(200,rng)
    o=simulate(d,p,TYP,charge=False)
    typ[name]=dict(Tpeak_med=float(np.median(o['Tpeak'])),Tpeak_p95=float(np.percentile(o['Tpeak'],95)),p_health=float(np.mean(o['Tpeak']<55)))
print("typical shift",typ); out['typical']=typ

# ---------- Safe operating map ----------
amb=[35,38,41,44,47,50]; crates=[0.5,1.0,1.5,2.0,2.5,3.0]
maps={}
for name in ["Baseline","Recommended"]:
    M=np.zeros((len(crates),len(amb)))
    for i,c in enumerate(crates):
        for j,a in enumerate(amb):
            prof=[(480,6.0*c),(1320,3.0)] if c>0.5 else [(1800,3.0)]
            rng=np.random.default_rng(3); p=sample_params(120,rng)
            o=simulate(ALL[name],p,prof,T_amb_duty=float(a),charge=False)
            hl=(o['Tpeak']<55)&(o['Ah_del']/o['Ah_req']>=0.95)
            sf=(o['Tpeak']<60)&(o['Ah_del']/o['Ah_req']>=0.95)
            # 2=healthy(>=90%), 1=safe-but-hot(>=95% <60), 0=risk
            M[i,j]=2 if hl.mean()>=0.9 else (1 if sf.mean()>=0.95 else 0)
    maps[name]=M.tolist()
out['maps']=maps; out['amb']=amb; out['crates']=crates
json.dump(out,open('results.json','w'),indent=1)
json.dump(curves,open('curves.json','w'))
for n,M in maps.items():
    print(n); 
    for r,c in zip(M,crates): print(c,r)
