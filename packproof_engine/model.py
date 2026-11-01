"""
PackProof model: 4S module thermal/electrical/charging model.

EVIDENCE CLASSES (every parameter carries one):
  DS  = taken from cell datasheet HX-TS-066 (Hunan Huaxing 32700-6000mAh)
  LIT = published literature / handbook value
  VEN = vendor listing (price / fan spec)
  ASM = assumption (range stated, covered by Monte Carlo, to be replaced by bench measurement)
"""
import numpy as np

# ---------------- verified inputs ----------------
CELL = dict(
    cap_Ah=6.0,            # DS 2.1
    v_nom=3.2,             # DS 2.4
    mass_kg=0.140,         # DS 2.6 (140 +/- 5 g)
    r_dc_max=0.016,        # DS 2.3  DC impedance <= 16 mOhm (1C, 10 s, 50% SOC, ~30C)
    i_cont_max=18.0,       # DS 2.11 18 A for <= 20 min
    t_dis_max=60.0,        # DS 2.14 discharge window -20..60 C
    d=0.0322, L=0.0705,    # DS 2.5 (m)
    cp=1109.0,             # LIT cylindrical LFP cell (Forgez 2010, tabulated in UMich paper)
    price_inr=315.0,       # VEN robu.in HX IFR32700 6000mAh 3C listing (incl GST)
)
CELL["area"] = np.pi*CELL["d"]*CELL["L"] + 2*np.pi*CELL["d"]**2/4   # m2 (geometry)
CELL["C_th"] = CELL["mass_kg"]*CELL["cp"]                           # J/K

# DS 2.14 charge window (T -> max C-rate)
def charge_limit_C(T):
    T = np.asarray(T)
    out = np.zeros_like(T, dtype=float)
    out = np.where(T < 0, 0.0, out)
    out = np.where((T >= 0) & (T < 10), 0.2, out)
    out = np.where((T >= 10) & (T < 15), 0.5, out)
    out = np.where((T >= 15) & (T < 45), 1.0, out)
    out = np.where((T >= 45) & (T <= 60), 0.5, out)
    return out

# DS 4.5 / 4.7: cycle life 2200 @ 30 C (1C), 1000 @ 55 C (1C); log-linear interpolation (ASM: interpolation)
def cycle_life(Tavg):
    slope = (np.log(1000) - np.log(2200)) / (55 - 30)
    return np.exp(np.log(2200) + slope*(np.asarray(Tavg) - 30))

AL = dict(rho=2700.0, cp=897.0, price_per_g=649/500)    # LIT / VEN (retail cut piece 300x300x2mm 500 g = Rs649)
FAN = dict(power_w=12*0.2, mass_kg=0.085, price_inr=75*1.18, h_forced=25.0)  # VEN electronicscomp 8025 12V 0.2A 85g Rs75+18%GST; h LIT lower bound of 25-250

# ---------------- designs ----------------
DESIGNS = {
    "Baseline":   dict(series=4, par=1, plate=False, fan=False, derate_start=None),
    "A: plate":   dict(series=4, par=1, plate=True,  fan=False, derate_start=None),
    "B: fan+BMS": dict(series=4, par=1, plate=False, fan=True,  derate_start=50.0),
    "C: 4S2P":    dict(series=4, par=2, plate=False, fan=False, derate_start=None),
}
PLATE_MASS = 0.130*0.070*0.002*AL["rho"]       # kg  (130x70x2 mm)
PLATE_AREA = 2*0.130*0.070                      # m2 both faces
PLATE_COST = PLATE_MASS*1000*AL["price_per_g"]

def sample_params(n, rng, scale=1.0):
    """ASM ranges (uniform). r_dc: DS upper bound 16 mOhm, lower bound ASM 10 mOhm."""
    return dict(
        r_dc=rng.uniform(0.010, 0.016, n),
        k_sus=rng.uniform(1.0, 1.5, n),        # sustained-vs-10s resistance factor
        h_nat=rng.uniform(3.0, 8.0, n),        # LIT 5-25 natural; enclosed -> low end
        h_forced=rng.uniform(15.0, 35.0, n),   # LIT forced >=25 typical; ASM spread
        g_nb=rng.uniform(0.03, 0.30, n),       # ASM cell-to-cell conduction W/K
        g_pad=rng.uniform(0.5, 3.0, n),        # ASM cell-plate pad conduction W/K
    )

def nominal_params():
    return dict(r_dc=np.array([0.013]), k_sus=np.array([1.25]), h_nat=np.array([5.0]),
                h_forced=np.array([25.0]), g_nb=np.array([0.1]), g_pad=np.array([1.5]))

EXPO = np.array([0.75, 0.5, 0.5, 0.75])   # ASM exposed fraction of cell area (end cells see more air)

def _G_amb(des, p, fan_on):
    par = des["par"]
    h = np.where(fan_on, p["h_forced"], p["h_nat"])[:, None] if des["fan"] else p["h_nat"][:, None]
    return h*CELL["area"]*EXPO[None, :]*(1.0 if par == 1 else 1.6)

def simulate(des, p, profile, T_amb_duty=45.0, T_amb_charge=35.0, dt=1.0, charge=True, T_ini=None):
    """profile: list of (duration_s, I_cell_pack_A) discharge current of the string (per series string total).
    Returns dict of arrays. Current per cell = I/par."""
    n = len(p["r_dc"])
    par = des["par"]
    C_node = CELL["C_th"]*par
    R_single = (p["r_dc"]*p["k_sus"])[:, None]        # per-cell sustained resistance (ohm)
    T = np.full((n, 4), T_amb_duty if T_ini is None else T_ini, float)
    Tp = np.full(n, T[:, 0].mean()) if des["plate"] else None
    C_pl = PLATE_MASS*AL["cp"]
    t = 0.0
    Ah_req = 0.0; Ah_del = np.zeros(n); Wh_del = np.zeros(n); Wh_fan = np.zeros(n)
    cutoff = np.zeros(n, bool)
    Tmax_hist = []; Tavg_acc = np.zeros(n); steps = 0
    Tmax_peak = T.max(axis=1).copy()

    def step(T, Tp, I_node_cell, T_amb, fan_on):
        # I_node_cell: (n,) current per cell (A)
        Q = par*(I_node_cell[:, None]**2)*R_single    # W per node (par cells, each carrying I_cell)
        G_amb = _G_amb(des, p, fan_on)
        dT = Q - G_amb*(T - T_amb)
        # neighbour conduction
        gnb = p["g_nb"][:, None]
        flux = gnb*(T[:, :-1] - T[:, 1:])
        dT[:, :-1] -= flux
        dT[:, 1:] += flux
        if des["plate"]:
            gp = p["g_pad"][:, None]
            fl = gp*(T - Tp[:, None])
            dT -= fl
            dTp = fl.sum(axis=1) - p["h_nat"]*PLATE_AREA*(Tp - T_amb)
            Tp = Tp + dt*dTp/C_pl
        T = T + dt*dT/C_node
        return T, Tp

    # ---- discharge ----
    for dur, I_req in profile:
        nsteps = int(dur/dt)
        for _ in range(nsteps):
            Tmax = T.max(axis=1)
            fan_on = Tmax > 40.0 if des["fan"] else np.zeros(n, bool)
            I = np.full(n, float(I_req))
            if des["derate_start"] is not None:
                ds = des["derate_start"]
                f = np.clip(1 - 0.8*(Tmax - ds)/(58.0 - ds), 0.2, 1.0)
                f = np.where(Tmax < ds, 1.0, f)
                I = I*f
            cutoff |= Tmax >= CELL["t_dis_max"]
            I = np.where(cutoff, 0.0, I)
            I_cell = I/par
            T, Tp = step(T, Tp, I_cell, T_amb_duty, fan_on)
            Ah_del += I*dt/3600.0
            v = CELL["v_nom"] - I_cell*R_single[:, 0]     # terminal V per cell (OCV = nominal, ASM)
            Wh_del += (v*4*I*dt/3600.0)
            Wh_fan += np.where(fan_on, FAN["power_w"]*dt/3600.0, 0.0) if des["fan"] else 0.0
            Tmax_peak = np.maximum(Tmax_peak, T.max(axis=1))
            Tavg_acc += T.mean(axis=1); steps += 1
            if int(t) % 10 == 0:
                Tmax_hist.append(T.max(axis=1).copy())
            t += dt
        Ah_req += I_req*dur/3600.0

    Wh_net = Wh_del - Wh_fan
    out = dict(Tpeak=Tmax_peak, Ah_del=Ah_del, Ah_req=Ah_req, Wh_net=Wh_net, cutoff=cutoff,
               hist=np.array(Tmax_hist), T_end=T.max(axis=1).copy(), Tavg_dis=Tavg_acc/max(steps, 1))

    # ---- charge-back: choose best of (charge now at window limit) vs (wait until <=44.5 then 1C) ----
    if charge:
        out["t_charge_s"], out["Tavg_chg"] = _charge(des, p, T, Tp, Ah_del, par, T_amb_charge, dt, step)
    return out

def _charge(des, p, T0, Tp0, Ah_need, par, T_amb, dt, step_fn):
    n = len(Ah_need)
    best = np.full(n, np.inf)
    best_Tavg = np.zeros(n)
    for strategy in ("now", "wait"):
        T = T0.copy(); Tp = None if Tp0 is None else Tp0.copy()
        rem = Ah_need.copy()
        t_used = np.zeros(n); Tacc = np.zeros(n)
        done = np.zeros(n, bool)
        for k in range(int(6*3600/dt)):
            Tmax = T.max(axis=1)
            lim = charge_limit_C(Tmax)*CELL["cap_Ah"]*par     # A for string
            if strategy == "wait":
                lim = np.where(Tmax > 44.5, 0.0, lim)
            I = np.minimum(lim, rem*3600.0/dt)
            I = np.where(done, 0.0, I)
            fan_on = (Tmax > 40.0) if des["fan"] else np.zeros(n, bool)
            T, Tp = step_fn(T, Tp, I/par, T_amb, fan_on)
            rem -= I*dt/3600.0
            newly = (rem <= 1e-6) & (~done)
            t_used = np.where(done, t_used, t_used + dt)
            Tacc += T.mean(axis=1)*(~done)
            done |= newly
            if done.all():
                break
        t_used = np.where(done, t_used, np.inf)
        better = t_used < best
        best = np.where(better, t_used, best)
        best_Tavg = np.where(better, Tacc/np.maximum(t_used, 1), best_Tavg)
    return best, best_Tavg

def design_cost_mass(des):
    ncell = des["series"]*des["par"]
    cost = ncell*CELL["price_inr"]
    mass = ncell*CELL["mass_kg"]
    if des["plate"]:
        cost += PLATE_COST; mass += PLATE_MASS
    if des["fan"]:
        cost += FAN["price_inr"]; mass += FAN["mass_kg"]
    return cost, mass

STRESS = [(480, 15.0), (1320, 3.0)]      # 8 min @ 2.5C then 22 min @ 0.5C (string current, A, for 6 Ah)
