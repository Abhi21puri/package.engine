"""PackProof prototype: Streamlit front-end for the battery-module engine (model.py).
Run:  pip install -r requirements.txt  &&  streamlit run app.py
"""
import os, sys, json, itertools, datetime, hashlib, threading
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

st.set_page_config(page_title="PackProof", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = {"DS": "Datasheet", "LIT": "Literature", "VEN": "Vendor listing", "ASM": "Assumed", "MEAS": "Measured"}

# ---------------- cell file (reusable: swap the JSON, same engine) ----------------
LOCK = threading.RLock()   # model.CELL is module-global: serialise access so parallel visitors cannot clash
REQUIRED = ("name", "cap_Ah", "v_nom", "mass_kg", "r_dc_max", "i_cont_max", "t_dis_max", "d", "L", "cp", "price_inr")
def apply_cell(cfg):
    for k in ("cap_Ah", "v_nom", "mass_kg", "r_dc_max", "i_cont_max", "t_dis_max", "d", "L", "cp", "price_inr"):
        M.CELL[k] = cfg[k]
    M.CELL["area"] = np.pi*cfg["d"]*cfg["L"] + 2*np.pi*cfg["d"]**2/4
    M.CELL["C_th"] = cfg["mass_kg"]*cfg["cp"]

st.sidebar.title("PackProof")
st.sidebar.caption("Test a battery design on a computer first, and know how sure we are.")
up = st.sidebar.file_uploader("Cell file (JSON)", type="json")
try:
    cfg = json.load(up) if up else json.load(open(os.path.join(HERE, "cell_hx32700.json")))
    miss = [k for k in REQUIRED if k not in cfg]
    bad = [k for k in REQUIRED[1:] if k in cfg and not (isinstance(cfg[k], (int, float)) and cfg[k] > 0)]
    if miss or bad:
        st.error(f"Cell file problem. Missing: {miss or 'none'}; not positive numbers: {bad or 'none'}"); st.stop()
except json.JSONDecodeError:
    st.error("Cell file is not valid JSON."); st.stop()
CELLID = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
st.info("Design aid built on datasheet values and stated assumptions. Results are simulated and not yet bench-measured. Not a safety certificate.")
st.sidebar.write(f"**Cell:** {cfg['name']}")
cap = cfg["cap_Ah"]

st.sidebar.subheader("Ride")
scen = st.sidebar.selectbox("Scenario", ["Stress hill-climb", "Normal e-rickshaw shift", "Custom"])
if scen == "Stress hill-climb":
    peakC, peak_min, cruiseC, cruise_min = 2.5, 8, 0.5, 22
elif scen == "Normal e-rickshaw shift":
    peakC, peak_min, cruiseC, cruise_min = 0.3, 180, 0.0, 0
else:
    peakC = st.sidebar.slider("Peak load (C of one cell)", 0.2, 3.0, 2.5, 0.1)
    peak_min = st.sidebar.slider("Peak minutes", 1, 60, 8)
    cruiseC = st.sidebar.slider("Cruise load (C)", 0.0, 1.0, 0.5, 0.1)
    cruise_min = st.sidebar.slider("Cruise minutes", 0, 60, 22)
amb = st.sidebar.slider("Outside temperature (C)", 25, 50, 45)
amb_chg = st.sidebar.slider("Temperature while charging (C)", 20, 45, 35)
profile = [(int(peak_min*60), peakC*cap)] + ([(int(cruise_min*60), cruiseC*cap)] if cruise_min else [])
N = st.sidebar.select_slider("Virtual cells (Monte Carlo)", [100, 200, 500, 1000], 200)
cal = st.sidebar.number_input("Resistance calibration factor (1.0 = datasheet only)", 0.3, 3.0, float(st.session_state.get("cal", 1.0)), 0.05)
st.sidebar.caption("Voltage is 12.8 V (4 cells in series). Load amps are set from the 6 Ah reference cell, so adding parallel cells shares the same current.")

def params(n, seed=7):
    p = M.sample_params(n, np.random.default_rng(seed))
    p["r_dc"] = p["r_dc"]*cal
    return p

@st.cache_data(show_spinner=False)
def run(des_t, profile_t, amb, amb_chg, n, cal_, cellname, charge=True):
    des = dict(des_t)
    with LOCK:
        apply_cell(cfg)
        o = M.simulate(des, params(n), list(profile_t), T_amb_duty=float(amb), T_amb_charge=float(amb_chg), charge=charge)
        cost, mass = M.design_cost_mass(des)
    d = o["Ah_del"]/max(o["Ah_req"], 1e-9)
    r = dict(peak_med=float(np.median(o["Tpeak"])), peak_p95=float(np.percentile(o["Tpeak"], 95)),
             p_safe=float(np.mean(o["Tpeak"] < cfg["t_dis_max"])),
             p_healthy=float(np.mean((o["Tpeak"] < 55) & (d >= 0.95))),
             ride=float(np.median(d)), wh=float(np.median(o["Wh_net"])), cost=cost, mass=mass,
             hist_med=np.median(o["hist"], 1).tolist(), hist_lo=np.percentile(o["hist"], 5, 1).tolist(),
             hist_hi=np.percentile(o["hist"], 95, 1).tolist())
    if charge:
        Tavg = (o["Tavg_dis"]*sum(x for x, _ in profile_t) + o["Tavg_chg"]*np.where(np.isinf(o["t_charge_s"]), 0, o["t_charge_s"]))/(sum(x for x, _ in profile_t) + np.where(np.isinf(o["t_charge_s"]), 0, o["t_charge_s"]))
        tc = o["t_charge_s"]; r["charge_fail"] = float(np.mean(np.isinf(tc)))
        tc = np.where(np.isinf(tc), np.nan, tc)
        r["charge_min"] = float(np.nanmedian(tc)/60) if not np.all(np.isnan(tc)) else float("nan"); r["life"] = float(np.median(M.cycle_life(Tavg)))
    return r

def key(des): return tuple(sorted(des.items()))

st.title("PackProof: virtual battery-module design lab")
tabs = st.tabs(["1 Design & run", "2 Compare designs", "3 Safe-zone map", "4 Calibrate on bench data", "5 Evidence & run record"])

# ---------------- tab 1 ----------------
with tabs[0]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Choose a design")
        par = st.radio("Cells in parallel", [1, 2], horizontal=True)
        plate = st.checkbox("Aluminium heat-spreader plate (130x70x2 mm)")
        fan = st.checkbox("12 V fan (on above 40 C)")
        ds = st.select_slider("BMS slows current above (C)", [None, 50.0, 53.0, 55.0], 53.0 if fan else None,
                              format_func=lambda v: "never" if v is None else f"{v:.0f}")
        des = dict(series=4, par=par, plate=plate, fan=fan, derate_start=ds)
        go = st.button("Run ride", type="primary") or ("last" not in st.session_state)
    with c2:
        if go or "last" in st.session_state:
            if go:
                with st.spinner("Simulating virtual cells..."):
                    st.session_state["last"] = (des, run(key(des), tuple(profile), amb, amb_chg, N, cal, CELLID))
            d0, r = st.session_state["last"]
            t = np.arange(len(r["hist_med"]))*10/60
            fig, ax = plt.subplots(figsize=(6.5, 3))
            ax.fill_between(t, r["hist_lo"], r["hist_hi"], alpha=.2, color="#E86A1C")
            ax.plot(t, r["hist_med"], color="#E86A1C", lw=2, label="hottest cell (typical)")
            ax.axhline(cfg["t_dis_max"], color="#C0392B", ls="--", label=f"{cfg['t_dis_max']:.0f} C datasheet limit")
            ax.axhline(55, color="#D99A1E", ls=":", label="55 C: life drops")
            ax.set_xlabel("minutes"); ax.set_ylabel("C"); ax.legend(frameon=False, fontsize=8)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
            st.pyplot(fig)
            a, b, c, d = st.columns(4)
            a.metric("Peak (typical)", f"{r['peak_med']:.1f} C", f"95% of runs below {r['peak_p95']:.1f}", delta_color="off")
            b.metric("Chance healthy", f"{r['p_healthy']*100:.0f}%", help="Stays under 55 C and finishes at least 95% of the ride")
            c.metric("Chance under limit", f"{r['p_safe']*100:.0f}%")
            d.metric("Recharge", "not possible" if np.isnan(r["charge_min"]) else f"{r['charge_min']:.0f} min", help="Constant-current phase only")
            e, f_, g, h = st.columns(4)
            e.metric("Energy out", f"{r['wh']:.1f} Wh"); f_.metric("Ride finished", f"{r['ride']*100:.0f}%")
            g.metric("Cost / weight", f"Rs {r['cost']:,.0f} / {r['mass']*1000:.0f} g"); h.metric("Est. life", f"{r['life']:,.0f} cycles")
            # datasheet rule check
            st.subheader("Datasheet rule check")
            I_cell = max(i for _, i in profile)/d0["par"]
            msgs = []
            long_hi = [m for m, i in profile if i/d0["par"] >= 0.99*cfg["i_cont_max"] and m > 20*60]
            if long_hi: msgs.append(("FAIL", "DS-2.11", "load at the 18 A rating lasts longer than the 20 minutes the datasheet allows"))
            if r.get("charge_fail", 0) > 0.05: msgs.append(("FAIL", "DS-2.14", "charging is not allowed at these cell temperatures in many runs"))
            if I_cell > cfg["i_cont_max"]: msgs.append(("FAIL", "DS-2.11", f"{I_cell:.1f} A per cell exceeds {cfg['i_cont_max']:.0f} A"))
            elif I_cell > 0.9*cfg["i_cont_max"]: msgs.append(("WARN", "DS-2.11", f"{I_cell:.1f} A per cell is within 10% of the {cfg['i_cont_max']:.0f} A limit"))
            if r["p_safe"] < 0.99: msgs.append(("FAIL" if r["p_safe"] < 0.95 else "WARN", "DS-2.14", f"only {r['p_safe']*100:.0f}% of runs stay under {cfg['t_dis_max']:.0f} C"))
            if r["peak_med"] >= 55: msgs.append(("WARN", "DS-4.5/4.7", "typical peak is at or above 55 C: datasheet cycle life falls from 2200 to 1000"))
            if r["ride"] < 0.95: msgs.append(("WARN", "interface", f"derating/cut-off trims the ride to {r['ride']*100:.0f}%"))
            if d0["fan"]: msgs.append(("INFO", "interface", "fan draws pack energy; included in Energy out"))
            if not msgs: msgs.append(("PASS", "all", "no datasheet rule broken in this ride"))
            for lvl, cid, m in msgs: st.write(f"**{lvl}** [{cid}] {m}")
            if scen == "Normal e-rickshaw shift" and r["peak_med"] - amb < 3:
                st.success(f"Cells warm only {r['peak_med']-amb:.1f} C on this ride: extra cooling is not needed.")
        else:
            st.info("Pick a design and press Run ride.")

# ---------------- tab 2 ----------------
with tabs[1]:
    st.write("Baseline, three options, and a sweep that finds the cheapest design meeting your rule.")
    c1, c2, c3 = st.columns(3)
    p_h = c1.slider("Min. share of runs healthy", 0.5, 1.0, 0.90, 0.01)
    p_s = c2.slider("Min. share of runs under limit", 0.8, 1.0, 0.99, 0.01)
    ride_min = c3.slider("Min. ride finished", 0.80, 1.0, 0.95, 0.01)
    if st.button("Compare + find recommended"):
        base = dict(series=4, par=1, plate=False, fan=False, derate_start=None)
        named = {"Baseline": base, "A: plate": {**base, "plate": True}, "B: fan + BMS": {**base, "fan": True, "derate_start": 50.0},
                 "C: twice the cells": {**base, "par": 2}}
        rows = []
        with st.spinner("Running designs..."):
            for nm, d in named.items():
                r = run(key(d), tuple(profile), amb, amb_chg, N, cal, CELLID); rows.append((nm, d, r))
            sweep = []
            for pl, fn, dsv in itertools.product([False, True], [False, True], [None, 53.0, 55.0]):
                d = {**base, "plate": pl, "fan": fn, "derate_start": dsv}
                r = run(key(d), tuple(profile), amb, amb_chg, N, cal, CELLID, charge=False); sweep.append((d, r))
        ok = [(d, r) for d, r in sweep if r["p_healthy"] >= p_h and r["p_safe"] >= p_s and r["ride"] >= ride_min]
        ok.sort(key=lambda x: (x[1]["cost"], x[1]["mass"]))
        if ok:
            rd = ok[0][0]; rr = run(key(rd), tuple(profile), amb, amb_chg, N, cal, CELLID)
            rows.append(("Recommended", rd, rr))
            st.success(f"Recommended: plate={rd['plate']}, fan={rd['fan']}, slow current above {rd['derate_start']} C")
        else:
            st.warning("No single-string design meets this rule. Relax a threshold or consider more parallel cells (design C).")
        b = rows[0][2]
        df = pd.DataFrame([{"Design": n, "Peak C": round(r["peak_med"], 1), "Healthy runs %": round(r["p_healthy"]*100),
                            "Under limit %": round(r["p_safe"]*100), "Energy Wh": round(r["wh"], 1), "Recharge min": (None if np.isnan(r["charge_min"]) else round(r["charge_min"])),
                            "Extra Rs": round(r["cost"]-b["cost"]), "Extra g": round((r["mass"]-b["mass"])*1000), "Est. life": round(r["life"])}
                           for n, d, r in rows])
        st.dataframe(df, hide_index=True, width='stretch')
        st.session_state["compare"] = df.to_dict("records")
        st.caption("Est. life is interpolated between the datasheet's 2,200-cycle (30 C) and 1,000-cycle (55 C) points.")

# ---------------- tab 3 ----------------
with tabs[2]:
    st.write("Where does a design stay healthy? Green: healthy. Amber: under limit but hot or ride trimmed. Red: under 95% chance below limit.")
    if st.button("Draw safe-zone map"):
        ambs = [35, 40, 45, 50]; cr = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        cols = st.columns(2)
        base = dict(series=4, par=1, plate=False, fan=False, derate_start=None)
        rec = dict(series=4, par=1, plate=True, fan=True, derate_start=53.0)
        with st.spinner("Mapping..."):
            for col, (nm, d) in zip(cols, [("Baseline", base), ("Plate + fan + BMS", rec)]):
                G = np.zeros((len(cr), len(ambs)))
                for i, c in enumerate(cr):
                    prof = [(480, c*cap), (1320, 0.5*cap)] if c > 0.5 else [(1800, 0.5*cap)]
                    for j, a in enumerate(ambs):
                        with LOCK:
                            apply_cell(cfg); o = M.simulate(d, params(120, 3), prof, T_amb_duty=float(a), charge=False)
                        hl = ((o["Tpeak"] < 55) & (o["Ah_del"]/o["Ah_req"] >= 0.95)).mean(); sf = (o["Tpeak"] < cfg["t_dis_max"]).mean()
                        G[i, j] = 2 if hl >= .9 else (1 if sf >= .95 else 0)
                fig, ax = plt.subplots(figsize=(4, 3.2))
                from matplotlib.colors import ListedColormap
                ax.imshow(G, origin="lower", cmap=ListedColormap(["#E4817A", "#F4C766", "#7CC99A"]), vmin=0, vmax=2, aspect="auto")
                ax.set_xticks(range(4)); ax.set_xticklabels(ambs); ax.set_yticks(range(6)); ax.set_yticklabels([f"{c}C" for c in cr])
                ax.set_xlabel("outside C"); ax.set_ylabel("peak load"); ax.set_title(nm)
                col.pyplot(fig)

# ---------------- tab 4 ----------------
with tabs[3]:
    st.write("Bench step: one cell, fused, supervised, ~25 C room, 15 A for 8 minutes (inside the 18 A / 20 min datasheet rating). "
             "Enter the measured temperature rise of the cell body to calibrate the model.")
    rng = np.random.default_rng(5); n = 4000
    r_ = rng.uniform(.010, .016, n)*cal; k_ = rng.uniform(1.0, 1.5, n); h_ = rng.uniform(5, 15, n)
    T = np.full(n, 25.0)
    area = np.pi*cfg["d"]*cfg["L"] + 2*np.pi*cfg["d"]**2/4; Cth = cfg["mass_kg"]*cfg["cp"]
    for _ in range(480): T += (15.0**2*r_*k_ - h_*area*(T-25.0))/Cth
    lo, med, hi = np.percentile(T-25, [5, 50, 95])
    st.info(f"Prediction on record: warms {lo:.1f} to {hi:.1f} K (typical {med:.1f} K).")
    meas = st.number_input("Measured temperature rise (K), 0 = not measured yet", 0.0, 60.0, 0.0, 0.5)
    if meas > 0:
        if lo <= meas <= hi: st.success("Inside the predicted band: model not contradicted. Tag heating as MEASURED.")
        else: st.warning("Outside the band: model needs calibration.")
        factor = round(meas/med, 2)
        st.write(f"Suggested resistance calibration factor: **{factor}**")
        if st.button("Apply factor"): st.session_state["cal"] = factor; st.rerun()

# ---------------- tab 5 ----------------
with tabs[4]:
    ev = pd.DataFrame([
        ("Capacity, mass, DC resistance max, 18 A limit, 60 C limit, charge window, cycle points", "DS"),
        ("Cell heat capacity", "LIT"), ("Aluminium properties; convection ranges", "LIT"),
        ("Cell, fan, plate prices", "VEN"),
        ("Cell resistance lower bound, sustained-resistance factor, enclosure convection, cell-to-cell conduction, pad conduction, exposed area", "ASM"),
        ("Heating of a cell at 2.5C", "MEAS" if "cal" in st.session_state and st.session_state["cal"] != 1.0 else "ASM")],
        columns=["Input", "Evidence"])
    ev["Evidence"] = ev["Evidence"].map(TAG)
    st.dataframe(ev, hide_index=True, width='stretch')
    if "last" in st.session_state:
        d0, r = st.session_state["last"]
        rec = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), cell=cfg, rules=cfg.get("rules"),
                   ride=dict(scenario=scen, profile_amps=profile, outside_C=amb, charge_C=amb_chg), runs=N, resistance_cal=cal,
                   design=d0, results={k: v for k, v in r.items() if not k.startswith("hist")}, compare=st.session_state.get("compare"),
                   evidence=ev.to_dict("records"), model_sha=hashlib.sha256(open(os.path.join(HERE, "model.py"), "rb").read()).hexdigest()[:12],
                   note="Design aid, not a safety certificate.")
        st.download_button("Download run record (JSON)", json.dumps(rec, indent=1), "packproof_run_record.json")
    else:
        st.caption("Run a ride first; the record links requirement, design, assumptions, results and evidence.")
