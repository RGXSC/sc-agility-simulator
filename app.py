import streamlit as st
import json, math
from datetime import datetime

st.set_page_config(layout="wide", page_title="Supply Chain Agility Simulator", page_icon="\U0001f3ed")

# ════════════════════════════════════════════════════════════════
# SIMULATION ENGINE — 2 STORES
# ════════════════════════════════════════════════════════════════
@st.cache_data
def run_simulation(weeks, init_store, init_cw, init_semi, init_rawmat,
                   order_freq, mat_lt, semi_lt, fp_lt, dist_lt,
                   cap_start, cap_ramp, base_forecast,
                   demand_mult, ramp_start, ramp_end,
                   price, var_cost, fixed_pct, store_a_pct, smart_distrib):

    phys_lt = mat_lt + semi_lt + fp_lt + dist_lt
    coverage = phys_lt + order_freq
    pct_a = store_a_pct / 100.0
    pct_b = 1.0 - pct_a

    # Demand profile (total)
    demand = {}
    for w in range(0, weeks + 1):
        if w < ramp_start:
            demand[w] = base_forecast
        elif ramp_start == ramp_end:
            demand[w] = round(base_forecast * demand_mult)
        elif w <= ramp_end:
            p = (w - ramp_start) / (ramp_end - ramp_start)
            demand[w] = round(base_forecast + (base_forecast * demand_mult - base_forecast) * p)
        else:
            demand[w] = round(base_forecast * demand_mult)

    # Pipelines
    mat_pipe = [0.0] * max(1, mat_lt)
    semi_pipe = [0.0] * max(1, semi_lt)
    fp_pipe = [0.0] * max(1, fp_lt)
    dist_pipe_a = [0.0] * max(1, dist_lt)
    dist_pipe_b = [0.0] * max(1, dist_lt)

    # Initial store stock: always 50/50 split (deliberate "push" allocation)
    store_a = float(init_store) / 2.0
    store_b = float(init_store) / 2.0
    raw_mat = float(init_rawmat)
    semi = float(init_semi)
    cw = float(init_cw)

    pb = 0.0
    pc = float(cap_start); ps = False
    sc_ = float(cap_start); ss_ = False
    fpc = float(cap_start); fps_ = False
    co = 0.0; cas = 0.0

    ow = list(range(order_freq, weeks + 1, order_freq)) if order_freq > 1 else list(range(1, weeks + 1))
    states = []

    for w in range(1, weeks + 1):
        s = {'week': w}
        dem_total = demand[w]
        dem_a = round(dem_total * pct_a)
        dem_b = dem_total - dem_a  # ensure exact total
        ff = dem_total
        s['demand'] = dem_total; s['demand_a'] = dem_a; s['demand_b'] = dem_b
        s['forecast'] = round(ff, 1)

        # 1. Arrivals from pipes
        m_arr = mat_pipe[0]; sm_arr = semi_pipe[0]; fp_arr = fp_pipe[0]
        da_arr = dist_pipe_a[0]; db_arr = dist_pipe_b[0]
        d_arr_total = da_arr + db_arr
        s['mat_arr'] = round(m_arr, 1); s['semi_arr'] = round(sm_arr, 1)
        s['fp_arr'] = round(fp_arr, 1)
        s['dist_arr_a'] = round(da_arr, 1); s['dist_arr_b'] = round(db_arr, 1)
        s['dist_arr'] = round(d_arr_total, 1)
        cas += d_arr_total

        # 2. Store A — sell
        avail_a = store_a + da_arr
        sales_a = min(dem_a, avail_a)
        missed_a = max(0, dem_a - sales_a)
        store_a = avail_a - sales_a

        # 3. Store B — sell
        avail_b = store_b + db_arr
        sales_b = min(dem_b, avail_b)
        missed_b = max(0, dem_b - sales_b)
        store_b = avail_b - sales_b

        sales = sales_a + sales_b
        missed = missed_a + missed_b
        store_total = store_a + store_b

        s.update({
            'store_a': round(store_a, 1), 'store_b': round(store_b, 1),
            'sales_a': round(sales_a, 1), 'sales_b': round(sales_b, 1),
            'missed_a': round(missed_a, 1), 'missed_b': round(missed_b, 1),
            'sales': round(sales, 1), 'missed': round(missed, 1),
            'store_stock': round(store_total, 1),
            'store_before': round(avail_a + avail_b, 1),
        })

        # 4. Supplier
        if pb > 0.01:
            if not ps: ps = True; pc = float(cap_start)
            else: pc = min(pc * (1 + cap_ramp), cap_start * 10)
            shipped = math.ceil(min(pb, pc)); pb -= shipped
        else:
            shipped = 0.0; ps = False; pc = float(cap_start)
        s['supplier_shipped'] = round(shipped, 1); s['supplier_cap'] = round(pc, 0)

        # 5. Raw mat receives
        raw_mat += m_arr; s['raw_mat_before_prod'] = round(raw_mat, 1)
        semi += sm_arr; cw += fp_arr

        # 6. Semi input
        if raw_mat > 0.01:
            if not ss_: ss_ = True; sc_ = float(cap_start)
            else: sc_ = min(sc_ * (1 + cap_ramp), cap_start * 10)
            si = math.ceil(min(raw_mat, sc_)); raw_mat -= si
        else:
            si = 0.0; ss_ = False; sc_ = float(cap_start)
        s['semi_input'] = round(si, 1); s['semi_cap'] = round(sc_, 0)
        s['raw_mat_stock'] = round(raw_mat, 1)

        # 7. FP input
        if semi > 0.01:
            if not fps_: fps_ = True; fpc = float(cap_start)
            else: fpc = min(fpc * (1 + cap_ramp), cap_start * 10)
            fi = math.ceil(min(semi, fpc)); semi -= fi
        else:
            fi = 0.0; fps_ = False; fpc = float(cap_start)
        s['fp_input'] = round(fi, 1); s['fp_cap'] = round(fpc, 0)
        s['semi_stock'] = round(semi, 1)

        # 8. CW — smart allocation to 2 dist pipes based on net need
        ship_out = math.ceil(cw); cw = 0.0
        s['cw_shipped'] = round(ship_out, 1); s['cw_stock'] = 0.0

        # Net need = (forecast_store × dist_lt) - current_store_stock
        # Smart: proportional allocation based on relative need
        # Dumb (push): always 50/50 regardless of actual need
        if smart_distrib:
            need_a = max(0, dem_a * dist_lt - store_a)
            need_b = max(0, dem_b * dist_lt - store_b)
            total_need = need_a + need_b
            if total_need > 0.01 and ship_out > 0:
                alloc_a = round(ship_out * (need_a / total_need))
                alloc_b = ship_out - alloc_a
            elif ship_out > 0:
                alloc_a = round(ship_out * pct_a)
                alloc_b = ship_out - alloc_a
            else:
                alloc_a = 0; alloc_b = 0
        else:
            # Push model: 50/50 split regardless of demand
            alloc_a = round(ship_out * 0.5)
            alloc_b = ship_out - alloc_a
        s['alloc_a'] = round(alloc_a, 1); s['alloc_b'] = round(alloc_b, 1)

        # 9. Update pipes
        mat_pipe = mat_pipe[1:] + [shipped]
        semi_pipe = semi_pipe[1:] + [si]
        fp_pipe = fp_pipe[1:] + [fi]
        dist_pipe_a = dist_pipe_a[1:] + [alloc_a]
        dist_pipe_b = dist_pipe_b[1:] + [alloc_b]

        s['mat_pipe'] = [round(x, 1) for x in mat_pipe]
        s['semi_pipe'] = [round(x, 1) for x in semi_pipe]
        s['fp_pipe'] = [round(x, 1) for x in fp_pipe]
        s['dist_pipe_a'] = [round(x, 1) for x in dist_pipe_a]
        s['dist_pipe_b'] = [round(x, 1) for x in dist_pipe_b]

        # 10. Order (based on total)
        if w in ow:
            pnd = co - cas; tgt = ff * coverage
            od = math.ceil(max(0, tgt - store_total - pnd))
            co += od; pb += od; s['order'] = round(od, 0)
        else:
            s['order'] = 0
        s['pending'] = round(co - cas, 0); s['backlog'] = round(pb, 0)
        s['coverage'] = coverage

        # Commentary
        parts = []
        # Store A
        if missed_a > 0.5:
            parts.append(f"\U0001f534 A: lost {missed_a:.0f}/{dem_a:.0f}.")
        else:
            parts.append(f"\U0001f7e2 A: sold {sales_a:.0f}/{dem_a:.0f}, stk {store_a:.0f}.")
        # Store B
        if missed_b > 0.5:
            parts.append(f"\U0001f534 B: lost {missed_b:.0f}/{dem_b:.0f}.")
        else:
            parts.append(f"\U0001f7e2 B: sold {sales_b:.0f}/{dem_b:.0f}, stk {store_b:.0f}.")
        if s['order'] > 0:
            parts.append(f"\U0001f4cb ORDER {od:.0f}.")
        if shipped > 0.5:
            parts.append(f"\U0001f69a Supplier {shipped:.0f} (cap {pc:.0f}).")
        if ship_out > 0.5:
            mode = "smart" if smart_distrib else "push 50/50"
            parts.append(f"\U0001f4e6 CW\u2192A:{alloc_a:.0f} B:{alloc_b:.0f} ({mode}).")
        s['comment'] = " ".join(parts)
        states.append(s)

    return states

def compute_kpis(states, price, var_cost, fixed_pct, base_forecast, weeks):
    ts = sum(s['sales'] for s in states)
    tm = sum(s['missed'] for s in states)
    td = sum(s['demand'] for s in states)
    tfp = sum(s['fp_input'] for s in states)
    rev = ts * price; vc = tfp * var_cost; gm = rev - vc
    fx = base_forecast * 52 * price * fixed_pct * (weeks / 52)
    mg = gm - fx
    return {
        'total_demand': td, 'total_sales': ts, 'total_missed': tm,
        'svc_level': ts / td if td > 0 else 0,
        'stockout_weeks': sum(1 for s in states if s['missed'] > 0.5),
        'revenue': rev, 'var_cost': vc, 'gm': gm, 'fixed': fx,
        'margin': mg, 'margin_pct': mg / rev if rev > 0 else 0,
        'produced': tfp, 'store_end': states[-1]['store_stock'],
        'lost_rev': tm * price,
        'missed_a': sum(s['missed_a'] for s in states),
        'missed_b': sum(s['missed_b'] for s in states),
        'sales_a': sum(s['sales_a'] for s in states),
        'sales_b': sum(s['sales_b'] for s in states),
    }

def cumulative_kpis(states, week, price, var_cost, fixed_pct, base_forecast, total_weeks):
    sub = states[:week]
    ts = sum(s['sales'] for s in sub); tm = sum(s['missed'] for s in sub)
    td = sum(s['demand'] for s in sub); tfp = sum(s['fp_input'] for s in sub)
    rev = ts * price; vc = tfp * var_cost; gm = rev - vc
    fx = base_forecast * 52 * price * fixed_pct * (week / 52)
    mg = gm - fx
    return {'sales': ts, 'missed': tm, 'demand': td, 'revenue': rev,
            'svc_level': ts / td if td > 0 else 0, 'margin': mg,
            'stockout_wks': sum(1 for s in sub if s['missed'] > 0.5),
            'missed_a': sum(s['missed_a'] for s in sub),
            'missed_b': sum(s['missed_b'] for s in sub)}


# ════════════════════════════════════════════════════════════════
# SC VISUALIZATION — SUPPLIER ▸▸▸ CW ▸▸ STORE A / STORE B
# ════════════════════════════════════════════════════════════════
def make_sc_html(state, params):
    cap_ref = max(1, params['cap_start'] * 2.5)

    def pipe_box_style(v, hue):
        if v > 0.5:
            light = min(88, 45 + int((v / cap_ref) * 43))
            return f"background:hsl({hue},48%,{light}%);color:{'#fff' if light<62 else '#333'};", f"{v:.0f}"
        return f"background:hsl({hue},8%,93%);color:transparent;", ""

    def pipe_html(pipe, hue, label):
        n = len(pipe)
        rpipe = list(reversed(pipe))
        def box(i):
            sty, txt = pipe_box_style(rpipe[i], hue)
            return f'<div style="width:28px;height:28px;{sty}border:1px solid hsl({hue},20%,80%);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;">{txt}</div>'
        inner = '<div style="display:flex;gap:2px;justify-content:center;">' + "".join(box(i) for i in range(n)) + '</div>'
        return f'<div style="text-align:center;flex:0 1 auto;min-width:32px;"><div style="font-size:7px;color:#8a96a6;margin-bottom:2px;font-weight:600;letter-spacing:0.3px;">{label}</div>{inner}</div>'

    def stage_card(title, stock, hue, icon, sub="", alert=""):
        is_alert = alert != ""
        bdr = "hsl(0,55%,60%)" if is_alert else f"hsl({hue},30%,75%)"
        bg = "linear-gradient(180deg,hsl(0,70%,97%),hsl(0,50%,94%))" if is_alert else f"linear-gradient(180deg,hsl({hue},20%,99%),hsl({hue},25%,95%))"
        return f'''<div style="background:{bg};border:2px solid {bdr};border-radius:10px;
            padding:7px 8px;min-width:72px;text-align:center;flex:0 0 auto;">
            <div style="font-size:15px;line-height:1;">{icon}</div>
            <div style="font-size:7px;font-weight:700;color:hsl({hue},35%,42%);text-transform:uppercase;letter-spacing:0.8px;margin:2px 0;">{title}</div>
            <div style="font-size:19px;font-weight:800;color:hsl({hue},40%,28%);">{stock:.0f}</div>
            {'<div style="font-size:7px;color:#7a8a9e;margin-top:1px;">'+sub+'</div>' if sub else ''}
            {'<div style="font-size:8px;color:hsl(0,60%,45%);font-weight:700;margin-top:1px;">'+alert+'</div>' if alert else ''}
        </div>'''

    arr = '<div style="color:#b0bac6;font-size:14px;display:flex;align-items:center;flex:0 0 auto;">\u25b8</div>'

    H_S = 215; H_SA = 210; H_SB = 270; H_DI = 255; H_CW = 42; H_FP = 38; H_SE = 24; H_RM = 18; H_SU = 145

    alert_a = f"LOST {state['missed_a']:.0f}" if state['missed_a'] > 0.5 else ""
    alert_b = f"LOST {state['missed_b']:.0f}" if state['missed_b'] > 0.5 else ""
    order_html = f"<b style='color:hsl(145,55%,35%);'>ORDER {state['order']:.0f}</b>" if state['order'] > 0 else "<span style='color:#b0b8c4;'>No order</span>"

    info_bar = f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 12px;
        background:linear-gradient(90deg,hsl(145,15%,97%),hsl(215,25%,96%));
        border:1px solid hsl(215,20%,89%);border-radius:7px;margin-bottom:6px;">
        <span style="font-size:9px;color:#556;">Backlog <b style="color:hsl(0,55%,50%);">{state['backlog']:.0f}</b></span>
        <span style="font-size:9px;color:#556;">Pending <b style="color:hsl(35,75%,45%);">{state['pending']:.0f}</b></span>
        <span style="font-size:9px;">{order_html}</span>
        <span style="font-size:9px;color:#556;">Forecast <b style="color:#1a2a40;">{state['forecast']:.0f}</b>/wk</span>
        <span style="font-size:9px;color:#556;">A:{params['store_a_pct']}% B:{100-params['store_a_pct']}% | {'Smart' if params['smart_distrib'] else 'Push 50/50'}</span>
        <span style="font-size:8px;color:#8a96a6;font-weight:700;letter-spacing:1px;">\u25c2 INFORMATION FLOW</span>
    </div>'''

    # Main upstream flow (Supplier → CW)
    upstream = f'''<div style="display:flex;align-items:center;gap:3px;">
        {stage_card("SUPPLIER", state['backlog'], H_SU, "\U0001f3ed", f"Cap {state['supplier_cap']:.0f}/wk")}
        {arr}
        {pipe_html(state['mat_pipe'], H_RM, f"Material {params['mat_lt']}wk")}
        {arr}
        {stage_card("RAW MAT", state['raw_mat_stock'], H_RM, "\U0001f4e6")}
        {arr}
        {pipe_html(state['semi_pipe'], H_SE, f"Semi {params['semi_lt']}wk")}
        {arr}
        {stage_card("SEMI", state['semi_stock'], H_SE, "\u2699\ufe0f", f"Cap {state['semi_cap']:.0f}/wk")}
        {arr}
        {pipe_html(state['fp_pipe'], H_FP, f"Finish {params['fp_lt']}wk")}
        {arr}
        {stage_card("CW", state['cw_stock'], H_CW, "\U0001f3ec", "Flow-thru")}
    </div>'''

    # Fork: CW → Dist A → Store A  /  CW → Dist B → Store B
    branch_a = f'''<div style="display:flex;align-items:center;gap:3px;">
        {arr}
        {pipe_html(state['dist_pipe_a'], H_SA, f"Dist A {params['dist_lt']}wk")}
        {arr}
        {stage_card("STORE A", state['store_a'], H_SA, "\U0001f6cd\ufe0f", f"Dem {state['demand_a']:.0f}/wk", alert_a)}
    </div>'''

    branch_b = f'''<div style="display:flex;align-items:center;gap:3px;">
        {arr}
        {pipe_html(state['dist_pipe_b'], H_SB, f"Dist B {params['dist_lt']}wk")}
        {arr}
        {stage_card("STORE B", state['store_b'], H_SB, "\U0001f3ea", f"Dem {state['demand_b']:.0f}/wk", alert_b)}
    </div>'''

    # Layout: upstream on left, fork on right
    flow = f'''<div style="display:flex;align-items:center;gap:0px;padding:8px 6px;
        background:linear-gradient(90deg,hsl(145,10%,97%),hsl(215,18%,97%));
        border:1px solid hsl(215,18%,90%);border-radius:10px;">
        {upstream}
        <div style="display:flex;flex-direction:column;gap:4px;margin-left:0px;">
            {branch_a}
            {branch_b}
        </div>
    </div>'''

    flow_label = '<div style="text-align:center;margin:3px 0;"><span style="font-size:7px;color:#a0aab4;font-weight:700;letter-spacing:2px;">\u25b8\u25b8\u25b8 PHYSICAL FLOW (GOODS) \u25b8\u25b8\u25b8</span></div>'

    comment = f'''<div style="padding:6px 12px;background:hsl(215,12%,96%);border:1px solid hsl(215,12%,91%);
        border-radius:7px;font-size:10px;color:#3a4a5e;line-height:1.5;">{state['comment']}</div>'''

    return f'''<!DOCTYPE html><html><head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        body {{ margin:0; padding:8px 10px; background:#f4f6f9; font-family:'Inter',system-ui,sans-serif; color:#1a2030; }}
    </style></head><body>{info_bar}{flow}{flow_label}{comment}</body></html>'''


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## \u2699\ufe0f Supply Chain Setup")
    weeks = st.select_slider("Simulation Length", options=list(range(12, 53, 2)), value=28)

    st.markdown("### \U0001f4e6 Initial Stock")
    init_store = st.slider("Store (total, split 50/50)", 0, 3000, 1500, 50)
    init_cw = st.slider("CW (Finished)", 0, 1000, 0, 50)
    init_semi = st.slider("Semi-Finished (WIP)", 0, 1000, 0, 50)
    init_rawmat = st.slider("Raw Material", 0, 1000, 0, 50)
    total_init = init_store + init_cw + init_semi + init_rawmat
    st.caption(f"Total initial stock: **{total_init}** pcs (Store: {init_store//2} + {init_store//2})")

    st.markdown("### \U0001f3ea Store Demand Split")
    store_a_pct = st.slider("Store A demand (%)", 0, 100, 50, 5)
    smart_distrib = st.toggle("Smart Distribution (need-based)", value=False)
    if smart_distrib:
        st.caption(f"A: **{store_a_pct}%** B: **{100-store_a_pct}%** — CW allocates by net need")
    else:
        st.caption(f"A: **{store_a_pct}%** B: **{100-store_a_pct}%** — CW allocates 50/50 (push)")

    st.markdown("### \U0001f517 Lead Times (weeks)")
    c1, c2 = st.columns(2)
    with c1:
        mat_lt = st.number_input("Material", 1, 12, 6)
        semi_lt = st.number_input("Semi-Fin", 1, 12, 3)
    with c2:
        fp_lt = st.number_input("Finishing", 1, 12, 1)
        dist_lt = st.number_input("Distribution", 1, 12, 1)
    phys_lt = mat_lt + semi_lt + fp_lt + dist_lt
    st.caption(f"Physical LT: **{phys_lt}** weeks")

    st.markdown("### \U0001f4cb Planning")
    order_freq = st.slider("Order Frequency (weeks)", 1, 4, 4)
    coverage = phys_lt + order_freq
    st.caption(f"Coverage target: **{coverage}** weeks (LT + freq)")

    st.markdown("### \U0001f4c8 Demand Profile")
    base_forecast = st.number_input("Base Forecast (pcs/wk)", 10, 1000, 100)
    demand_mult = st.slider("Demand Multiplier (end)", 0.3, 5.0, 4.0, 0.1)
    c3, c4 = st.columns(2)
    with c3: ramp_start = st.number_input("Ramp Start Week", 1, 52, 3)
    with c4: ramp_end = st.number_input("Ramp End Week", 1, 52, 5)
    if ramp_end < ramp_start: ramp_end = ramp_start
    st.caption(f"Demand: {base_forecast} \u2192 {base_forecast * demand_mult:.0f} (W{ramp_start}\u2013W{ramp_end})")

    st.markdown("### \U0001f3ed Capacity")
    cap_start = st.number_input("Starting Capacity (pcs/wk)", 10, 1000, 100)
    cap_ramp = st.slider("Max Ramp-up (%/week)", 0, 50, 20) / 100

    st.markdown("### \U0001f4b0 Economics")
    price = st.number_input("Selling Price (\u20ac)", 100, 10000, 1000, 100)
    var_cost = st.number_input("Variable Cost (\u20ac)", 10, 5000, 200, 10)
    fixed_pct = st.slider("Fixed Cost (% annual fcst rev)", 0, 100, 45) / 100

params = {
    'weeks': weeks, 'init_store': init_store, 'init_cw': init_cw,
    'init_semi': init_semi, 'init_rawmat': init_rawmat,
    'order_freq': order_freq, 'mat_lt': mat_lt, 'semi_lt': semi_lt,
    'fp_lt': fp_lt, 'dist_lt': dist_lt, 'cap_start': cap_start,
    'cap_ramp': cap_ramp, 'base_forecast': base_forecast,
    'demand_mult': demand_mult, 'ramp_start': ramp_start,
    'ramp_end': ramp_end, 'price': price, 'var_cost': var_cost,
    'fixed_pct': fixed_pct, 'store_a_pct': store_a_pct,
    'smart_distrib': smart_distrib,
}


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════
states = run_simulation(**params)
final_kpis = compute_kpis(states, price, var_cost, fixed_pct, base_forecast, weeks)


# ════════════════════════════════════════════════════════════════
# STYLES
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #eaeff5, #f0f3f8); }
    h1 { color: #1a2a40 !important; }
    h2, h3, h4 { color: #2c3e56 !important; }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff, #f7f9fc);
        border-radius: 10px; padding: 14px 8px;
        border: 1px solid #dde3ed; text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .kpi-value { font-size: 21px; font-weight: 800; margin-top: 2px; }
    .kpi-label { font-size: 8px; color: #7a8a9e; text-transform: uppercase;
        letter-spacing: 1.2px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HEADER + WEEK SELECTOR
# ════════════════════════════════════════════════════════════════
st.markdown("# \U0001f3ed Supply Chain Agility Simulator")
distrib_mode = "Smart" if smart_distrib else "Push 50/50"
st.markdown(f"*Luxury Industry \u00b7 LT = **{phys_lt}**wk \u00b7 Coverage = **{coverage}**wk \u00b7 Store A: **{store_a_pct}%** / B: **{100-store_a_pct}%** \u00b7 Init stock 50/50 \u00b7 Distrib: **{distrib_mode}***")

week = st.slider("\U0001f4c5 Week", 1, weeks, 1, key="week_slider")
state = states[week - 1]
cum = cumulative_kpis(states, week, price, var_cost, fixed_pct, base_forecast, weeks)


# ════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════
def kpi_card(label, value, color="#1a2a40"):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color};">{value}</div></div>'

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
with k1:
    svc = cum['svc_level']
    c = "#c0392b" if svc < 0.6 else ("#d4850a" if svc < 0.85 else "#1a8a4a")
    st.markdown(kpi_card("Service Level", f"{svc*100:.1f}%", c), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Cumul. Sales", f"{round(cum['sales'], -1):,.0f}", "#2c5f8a"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Missed Total", f"{round(cum['missed'], -1):,.0f}", "#c0392b"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Missed A", f"{round(cum['missed_a'], -1):,.0f}", "#c0392b"), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_card("Missed B", f"{round(cum['missed_b'], -1):,.0f}", "#7b2d8e"), unsafe_allow_html=True)
with k6:
    mc = "#1a6b3c" if cum['margin'] > 0 else "#c0392b"
    st.markdown(kpi_card("Net Margin", f"\u20ac{cum['margin']:,.0f}", mc), unsafe_allow_html=True)
with k7:
    sc = "#c0392b" if cum['stockout_wks'] > 0 else "#1a8a4a"
    st.markdown(kpi_card("Stockout Wks", f"{cum['stockout_wks']}/{week}", sc), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SC FLOW VISUALIZATION
# ════════════════════════════════════════════════════════════════
st.markdown("")
st.components.v1.html(make_sc_html(state, params), height=280, scrolling=False)


# ════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════
import altair as alt
import pandas as pd

ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("#### Demand vs Fulfillment (per store)")
    rows = []
    for s in states:
        rows.append({'Week': s['week'], 'Group': 'Dem A', 'Component': 'Demand A', 'Value': s['demand_a']})
        rows.append({'Week': s['week'], 'Group': 'Fill A', 'Component': 'Sales A', 'Value': s['sales_a']})
        rows.append({'Week': s['week'], 'Group': 'Fill A', 'Component': 'Lost A', 'Value': s['missed_a']})
        rows.append({'Week': s['week'], 'Group': 'Dem B', 'Component': 'Demand B', 'Value': s['demand_b']})
        rows.append({'Week': s['week'], 'Group': 'Fill B', 'Component': 'Sales B', 'Value': s['sales_b']})
        rows.append({'Week': s['week'], 'Group': 'Fill B', 'Component': 'Lost B', 'Value': s['missed_b']})
    df_bars = pd.DataFrame(rows)

    bars = alt.Chart(df_bars).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X('Week:O', title='Week'),
        y=alt.Y('Value:Q', title='Units', stack=True),
        color=alt.Color('Component:N',
            scale=alt.Scale(
                domain=['Demand A', 'Sales A', 'Lost A', 'Demand B', 'Sales B', 'Lost B'],
                range=['#90a4ae', '#2c5f8a', '#c0392b', '#b39ddb', '#6a3d9a', '#e74c8c']),
            legend=alt.Legend(orient='top', title=None, columns=3)),
        xOffset='Group:N',
    ).properties(height=260)

    rule = alt.Chart(pd.DataFrame({'Week': [week]})).mark_rule(
        color='#d4850a', strokeWidth=2, strokeDash=[4,2]).encode(x='Week:O')
    st.altair_chart(bars + rule, use_container_width=True)

with ch2:
    st.markdown("#### Store Stocks & Orders")
    stock_data = pd.DataFrame({
        'Week': [s['week'] for s in states],
        'Store A': [s['store_a'] for s in states],
        'Store B': [s['store_b'] for s in states],
        'Order': [s['order'] for s in states],
    })
    melted = stock_data.melt('Week', ['Store A', 'Store B'], var_name='Store', value_name='Stock')

    lines = alt.Chart(melted).mark_area(opacity=0.25).encode(
        x=alt.X('Week:O', title='Week'),
        y=alt.Y('Stock:Q', title='Units', stack=False),
        color=alt.Color('Store:N',
            scale=alt.Scale(domain=['Store A', 'Store B'], range=['#2c5f8a', '#6a3d9a']),
            legend=alt.Legend(orient='top', title=None)),
    ).properties(height=260)

    order_bars = alt.Chart(stock_data[stock_data['Order'] > 0]).mark_bar(
        color='#1a8a4a', opacity=0.4, cornerRadiusTopLeft=2, cornerRadiusTopRight=2
    ).encode(x='Week:O', y='Order:Q')

    rule2 = alt.Chart(pd.DataFrame({'Week': [week]})).mark_rule(
        color='#d4850a', strokeWidth=2, strokeDash=[4,2]).encode(x='Week:O')
    st.altair_chart(lines + order_bars + rule2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# EXPANDABLE SECTIONS
# ════════════════════════════════════════════════════════════════
with st.expander("\U0001f4cb Full Simulation Summary", expanded=False):
    fk = final_kpis
    def r10(v): return round(v, -1)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Demand", f"{r10(fk['total_demand']):,.0f}")
        st.metric("Total Sales", f"{r10(fk['total_sales']):,.0f}")
        st.metric("Service Level", f"{fk['svc_level']*100:.1f}%")
    with c2:
        st.metric("Total Missed", f"{r10(fk['total_missed']):,.0f}")
        st.metric("Missed A", f"{r10(fk['missed_a']):,.0f}")
        st.metric("Missed B", f"{r10(fk['missed_b']):,.0f}")
    with c3:
        st.metric("Revenue", f"\u20ac{fk['revenue']:,.0f}")
        st.metric("Gross Margin", f"\u20ac{fk['gm']:,.0f}")
        st.metric("Variable Costs", f"\u20ac{fk['var_cost']:,.0f}")
    with c4:
        st.metric("Fixed Costs", f"\u20ac{fk['fixed']:,.0f}")
        st.metric("Net Margin", f"\u20ac{fk['margin']:,.0f}")
        st.metric("Margin %", f"{fk['margin_pct']*100:.1f}%")

with st.expander("\U0001f4ca Detailed Week-by-Week Data", expanded=False):
    table_data = [{
        'Week': s['week'],
        'Dem Tot': s['demand'], 'Dem A': s['demand_a'], 'Dem B': s['demand_b'],
        'Sales A': s['sales_a'], 'Sales B': s['sales_b'], 'Sales': s['sales'],
        'Miss A': s['missed_a'], 'Miss B': s['missed_b'], 'Missed': s['missed'],
        'Stk A': s['store_a'], 'Stk B': s['store_b'],
        'Alloc A': s['alloc_a'], 'Alloc B': s['alloc_b'],
        'Order': s['order'], 'Pending': s['pending'],
        'Ship': s['supplier_shipped'], 'Mat In': s['mat_arr'],
        'Semi': s['semi_input'], 'FP': s['fp_input'],
        'Comment': s['comment'],
    } for s in states]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=500)

with st.expander("\U0001f4be Save Scenario for Comparison", expanded=False):
    now_str = datetime.now().strftime("%H:%M:%S")
    scenario_name = st.text_input("Scenario Name", f"SC_{order_freq}wk_{init_store}st_A{store_a_pct}_{'smart' if smart_distrib else 'push'}_{now_str}")
    if st.button("Save Current Scenario"):
        if 'saved_scenarios' not in st.session_state:
            st.session_state.saved_scenarios = {}
        st.session_state.saved_scenarios[scenario_name] = {
            'params': params.copy(), 'kpis': final_kpis.copy()}
        st.success(f"Saved '{scenario_name}'!")

    if 'saved_scenarios' in st.session_state and len(st.session_state.saved_scenarios) > 0:
        st.markdown("### Comparison")
        saved_list = list(st.session_state.saved_scenarios.items())
        first_margin = saved_list[0][1]['kpis']['margin']
        comp = []
        for i, (n, d) in enumerate(saved_list):
            k = d['kpis']; p = d['params']
            delta = k['margin'] - first_margin
            delta_str = f"\u20ac{delta:+,.0f}" if i > 0 else "Baseline"
            comp.append({
                'Scenario': n,
                'Svc%': f"{k['svc_level']*100:.1f}%",
                'Sales': f"{round(k['total_sales'], -1):,.0f}",
                'Missed': f"{round(k['total_missed'], -1):,.0f}",
                'Miss A': f"{round(k['missed_a'], -1):,.0f}",
                'Miss B': f"{round(k['missed_b'], -1):,.0f}",
                'Revenue': f"\u20ac{k['revenue']:,.0f}",
                'Margin': f"\u20ac{k['margin']:,.0f}",
                'Margin %': f"{k['margin_pct']*100:.1f}%",
                '\u0394 vs Base': delta_str,
                'StoreA%': f"{p['store_a_pct']}%",
                'Distrib': 'Smart' if p.get('smart_distrib', False) else 'Push',
                'Store Init': p['init_store'],
                'CW': p['init_cw'], 'Semi': p['init_semi'], 'RM': p['init_rawmat'],
                'Freq': f"{p['order_freq']}wk",
                'Mat LT': p['mat_lt'], 'Semi LT': p['semi_lt'],
                'FP LT': p['fp_lt'], 'Dist LT': p['dist_lt'],
                'Fcst': p['base_forecast'], 'Mult': f"{p['demand_mult']}x",
                'Ramp': f"W{p['ramp_start']}-{p['ramp_end']}",
                'Cap': p['cap_start'], 'Ramp%': f"{p['cap_ramp']*100:.0f}%",
                'Price': f"\u20ac{p['price']}", 'VarC': f"\u20ac{p['var_cost']}",
                'Fix%': f"{p['fixed_pct']*100:.0f}%", 'Wks': p['weeks'],
            })
        st.dataframe(pd.DataFrame(comp), use_container_width=True)
        if st.button("Clear All"):
            st.session_state.saved_scenarios = {}
            st.rerun()
