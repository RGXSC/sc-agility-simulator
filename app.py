import streamlit as st
import json, math

st.set_page_config(layout="wide", page_title="Supply Chain Agility Simulator", page_icon="🏭")

# ════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ════════════════════════════════════════════════════════════════
@st.cache_data
def run_simulation(weeks, init_store, init_cw, init_semi, init_rawmat,
                   order_freq, mat_lt, semi_lt, fp_lt, dist_lt,
                   cap_start, cap_ramp, base_forecast,
                   demand_mult, ramp_start, ramp_end,
                   price, var_cost, fixed_pct):
    
    phys_lt = mat_lt + semi_lt + fp_lt + dist_lt
    coverage = phys_lt + order_freq
    
    # Demand profile
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
    
    # Pipelines (FIFO: index 0 exits next, index -1 just entered)
    mat_pipe = [0.0] * max(1, mat_lt)
    semi_pipe = [0.0] * max(1, semi_lt)
    fp_pipe = [0.0] * max(1, fp_lt)
    dist_pipe = [0.0] * max(1, dist_lt)
    
    store = float(init_store)
    raw_mat = float(init_rawmat)
    semi = float(init_semi)
    cw = float(init_cw)
    
    pb = 0.0  # supplier backlog
    pc = float(cap_start); ps = False
    sc_ = float(cap_start); ss_ = False
    fpc = float(cap_start); fps = False
    co = 0.0; cas = 0.0
    
    ow = list(range(order_freq, weeks + 1, order_freq)) if order_freq > 1 else list(range(1, weeks + 1))
    
    states = []
    
    for w in range(1, weeks + 1):
        s = {'week': w, 'demand': round(demand[w], 1)}
        ff = demand[w]
        s['forecast'] = round(ff, 1)
        
        # 1. Read pipe fronts (arrivals)
        d_arr = dist_pipe[0]
        m_arr = mat_pipe[0]
        sm_arr = semi_pipe[0]
        fp_arr = fp_pipe[0]
        
        s['dist_arr'] = round(d_arr, 1)
        s['mat_arr'] = round(m_arr, 1)
        s['semi_arr'] = round(sm_arr, 1)
        s['fp_arr'] = round(fp_arr, 1)
        
        cas += d_arr
        
        # 2. Store
        avail = store + d_arr
        sales = min(demand[w], avail)
        missed = max(0, demand[w] - sales)
        store = avail - sales
        s.update({'store_before': round(avail, 1), 'sales': round(sales, 1),
                  'missed': round(missed, 1), 'store_stock': round(store, 1)})
        
        # 3. Supplier ships from existing backlog
        if pb > 0.01:
            if not ps: ps = True; pc = float(cap_start)
            else: pc = min(pc * (1 + cap_ramp), cap_start * 10)  # cap max 10x
            shipped = math.ceil(min(pb, pc)); pb -= shipped
        else:
            shipped = 0.0
            ps = False; pc = float(cap_start)  # reset when idle
        s['supplier_shipped'] = round(shipped, 1)
        s['supplier_cap'] = round(pc, 0)
        
        # 4. Raw material receives
        raw_mat += m_arr
        s['raw_mat_before_prod'] = round(raw_mat, 1)
        
        # 5. Semi receives
        semi += sm_arr
        
        # 6. CW receives FP
        cw += fp_arr
        
        # 7. Semi input (raw mat → semi pipe, capacity limited)
        if raw_mat > 0.01:
            if not ss_: ss_ = True; sc_ = float(cap_start)
            else: sc_ = min(sc_ * (1 + cap_ramp), cap_start * 10)
            si = math.ceil(min(raw_mat, sc_)); raw_mat -= si
        else:
            si = 0.0
            ss_ = False; sc_ = float(cap_start)  # reset when idle
        s['semi_input'] = round(si, 1)
        s['semi_cap'] = round(sc_, 0)
        s['raw_mat_stock'] = round(raw_mat, 1)
        
        # 8. FP input (semi → fp pipe, capacity limited)
        if semi > 0.01:
            if not fps: fps = True; fpc = float(cap_start)
            else: fpc = min(fpc * (1 + cap_ramp), cap_start * 10)
            fi = math.ceil(min(semi, fpc)); semi -= fi
        else:
            fi = 0.0
            fps = False; fpc = float(cap_start)  # reset when idle
        s['fp_input'] = round(fi, 1)
        s['fp_cap'] = round(fpc, 0)
        s['semi_stock'] = round(semi, 1)
        
        # 9. CW ships everything → dist pipe
        ship_out = math.ceil(cw); cw = 0.0
        s['cw_shipped'] = round(ship_out, 1)
        s['cw_stock'] = 0.0
        
        # 10. Update pipes
        mat_pipe = mat_pipe[1:] + [shipped]
        semi_pipe = semi_pipe[1:] + [si]
        fp_pipe = fp_pipe[1:] + [fi]
        dist_pipe = dist_pipe[1:] + [ship_out]
        
        s['mat_pipe'] = [round(x, 1) for x in mat_pipe]
        s['semi_pipe'] = [round(x, 1) for x in semi_pipe]
        s['fp_pipe'] = [round(x, 1) for x in fp_pipe]
        s['dist_pipe'] = [round(x, 1) for x in dist_pipe]
        
        # 11. SC Order (Friday)
        if w in ow:
            pnd = co - cas
            tgt = ff * coverage
            od = math.ceil(max(0, tgt - store - pnd))
            co += od; pb += od
            s['order'] = round(od, 0)
        else:
            s['order'] = 0
        s['pending'] = round(co - cas, 0)
        s['backlog'] = round(pb, 0)
        s['coverage'] = coverage
        
        # Commentary
        parts = []
        if missed > 0.5 and avail < 0.5:
            parts.append(f"🔴 STOCKOUT: Store empty. Demand={demand[w]:.0f}, lost {missed:.0f} sales.")
        elif missed > 0.5:
            parts.append(f"🟡 PARTIAL: Only {avail:.0f} avail vs {demand[w]:.0f} demand. Lost {missed:.0f}.")
        else:
            parts.append(f"🟢 Sold {sales:.0f}/{demand[w]:.0f}. Store: {store:.0f}.")
        if s['order'] > 0:
            parts.append(f"📋 ORDER: Fcst={ff:.0f}×{coverage}={ff*coverage:.0f}. Store={store:.0f}+pend={co-cas-od:.0f}. Gap={od:.0f}.")
        if shipped > 0.5:
            parts.append(f"🚚 Supplier ships {shipped:.0f} (cap={pc:.0f})→W{w+mat_lt}.")
        if m_arr > 0.5:
            parts.append(f"📦 Material arrives: {m_arr:.0f}.")
        if si > 0.5:
            parts.append(f"⚙️ Semi prod: {si:.0f} (cap={sc_:.0f})→W{w+semi_lt}.")
        if fi > 0.5:
            parts.append(f"🔧 Finishing: {fi:.0f} (cap={fpc:.0f})→W{w+fp_lt}.")
        if d_arr > 0.5:
            parts.append(f"📬 {d_arr:.0f} delivered to store.")
        s['comment'] = " ".join(parts)
        
        states.append(s)
    
    return states

def compute_kpis(states, price, var_cost, fixed_pct, base_forecast, weeks):
    ts = sum(s['sales'] for s in states)
    tm = sum(s['missed'] for s in states)
    td = sum(s['demand'] for s in states)
    tfp = sum(s['fp_input'] for s in states)
    rev = ts * price
    vc = tfp * var_cost
    gm = rev - vc
    annual_fcst_rev = base_forecast * 52 * price
    fx = annual_fcst_rev * fixed_pct * (weeks / 52)
    mg = gm - fx
    return {
        'total_demand': td, 'total_sales': ts, 'total_missed': tm,
        'svc_level': ts / td if td > 0 else 0,
        'stockout_weeks': sum(1 for s in states if s['missed'] > 0.5),
        'revenue': rev, 'var_cost': vc, 'gm': gm, 'fixed': fx,
        'margin': mg, 'margin_pct': mg / rev if rev > 0 else 0,
        'produced': tfp, 'store_end': states[-1]['store_stock'],
        'lost_rev': tm * price,
    }

def cumulative_kpis(states, week, price, var_cost, fixed_pct, base_forecast, total_weeks):
    sub = states[:week]
    ts = sum(s['sales'] for s in sub)
    tm = sum(s['missed'] for s in sub)
    td = sum(s['demand'] for s in sub)
    tfp = sum(s['fp_input'] for s in sub)
    rev = ts * price
    vc = tfp * var_cost
    gm = rev - vc
    fx = base_forecast * 52 * price * fixed_pct * (week / 52)
    mg = gm - fx
    return {'sales': ts, 'missed': tm, 'demand': td, 'revenue': rev,
            'svc_level': ts / td if td > 0 else 0, 'margin': mg,
            'stockout_wks': sum(1 for s in sub if s['missed'] > 0.5)}


# ════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════
def make_sc_html(state, params):
    w = state['week']
    
    def bar(val, max_val, color, label=""):
        pct = min(100, (val / max_val * 100)) if max_val > 0 else 0
        txt_color = "#fff" if pct > 20 else "#333"
        return f'''<div style="background:#1a1a2e;border-radius:4px;height:28px;width:100%;position:relative;overflow:hidden;margin:2px 0;">
            <div style="background:{color};height:100%;width:{pct}%;border-radius:4px;transition:width 0.3s;"></div>
            <div style="position:absolute;top:0;left:0;right:0;height:100%;display:flex;align-items:center;justify-content:center;font-size:11px;color:{txt_color};font-weight:600;">{val:.0f}{" "+label if label else ""}</div>
        </div>'''
    
    def pipe_boxes(pipe, color, lt_label):
        boxes = ""
        for i, v in enumerate(pipe):
            opacity = min(1.0, 0.15 + (v / max(1, params['cap_start'] * 3)) * 0.85) if v > 0.5 else 0.08
            txt = f"{v:.0f}" if v > 0.5 else ""
            boxes += f'''<div style="width:32px;height:32px;background:rgba({color},{opacity});
                border:1px solid rgba({color},0.4);border-radius:3px;display:flex;align-items:center;
                justify-content:center;font-size:9px;color:#e0e0e0;font-weight:600;margin:1px;">{txt}</div>'''
        n = len(pipe)
        if n <= 4:
            return f'''<div style="text-align:center;">
                <div style="font-size:8px;color:#888;margin-bottom:2px;">{lt_label}</div>
                <div style="display:flex;gap:0px;justify-content:center;">{boxes}</div>
            </div>'''
        else:
            half = math.ceil(n / 2)
            top_boxes = "".join(f'''<div style="width:32px;height:32px;background:rgba({color},{min(1.0, 0.15 + (pipe[i] / max(1, params["cap_start"] * 3)) * 0.85) if pipe[i] > 0.5 else 0.08});
                border:1px solid rgba({color},0.4);border-radius:3px;display:flex;align-items:center;
                justify-content:center;font-size:9px;color:#e0e0e0;font-weight:600;margin:1px;">{f"{pipe[i]:.0f}" if pipe[i] > 0.5 else ""}</div>''' for i in range(half))
            bot_boxes = "".join(f'''<div style="width:32px;height:32px;background:rgba({color},{min(1.0, 0.15 + (pipe[i] / max(1, params["cap_start"] * 3)) * 0.85) if pipe[i] > 0.5 else 0.08});
                border:1px solid rgba({color},0.4);border-radius:3px;display:flex;align-items:center;
                justify-content:center;font-size:9px;color:#e0e0e0;font-weight:600;margin:1px;">{f"{pipe[i]:.0f}" if pipe[i] > 0.5 else ""}</div>''' for i in range(half, n))
            return f'''<div style="text-align:center;">
                <div style="font-size:8px;color:#888;margin-bottom:2px;">{lt_label}</div>
                <div style="display:flex;gap:0px;justify-content:center;">{top_boxes}</div>
                <div style="display:flex;gap:0px;justify-content:center;margin-top:1px;">{bot_boxes}</div>
            </div>'''
    
    def stage_card(title, stock, color_hex, icon, cap_text="", extra=""):
        is_stockout = title == "STORE" and state['missed'] > 0.5
        border_color = "#ff4444" if is_stockout else color_hex
        bg = "rgba(255,50,50,0.15)" if is_stockout else f"rgba(30,30,50,0.9)"
        return f'''<div style="background:{bg};border:2px solid {border_color};border-radius:8px;
            padding:8px;min-width:80px;text-align:center;">
            <div style="font-size:18px;">{icon}</div>
            <div style="font-size:10px;font-weight:700;color:{color_hex};text-transform:uppercase;letter-spacing:1px;">{title}</div>
            <div style="font-size:22px;font-weight:800;color:#fff;margin:4px 0;">{stock:.0f}</div>
            {f'<div style="font-size:8px;color:#aaa;">{cap_text}</div>' if cap_text else ""}
            {f'<div style="font-size:9px;color:#ff8888;font-weight:600;">{extra}</div>' if extra else ""}
        </div>'''
    
    arrow = '<div style="color:#555;font-size:18px;display:flex;align-items:center;margin:0 2px;">→</div>'
    
    # Colors (RGB for rgba)
    c_supplier = "100,160,80"
    c_mat = "130,60,12"
    c_semi = "197,90,17"
    c_fp = "191,143,0"
    c_dist = "112,48,160"
    c_store = "68,114,196"
    
    store_extra = f"🔴 LOST: {state['missed']:.0f}" if state['missed'] > 0.5 else ""
    
    # Info flow (top)
    order_display = f"<span style='color:#66ff66;font-weight:800;font-size:14px;'>ORDER: {state['order']:.0f}</span>" if state['order'] > 0 else "<span style='color:#666;'>No order this week</span>"
    info_flow = f'''<div style="display:flex;justify-content:space-between;align-items:center;
        padding:6px 12px;background:rgba(50,80,50,0.2);border:1px solid rgba(100,160,80,0.3);border-radius:6px;margin-bottom:8px;">
        <div style="font-size:10px;color:#aaa;">◀ INFORMATION FLOW</div>
        <div style="font-size:11px;color:#ccc;">Forecast: <b style="color:#fff;">{state['forecast']:.0f}</b>/wk</div>
        <div>{order_display}</div>
        <div style="font-size:11px;color:#ccc;">Pending: <b style="color:#ffaa00;">{state['pending']:.0f}</b></div>
        <div style="font-size:11px;color:#ccc;">Backlog: <b style="color:#ff6666;">{state['backlog']:.0f}</b></div>
    </div>'''
    
    # Physical flow (main visualization)
    physical = f'''<div style="display:flex;align-items:center;justify-content:center;gap:4px;flex-wrap:nowrap;">
        {stage_card("SUPPLIER", state['backlog'], "#64A050", "🏭", f"Cap: {state['supplier_cap']:.0f}/wk")}
        {arrow}
        {pipe_boxes(state['mat_pipe'], c_mat, f"Material LT {params['mat_lt']}wk")}
        {arrow}
        {stage_card("RAW MAT", state['raw_mat_stock'], "#843C0C", "📦")}
        {arrow}
        {pipe_boxes(state['semi_pipe'], c_semi, f"Semi-Fin LT {params['semi_lt']}wk")}
        {arrow}
        {stage_card("SEMI STK", state['semi_stock'], "#C55A11", "⚙️", f"Cap: {state['semi_cap']:.0f}/wk")}
        {arrow}
        {pipe_boxes(state['fp_pipe'], c_fp, f"Finish {params['fp_lt']}wk")}
        {arrow}
        {stage_card("CW", state['cw_stock'], "#BF8F00", "🏬")}
        {arrow}
        {pipe_boxes(state['dist_pipe'], c_dist, f"Dist {params['dist_lt']}wk")}
        {arrow}
        {stage_card("STORE", state['store_stock'], "#4472C4", "🛍️", f"Dem: {state['demand']:.0f}", store_extra)}
    </div>'''
    
    # Demand vs Sales bar
    max_dem = max(state['demand'], 1)
    demand_bar = f'''<div style="margin-top:10px;padding:6px 12px;background:rgba(30,30,50,0.5);border-radius:6px;">
        <div style="display:flex;gap:20px;align-items:center;">
            <div style="flex:1;">
                <div style="font-size:9px;color:#888;margin-bottom:2px;">DEMAND: {state['demand']:.0f}</div>
                {bar(state['demand'], max_dem, '#555', '')}
            </div>
            <div style="flex:1;">
                <div style="font-size:9px;color:#888;margin-bottom:2px;">SALES: {state['sales']:.0f}</div>
                {bar(state['sales'], max_dem, '#4472C4' if state['missed'] < 0.5 else '#ff4444', '')}
            </div>
            <div style="flex:1;">
                <div style="font-size:9px;color:#888;margin-bottom:2px;">MISSED: {state['missed']:.0f}</div>
                {bar(state['missed'], max_dem, '#ff4444', '')}
            </div>
        </div>
    </div>'''
    
    # Commentary
    comment = f'''<div style="margin-top:6px;padding:8px 12px;background:rgba(30,30,50,0.4);border-radius:6px;
        font-size:11px;color:#ccc;line-height:1.5;">{state['comment']}</div>'''
    
    html = f'''<!DOCTYPE html><html><head><style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap');
        body {{ margin:0; padding:12px; background:#0d0d1a; font-family:'JetBrains Mono',monospace; color:#fff; }}
    </style></head><body>
        {info_flow}
        {physical}
        {demand_bar}
        {comment}
    </body></html>'''
    return html


# ════════════════════════════════════════════════════════════════
# SIDEBAR PARAMETERS
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Supply Chain Setup")
    
    weeks = st.select_slider("Simulation Length", options=list(range(12, 53, 4)), value=28)
    
    st.markdown("### 📦 Initial Stock")
    init_store = st.slider("Store", 0, 3000, 1500, 50)
    init_cw = st.slider("CW (Finished)", 0, 1000, 0, 50)
    init_semi = st.slider("Semi-Finished (WIP)", 0, 1000, 0, 50)
    init_rawmat = st.slider("Raw Material", 0, 1000, 0, 50)
    total_init = init_store + init_cw + init_semi + init_rawmat
    st.caption(f"Total initial stock: **{total_init}** pcs")
    
    st.markdown("### 🔗 Lead Times (weeks)")
    c1, c2 = st.columns(2)
    with c1:
        mat_lt = st.number_input("Material", 1, 12, 6)
        semi_lt = st.number_input("Semi-Fin", 1, 12, 3)
    with c2:
        fp_lt = st.number_input("Finishing", 1, 12, 1)
        dist_lt = st.number_input("Distribution", 1, 12, 1)
    phys_lt = mat_lt + semi_lt + fp_lt + dist_lt
    st.caption(f"Physical LT: **{phys_lt}** weeks")
    
    st.markdown("### 📋 Planning")
    order_freq = st.slider("Order Frequency (weeks)", 1, 4, 4)
    coverage = phys_lt + order_freq
    st.caption(f"Coverage target: **{coverage}** weeks (LT + freq)")
    
    st.markdown("### 📈 Demand Profile")
    base_forecast = st.number_input("Base Forecast (pcs/wk)", 10, 1000, 100)
    demand_mult = st.slider("Demand Multiplier (end)", 0.3, 5.0, 4.0, 0.1)
    c3, c4 = st.columns(2)
    with c3:
        ramp_start = st.number_input("Ramp Start Week", 1, 52, 3)
    with c4:
        ramp_end = st.number_input("Ramp End Week", 1, 52, 5)
    if ramp_end < ramp_start:
        ramp_end = ramp_start
    st.caption(f"Demand: {base_forecast} → {base_forecast * demand_mult:.0f} (W{ramp_start}–W{ramp_end})")
    
    st.markdown("### 🏭 Capacity")
    cap_start = st.number_input("Starting Capacity (pcs/wk)", 10, 1000, 100)
    cap_ramp = st.slider("Max Ramp-up (%/week)", 0, 50, 20) / 100
    
    st.markdown("### 💰 Economics")
    price = st.number_input("Selling Price (€)", 100, 10000, 1000, 100)
    var_cost = st.number_input("Variable Cost (€)", 10, 5000, 200, 10)
    fixed_pct = st.slider("Fixed Cost (% annual fcst rev)", 0, 100, 45) / 100

params = {
    'weeks': weeks, 'init_store': init_store, 'init_cw': init_cw,
    'init_semi': init_semi, 'init_rawmat': init_rawmat,
    'order_freq': order_freq, 'mat_lt': mat_lt, 'semi_lt': semi_lt,
    'fp_lt': fp_lt, 'dist_lt': dist_lt, 'cap_start': cap_start,
    'cap_ramp': cap_ramp, 'base_forecast': base_forecast,
    'demand_mult': demand_mult, 'ramp_start': ramp_start,
    'ramp_end': ramp_end, 'price': price, 'var_cost': var_cost,
    'fixed_pct': fixed_pct,
}


# ════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ════════════════════════════════════════════════════════════════
states = run_simulation(**params)
final_kpis = compute_kpis(states, price, var_cost, fixed_pct, base_forecast, weeks)


# ════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0d0d1a; }
    h1, h2, h3 { color: #c8d6e5 !important; }
    .metric-card { background: rgba(30,30,50,0.8); border-radius: 8px; padding: 12px;
        border: 1px solid rgba(100,100,150,0.3); text-align: center; }
    .metric-value { font-size: 24px; font-weight: 800; }
    .metric-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏭 Supply Chain Agility Simulator")
st.markdown(f"*Luxury Industry · Physical LT = {phys_lt}wk · Coverage = {coverage}wk · Order every {order_freq}wk*")

# Week selector
week = st.slider("📅 Week", 1, weeks, 1, key="week_slider")
state = states[week - 1]
cum = cumulative_kpis(states, week, price, var_cost, fixed_pct, base_forecast, weeks)

# KPI row (cumulative up to current week)
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    color = "#ff4444" if cum['svc_level'] < 0.6 else ("#ffaa00" if cum['svc_level'] < 0.85 else "#44ff44")
    st.markdown(f'''<div class="metric-card">
        <div class="metric-label">Service Level</div>
        <div class="metric-value" style="color:{color};">{cum['svc_level']*100:.1f}%</div>
    </div>''', unsafe_allow_html=True)
with k2:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-label">Cumul. Sales</div>
        <div class="metric-value" style="color:#4472C4;">{cum['sales']:,.0f}</div>
    </div>''', unsafe_allow_html=True)
with k3:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-label">Cumul. Missed</div>
        <div class="metric-value" style="color:#ff4444;">{cum['missed']:,.0f}</div>
    </div>''', unsafe_allow_html=True)
with k4:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-label">Revenue</div>
        <div class="metric-value" style="color:#44ff88;">€{cum['revenue']:,.0f}</div>
    </div>''', unsafe_allow_html=True)
with k5:
    mg_color = "#44ff88" if cum['margin'] > 0 else "#ff4444"
    st.markdown(f'''<div class="metric-card">
        <div class="metric-label">Net Margin</div>
        <div class="metric-value" style="color:{mg_color};">€{cum['margin']:,.0f}</div>
    </div>''', unsafe_allow_html=True)
with k6:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-label">Stockout Weeks</div>
        <div class="metric-value" style="color:{"#ff4444" if cum["stockout_wks"]>0 else "#44ff44"};">{cum['stockout_wks']}/{week}</div>
    </div>''', unsafe_allow_html=True)


# Supply Chain Visualization
st.components.v1.html(make_sc_html(state, params), height=340, scrolling=False)


# ════════════════════════════════════════════════════════════════
# TIMELINE CHARTS
# ════════════════════════════════════════════════════════════════
import altair as alt
import pandas as pd

chart_data = pd.DataFrame({
    'Week': [s['week'] for s in states],
    'Demand': [s['demand'] for s in states],
    'Sales': [s['sales'] for s in states],
    'Missed': [s['missed'] for s in states],
    'Store Stock': [s['store_stock'] for s in states],
    'Order': [s['order'] for s in states],
    'Supplier Shipped': [s['supplier_shipped'] for s in states],
    'FP Produced': [s['fp_input'] for s in states],
})

ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("#### 📊 Demand vs Sales")
    melted = chart_data[['Week', 'Demand', 'Sales', 'Missed']].melt('Week', var_name='Metric', value_name='Units')
    colors = {'Demand': '#555555', 'Sales': '#4472C4', 'Missed': '#ff4444'}
    
    base = alt.Chart(melted).mark_bar(opacity=0.8).encode(
        x=alt.X('Week:O'),
        y=alt.Y('Units:Q'),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values()))),
    ).properties(height=250)
    
    rule = alt.Chart(pd.DataFrame({'Week': [week]})).mark_rule(color='#ffaa00', strokeWidth=2).encode(x='Week:O')
    st.altair_chart(base + rule, use_container_width=True)

with ch2:
    st.markdown("#### 📦 Store Stock & Orders")
    stock_chart = alt.Chart(chart_data).mark_area(opacity=0.4, color='#4472C4').encode(
        x=alt.X('Week:O'), y=alt.Y('Store Stock:Q')
    ).properties(height=250)
    
    order_bars = alt.Chart(chart_data[chart_data['Order'] > 0]).mark_bar(color='#44ff88', opacity=0.7).encode(
        x=alt.X('Week:O'), y=alt.Y('Order:Q')
    )
    
    rule2 = alt.Chart(pd.DataFrame({'Week': [week]})).mark_rule(color='#ffaa00', strokeWidth=2).encode(x='Week:O')
    st.altair_chart(stock_chart + order_bars + rule2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# FINAL KPI SUMMARY (full simulation)
# ════════════════════════════════════════════════════════════════
with st.expander("📋 Full Simulation Summary (all weeks)", expanded=False):
    fk = final_kpis
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Demand", f"{fk['total_demand']:,.0f}")
        st.metric("Total Sales", f"{fk['total_sales']:,.0f}")
        st.metric("Total Missed", f"{fk['total_missed']:,.0f}")
        st.metric("Service Level", f"{fk['svc_level']*100:.1f}%")
    with c2:
        st.metric("Revenue", f"€{fk['revenue']:,.0f}")
        st.metric("Variable Costs", f"€{fk['var_cost']:,.0f}")
        st.metric("Gross Margin", f"€{fk['gm']:,.0f}")
    with c3:
        st.metric("Fixed Costs", f"€{fk['fixed']:,.0f}")
        st.metric("Net Margin", f"€{fk['margin']:,.0f}")
        st.metric("Margin %", f"{fk['margin_pct']*100:.1f}%")
        st.metric("Lost Revenue", f"€{fk['lost_rev']:,.0f}")


# ════════════════════════════════════════════════════════════════
# WEEK-BY-WEEK TABLE
# ════════════════════════════════════════════════════════════════
with st.expander("📊 Detailed Week-by-Week Data", expanded=False):
    table_data = []
    for s in states:
        table_data.append({
            'Week': s['week'], 'Demand': s['demand'], 'Forecast': s['forecast'],
            'Sales': s['sales'], 'Missed': s['missed'], 'Store Stock': s['store_stock'],
            'Order': s['order'], 'Pending': s['pending'],
            'Suppl Ship': s['supplier_shipped'], 'Mat Arrived': s['mat_arr'],
            'Semi Input': s['semi_input'], 'FP Input': s['fp_input'],
            'Comment': s['comment'],
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=500)


# ════════════════════════════════════════════════════════════════
# SAVE & COMPARE (nice to have)
# ════════════════════════════════════════════════════════════════
with st.expander("💾 Save Scenario for Comparison", expanded=False):
    scenario_name = st.text_input("Scenario Name", f"SC_{order_freq}wk_{init_store}store")
    if st.button("Save Current Scenario"):
        if 'saved_scenarios' not in st.session_state:
            st.session_state.saved_scenarios = {}
        st.session_state.saved_scenarios[scenario_name] = {
            'params': params.copy(),
            'kpis': final_kpis.copy(),
        }
        st.success(f"Saved '{scenario_name}'!")
    
    if 'saved_scenarios' in st.session_state and len(st.session_state.saved_scenarios) > 0:
        st.markdown("### Saved Scenarios Comparison")
        comp_data = []
        for name, data in st.session_state.saved_scenarios.items():
            k = data['kpis']
            p = data['params']
            comp_data.append({
                'Scenario': name,
                'Order Freq': f"{p['order_freq']}wk",
                'Init Stock': p['init_store'],
                'Service %': f"{k['svc_level']*100:.1f}%",
                'Sales': f"{k['total_sales']:,.0f}",
                'Missed': f"{k['total_missed']:,.0f}",
                'Revenue': f"€{k['revenue']:,.0f}",
                'Margin': f"€{k['margin']:,.0f}",
                'Margin %': f"{k['margin_pct']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
        if st.button("Clear All Saved"):
            st.session_state.saved_scenarios = {}
            st.rerun()
