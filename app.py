import streamlit as st
import json, math
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Supply Chain Agility Simulator", page_icon="\U0001f3ed")

# ════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "mat_lt": 6, "semi_lt": 3, "fp_lt": 1, "dist_lt": 1,
    "order_freq": 4,
    "total_stock": 1500,
    "store_pct": 60, "wh_pct": 20, "semi_pct": 10,
    "store_a_pct": 50, "smart_distrib": False,
    "demand_shape": "\u27a1\ufe0f Flat (constant demand)",
}

DEMAND_SHAPES = [
    "\u27a1\ufe0f Flat (constant demand)",
    "\U0001f4c8 Linear ramp then flat",
    "\U0001f4c9 Linear drop then flat",
    "\U0001f514 Poisson curve (launch peak)",
]
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

BASE_FORECAST = 100  # Fixed (point 3)

# ════════════════════════════════════════════════════════════════
# PRESET SCENARIOS (3 Lead Time × 3 Demand)
# ════════════════════════════════════════════════════════════════
LT_PROFILES = {
    "Agile": {"mat_lt": 4, "semi_lt": 2, "fp_lt": 1, "dist_lt": 1, "order_freq": 1},
    "Medium": {"mat_lt": 8, "semi_lt": 4, "fp_lt": 2, "dist_lt": 2, "order_freq": 2},
    "Push":   {"mat_lt": 12, "semi_lt": 6, "fp_lt": 3, "dist_lt": 3, "order_freq": 4},
}


# ════════════════════════════════════════════════════════════════
# VALORIZATION CONSTANTS
# ════════════════════════════════════════════════════════════════
VALOR_RAW_MAT = 0.50
VALOR_SEMI    = 0.75
VALOR_FINISHED = 1.00

# ════════════════════════════════════════════════════════════════
# SIMULATION ENGINE — with stage-specific push logic (points 9, 10)
# ════════════════════════════════════════════════════════════════
@st.cache_data
def run_simulation(weeks, init_store, init_cw, init_semi, init_rawmat,
                   order_freq, mat_lt, semi_lt, fp_lt, dist_lt,
                   cap_start, cap_ramp, base_forecast,
                   demand_mult, ramp_start, ramp_end,
                   price, var_cost, fixed_pct, store_a_pct, smart_distrib,
                   custom_demand=None):

    phys_lt = mat_lt + semi_lt + fp_lt + dist_lt
    coverage = phys_lt + order_freq
    pct_a = store_a_pct / 100.0
    pct_b = 1.0 - pct_a


    demand = {}
    if custom_demand is not None:
        for w in range(0, weeks + 1):
            demand[w] = int(custom_demand[w]) if w < len(custom_demand) else int(custom_demand[-1])
    else:
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

    mat_pipe = [0.0] * max(1, mat_lt)
    semi_pipe = [0.0] * max(1, semi_lt)
    fp_pipe = [0.0] * max(1, fp_lt)
    dist_pipe_a = [0.0] * max(1, dist_lt)
    dist_pipe_b = [0.0] * max(1, dist_lt)

    # Initial store stock: ALWAYS 50/50 (push reality — planner hasn't reviewed yet)
    store_a = float(init_store) / 2.0
    store_b = float(init_store) / 2.0
    raw_mat = float(init_rawmat)
    semi = float(init_semi)
    cw = float(init_cw)

    pb = 0.0  # supplier backlog
    pn = 0; pc = float(cap_start)
    sn = 0; sc_ = float(cap_start)
    fn = 0; fpc = float(cap_start)
    co = 0.0; cas = 0.0
    smart_discovered = False
    first_order_week = None  # capacity ramp starts the week AFTER first order

    ow = list(range(order_freq, weeks + 1, order_freq)) if order_freq > 1 else list(range(1, weeks + 1))
    states = []
    ff = float(base_forecast)  # forecast: starts at base, updates at planning freq only

    s0 = {
        'week': 0,
        'demand': 0, 'demand_a': 0, 'demand_b': 0,
        'forecast': base_forecast,
        'mat_arr': 0, 'semi_arr': 0, 'fp_arr': 0,
        'dist_arr_a': 0, 'dist_arr_b': 0, 'dist_arr': 0,
        'sales_a': 0, 'sales_b': 0, 'sales': 0,
        'missed_a': 0, 'missed_b': 0, 'missed': 0,
        'store_a': store_a, 'store_b': store_b, 'store_stock': store_a + store_b,
        'supplier_shipped': 0, 'supplier_cap': cap_start,
        'raw_mat_before_prod': raw_mat, 'raw_mat_stock': raw_mat,
        'semi_input': 0, 'semi_cap': cap_start, 'semi_stock': semi,
        'fp_input': 0, 'fp_cap': cap_start,
        'cw_shipped': 0, 'cw_stock': cw,
        'alloc_a': 0, 'alloc_b': 0,
        'mat_pipe': [0.0] * max(1, mat_lt),
        'semi_pipe': [0.0] * max(1, semi_lt),
        'fp_pipe': [0.0] * max(1, fp_lt),
        'dist_pipe_a': [0.0] * max(1, dist_lt),
        'dist_pipe_b': [0.0] * max(1, dist_lt),
        'order': 0, 'pending': 0, 'backlog': 0,
        'wip_total': 0,
        # Point 4: initial stock costs (valorized at their stage)
        'cost_mat': round(init_rawmat * var_cost * VALOR_RAW_MAT, 1),
        'cost_semi': round(init_semi * var_cost * VALOR_SEMI, 1),
        'cost_fp': round((init_store + init_cw) * var_cost * VALOR_FINISHED, 1),
        'coverage': coverage,
        'comment': "Week 0 - Initial state.",
    }
    states.append(s0)

    for w in range(1, weeks + 1):
        s = {'week': w}
        dem_total = demand[w]
        dem_a = round(dem_total * pct_a)
        dem_b = dem_total - dem_a

        # Forecast: updates only at planning frequency (periodic review)
        # Between reviews, system is blind to demand changes
        if w in ow:
            ff = float(dem_total)  # update forecast to current demand curve
            if smart_distrib and not smart_discovered:
                smart_discovered = True  # planner discovers demand imbalance
        s['demand'] = dem_total; s['demand_a'] = dem_a; s['demand_b'] = dem_b
        s['forecast'] = round(ff, 1)

        # 1. Arrivals from pipes
        m_arr = mat_pipe[0]; sm_arr = semi_pipe[0]; fp_arr = fp_pipe[0]
        da_arr = dist_pipe_a[0]; db_arr = dist_pipe_b[0]
        d_arr_total = da_arr + db_arr
        # Clear fronts — these units have arrived, no longer in transit
        mat_pipe[0] = 0.0; semi_pipe[0] = 0.0; fp_pipe[0] = 0.0
        dist_pipe_a[0] = 0.0; dist_pipe_b[0] = 0.0
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
        })

        # 4. Supplier — capacity ramps starting the week AFTER first order
        if first_order_week is not None and w > first_order_week:
            pn += 1
        pc = cap_start * (1 + pn * cap_ramp) if pn > 0 else float(cap_start)
        pc = min(pc, cap_start * 10)
        if pb > 0.01:
            shipped = math.ceil(min(pb, pc)); pb -= shipped
        else:
            shipped = 0.0
        s['supplier_shipped'] = round(shipped, 1); s['supplier_cap'] = round(pc, 0)

        # 5. Arrivals update stocks
        raw_mat += m_arr; s['raw_mat_before_prod'] = round(raw_mat, 1)
        semi += sm_arr
        cw += fp_arr

        # 6. Semi — process RM into semi (capacity-limited). Cost at TRANSFORMATION.
        sc_ = min(cap_start * (1 + sn * cap_ramp), cap_start * 10)
        if raw_mat > 0.01:
            si = math.ceil(min(raw_mat, sc_))
            raw_mat -= si
        else:
            si = 0.0
        sn += 1  # ramp for next week (monotonic, always increments)
        s['semi_input'] = round(si, 1); s['semi_cap'] = round(sc_, 0)
        s['raw_mat_stock'] = round(raw_mat, 1)
        s['cost_semi'] = round(si * var_cost * (VALOR_SEMI - VALOR_RAW_MAT), 1)

        # 7. FP — process semi into finished (capacity-limited). Cost at TRANSFORMATION.
        fpc = min(cap_start * (1 + fn * cap_ramp), cap_start * 10)
        if semi > 0.01:
            fi = math.ceil(min(semi, fpc))
            semi -= fi
        else:
            fi = 0.0
        fn += 1  # ramp for next week (monotonic, always increments)
        s['fp_input'] = round(fi, 1); s['fp_cap'] = round(fpc, 0)
        s['semi_stock'] = round(semi, 1)
        s['cost_fp'] = round(fi * var_cost * (VALOR_FINISHED - VALOR_SEMI), 1)

        # 8. CW — push everything to stores, allocate per-store
        ship_out = math.ceil(cw) if cw > 0.01 else 0.0
        cw -= ship_out
        s['cw_shipped'] = round(ship_out, 1); s['cw_stock'] = round(cw, 1)

        # Allocate: Before first planning review = 50/50 (blind)
        # After first review: equalize weeks-of-cover, then split by demand rate
        if ship_out > 0:
            if smart_distrib and smart_discovered:
                dem_a_wk = max(ff * pct_a, 0.01)
                dem_b_wk = max(ff * pct_b, 0.01)
                cover_a = (store_a + sum(dist_pipe_a)) / dem_a_wk
                cover_b = (store_b + sum(dist_pipe_b)) / dem_b_wk
                # Priority: fill the worst-covered store first to equalize
                if cover_a < cover_b:
                    gap = math.ceil(max(0, (cover_b - cover_a) * dem_a_wk))
                    priority_a = min(ship_out, gap)
                    remaining = ship_out - priority_a
                    alloc_a = priority_a + round(remaining * pct_a)
                    alloc_b = ship_out - alloc_a
                elif cover_b < cover_a:
                    gap = math.ceil(max(0, (cover_a - cover_b) * dem_b_wk))
                    priority_b = min(ship_out, gap)
                    remaining = ship_out - priority_b
                    alloc_b = priority_b + round(remaining * pct_b)
                    alloc_a = ship_out - alloc_b
                else:
                    alloc_a = round(ship_out * pct_a)
                    alloc_b = ship_out - alloc_a
            else:
                # Push / not yet discovered: 50/50 blind
                alloc_a = round(ship_out * 0.5)
                alloc_b = ship_out - alloc_a
        else:
            alloc_a = 0; alloc_b = 0
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

        # RM cost: anticipated 1 week — book cost of what will arrive NEXT week
        s['cost_mat'] = round(mat_pipe[0] * var_cost * VALOR_RAW_MAT, 1)

        # 10. Order — ACCOUNT FOR ALL WIP INCLUDING SUPPLIER BACKLOG
        total_wip = (sum(mat_pipe) + sum(semi_pipe) + sum(fp_pipe)
                     + sum(dist_pipe_a) + sum(dist_pipe_b)
                     + raw_mat + semi + cw + pb)  # pb = units ordered, not yet shipped
        s['wip_total'] = round(total_wip, 1)

        if w in ow:
            tgt = ff * coverage
            existing = store_total + total_wip
            od = math.ceil(max(0, tgt - existing))
            co += od; pb += od; s['order'] = round(od, 0)
            if od > 0 and first_order_week is None:
                first_order_week = w  # capacity starts ramping next week
        else:
            s['order'] = 0
        s['pending'] = round(co - cas, 0); s['backlog'] = round(pb, 0)
        s['coverage'] = coverage

        # Commentary
        parts = []
        if missed_a > 0.5:
            parts.append(f"A: lost {missed_a:.0f}/{dem_a:.0f}.")
        else:
            parts.append(f"A: sold {sales_a:.0f}/{dem_a:.0f}, stk {store_a:.0f}.")
        if missed_b > 0.5:
            parts.append(f"B: lost {missed_b:.0f}/{dem_b:.0f}.")
        else:
            parts.append(f"B: sold {sales_b:.0f}/{dem_b:.0f}, stk {store_b:.0f}.")
        if s['order'] > 0:
            parts.append(f"ORDER {od:.0f}.")
        if shipped > 0.5:
            parts.append(f"Supplier {shipped:.0f}.")
        if ship_out > 0.5:
            if smart_distrib and smart_discovered:
                mode = "smart"
            else:
                mode = "push 50/50"
            parts.append(f"WH\u2192A:{alloc_a:.0f} B:{alloc_b:.0f} ({mode}).")
        s['comment'] = " ".join(parts)
        states.append(s)

    return states


def compute_kpis(states, price, var_cost, fixed_pct, base_forecast, weeks,
                 init_store=0, init_cw=0, init_semi=0, init_rawmat=0):
    ts = sum(s['sales'] for s in states)
    tm = sum(s['missed'] for s in states)
    td = sum(s['demand'] for s in states)
    tfp = sum(s['fp_input'] for s in states)

    # Initial stock: valorized at its stage rate (already invested)
    init_stock_value = (init_store * var_cost * VALOR_FINISHED
                       + init_cw * var_cost * VALOR_FINISHED
                       + init_semi * var_cost * VALOR_SEMI
                       + init_rawmat * var_cost * VALOR_RAW_MAT)

    # Production cost = SUM OF INCREMENTAL costs at each stage (no double counting)
    # Supplier ships RM: +50% per unit
    # Semi processing:   +25% per unit (50%→75%)
    # Finishing:          +25% per unit (75%→100%)
    cost_mat_total = sum(s.get('cost_mat', 0) for s in states)
    cost_semi_total = sum(s.get('cost_semi', 0) for s in states)
    cost_fp_total = sum(s.get('cost_fp', 0) for s in states)
    prod_cost = cost_mat_total + cost_semi_total + cost_fp_total

    vc = init_stock_value + prod_cost

    rev = ts * price
    gm = rev - vc
    fx = base_forecast * weeks * price * fixed_pct
    mg = gm - fx

    # End stock
    last = states[-1] if states else {}
    end_store = last.get('store_a', 0) + last.get('store_b', 0)
    end_cw = last.get('cw_stock', 0)
    end_semi = last.get('semi_stock', 0)
    end_rawmat = last.get('raw_mat_stock', 0)
    end_stock_units = end_store + end_cw + end_semi + end_rawmat

    end_stock_value = (end_store * var_cost * VALOR_FINISHED
                      + end_cw * var_cost * VALOR_FINISHED
                      + end_semi * var_cost * VALOR_SEMI
                      + end_rawmat * var_cost * VALOR_RAW_MAT)

    end_pipe_units = (sum(last.get('mat_pipe', [0]))
                     + sum(last.get('semi_pipe', [0]))
                     + sum(last.get('fp_pipe', [0]))
                     + sum(last.get('dist_pipe_a', [0]))
                     + sum(last.get('dist_pipe_b', [0])))

    # Pipeline value (valorized by position)
    end_pipe_value = (sum(last.get('mat_pipe', [0])) * var_cost * VALOR_RAW_MAT
                     + sum(last.get('semi_pipe', [0])) * var_cost * VALOR_SEMI
                     + sum(last.get('fp_pipe', [0])) * var_cost * VALOR_FINISHED
                     + sum(last.get('dist_pipe_a', [0])) * var_cost * VALOR_FINISHED
                     + sum(last.get('dist_pipe_b', [0])) * var_cost * VALOR_FINISHED)

    total_system_units = ts + end_stock_units + end_pipe_units
    useful_pct = (ts / total_system_units * 100) if total_system_units > 0 else 0
    useless_units = end_stock_units + end_pipe_units
    useless_pct = (useless_units / total_system_units * 100) if total_system_units > 0 else 0

    # P&L: cost of sold = sold units * full cost; cost of unsold = total VC - cost of sold
    cost_of_sold = ts * var_cost
    cost_of_unsold = max(0, vc - cost_of_sold)

    return {
        'total_demand': td, 'total_sales': ts, 'total_missed': tm,
        'svc_level': ts / td if td > 0 else 0,
        'stockout_weeks': sum(1 for s in states if s['missed'] > 0.5),
        'revenue': rev, 'var_cost': vc, 'gm': gm, 'fixed': fx,
        'margin': mg, 'margin_pct': mg / rev if rev > 0 else 0,
        'produced': tfp,
        'init_stock_value': init_stock_value,
        'prod_cost': prod_cost,
        'cost_mat_total': cost_mat_total,
        'cost_semi_total': cost_semi_total,
        'cost_fp_total': cost_fp_total,
        'end_stock_value': end_stock_value,
        'end_pipe_value': end_pipe_value,
        'end_stock_units': end_stock_units,
        'end_pipe_units': end_pipe_units,
        'store_end': end_store,
        'lost_rev': tm * price,
        'missed_a': sum(s['missed_a'] for s in states),
        'missed_b': sum(s['missed_b'] for s in states),
        'sales_a': sum(s['sales_a'] for s in states),
        'sales_b': sum(s['sales_b'] for s in states),
        'useful_pct': useful_pct, 'useless_pct': useless_pct,
        'useful_units': ts, 'useless_units': useless_units,
        'total_system_units': total_system_units,
        'cost_of_sold': cost_of_sold, 'cost_of_unsold': cost_of_unsold,
    }


def cumulative_kpis(states, week, price, var_cost, fixed_pct, base_forecast, total_weeks,
                    init_store=0, init_cw=0, init_semi=0, init_rawmat=0):
    sub = states[:week]
    ts = sum(s['sales'] for s in sub); tm = sum(s['missed'] for s in sub)
    td = sum(s['demand'] for s in sub); tfp = sum(s['fp_input'] for s in sub)
    init_stock_value = (init_store * var_cost * VALOR_FINISHED
                       + init_cw * var_cost * VALOR_FINISHED
                       + init_semi * var_cost * VALOR_SEMI
                       + init_rawmat * var_cost * VALOR_RAW_MAT)
    # Incremental production costs (no double counting)
    prod_cost = (sum(s.get('cost_mat', 0) for s in sub)
                + sum(s.get('cost_semi', 0) for s in sub)
                + sum(s.get('cost_fp', 0) for s in sub))
    vc = init_stock_value + prod_cost
    rev = ts * price; gm = rev - vc
    fx = base_forecast * week * price * fixed_pct
    mg = gm - fx

    init_total = init_store + init_cw + init_semi + init_rawmat
    total_in = init_total + tfp
    useful_pct = (ts / total_in * 100) if total_in > 0 else 0

    return {'sales': ts, 'missed': tm, 'demand': td, 'revenue': rev,
            'svc_level': ts / td if td > 0 else 0, 'margin': mg,
            'stockout_wks': sum(1 for s in sub if s['missed'] > 0.5),
            'missed_a': sum(s['missed_a'] for s in sub),
            'missed_b': sum(s['missed_b'] for s in sub),
            'useful_pct': useful_pct, 'useless_pct': 100 - useful_pct}


# ════════════════════════════════════════════════════════════════
# SC VISUALIZATION
# ════════════════════════════════════════════════════════════════
def make_sc_html(state, params):
    var_cost = params.get('var_cost', 200)
    cap_ref = max(1, params['cap_start'] * 2.5)
    mat_lt = params['mat_lt']; semi_lt = params['semi_lt']
    fp_lt = params['fp_lt']; dist_lt = params['dist_lt']

    # Grey-blue LVMH palette
    C_TXT = '#4a5a6e'; C_TXT_L = '#8a96a6'; C_ARR = '#a0aab8'
    C_SUP_BG = '#1a2744'; C_SUP_FG = '#fff'
    C_BUF_BG = '#f2f4f8'; C_BUF_BDR = '#a0aab8'; C_BUF_FG = '#3a4a5c'
    C_BAND_BG = '#e8ecf2'; C_PROC_BG = '#ccd4e0'
    C_BOX_EMPTY = '#d0d8e4'; C_BOX_FULL = '#6a7e96'; C_BOX_BDR = '#8a96a8'
    C_PROC_BDR = '#4a5e74'
    C_LOST_BG = '#f8e8e8'; C_LOST_BDR = '#c05050'; C_LOST_FG = '#8a2020'

    arr = f'<div style="color:{C_ARR};font-size:16px;display:flex;align-items:center;flex:0 0 auto;padding:0 2px;">\u25b8</div>'

    def transit_box(v, size="32px"):
        if v > 0.5:
            return (f'<div style="width:{size};height:{size};background:{C_BOX_FULL};'
                    f'border:1px solid {C_BOX_BDR};border-radius:3px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:9px;font-weight:700;color:#fff;">{v:.0f}</div>')
        return (f'<div style="width:{size};height:{size};background:{C_BOX_EMPTY};'
                f'border:1px solid {C_BOX_BDR};border-radius:3px;"></div>')

    def proc_box(v, size="32px"):
        if v > 0.5:
            return (f'<div style="width:{size};height:{size};background:{C_BOX_FULL};'
                    f'border:2px dashed {C_PROC_BDR};border-radius:3px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:9px;font-weight:700;color:#fff;">{v:.0f}</div>')
        return (f'<div style="width:{size};height:{size};background:{C_BOX_EMPTY};opacity:0.3;'
                f'border:2px dashed {C_PROC_BDR};border-radius:3px;"></div>')

    def make_band(pipe, label, lt):
        transit = pipe[1:] if len(pipe) > 1 else []
        proc_val = pipe[0] if pipe else 0
        n = len(transit)
        bsize = "24px" if lt > 8 else "32px"
        # Transit boxes
        t_html = ""
        if n > 0:
            if n > 6:
                mid = (n + 1) // 2
                r1 = "".join(transit_box(transit[-(i+1)], "22px") for i in range(mid))
                r2 = "".join(transit_box(transit[-(i+1)], "22px") for i in range(mid, n))
                t_html = (f'<div style="display:flex;flex-direction:column;gap:1px;">'
                    f'<div style="display:flex;gap:1px;">{r1}</div>'
                    f'<div style="display:flex;gap:1px;">{r2}</div></div>')
            else:
                boxes = "".join(transit_box(transit[-(i+1)], bsize) for i in range(n))
                t_html = f'<div style="display:flex;gap:2px;">{boxes}</div>'
        # Processing box
        p_html = proc_box(proc_val, bsize)
        # Band assembly
        parts = []
        if t_html:
            parts.append(f'<div style="padding:4px 6px;background:{C_BAND_BG};border-radius:8px 0 0 8px;display:flex;align-items:center;gap:2px;">{t_html}</div>')
        parts.append(f'<div style="padding:4px 6px;background:{C_PROC_BG};{"border-radius:8px;" if not t_html else "border-radius:0 8px 8px 0;"}'
                    f'display:flex;align-items:center;flex-direction:column;gap:1px;">'
                    f'<div style="font-size:6px;color:{C_PROC_BDR};font-weight:700;letter-spacing:0.5px;text-transform:uppercase;">proc</div>'
                    f'{p_html}</div>')
        return (f'<div style="display:flex;flex-direction:column;align-items:center;flex:1 1 auto;">'
                f'<div style="font-size:8px;color:{C_TXT_L};margin-bottom:2px;font-weight:600;letter-spacing:0.3px;">{label}</div>'
                f'<div style="display:flex;border:1px solid {C_BOX_BDR};border-radius:8px;overflow:hidden;">'
                f'{"".join(parts)}</div></div>')

    def buffer_card(title, stock, valor_rate=1.0, sub=""):
        valor_eur = stock * var_cost * valor_rate
        valor_txt = f"\u20ac{valor_eur:,.0f}" if stock > 0.5 else ""
        return (f'<div style="background:{C_BUF_BG};border:1.5px solid {C_BUF_BDR};border-radius:10px;'
                f'padding:6px 10px;min-width:52px;text-align:center;flex:0 0 auto;">'
                f'<div style="font-size:7px;font-weight:700;color:{C_TXT_L};text-transform:uppercase;letter-spacing:0.5px;">{title}</div>'
                f'<div style="font-size:20px;font-weight:800;color:{C_BUF_FG};">{stock:.0f}</div>'
                f'{"<div style=font-size:7px;color:"+C_TXT_L+";>"+sub+"</div>" if sub else ""}'
                f'{"<div style=font-size:7px;color:"+C_TXT_L+";font-style:italic;>"+valor_txt+"</div>" if valor_txt else ""}'
                f'</div>')

    def store_card(title, stock, dem, alert="", processing=0):
        is_alert = alert != ""
        bg = C_LOST_BG if is_alert else C_BUF_BG
        bdr = C_LOST_BDR if is_alert else C_BUF_BDR
        valor_eur = (stock + processing) * var_cost * VALOR_FINISHED
        valor_txt = f"\u20ac{valor_eur:,.0f}" if (stock + processing) > 0.5 else ""
        proc_html = ""
        if processing > 0.5:
            proc_html = (f'<div style="margin-top:2px;font-size:7px;color:{C_TXT_L};">\u2699 proc</div>'
                        f'<div style="width:26px;height:26px;background:{C_BOX_FULL};'
                        f'border:1.5px dashed {C_PROC_BDR};border-radius:3px;'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-size:8px;font-weight:700;color:#fff;margin:1px auto;">{processing:.0f}</div>')
        return (f'<div style="background:{bg};border:1.5px solid {bdr};border-radius:10px;'
                f'padding:6px 8px;min-width:64px;text-align:center;flex:0 0 auto;">'
                f'<div style="font-size:7px;font-weight:700;color:{C_TXT_L};text-transform:uppercase;letter-spacing:0.5px;">{title}</div>'
                f'<div style="font-size:20px;font-weight:800;color:{C_BUF_FG};">{stock:.0f}</div>'
                f'{proc_html}'
                f'<div style="font-size:7px;color:{C_TXT_L};">Dem {dem:.0f}/wk</div>'
                f'{"<div style=font-size:7px;color:"+C_TXT_L+";font-style:italic;>"+valor_txt+"</div>" if valor_txt else ""}'
                f'{"<div style=font-size:8px;color:"+C_LOST_FG+";font-weight:700;>"+alert+"</div>" if alert else ""}'
                f'</div>')

    alert_a = f"LOST {state['missed_a']:.0f}" if state['missed_a'] > 0.5 else ""
    alert_b = f"LOST {state['missed_b']:.0f}" if state['missed_b'] > 0.5 else ""
    order_html = f"<b style='color:#2a5a3a;font-size:13px;'>ORDER {state['order']:.0f}</b>" if state['order'] > 0 else f"<span style='color:{C_TXT_L};'>No order</span>"
    wip_val = state.get('wip_total', 0)

    # Supplier
    sup_cap = state['supplier_cap']
    sup_val = state['backlog'] * var_cost * VALOR_RAW_MAT
    sup_html = (f'<div style="background:{C_SUP_BG};border:1.5px solid #0d1a30;border-radius:10px;'
                f'padding:8px 10px;min-width:58px;text-align:center;flex:0 0 auto;">'
                f'<div style="font-size:14px;line-height:1;">\U0001f3ed</div>'
                f'<div style="font-size:7px;font-weight:700;color:#8aa0c0;text-transform:uppercase;letter-spacing:0.5px;">Supplier</div>'
                f'<div style="font-size:20px;font-weight:800;color:{C_SUP_FG};">{state["backlog"]:.0f}</div>'
                f'<div style="font-size:7px;color:#8aa0c0;">Cap {sup_cap:.0f}/wk</div>'
                f'{"<div style=font-size:7px;color:#8aa0c0;font-style:italic;>\u20ac"+f"{sup_val:,.0f}"+"</div>" if state["backlog"]>0.5 else ""}'
                f'</div>')

    # Info bar
    info_bar = f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 16px;
        background:linear-gradient(90deg,#f4f6f9,#eef1f6);
        border:1px solid #dde2ea;border-radius:8px;margin-bottom:8px;">
        <span style="font-size:11px;color:{C_TXT};">Backlog <b style="color:#8a3030;">{state['backlog']:.0f}</b></span>
        <span style="font-size:11px;color:{C_TXT};">Pending <b style="color:#8a6a20;">{state['pending']:.0f}</b></span>
        <span style="font-size:11px;color:{C_TXT};">WIP <b style="color:#2a5a8a;">{wip_val:.0f}</b></span>
        <span style="font-size:11px;">{order_html}</span>
        <span style="font-size:11px;color:{C_TXT};">Forecast <b style="color:#1a2a40;">{state['forecast']:.0f}</b>/wk</span>
        <span style="font-size:11px;color:{C_TXT};">A:{params['store_a_pct']}% B:{100-params['store_a_pct']}%</span>
        <span style="font-size:9px;color:{C_TXT_L};font-weight:700;letter-spacing:1px;">\u25c2 INFO FLOW</span>
    </div>'''

    # Build bands
    mat_band = make_band(state['mat_pipe'], f"Material {mat_lt}wk", mat_lt)
    rm_card = buffer_card("Raw Mat", state['raw_mat_stock'], VALOR_RAW_MAT)
    semi_band = make_band(state['semi_pipe'], f"Semi {semi_lt}wk", semi_lt)
    fp_band = make_band(state['fp_pipe'], f"Finish {fp_lt}wk", fp_lt)
    cw_card = buffer_card("Central WH", state['cw_stock'], VALOR_FINISHED,
                          f"A:{state['alloc_a']:.0f} B:{state['alloc_b']:.0f}")

    upstream = (f'<div style="display:flex;align-items:center;gap:3px;flex:1 1 auto;">'
                f'{sup_html}{arr}{mat_band}{arr}{rm_card}{arr}{semi_band}{arr}{fp_band}{arr}{cw_card}</div>')

    # Store branches
    aa = state['alloc_a']; ab = state['alloc_b']
    arr_a = (f'<div style="color:{C_ARR};font-size:14px;display:flex;flex-direction:column;align-items:center;'
             f'flex:0 0 auto;padding:0 1px;"><span>\u25b8</span>'
             f'{"<span style=font-size:8px;color:#2a5a8a;font-weight:700;>"+str(int(aa))+"</span>" if aa > 0.5 else ""}</div>')
    arr_b = (f'<div style="color:{C_ARR};font-size:14px;display:flex;flex-direction:column;align-items:center;'
             f'flex:0 0 auto;padding:0 1px;"><span>\u25b8</span>'
             f'{"<span style=font-size:8px;color:#5a4a8a;font-weight:700;>"+str(int(ab))+"</span>" if ab > 0.5 else ""}</div>')

    dt_a = state['dist_pipe_a'][1:] if len(state['dist_pipe_a']) > 1 else []
    dt_b = state['dist_pipe_b'][1:] if len(state['dist_pipe_b']) > 1 else []
    dp_a = state['dist_pipe_a'][0] if state['dist_pipe_a'] else 0
    dp_b = state['dist_pipe_b'][0] if state['dist_pipe_b'] else 0

    def dist_transit_html(tlist):
        if not tlist: return ''
        boxes = "".join(transit_box(tlist[-(i+1)], "28px") for i in range(len(tlist)))
        return f'<div style="display:flex;gap:2px;">{boxes}</div>{arr}'

    sa = store_card("Store A", state['store_a'], state['demand_a'], alert_a, dp_a)
    sb = store_card("Store B", state['store_b'], state['demand_b'], alert_b, dp_b)

    branch_a = f'<div style="display:flex;align-items:center;gap:3px;">{arr_a}{dist_transit_html(dt_a)}{sa}</div>'
    branch_b = f'<div style="display:flex;align-items:center;gap:3px;">{arr_b}{dist_transit_html(dt_b)}{sb}</div>'

    flow = f'''<div style="display:flex;align-items:center;gap:0px;padding:12px 10px;width:100%;box-sizing:border-box;
        background:linear-gradient(90deg,#f6f8fa,#f0f2f6);
        border:1px solid #dde2ea;border-radius:12px;">
        {upstream}
        <div style="display:flex;flex-direction:column;gap:5px;flex:0 0 auto;">
            {branch_a}
            {branch_b}
        </div>
    </div>'''

    comment = state.get('comment', '')
    comment_html = f'<div style="padding:8px 16px;font-size:11px;color:{C_TXT};line-height:1.5;background:#f8f9fb;border:1px solid #e8ecf0;border-radius:6px;margin-top:6px;">{comment}</div>'
    physical_flow = f'<div style="text-align:center;padding:8px 0;"><span style="font-size:10px;color:{C_TXT_L};letter-spacing:2px;font-weight:700;">- - - PHYSICAL FLOW (GOODS) - - -</span></div>'

    return f'<div style="font-family:Arial,Helvetica,sans-serif;">{info_bar}<div style="padding:4px 0;">{flow}</div>{physical_flow}{comment_html}</div>'

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## \u2699\ufe0f Supply Chain Setup")
    weeks = st.select_slider("Simulation Length (weeks)", options=[13, 26, 39, 52], value=26)

    # 1. LEAD TIMES
    st.markdown("### \U0001f517 Lead Times (weeks)")
    c1, c2 = st.columns(2)
    with c1:
        mat_lt = st.number_input("Material", min_value=1, max_value=24, step=1, key="mat_lt")
        semi_lt = st.number_input("Semi-Fin", min_value=1, max_value=12, step=1, key="semi_lt")
    with c2:
        fp_lt = st.number_input("Finishing", min_value=1, max_value=12, step=1, key="fp_lt")
        dist_lt = st.number_input("Distribution", min_value=1, max_value=12, step=1, key="dist_lt")
    phys_lt = mat_lt + semi_lt + fp_lt + dist_lt
    st.caption(f"Physical LT: **{phys_lt}** weeks")

    # 2. PLANNING (base forecast fixed at 100 — point 3)
    st.markdown("### \U0001f4cb Planning")
    order_freq = st.slider("Order / Replenishment Frequency (weeks)", min_value=1, max_value=4, step=1, key="order_freq")
    base_forecast = BASE_FORECAST
    coverage = phys_lt + order_freq
    st.caption(f"Base forecast: **{base_forecast}** pcs/wk (fixed)")
    st.caption(f"Coverage target: **{coverage}** weeks (LT {phys_lt} + freq {order_freq})")

    # 3. INITIAL STOCK
    st.markdown("### \U0001f4e6 Initial Stock")
    recommended_stock = base_forecast * coverage
    st.markdown(
        f'<div style="color:#8a96a6;font-size:12px;font-style:italic;margin-bottom:8px;">'
        f'[Recommended: <b>{recommended_stock:,.0f}</b> pcs = '
        f'{base_forecast}/wk \u00d7 {coverage} wks coverage]</div>',
        unsafe_allow_html=True)
    total_stock = st.slider("Total Initial Stock (pcs)", min_value=0, max_value=10000, step=50, key="total_stock")

    st.caption("Distribution (% of total):")
    _sp = st.session_state.get("store_pct", 60)
    _wp = st.session_state.get("wh_pct", 20)
    _sep = st.session_state.get("semi_pct", 10)
    if _wp > 100 - _sp:
        st.session_state["wh_pct"] = max(0, 100 - _sp)
    if _sep > 100 - _sp - st.session_state.get("wh_pct", 0):
        st.session_state["semi_pct"] = max(0, 100 - _sp - st.session_state.get("wh_pct", 0))

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        store_pct = st.number_input("Store %", min_value=0, max_value=100, step=5, key="store_pct")
    with sc2:
        wh_max = max(0, 100 - store_pct)
        warehouse_pct = st.number_input("Warehouse %", min_value=0, max_value=wh_max, step=5, key="wh_pct")
    with sc3:
        semi_max = max(0, 100 - store_pct - warehouse_pct)
        semi_pct = st.number_input("Semi-Fin %", min_value=0, max_value=semi_max, step=5, key="semi_pct")
    rawmat_pct = 100 - store_pct - warehouse_pct - semi_pct

    init_store = int(round(total_stock * store_pct / 100))
    init_cw = int(round(total_stock * warehouse_pct / 100))
    init_semi = int(round(total_stock * semi_pct / 100))
    init_rawmat = total_stock - init_store - init_cw - init_semi

    st.markdown(
        f'<div style="background:#f0f2f5;border-radius:8px;padding:8px 12px;font-size:11px;line-height:1.8;">'
        f'<b>Store:</b> {init_store} ({store_pct}%) @ 100% <i>(always 50/50 initial)</i> | '
        f'<b>WH:</b> {init_cw} ({warehouse_pct}%) @ 100% | '
        f'<b>Semi:</b> {init_semi} ({semi_pct}%) @ 75% | '
        f'<b>RM:</b> {init_rawmat} ({rawmat_pct}%) @ 50%</div>',
        unsafe_allow_html=True)

    # 4. STORE DEMAND SPLIT
    st.markdown("### \U0001f3ea Store Demand Split")
    store_a_pct = st.slider("Store A demand (%)", 0, 100, step=5, key="store_a_pct")
    smart_distrib = st.toggle("Smart Distribution (need-based)", key="smart_distrib")
    if smart_distrib:
        st.caption(f"A: **{store_a_pct}%** B: **{100-store_a_pct}%** \u2014 Stores start 50/50, CW rebalances at first planning review")
    else:
        st.caption(f"A: **{store_a_pct}%** B: **{100-store_a_pct}%** \u2014 Push 50/50 always")

    # 5. DEMAND PROFILE (point 1, 2: correct slider ranges)
    st.markdown("### \U0001f4c8 Demand Profile")
    custom_demand = None
    demand_mult = 1.0; ramp_start = 1; ramp_end = 1
    demand_description = ""  # point 4: for saving

    bf = base_forecast
    preset_shape = st.selectbox("Demand shape", DEMAND_SHAPES, key="demand_shape")

    if "Flat" in preset_shape:
        init_demand = [0] + [bf] * weeks
        demand_description = f"Flat {bf}/wk for {weeks} wks"

    elif "Linear ramp" in preset_shape:
        if "lr_end" not in st.session_state:
            st.session_state["lr_end"] = min(bf * 3, 1000)
        if "lr_wks" not in st.session_state:
            st.session_state["lr_wks"] = min(weeks // 3, 8)
        end_dem = st.slider("Target demand (pcs/wk)", min_value=bf, max_value=1000, step=10, key="lr_end")
        ramp_wks = st.slider("Ramp duration (weeks)", min_value=1, max_value=weeks, step=1, key="lr_wks")
        init_demand = [0]
        for w in range(1, weeks + 1):
            if w <= ramp_wks:
                val = bf + (end_dem - bf) * w / ramp_wks
            else:
                val = end_dem
            init_demand.append(max(0, int(round(val))))
        demand_description = f"Ramp {bf}\u2192{end_dem} in {ramp_wks}wk"

    elif "Linear drop" in preset_shape:
        if "ld_end" not in st.session_state:
            st.session_state["ld_end"] = max(0, bf // 3)
        if "ld_wks" not in st.session_state:
            st.session_state["ld_wks"] = 1
        drop_dem = st.slider("Floor demand (pcs/wk)", min_value=0, max_value=bf, step=10, key="ld_end")
        drop_wks = st.slider("Drop duration (weeks)", min_value=1, max_value=weeks, step=1, key="ld_wks")
        init_demand = [0]
        for w in range(1, weeks + 1):
            if w <= drop_wks:
                val = bf + (drop_dem - bf) * w / drop_wks
            else:
                val = drop_dem
            init_demand.append(max(0, int(round(val))))
        demand_description = f"Drop {bf}\u2192{drop_dem} in {drop_wks}wk"

    else:  # Poisson
        if "pk_wk" not in st.session_state:
            st.session_state["pk_wk"] = min(weeks // 3, 8)
        if "pk_h" not in st.session_state:
            st.session_state["pk_h"] = min(bf * 4, 1000)
        if "pk_sp" not in st.session_state:
            st.session_state["pk_sp"] = 4.0
        peak_wk = st.slider("Peak week", min_value=1, max_value=weeks, step=1, key="pk_wk")
        peak_h = st.slider("Peak demand (pcs/wk)", min_value=bf, max_value=1000, step=10, key="pk_h")
        spread = st.slider("Spread (weeks)", min_value=1.0, max_value=float(max(2, weeks // 2)), step=0.5, key="pk_sp")
        init_demand = [0]
        for w in range(1, weeks + 1):
            val = peak_h * np.exp(-0.5 * ((w - peak_wk) / spread) ** 2) + bf * 0.05
            init_demand.append(max(int(round(val)), 0))
        demand_description = f"Poisson peak {peak_h} at W{peak_wk}"

    st.caption(f"**{demand_description}**")

    # Editable table
    st.caption("\u270f\ufe0f Edit demand per week:")
    demand_df = pd.DataFrame({
        "Week": list(range(1, weeks + 1)),
        "Demand (pcs)": init_demand[1:weeks + 1],
    })
    edited = st.data_editor(
        demand_df,
        column_config={
            "Week": st.column_config.NumberColumn(disabled=True, width="small"),
            "Demand (pcs)": st.column_config.NumberColumn(min_value=0, max_value=9999, step=10, width="medium"),
        },
        hide_index=True, use_container_width=True, height=min(300, weeks * 35 + 40),
        key="demand_editor",
    )

    custom_demand = [0]
    for _, row in edited.iterrows():
        custom_demand.append(int(row["Demand (pcs)"]))

    total_dem = sum(custom_demand[1:])
    avg_dem = total_dem / max(weeks, 1)
    peak_val = max(custom_demand[1:]) if weeks > 0 else 0
    peak_wk_idx = custom_demand[1:].index(peak_val) + 1 if peak_val > 0 else 0

    # 6. CAPACITY
    st.markdown("### \U0001f3ed Capacity")
    cap_start = st.number_input("Starting Capacity (pcs/wk)", 10, 1000, 100)
    cap_ramp = st.slider("Ramp-up (% vs starting capacity, linear every week)", 0, 50, 20, 5) / 100

    # 7. ECONOMICS
    st.markdown("### \U0001f4b0 Economics")
    price = st.number_input("Selling Price (\u20ac)", 100, 10000, 1000, 100)
    var_cost = st.number_input("Variable Cost / Finished Product (\u20ac)", 10, 5000, 200, 10)
    st.markdown(
        f'<div style="color:#8a96a6;font-size:11px;font-style:italic;">'
        f'RM: \u20ac{var_cost * VALOR_RAW_MAT:.0f} (50%) | '
        f'Semi: \u20ac{var_cost * VALOR_SEMI:.0f} (75%) | '
        f'Finished: \u20ac{var_cost:.0f} (100%)</div>',
        unsafe_allow_html=True)
    fixed_pct = st.slider("Fixed Cost (% of sim period fcst rev)", 0, 100, 45) / 100

    # 8. QUICK SCENARIOS
    st.markdown("---")
    st.markdown("### \U0001f3af Quick Scenarios")
    st.caption("3 Lead Time \u00d7 3 Demand \u2014 click to load:")

    def apply_preset(lt_name, dem_name):
        lt = LT_PROFILES[lt_name]
        st.session_state["mat_lt"] = lt["mat_lt"]
        st.session_state["semi_lt"] = lt["semi_lt"]
        st.session_state["fp_lt"] = lt["fp_lt"]
        st.session_state["dist_lt"] = lt["dist_lt"]
        st.session_state["order_freq"] = lt["order_freq"]
        total_lt = lt["mat_lt"] + lt["semi_lt"] + lt["fp_lt"] + lt["dist_lt"]
        cov = total_lt + lt["order_freq"]
        rec = BASE_FORECAST * cov
        st.session_state["total_stock"] = min(rec, 10000)
        st.session_state["store_a_pct"] = 60
        st.session_state["smart_distrib"] = True
        if lt_name == "Push":
            st.session_state["store_pct"] = 100
            st.session_state["wh_pct"] = 0
            st.session_state["semi_pct"] = 0
        elif lt_name == "Medium":
            st.session_state["store_pct"] = 80
            st.session_state["wh_pct"] = 20
            st.session_state["semi_pct"] = 0
        else:
            st.session_state["store_pct"] = 60
            st.session_state["wh_pct"] = 20
            st.session_state["semi_pct"] = 10
        # Set demand shape selectbox + slider values directly
        if "Flat" in dem_name:
            st.session_state["demand_shape"] = DEMAND_SHAPES[0]  # Flat
        elif "Growth" in dem_name:
            st.session_state["demand_shape"] = DEMAND_SHAPES[1]  # Linear ramp
            st.session_state["lr_end"] = 300
            st.session_state["lr_wks"] = 5
        elif "Drop" in dem_name:
            st.session_state["demand_shape"] = DEMAND_SHAPES[2]  # Linear drop
            st.session_state["ld_end"] = 40
            st.session_state["ld_wks"] = 1

    h1, h2, h3, h4 = st.columns([1.2, 1, 1, 1])
    with h2: st.markdown("**Flat**")
    with h3: st.markdown("**Growth**")
    with h4: st.markdown("**Drop**")

    a1, a2, a3, a4 = st.columns([1.2, 1, 1, 1])
    with a1: st.markdown("\U0001f7e2 **Agile**\n\n*LT=8, f=1*")
    with a2: st.button("\u26a1", key="p_af", use_container_width=True, on_click=apply_preset, args=("Agile", "Flat 100"))
    with a3: st.button("\u26a1", key="p_ag", use_container_width=True, on_click=apply_preset, args=("Agile", "Growth \u2192300"))
    with a4: st.button("\u26a1", key="p_ad", use_container_width=True, on_click=apply_preset, args=("Agile", "Drop \u219240"))

    m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1])
    with m1: st.markdown("\U0001f7e1 **Medium**\n\n*LT=16, f=2*")
    with m2: st.button("\U0001f536", key="p_mf", use_container_width=True, on_click=apply_preset, args=("Medium", "Flat 100"))
    with m3: st.button("\U0001f536", key="p_mg", use_container_width=True, on_click=apply_preset, args=("Medium", "Growth \u2192300"))
    with m4: st.button("\U0001f536", key="p_md", use_container_width=True, on_click=apply_preset, args=("Medium", "Drop \u219240"))

    p1, p2, p3, p4 = st.columns([1.2, 1, 1, 1])
    with p1: st.markdown("\U0001f534 **Push**\n\n*LT=24, f=4*")
    with p2: st.button("\U0001f9f1", key="p_pf", use_container_width=True, on_click=apply_preset, args=("Push", "Flat 100"))
    with p3: st.button("\U0001f9f1", key="p_pg", use_container_width=True, on_click=apply_preset, args=("Push", "Growth \u2192300"))
    with p4: st.button("\U0001f9f1", key="p_pd", use_container_width=True, on_click=apply_preset, args=("Push", "Drop \u219240"))

    st.caption(
        "All: **A=60%, Smart ON**\n\n"
        "\U0001f7e2 Agile: 60/20/10/10 | "
        "\U0001f7e1 Medium: 80/20 | "
        "\U0001f534 Push: 100% store")


# ════════════════════════════════════════════════════════════════
# BUILD PARAMS & RUN
# ════════════════════════════════════════════════════════════════
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
    'custom_demand': tuple(custom_demand) if custom_demand is not None else None,
}

states = run_simulation(**params)
final_kpis = compute_kpis(states[1:], price, var_cost, fixed_pct, base_forecast, weeks,
                          init_store, init_cw, init_semi, init_rawmat)

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
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("# \U0001f3ed Supply Chain Agility Simulator")
distrib_mode = "Smart" if smart_distrib else "Push 50/50"
st.markdown(f"*LT = **{phys_lt}**wk | Coverage = **{coverage}**wk | Demand: **{demand_description}** | A: **{store_a_pct}%** / B: **{100-store_a_pct}%** | {distrib_mode}*")

# ════════════════════════════════════════════════════════════════
# WEEK NAVIGATION
# ════════════════════════════════════════════════════════════════
if "week_num" not in st.session_state:
    st.session_state.week_num = 0
if st.session_state.week_num > weeks:
    st.session_state.week_num = weeks
if st.session_state.week_num < 0:
    st.session_state.week_num = 0

def _nav_w0(): st.session_state.week_num = 0
def _nav_minus(): st.session_state.week_num = max(0, st.session_state.week_num - 1)
def _nav_plus(): st.session_state.week_num = min(weeks, st.session_state.week_num + 1)
def _nav_end(): st.session_state.week_num = weeks

b1, b2, b3, b4, info = st.columns([1, 1, 1, 1, 2])
with b1: st.button("\u23ee W0", use_container_width=True, disabled=st.session_state.week_num == 0, on_click=_nav_w0)
with b2: st.button("\u25c0 \u22121", use_container_width=True, disabled=st.session_state.week_num <= 0, on_click=_nav_minus)
with b3: st.button("+1 \u25b6", use_container_width=True, disabled=st.session_state.week_num >= weeks, on_click=_nav_plus)
with b4: st.button(f"W{weeks} \u23ed", use_container_width=True, disabled=st.session_state.week_num >= weeks, on_click=_nav_end)
with info:
    pct = st.session_state.week_num / max(weeks, 1)
    bar_w = int(pct * 100)
    st.markdown(
        f"<div style='padding:8px 0;'>"
        f"<div style='font-size:20px;font-weight:800;color:#1a2a40;text-align:center;'>Week {st.session_state.week_num} <span style='font-size:13px;color:#7a8a9e;'>/ {weeks}</span></div>"
        f"<div style='background:#e0e4ea;border-radius:4px;height:6px;margin-top:4px;'>"
        f"<div style='background:#4a90d9;height:6px;border-radius:4px;width:{bar_w}%;'></div></div></div>",
        unsafe_allow_html=True)

week = st.session_state.week_num
state = states[week]
cum = cumulative_kpis(states[1:], week, price, var_cost, fixed_pct, base_forecast, weeks,
                      init_store, init_cw, init_semi, init_rawmat)

# ════════════════════════════════════════════════════════════════
# KPI CARDS (point 11: no margin here, just operational KPIs)
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
    sc = "#c0392b" if cum['stockout_wks'] > 0 else "#1a8a4a"
    st.markdown(kpi_card("Stockout Wks", f"{cum['stockout_wks']}/{week}", sc), unsafe_allow_html=True)
with k7:
    uf = cum['useful_pct']
    uc = "#1a8a4a" if uf > 80 else ("#d4850a" if uf > 50 else "#c0392b")
    st.markdown(kpi_card("Useful Prod.", f"{uf:.0f}%", uc), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SC FLOW VISUALIZATION
# ════════════════════════════════════════════════════════════════
st.markdown("")
st.components.v1.html(make_sc_html(state, params), height=440, scrolling=False)

# ════════════════════════════════════════════════════════════════
# DEMAND CHART (point 7: hidden by default in expander)
# ════════════════════════════════════════════════════════════════
import altair as alt

with st.expander("\U0001f4c8 Charts: Demand, Fulfillment, Stocks", expanded=False):
    dem_chart_data = pd.DataFrame({
        "Week": list(range(1, weeks + 1)),
        "Demand": [states[i]["demand"] for i in range(1, weeks + 1)],
        "Sales": [states[i]["sales"] for i in range(1, weeks + 1)],
        "Missed": [states[i]["missed"] for i in range(1, weeks + 1)],
    })
    y_max = max(dem_chart_data["Demand"].max(), 1) * 1.15
    y_scale = alt.Scale(domain=[0, y_max])

    bar_data = dem_chart_data.melt("Week", ["Sales", "Missed"], var_name="Type", value_name="Units")
    stacked_bars = alt.Chart(bar_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Week:O", title="Week"),
        y=alt.Y("Units:Q", title="Units/week", scale=y_scale, stack=True),
        color=alt.Color("Type:N",
            scale=alt.Scale(domain=["Sales", "Missed"], range=["#1a8a4a", "#c0392b"]),
            legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("Type:N", sort="descending"),
    )
    demand_line = alt.Chart(dem_chart_data).mark_line(color="#4a90d9", strokeWidth=3, strokeDash=[6,3]).encode(
        x=alt.X("Week:O"), y=alt.Y("Demand:Q", scale=y_scale))
    demand_dots = alt.Chart(dem_chart_data).mark_circle(color="#4a90d9", size=40).encode(
        x="Week:O", y=alt.Y("Demand:Q", scale=y_scale))
    rule_dc = alt.Chart(pd.DataFrame({"Week": [week]})).mark_rule(
        color="#d4850a", strokeWidth=2, strokeDash=[4,2]).encode(x="Week:O")

    st.markdown("#### Demand vs Sales vs Missed")
    st.altair_chart((stacked_bars + demand_line + demand_dots + rule_dc).properties(height=300), use_container_width=True)

    # Store-level charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### Demand vs Fulfillment (per store)")
        rows = []
        for s in states[1:]:
            rows.append({'Week': s['week'], 'Group': 'Fill A', 'Component': 'Sales A', 'Value': s['sales_a']})
            rows.append({'Week': s['week'], 'Group': 'Fill A', 'Component': 'Lost A', 'Value': s['missed_a']})
            rows.append({'Week': s['week'], 'Group': 'Fill B', 'Component': 'Sales B', 'Value': s['sales_b']})
            rows.append({'Week': s['week'], 'Group': 'Fill B', 'Component': 'Lost B', 'Value': s['missed_b']})
        df_bars = pd.DataFrame(rows)
        bars = alt.Chart(df_bars).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X('Week:O'), y=alt.Y('Value:Q', title='Units', stack=True),
            color=alt.Color('Component:N',
                scale=alt.Scale(domain=['Sales A','Lost A','Sales B','Lost B'],
                                range=['#2c5f8a','#c0392b','#6a3d9a','#e74c8c']),
                legend=alt.Legend(orient='top', title=None, columns=2)),
            xOffset='Group:N',
        ).properties(height=260)
        rule = alt.Chart(pd.DataFrame({'Week': [week]})).mark_rule(color='#d4850a', strokeWidth=2, strokeDash=[4,2]).encode(x='Week:O')
        st.altair_chart(bars + rule, use_container_width=True)

    with ch2:
        st.markdown("#### Store Stocks & Orders")
        stock_data = pd.DataFrame({
            'Week': [s['week'] for s in states], 'Store A': [s['store_a'] for s in states],
            'Store B': [s['store_b'] for s in states], 'Order': [s['order'] for s in states],
        })
        melted = stock_data.melt('Week', ['Store A','Store B'], var_name='Store', value_name='Stock')
        lines = alt.Chart(melted).mark_area(opacity=0.25).encode(
            x=alt.X('Week:O'), y=alt.Y('Stock:Q', title='Units', stack=False),
            color=alt.Color('Store:N', scale=alt.Scale(domain=['Store A','Store B'], range=['#2c5f8a','#6a3d9a']),
                legend=alt.Legend(orient='top', title=None)),
        ).properties(height=260)
        order_bars = alt.Chart(stock_data[stock_data['Order'] > 0]).mark_bar(
            color='#1a8a4a', opacity=0.4, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(x='Week:O', y='Order:Q')
        rule2 = alt.Chart(pd.DataFrame({'Week': [week]})).mark_rule(color='#d4850a', strokeWidth=2, strokeDash=[4,2]).encode(x='Week:O')
        st.altair_chart(lines + order_bars + rule2, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# P&L SUMMARY (point 12: at end, like a proper P&L)
# ════════════════════════════════════════════════════════════════
with st.expander("\U0001f4cb P&L Summary (end of simulation)", expanded=False):
    fk = final_kpis
    st.markdown("#### Profit & Loss Statement")
    pl_data = {
        "Line": [
            "\U0001f4b0 Revenue",
            "",
            "\u2796 Initial Stock (pre-invested)",
            "\u2796 Purchasing (RM @50%)",
            "\u2796 Semi Processing (+25%)",
            "\u2796 Finishing (+25%)",
            "= Total Variable Cost",
            "",
            "= Gross Margin",
            "\u2796 Fixed Costs",
            "",
            "= **Net Margin**",
        ],
        "Amount (\u20ac)": [
            f"{fk['revenue']:,.0f}",
            "",
            f"-{fk['init_stock_value']:,.0f}",
            f"-{fk['cost_mat_total']:,.0f}",
            f"-{fk['cost_semi_total']:,.0f}",
            f"-{fk['cost_fp_total']:,.0f}",
            f"-{fk['var_cost']:,.0f}",
            "",
            f"{fk['gm']:,.0f}",
            f"-{fk['fixed']:,.0f}",
            "",
            f"{fk['margin']:,.0f}",
        ],
        "Detail": [
            f"{fk['total_sales']:,.0f} pcs x \u20ac{price}",
            "",
            f"Store/WH @100% + Semi @75% + RM @50%",
            f"{sum(s.get('supplier_shipped',0) for s in states[1:]):.0f} pcs x \u20ac{var_cost*VALOR_RAW_MAT:.0f}",
            f"Incremental \u20ac{var_cost*(VALOR_SEMI-VALOR_RAW_MAT):.0f}/pc",
            f"Incremental \u20ac{var_cost*(VALOR_FINISHED-VALOR_SEMI):.0f}/pc",
            f"Init stock + production costs",
            "",
            f"Revenue - Variable Costs",
            f"{fixed_pct*100:.0f}% of simulation forecast revenue",
            "",
            f"{fk['margin_pct']*100:.1f}% of revenue",
        ],
    }
    st.table(pd.DataFrame(pl_data).set_index("Line"))

    st.markdown("---")
    st.markdown("#### Production Efficiency")
    u1, u2, u3 = st.columns(3)
    with u1:
        st.metric("Service Level", f"{fk['svc_level']*100:.1f}%")
    with u2:
        st.metric("\u2705 Sold (Useful)", f"{fk['useful_units']:,.0f} pcs ({fk['useful_pct']:.0f}%)")
    with u3:
        st.metric("\u274c Unsold", f"{fk['useless_units']:,.0f} pcs ({fk['useless_pct']:.0f}%)")

# ════════════════════════════════════════════════════════════════
# WEEK-BY-WEEK TABLE (points 5, 6: add RM, Semi in W0; add revenue/cost/margin cols)
# ════════════════════════════════════════════════════════════════
with st.expander("\U0001f4ca Detailed Week-by-Week Data", expanded=False):
    table_data = []
    for s in states:
        wk_rev = s['sales'] * price
        wk_vc = s.get('cost_mat', 0) + s.get('cost_semi', 0) + s.get('cost_fp', 0)
        wk_margin = wk_rev - wk_vc
        in_mat_pipe = sum(s.get('mat_pipe', []))
        in_semi_pipe = sum(s.get('semi_pipe', []))
        in_fp_pipe = sum(s.get('fp_pipe', []))
        table_data.append({
            'Week': s['week'],
            'Demand': s['demand'], 'Dem A': s['demand_a'], 'Dem B': s['demand_b'],
            'Sales': s['sales'], 'Sales A': s['sales_a'], 'Sales B': s['sales_b'],
            'Missed': s['missed'], 'Miss A': s['missed_a'], 'Miss B': s['missed_b'],
            # Stores
            'Stk A': s['store_a'], 'Stk B': s['store_b'],
            'Alloc A': s['alloc_a'], 'Alloc B': s['alloc_b'],
            # CW stage
            'CW': s.get('cw_stock', 0), 'CW Ship': s.get('cw_shipped', 0),
            # FP pipe
            'FP Pipe': round(in_fp_pipe, 1),
            # Semi stage (wait = buffer stock, proc = transformation AT semi stage)
            'Semi Wait': s.get('semi_stock', 0), 'Semi Pipe': round(in_semi_pipe, 1), 'Semi Proc': s.get('fp_input', 0),
            # RM stage (wait = buffer stock, proc = transformation AT RM stage)
            'RM Wait': s.get('raw_mat_stock', 0), 'RM Proc': s.get('semi_input', 0), 'Mat Pipe': round(in_mat_pipe, 1),
            # Totals
            'WIP': s.get('wip_total', 0),
            'Order': s['order'], 'Pending': s['pending'],
            'Sup Cap': s.get('supplier_cap', 0),
            # Financials
            'Revenue': round(wk_rev),
            'Cost RM': round(s.get('cost_mat', 0)),
            'Cost Semi': round(s.get('cost_semi', 0)),
            'Cost FP': round(s.get('cost_fp', 0)),
            'Tot VC': round(wk_vc),
            'Margin': round(wk_margin),
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=500)
    st.caption("**RM Proc** = units transformed at RM stage (RM\u2192Semi). "
               "**Semi Proc** = units transformed at Semi stage (Semi\u2192FP). "
               "**CW Ship** = units shipped to stores. "
               "**Costs**: RM @50% anticipated 1wk, Semi +25% at RM Proc, FP +25% at Semi Proc.")

# ════════════════════════════════════════════════════════════════
# SAVE SCENARIO (point 4: include demand description)
# ════════════════════════════════════════════════════════════════
with st.expander("\U0001f4be Save Scenario for Comparison", expanded=False):
    now_str = datetime.now().strftime("%H:%M:%S")
    default_name = f"LT{phys_lt}_f{order_freq}_{demand_description}_{now_str}"
    scenario_name = st.text_input("Scenario Name", default_name)
    if st.button("Save Current Scenario"):
        if 'saved_scenarios' not in st.session_state:
            st.session_state.saved_scenarios = {}
        st.session_state.saved_scenarios[scenario_name] = {
            'params': params.copy(), 'kpis': final_kpis.copy(),
            'demand_desc': demand_description}
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
                'Demand': d.get('demand_desc', ''),
                'Svc%': f"{k['svc_level']*100:.1f}%",
                'Sales': f"{round(k['total_sales'], -1):,.0f}",
                'Missed': f"{round(k['total_missed'], -1):,.0f}",
                'Revenue': f"\u20ac{k['revenue']:,.0f}",
                'Margin': f"\u20ac{k['margin']:,.0f}",
                '\u0394 vs Base': delta_str,
                'Useful%': f"{k['useful_pct']:.0f}%",
                'Stock': p['init_store'] + p['init_cw'] + p['init_semi'] + p['init_rawmat'],
                'Freq': f"{p['order_freq']}wk",
                'Tot LT': p['mat_lt'] + p['semi_lt'] + p['fp_lt'] + p['dist_lt'],
            })
        st.dataframe(pd.DataFrame(comp), use_container_width=True)
        if st.button("Clear All"):
            st.session_state.saved_scenarios = {}
            st.rerun()
