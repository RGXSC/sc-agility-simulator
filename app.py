import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import itertools
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestRegressor

from sim_engine import sim_fast, worker_chunk

st.set_page_config(layout="wide", page_title="SC Batch Simulator v4", page_icon="📊")

# ════════════════════════════════════════════════════════════════
# STOCK DISTRIBUTION GENERATOR
# ════════════════════════════════════════════════════════════════
def generate_stock_distribs(store_vals, wh_vals, semi_vals):
    """Generate all valid (store%, wh%, semi%, rm%) combos that sum to 100."""
    combos = set()
    for sp in store_vals:
        for wp in wh_vals:
            for sep in semi_vals:
                rp = 100 - sp - wp - sep
                if rp >= 0:
                    combos.add((sp, wp, sep, rp))
    return sorted(combos, key=lambda x: (-x[0], -x[1], -x[2]))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Human-readable column mapping for output
PHYS_COLS = [
    'demand_level', 'sim_weeks', 'planning_freq_wks', 'capacity_ramp_pct',
    'smart_allocation',
    'lt_raw_material', 'lt_semifinished', 'lt_finished_product', 'lt_distribution',
    'lt_total_weeks',
    'stock_index', 'pct_in_stores', 'pct_in_warehouse', 'pct_in_semifinished',
    'pct_in_raw_material', 'initial_stock_total',
    'units_sold', 'units_missed', 'units_demanded', 'units_produced',
    'weeks_with_stockout', 'units_missed_store_a', 'units_missed_store_b',
    'demand_split_store_a_pct'
]

# Feature names for ML (must match column names)
ML_FEATURES = [
    'demand_level', 'sim_weeks', 'planning_freq_wks', 'capacity_ramp_pct',
    'smart_allocation',
    'lt_raw_material', 'lt_semifinished', 'lt_finished_product', 'lt_distribution',
    'lt_total_weeks',
    'initial_stock_total', 'pct_in_stores', 'pct_in_warehouse', 'pct_in_semifinished',
    'pct_in_raw_material', 'fixed_cost_pct', 'demand_split_store_a_pct'
]

# Pretty labels for display
PRETTY_LABELS = {
    'demand_level': 'Demand Level (×forecast)',
    'sim_weeks': 'Simulation Period (weeks)',
    'planning_freq_wks': 'Planning Frequency (weeks)',
    'capacity_ramp_pct': 'Capacity Ramp (%/week)',
    'smart_allocation': 'Smart Allocation (0/1)',
    'lt_raw_material': 'LT Raw Material (wks)',
    'lt_semifinished': 'LT Semi-Finished (wks)',
    'lt_finished_product': 'LT Finished Product (wks)',
    'lt_distribution': 'LT Distribution (wks)',
    'lt_total_weeks': 'LT Total (wks)',
    'initial_stock_total': 'Initial Stock (units)',
    'pct_in_stores': '% Stock in Stores',
    'pct_in_warehouse': '% Stock in Warehouse',
    'pct_in_semifinished': '% Stock in Semi-Finished',
    'pct_in_raw_material': '% Stock in Raw Material',
    'fixed_cost_pct': 'Fixed Cost (%)',
    'demand_split_store_a_pct': 'Demand Split Store A (%)',
}


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def compute_lognormal_weights(demand_mults, sigma=0.5, center=1.0):
    """Log-normal weights with adjustable mode (center).
    mode = exp(mu - sigma^2) = center → mu = ln(center) + sigma^2"""
    mu = np.log(max(center, 0.01)) + sigma ** 2
    pdf_vals = stats.lognorm.pdf(demand_mults, s=sigma, scale=np.exp(mu))
    total = pdf_vals.sum()
    return (pdf_vals / total).tolist() if total > 0 else [1/len(demand_mults)]*len(demand_mults)


def weighted_mean(df, col, wc='demand_weight'):
    w = df[wc]; s = w.sum()
    return (df[col]*w).sum()/s if s > 0 else df[col].mean()


def generate_jobs(params):
    jobs = []
    for dm in params['demand_mults']:
        for wks in params['sim_weeks']:
            for pf in params['plan_freqs']:
                for cr in params['cap_ramps']:
                    for sm_d in params['smart_opts']:
                        for lt in params['lt_combos']:
                            m, s, f, d = lt
                            for stk in params['stock_levels']:
                                for sd in params['stock_distribs']:
                                    sp_, wp_, sep_, rp_ = sd
                                    for dpa in params['demand_splits']:
                                        jobs.append((
                                            dm, wks, pf, cr, sm_d,
                                            m, s, f, d, sum(lt),
                                            stk/1000.0, sp_, wp_, sep_, rp_, stk,
                                            params['cap_start'], params['base_forecast'],
                                            dpa
                                        ))
    return jobs


# ════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ════════════════════════════════════════════════════════════════
def run_batch(params, progress_bar, status_text):
    all_jobs = generate_jobs(params)
    total = len(all_jobs)
    status_text.text(f"⏳ {total:,.0f} jobs...")

    nc = mp.cpu_count(); nw = max(1, nc-1)
    cs = max(1, total // (nw*4))
    job_chunks = [all_jobs[i:i+cs] for i in range(0, total, cs)]
    progress_bar.progress(0.02); t0 = time.time()
    all_results = []; use_mp = True

    try:
        ctx = mp.get_context('fork') if sys.platform != 'win32' else mp.get_context('spawn')
        with ctx.Pool(processes=nw) as pool:
            for cr in pool.imap_unordered(worker_chunk, job_chunks):
                all_results.extend(cr)
                done = len(all_results); pct = done/total; el = time.time()-t0
                eta = (el/pct*(1-pct)) if pct > 0.05 else 0
                progress_bar.progress(min(pct,0.99))
                status_text.text(f"⏳ {done:,.0f}/{total:,.0f} ({pct:.1%}) — {el:.0f}s — ETA {eta:.0f}s — {nw}w — {done/max(el,0.1):,.0f}/s")
    except Exception as e:
        use_mp = False; all_results = []
        for ch in job_chunks:
            all_results.extend(worker_chunk(ch))
            done = len(all_results); pct = done/total; el = time.time()-t0
            eta = (el/pct*(1-pct)) if pct > 0.05 else 0
            progress_bar.progress(min(pct,0.99))
            status_text.text(f"⏳ {done:,.0f}/{total:,.0f} ({pct:.1%}) — {el:.0f}s — ETA {eta:.0f}s — 1t")

    progress_bar.progress(1.0); el = time.time()-t0
    status_text.text(f"✅ {len(all_results):,.0f} sims — {el:.0f}s ({el/60:.1f}min) — {len(all_results)/max(el,0.1):,.0f}/s")

    df = pd.DataFrame(np.array(all_results, dtype=np.float32), columns=PHYS_COLS)
    for c in ['sim_weeks','planning_freq_wks','smart_allocation',
              'lt_raw_material','lt_semifinished','lt_finished_product','lt_distribution',
              'lt_total_weeks','pct_in_stores','pct_in_warehouse','pct_in_semifinished',
              'pct_in_raw_material','weeks_with_stockout','initial_stock_total',
              'demand_split_store_a_pct']:
        df[c] = df[c].astype(int)
    return df


def expand_financials(df_phys, price, var_cost, fixed_pcts, base_forecast, max_rows=3_000_000):
    n = len(fixed_pcts)
    df_s = df_phys.sample(max_rows//n, random_state=42) if len(df_phys)*n > max_rows else df_phys
    rows = []
    for fp in fixed_pcts:
        sub = df_s.copy()
        sub['selling_price'] = price
        sub['variable_cost_per_unit'] = var_cost
        sub['fixed_cost_pct'] = fp
        sub['total_revenue'] = sub['units_sold'] * price
        sub['total_variable_cost'] = (sub['units_produced'] + sub['initial_stock_total']) * var_cost
        sub['initial_stock_cost'] = sub['initial_stock_total'] * var_cost
        sub['production_cost'] = sub['units_produced'] * var_cost
        sub['gross_profit'] = sub['total_revenue'] - sub['total_variable_cost']
        sub['total_fixed_cost'] = base_forecast * 52 * price * fp * (sub['sim_weeks'] / 52)
        sub['net_profit'] = sub['gross_profit'] - sub['total_fixed_cost']
        sub['net_margin_pct'] = sub['net_profit'] / sub['total_revenue'].replace(0, np.nan)
        sub['service_level'] = sub['units_sold'] / sub['units_demanded'].replace(0, np.nan)
        sub['lost_sales_revenue'] = sub['units_missed'] * price
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def add_demand_weights(df, demand_mults, weights):
    df['demand_weight'] = df['demand_level'].map(dict(zip(demand_mults, weights))).fillna(0)
    return df


# ════════════════════════════════════════════════════════════════
# STRATEGY LABELS
# ════════════════════════════════════════════════════════════════
def label_strategies(df):
    """Add clear strategy labels for the 2×2 matrix."""
    median_lt = df['lt_total_weeks'].median()

    # Stock positioning: Push to Store (≥80% in stores) vs Spread along SC (<80%)
    df['stock_strategy'] = np.where(df['pct_in_stores'] >= 80,
                                     '🏪 Push to Store (≥80% in stores)',
                                     '🔗 Spread along SC (<80% in stores)')

    # Speed: Fast (<median LT) vs Slow (≥median LT)
    df['speed_strategy'] = np.where(df['lt_total_weeks'] < median_lt,
                                     '⚡ Fast SC (LT<' + str(int(median_lt)) + 'wk)',
                                     '🐢 Slow SC (LT≥' + str(int(median_lt)) + 'wk)')

    # Combined 2×2 label
    df['strategy_quadrant'] = df['stock_strategy'] + ' × ' + df['speed_strategy']
    return df, median_lt


# ════════════════════════════════════════════════════════════════
# ML ANALYSIS
# ════════════════════════════════════════════════════════════════
def run_analysis(df_full):
    dc = df_full.dropna(subset=['net_margin_pct']).copy()
    dc['smart_allocation'] = dc['smart_allocation'].astype(int)
    X = dc[ML_FEATURES].values; y = dc['net_margin_pct'].values
    w = dc['demand_weight'].values if 'demand_weight' in dc.columns else None

    if len(X) > 2_000_000:
        idx = np.random.RandomState(42).choice(len(X), 2_000_000, replace=False)
        X, y = X[idx], y[idx]
        if w is not None: w = w[idx]

    # Tree with pretty feature names
    pretty_names = [PRETTY_LABELS.get(f, f) for f in ML_FEATURES]

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=500, random_state=42)
    tree.fit(X, y, sample_weight=w)

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=200,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y, sample_weight=w)

    corr = {f: np.corrcoef(X[:, i], y)[0, 1] for i, f in enumerate(ML_FEATURES)}

    # Visual tree figure
    fig, ax = plt.subplots(figsize=(28, 12), dpi=100)
    plot_tree(tree, feature_names=pretty_names, filled=True, rounded=True,
              ax=ax, fontsize=8, impurity=False, proportion=True,
              label='root', precision=2)
    ax.set_title("Regression Tree — Net Margin % Drivers", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return {
        'tree_r2': tree.score(X, y),
        'tree_text': export_text(tree, feature_names=pretty_names, max_depth=4),
        'tree_imp': dict(zip(ML_FEATURES, tree.feature_importances_)),
        'tree_fig': fig,
        'rf_r2': rf.score(X, y),
        'rf_imp': dict(zip(ML_FEATURES, rf.feature_importances_)),
        'features': ML_FEATURES, 'n_samples': len(X), 'correlations': corr
    }


# ════════════════════════════════════════════════════════════════
# STRATEGY 2×2 MATRIX
# ════════════════════════════════════════════════════════════════
def strategy_matrix(df):
    """Compute the 2×2 strategy comparison with clear metrics."""
    has_w = 'demand_weight' in df.columns
    results = {}

    for quad in df['strategy_quadrant'].unique():
        g = df[df['strategy_quadrant'] == quad]
        if len(g) == 0: continue
        r = {
            'Scenarios': len(g),
            'Weighted Avg Net Margin %': weighted_mean(g, 'net_margin_pct') if has_w else g['net_margin_pct'].mean(),
            'Median Net Margin %': g['net_margin_pct'].median(),
            'Weighted Avg Service Level': weighted_mean(g, 'service_level') if has_w else g['service_level'].mean(),
            '% Scenarios Profitable': (g['net_margin_pct'] > 0).mean(),
            'Weighted Avg Net Profit €': weighted_mean(g, 'net_profit') if has_w else g['net_profit'].mean(),
            'Std Margin %': g['net_margin_pct'].std(),
            'Avg Initial Stock': g['initial_stock_total'].mean(),
            'Avg Total LT (wks)': g['lt_total_weeks'].mean(),
            'Avg % in Stores': g['pct_in_stores'].mean(),
        }
        results[quad] = r

    return pd.DataFrame(results).T


def paired_comparison(df):
    """Paired comparison: Push to Store+Slow vs Spread along SC+Fast at equal stock."""
    has_w = 'demand_weight' in df.columns
    cent_slow = df[(df['pct_in_stores'] >= 80) & (df['lt_total_weeks'] >= df['lt_total_weeks'].median())]
    dist_fast = df[(df['pct_in_stores'] < 80) & (df['lt_total_weeks'] < df['lt_total_weeks'].median())]

    mk = ['demand_level', 'sim_weeks', 'initial_stock_total', 'fixed_cost_pct']
    cs_a = cent_slow.groupby(mk)['net_margin_pct'].mean().reset_index()
    cs_a.columns = mk + ['margin_push_slow']
    df_a = dist_fast.groupby(mk)['net_margin_pct'].mean().reset_index()
    df_a.columns = mk + ['margin_spread_fast']

    p = cs_a.merge(df_a, on=mk, how='inner')
    if len(p) == 0:
        return {'pairs': 0, 'spread_wins': 0, 'push_wins': 0,
                'pct_spread': 0, 'avg_delta': 0}

    if has_w:
        wmap = dict(zip(df['demand_level'].unique(),
                       [df[df['demand_level']==dm]['demand_weight'].iloc[0] for dm in df['demand_level'].unique()]))
        p['w'] = p['demand_level'].map(wmap).fillna(1)
    else:
        p['w'] = 1.0

    p['sw_flag'] = (p['margin_spread_fast'] > p['margin_push_slow']).astype(float)
    ws = p['w'].sum()
    return {
        'pairs': len(p),
        'spread_wins': (p['sw_flag'] * p['w']).sum() / ws,
        'push_wins': 1 - (p['sw_flag'] * p['w']).sum() / ws,
        'pct_spread': (p['sw_flag'] * p['w']).sum() / ws,
        'avg_delta': ((p['margin_spread_fast'] - p['margin_push_slow']) * p['w']).sum() / ws,
    }


def smart_comparison(df):
    ag = df[df['pct_in_stores'] < 80]
    has_w = 'demand_weight' in ag.columns
    on = ag[ag['smart_allocation']==1]; off = ag[ag['smart_allocation']==0]
    return {
        'smart_margin': weighted_mean(on, 'net_margin_pct') if has_w and len(on)>0 else on['net_margin_pct'].mean() if len(on)>0 else 0,
        'push_margin': weighted_mean(off, 'net_margin_pct') if has_w and len(off)>0 else off['net_margin_pct'].mean() if len(off)>0 else 0,
        'smart_svc': weighted_mean(on, 'service_level') if has_w and len(on)>0 else on['service_level'].mean() if len(on)>0 else 0,
        'push_svc': weighted_mean(off, 'service_level') if has_w and len(off)>0 else off['service_level'].mean() if len(off)>0 else 0,
    }


# ════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════════════════════
st.markdown("""<style>
.stApp{background:#f4f6f9}h1{color:#1a2a40}
h2{color:#2a3a50;border-bottom:2px solid #e0e4ea;padding-bottom:6px}
</style>""", unsafe_allow_html=True)
st.title("📊 SC Batch Simulator v4")
st.markdown("*Weighted demand · Absolute stock · Store A/B split · Visual tree · Clear strategy matrix*")

with st.sidebar:
    st.header("🎛️ Parameters")

    st.subheader("💰 Pricing")
    price = st.number_input("Selling price €", value=1000, step=100)
    var_cost = st.number_input("Variable cost €/unit", value=200, step=50)

    st.subheader("📈 Demand Profile")
    dm_min = st.number_input("Min demand ×", value=0.30, step=0.05, format="%.2f")
    dm_max = st.number_input("Max demand ×", value=4.00, step=0.10, format="%.2f")
    dm_step = st.number_input("Step", value=0.10, step=0.05, format="%.2f", key="dms")
    demand_mults = [round(dm_min + i*dm_step, 2) for i in range(int((dm_max-dm_min)/dm_step)+1)]
    demand_mults = [d for d in demand_mults if d <= dm_max+0.001]

    sigma = st.slider("σ (uncertainty spread)", 0.1, 1.5, 0.5, 0.05)
    center = st.slider("📍 Curve center (most likely demand)", 0.30, 1.50, 0.70, 0.05,
                        help="0.70 = products most often sell at 70% of forecast. 1.0 = optimistic (on forecast)")
    default_weights = compute_lognormal_weights(demand_mults, sigma, center)

    expected_dem = sum(d*w for d,w in zip(demand_mults, default_weights))
    st.caption(f"Peak ≈ **{center:.0%}** of forecast · Expected demand: **{expected_dem:.2f}×**")

    with st.expander("🎚️ Edit demand weights"):
        st.caption("Adjust %. Auto-renormalized.")
        ew = {}
        for dm, dw in zip(demand_mults, default_weights):
            ew[dm] = st.number_input(f"{dm:.2f}×", value=round(dw*100,1), step=0.5,
                                     format="%.1f", key=f"w_{dm}_s{sigma:.2f}_c{center:.2f}")
        raw = [ew[dm] for dm in demand_mults]; tot = sum(raw)
        weights = [w/tot for w in raw] if tot > 0 else default_weights
    w_df = pd.DataFrame({'Demand ×': demand_mults, 'Prob %': [w*100 for w in weights]})
    st.bar_chart(w_df, x='Demand ×', y='Prob %', height=130, color='#4a90d9')

    st.subheader("🏪 Demand Split A/B")
    st.caption("Test different demand concentrations between stores")
    split_options = st.multiselect("Store A demand %",
        [50, 60, 70, 80], default=[50, 70],
        help="50=balanced, 70=Store A gets 70% of demand")
    demand_splits = split_options if split_options else [50]

    st.subheader("📅 Simulation length")
    sim_weeks = []
    if st.checkbox("13 wks", value=True): sim_weeks.append(13)
    if st.checkbox("26 wks", value=True): sim_weeks.append(26)
    if st.checkbox("39 wks", value=False): sim_weeks.append(39)
    if st.checkbox("52 wks", value=True): sim_weeks.append(52)
    if not sim_weeks: sim_weeks = [52]

    st.subheader("⏱️ Lead Times per Stage")
    lt_configs = {}
    for stage, abbr, dmin, dmax, dstp in [
        ("Raw Material", "rm", 1, 11, 2),
        ("Semi-Finished", "sf", 1, 11, 2),
        ("Finished Product", "fp", 1, 7, 2),
        ("Distribution", "di", 1, 5, 2),
    ]:
        with st.expander(f"🔧 {stage}"):
            c1,c2,c3 = st.columns(3)
            lmin = c1.number_input("Min", value=dmin, min_value=1, step=1, key=f"lt_{abbr}_min")
            lmax = c2.number_input("Max", value=dmax, step=1, key=f"lt_{abbr}_max")
            lstp = c3.number_input("Step", value=dstp, min_value=1, step=1, key=f"lt_{abbr}_stp")
            lt_configs[stage] = list(range(lmin, lmax+1, lstp))
            st.caption(f"→ {lt_configs[stage]}")

    lt_combos = list(itertools.product(
        lt_configs["Raw Material"], lt_configs["Semi-Finished"],
        lt_configs["Finished Product"], lt_configs["Distribution"]))
    if len(lt_combos) > 200:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(lt_combos), 200, replace=False)
        lt_combos = [lt_combos[i] for i in sorted(idx)]
        st.warning(f"Sampled 200 / {len(list(itertools.product(*lt_configs.values())))} LT combos")
    st.caption(f"**{len(lt_combos)}** LT combinations")

    st.subheader("📦 Initial Stock (absolute)")
    base_forecast = st.number_input("Base forecast/wk", value=100, step=25)
    stk_min = st.number_input("Min stock", value=500, step=100)
    stk_max = st.number_input("Max stock", value=2500, step=100)
    stk_stp = st.number_input("Step", value=500, step=100, key="stk_s")
    stock_levels = list(range(stk_min, stk_max+1, stk_stp))
    bf = max(base_forecast, 1)
    stk_info = ", ".join(f"{s} ({s/bf:.0f}wks)" for s in stock_levels)
    st.caption(f"Levels: {stk_info}")

    st.subheader("📦 Stock Distribution along SC")
    st.caption("How initial stock is split: Stores / Warehouse / Semi-Finished / Raw Material (auto = 100% − rest)")

    sd_c1, sd_c2, sd_c3 = st.columns(3)
    with sd_c1:
        st.markdown("**🏪 % in Stores**")
        sd_store_min = st.number_input("Min %", value=30, step=10, key="sd_st_min")
        sd_store_max = st.number_input("Max %", value=100, step=10, key="sd_st_max")
        sd_store_stp = st.number_input("Step %", value=10, step=5, key="sd_st_stp")
    with sd_c2:
        st.markdown("**🏭 % in Warehouse**")
        sd_wh_min = st.number_input("Min %", value=0, step=10, key="sd_wh_min")
        sd_wh_max = st.number_input("Max %", value=30, step=10, key="sd_wh_max")
        sd_wh_stp = st.number_input("Step %", value=15, step=5, key="sd_wh_stp")
    with sd_c3:
        st.markdown("**🔧 % in Semi-Finished**")
        sd_se_min = st.number_input("Min %", value=0, step=10, key="sd_se_min")
        sd_se_max = st.number_input("Max %", value=30, step=10, key="sd_se_max")
        sd_se_stp = st.number_input("Step %", value=15, step=5, key="sd_se_stp")

    store_vals = list(range(sd_store_min, sd_store_max+1, max(1,sd_store_stp)))
    wh_vals = list(range(sd_wh_min, sd_wh_max+1, max(1,sd_wh_stp)))
    semi_vals = list(range(sd_se_min, sd_se_max+1, max(1,sd_se_stp)))
    stock_distribs = generate_stock_distribs(store_vals, wh_vals, semi_vals)

    with st.expander(f"📋 {len(stock_distribs)} distribution combos", expanded=False):
        sd_df = pd.DataFrame(stock_distribs, columns=['Stores %','Warehouse %','Semi %','Raw Mat %'])
        st.dataframe(sd_df, hide_index=True, use_container_width=True, height=200)
    if len(stock_distribs) == 0:
        st.error("⚠️ No valid combos! Store + Warehouse + Semi must ≤ 100%")
        stock_distribs = [(100, 0, 0, 0)]

    st.subheader("🔄 Planning")
    pf = []
    if st.checkbox("Every week", value=True): pf.append(1)
    if st.checkbox("Every 2 weeks", value=True): pf.append(2)
    if st.checkbox("Every 4 weeks", value=True): pf.append(4)
    if not pf: pf = [1]

    st.subheader("🏭 Capacity")
    cr_min = st.number_input("Min ramp %/wk", value=5, step=5)
    cr_max = st.number_input("Max ramp %/wk", value=25, step=5)
    cr_stp = st.number_input("Ramp step %", value=10, step=5, key="cr_s")
    cap_ramps = [r/100 for r in range(cr_min, cr_max+1, cr_stp)]
    cap_start = st.number_input("Initial capacity/wk", value=100, step=25)

    st.subheader("📊 Fixed costs")
    fx_min = st.number_input("Min fixed %", value=20, step=5)
    fx_max = st.number_input("Max fixed %", value=60, step=5)
    fx_stp = st.number_input("Fixed step %", value=10, step=5, key="fx_s")
    fixed_pcts = [f/100 for f in range(fx_min, fx_max+1, fx_stp)]

    st.subheader("🧠 Distribution")
    smart_opts = []
    if st.checkbox("Push 50/50", value=True): smart_opts.append(False)
    if st.checkbox("Smart allocation", value=True): smart_opts.append(True)
    if not smart_opts: smart_opts = [False]


# ─── SCENARIO COUNT ───
params = {
    'demand_mults': demand_mults, 'sim_weeks': sim_weeks,
    'plan_freqs': pf, 'cap_ramps': cap_ramps,
    'smart_opts': smart_opts, 'lt_combos': lt_combos,
    'stock_levels': stock_levels, 'stock_distribs': stock_distribs,
    'cap_start': cap_start, 'base_forecast': base_forecast,
    'demand_splits': demand_splits,
}

n_phys = (len(demand_mults) * len(sim_weeks) * len(pf) * len(cap_ramps)
          * len(smart_opts) * len(lt_combos) * len(stock_levels)
          * len(stock_distribs) * len(demand_splits))
n_fin = n_phys * len(fixed_pcts)
nc = mp.cpu_count(); nw = max(1, nc-1)
est = n_phys * 65e-6 / 60 / nw

st.info(f"""
**{n_phys:,.0f}** physical sims × {len(fixed_pcts)} fixed costs = **{n_fin:,.0f}** financial scenarios  
💰 €{price} price · €{var_cost} var cost · Stock: {stock_levels} · {len(stock_distribs)} distributions · Splits A: {demand_splits}%  
📈 Demand: center **{center:.0%}**, σ={sigma}, expected={sum(d*w for d,w in zip(demand_mults,weights)):.2f}× · 🖥️ {nc} cores → ~**{max(1,est):.0f} min**
""")


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════
if st.button("🚀 Run All Simulations", type="primary", use_container_width=True):
    st.divider()

    st.subheader("Phase 1: Physical Simulations")
    pb_bar = st.progress(0); stxt = st.empty()
    t0 = time.time()
    df_phys = run_batch(params, pb_bar, stxt)

    st.subheader("Phase 2: Financial Expansion + Demand Weights")
    with st.spinner("..."):
        df_full = expand_financials(df_phys, price, var_cost, fixed_pcts, base_forecast)
        df_full = add_demand_weights(df_full, demand_mults, weights)
        st.success(f"✅ {len(df_full):,.0f} scenarios")

    st.subheader("Phase 3: Strategy Labeling")
    with st.spinner("..."):
        df_full, median_lt = label_strategies(df_full)

    st.subheader("Phase 4: ML Analysis")
    with st.spinner("Training tree + random forest..."):
        analysis = run_analysis(df_full)
        st.success(f"✅ Tree R²={analysis['tree_r2']:.3f}, Forest R²={analysis['rf_r2']:.3f}")

    st.subheader("Phase 5: Strategy Comparison")
    with st.spinner("..."):
        matrix_df = strategy_matrix(df_full)
        pairs = paired_comparison(df_full)
        smart = smart_comparison(df_full)

    tt = time.time()-t0
    st.success(f"🏁 **Done in {tt:.0f}s ({tt/60:.1f} min)**")

    # ════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.header("📈 Results (demand-weighted)")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Scenarios", f"{len(df_full):,.0f}")
    c2.metric("Wtd Avg Net Margin", f"{weighted_mean(df_full, 'net_margin_pct'):.1%}")
    c3.metric("Wtd Avg Service Level", f"{weighted_mean(df_full, 'service_level'):.1%}")
    c4.metric("% Profitable Scenarios", f"{(df_full['net_margin_pct']>0).mean():.1%}")

    # ─── VISUAL REGRESSION TREE ───
    st.subheader("🌳 Decision Tree — What Drives Net Margin?")
    st.pyplot(analysis['tree_fig'], use_container_width=True)
    plt.close(analysis['tree_fig'])

    with st.expander("📝 Tree Rules (text)"):
        st.code(analysis['tree_text'], language='text')

    # ─── FEATURE IMPORTANCE ───
    st.subheader("🔑 Feature Importance + Direction")
    imp_data = []
    for f in ML_FEATURES:
        imp_data.append({
            'Feature': PRETTY_LABELS.get(f, f),
            'Tree Importance': analysis['tree_imp'].get(f, 0),
            'Forest Importance': analysis['rf_imp'].get(f, 0),
            'Effect on Margin': ('📈 Increases margin' if analysis['correlations'][f] > 0.01
                                 else '📉 Decreases margin' if analysis['correlations'][f] < -0.01
                                 else '↔️ Neutral'),
            'Correlation': analysis['correlations'][f],
        })
    imp_df = pd.DataFrame(imp_data).sort_values('Forest Importance', ascending=False)
    imp_df = imp_df[imp_df['Forest Importance'] > 0.005]
    st.dataframe(imp_df.style.format({
        'Tree Importance': '{:.3f}', 'Forest Importance': '{:.3f}', 'Correlation': '{:+.3f}'
    }).bar(subset=['Forest Importance'], color='#4a90d9'),
        hide_index=True, use_container_width=True)

    # ─── 2×2 STRATEGY MATRIX ───
    st.subheader("⚔️ Strategy Matrix: Stock Position × Supply Chain Speed")
    st.caption(f"Push to Store = ≥80% in stores · Fast = LT < {int(median_lt)} wks · **Same stock levels for all**")

    # Format the matrix
    fmt_matrix = matrix_df.copy()
    for col in ['Weighted Avg Net Margin %', 'Median Net Margin %', 'Weighted Avg Service Level',
                '% Scenarios Profitable', 'Std Margin %']:
        if col in fmt_matrix.columns:
            fmt_matrix[col] = fmt_matrix[col].map(lambda x: f"{x:.1%}")
    for col in ['Weighted Avg Net Profit €']:
        if col in fmt_matrix.columns:
            fmt_matrix[col] = fmt_matrix[col].map(lambda x: f"€{x:,.0f}")
    for col in ['Scenarios', 'Avg Initial Stock']:
        if col in fmt_matrix.columns:
            fmt_matrix[col] = fmt_matrix[col].map(lambda x: f"{x:,.0f}")
    if 'Avg Total LT (wks)' in fmt_matrix.columns:
        fmt_matrix['Avg Total LT (wks)'] = fmt_matrix['Avg Total LT (wks)'].map(lambda x: f"{x:.0f}")
    if 'Avg % in Stores' in fmt_matrix.columns:
        fmt_matrix['Avg % in Stores'] = fmt_matrix['Avg % in Stores'].map(lambda x: f"{x:.0f}%")
    st.dataframe(fmt_matrix, use_container_width=True)

    # ─── HEAD-TO-HEAD ───
    st.subheader("🏆 Head-to-Head: Push to Store+Slow vs Spread along SC+Fast")
    h1,h2,h3,h4 = st.columns(4)
    h1.metric("Matched Pairs", f"{pairs['pairs']:,.0f}")
    h2.metric("Spread along SC+Fast Wins", f"{pairs['pct_spread']:.1%}")
    h3.metric("Push to Store+Slow Wins", f"{pairs['push_wins']:.1%}")
    h4.metric("Wtd Δ Margin (Dist−Cent)", f"{pairs['avg_delta']:+.1%}")

    # ─── SMART vs PUSH ───
    st.subheader("🧠 Smart Allocation vs Push 50/50")
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Smart Avg Margin", f"{smart['smart_margin']:.1%}")
    s2.metric("Push 50/50 Avg Margin", f"{smart['push_margin']:.1%}")
    s3.metric("Smart Avg Service", f"{smart['smart_svc']:.1%}")
    s4.metric("Push 50/50 Avg Service", f"{smart['push_svc']:.1%}")

    # ─── BREAKDOWN TABLES ───
    st.subheader("📊 Net Margin by Stock Level")
    by_stk = []
    for stk in sorted(df_full['initial_stock_total'].unique()):
        g = df_full[df_full['initial_stock_total']==stk]
        by_stk.append({'Initial Stock (units)': int(stk),
                        'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                        'Wtd Avg Service Level': weighted_mean(g, 'service_level'),
                        'Wtd Avg Net Profit €': weighted_mean(g, 'net_profit')})
    st.dataframe(pd.DataFrame(by_stk).set_index('Initial Stock (units)').style.format({
        'Wtd Avg Net Margin %':'{:.1%}', 'Wtd Avg Service Level':'{:.1%}', 'Wtd Avg Net Profit €':'€{:,.0f}'
    }), use_container_width=True)

    st.subheader("📊 Net Margin by % Stock in Stores")
    by_sp = []
    for sp in sorted(df_full['pct_in_stores'].unique()):
        g = df_full[df_full['pct_in_stores']==sp]
        by_sp.append({'% in Stores': int(sp),
                       'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                       'Wtd Avg Service Level': weighted_mean(g, 'service_level'),
                       'Wtd Avg Net Profit €': weighted_mean(g, 'net_profit')})
    st.dataframe(pd.DataFrame(by_sp).set_index('% in Stores').style.format({
        'Wtd Avg Net Margin %':'{:.1%}', 'Wtd Avg Service Level':'{:.1%}', 'Wtd Avg Net Profit €':'€{:,.0f}'
    }), use_container_width=True)

    if df_full['pct_in_warehouse'].nunique() > 1:
        st.subheader("📊 Net Margin by % Stock in Warehouse")
        by_wh = []
        for wp in sorted(df_full['pct_in_warehouse'].unique()):
            g = df_full[df_full['pct_in_warehouse']==wp]
            by_wh.append({'% in Warehouse': int(wp),
                           'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                           'Wtd Avg Service Level': weighted_mean(g, 'service_level'),
                           'Wtd Avg Net Profit €': weighted_mean(g, 'net_profit')})
        st.dataframe(pd.DataFrame(by_wh).set_index('% in Warehouse').style.format({
            'Wtd Avg Net Margin %':'{:.1%}', 'Wtd Avg Service Level':'{:.1%}', 'Wtd Avg Net Profit €':'€{:,.0f}'
        }), use_container_width=True)

    if df_full['pct_in_semifinished'].nunique() > 1:
        st.subheader("📊 Net Margin by % Stock in Semi-Finished")
        by_se = []
        for sep in sorted(df_full['pct_in_semifinished'].unique()):
            g = df_full[df_full['pct_in_semifinished']==sep]
            by_se.append({'% in Semi-Finished': int(sep),
                           'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                           'Wtd Avg Service Level': weighted_mean(g, 'service_level'),
                           'Wtd Avg Net Profit €': weighted_mean(g, 'net_profit')})
        st.dataframe(pd.DataFrame(by_se).set_index('% in Semi-Finished').style.format({
            'Wtd Avg Net Margin %':'{:.1%}', 'Wtd Avg Service Level':'{:.1%}', 'Wtd Avg Net Profit €':'€{:,.0f}'
        }), use_container_width=True)

    st.subheader("📊 Net Margin by Total Lead Time")
    by_lt = []
    for lt in sorted(df_full['lt_total_weeks'].unique()):
        g = df_full[df_full['lt_total_weeks']==lt]
        by_lt.append({'Total Lead Time (wks)': int(lt),
                       'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                       'Wtd Avg Service Level': weighted_mean(g, 'service_level'),
                       'Wtd Avg Net Profit €': weighted_mean(g, 'net_profit')})
    st.dataframe(pd.DataFrame(by_lt).set_index('Total Lead Time (wks)').style.format({
        'Wtd Avg Net Margin %':'{:.1%}', 'Wtd Avg Service Level':'{:.1%}', 'Wtd Avg Net Profit €':'€{:,.0f}'
    }), use_container_width=True)

    if len(demand_splits) > 1:
        st.subheader("📊 Net Margin by Demand Split (Store A %)")
        by_ds = []
        for ds in sorted(df_full['demand_split_store_a_pct'].unique()):
            g = df_full[df_full['demand_split_store_a_pct']==ds]
            by_ds.append({'Store A Demand %': int(ds),
                           'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                           'Wtd Avg Service Level': weighted_mean(g, 'service_level')})
        st.dataframe(pd.DataFrame(by_ds).set_index('Store A Demand %').style.format({
            'Wtd Avg Net Margin %':'{:.1%}', 'Wtd Avg Service Level':'{:.1%}'
        }), use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.header("📋 Executive Summary")

    t5 = sorted(analysis['rf_imp'].items(), key=lambda x:-x[1])[:5]
    dw = pairs['pct_spread']*100; cw = pairs['push_wins']*100
    sm_d = smart['smart_margin']-smart['push_margin']

    pos_f = [(f,analysis['correlations'][f]) for f in ML_FEATURES
             if analysis['rf_imp'][f]>0.01 and analysis['correlations'][f]>0.01]
    neg_f = [(f,analysis['correlations'][f]) for f in ML_FEATURES
             if analysis['rf_imp'][f]>0.01 and analysis['correlations'][f]<-0.01]
    pos_f.sort(key=lambda x:-analysis['rf_imp'][x[0]])
    neg_f.sort(key=lambda x:-analysis['rf_imp'][x[0]])

    ex = f"""
### Setup
**{len(df_full):,.0f}** scenarios · €{price} price · €{var_cost} var cost · Stock {stock_levels} (absolute) · Store A splits: {demand_splits}%
Demand weighted (center={center:.0%}, σ={sigma}) · **Initial stock cost included** in variable costs

### Key Drivers (Random Forest R²={analysis['rf_r2']:.3f})
| Feature | Importance | Effect |
|---------|-----------|--------|
"""
    for f, imp in t5:
        d = '📈 Increases' if analysis['correlations'][f]>0.01 else '📉 Decreases' if analysis['correlations'][f]<-0.01 else '↔️'
        ex += f"| {PRETTY_LABELS.get(f,f)} | {imp:.3f} | {d} margin |\n"

    ex += f"""
### Factors that INCREASE margin: {', '.join(PRETTY_LABELS.get(f,f) for f,_ in pos_f[:4]) or 'None identified'}
### Factors that DECREASE margin: {', '.join(PRETTY_LABELS.get(f,f) for f,_ in neg_f[:4]) or 'None identified'}

### Strategy Comparison — {pairs['pairs']:,.0f} matched pairs (equal stock)
**Spread along SC+Fast wins {dw:.1f}%** · Push to Store+Slow wins {cw:.1f}%
Weighted Δ margin: **{pairs['avg_delta']:+.1%}** in favor of {'Spread along SC+Fast' if pairs['avg_delta']>0 else 'Push to Store+Slow'}

### Smart Allocation: Δ margin **{sm_d:+.1%}** vs push 50/50

### Verdict
{"**Agility wins.** Spreading stock + short LTs outperforms Push to Store on a weighted, equal-stock basis." if dw>55 else "**Mixed.** Neither strategy dominates — optimal choice depends on demand and cost structure." if 45<dw<55 else "**Push to Store holds.** Pre-loading stores is competitive even at equal stock investment."}
"""
    st.markdown(ex)

    # ─── EXPORT ───
    st.divider()
    st.subheader("💾 Export")

    # Build standalone HTML report
    import base64, io

    # Tree image as base64
    buf = io.BytesIO()
    analysis['tree_fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
    tree_b64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()

    # Demand chart as base64
    fig_dem, ax_dem = plt.subplots(figsize=(8, 2.5), dpi=100)
    ax_dem.bar(demand_mults, [w*100 for w in weights], width=0.06, color='#4a90d9')
    ax_dem.set_xlabel('Demand ×'); ax_dem.set_ylabel('Probability %')
    ax_dem.set_title(f'Demand Profile (center={center:.0%}, σ={sigma})')
    plt.tight_layout()
    buf2 = io.BytesIO()
    fig_dem.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    dem_b64 = base64.b64encode(buf2.getvalue()).decode()
    buf2.close(); plt.close(fig_dem)

    # Helper: dataframe to HTML table
    def df_to_html(df, pct_cols=None, eur_cols=None, int_cols=None):
        pct_cols = pct_cols or []; eur_cols = eur_cols or []; int_cols = int_cols or []
        h = '<table><thead><tr>'
        for c in df.columns:
            h += f'<th>{c}</th>'
        h += '</tr></thead><tbody>'
        for _, row in df.iterrows():
            h += '<tr>'
            for c in df.columns:
                v = row[c]
                if c in pct_cols and isinstance(v, (int,float)):
                    h += f'<td>{v:.1%}</td>'
                elif c in eur_cols and isinstance(v, (int,float)):
                    h += f'<td>€{v:,.0f}</td>'
                elif c in int_cols and isinstance(v, (int,float)):
                    h += f'<td>{v:,.0f}</td>'
                else:
                    h += f'<td>{v}</td>'
            h += '</tr>'
        h += '</tbody></table>'
        return h

    # Build importance table
    imp_html_data = []
    for f in ML_FEATURES:
        rf_imp = analysis['rf_imp'].get(f, 0)
        if rf_imp < 0.005: continue
        corr_v = analysis['correlations'][f]
        direction = '📈 Increases margin' if corr_v > 0.01 else '📉 Decreases margin' if corr_v < -0.01 else '↔️ Neutral'
        imp_html_data.append({
            'Feature': PRETTY_LABELS.get(f, f),
            'Forest Importance': rf_imp,
            'Tree Importance': analysis['tree_imp'].get(f, 0),
            'Direction': direction,
            'Correlation': corr_v,
        })
    imp_html_df = pd.DataFrame(imp_html_data).sort_values('Forest Importance', ascending=False)

    # Build breakdown tables
    def build_breakdown(df_src, group_col, label):
        rows = []
        for val in sorted(df_src[group_col].unique()):
            g = df_src[df_src[group_col]==val]
            rows.append({
                label: int(val) if isinstance(val, (int, float, np.integer, np.floating)) else val,
                'Wtd Avg Net Margin %': weighted_mean(g, 'net_margin_pct'),
                'Wtd Avg Service Level': weighted_mean(g, 'service_level'),
                'Wtd Avg Net Profit €': weighted_mean(g, 'net_profit'),
            })
        return pd.DataFrame(rows)

    bd_stock = build_breakdown(df_full, 'initial_stock_total', 'Initial Stock')
    bd_store = build_breakdown(df_full, 'pct_in_stores', '% in Stores')
    bd_lt = build_breakdown(df_full, 'lt_total_weeks', 'Total LT (wks)')

    bd_wh = build_breakdown(df_full, 'pct_in_warehouse', '% in Warehouse') if df_full['pct_in_warehouse'].nunique() > 1 else None
    bd_semi = build_breakdown(df_full, 'pct_in_semifinished', '% in Semi-Finished') if df_full['pct_in_semifinished'].nunique() > 1 else None
    bd_split = build_breakdown(df_full, 'demand_split_store_a_pct', 'Store A Demand %') if len(demand_splits) > 1 else None

    # Strategy matrix HTML
    matrix_html_df = matrix_df.copy().reset_index().rename(columns={'index': 'Strategy'})

    # Format metrics
    wm = weighted_mean(df_full, 'net_margin_pct')
    ws = weighted_mean(df_full, 'service_level')
    ppos = (df_full['net_margin_pct'] > 0).mean()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SC Agility — Batch Results</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 0; padding: 20px 40px; background: #f4f6f9; color: #1a2a40; }}
h1 {{ color: #1a2a40; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
h2 {{ color: #2a3a50; margin-top: 35px; border-bottom: 2px solid #e0e4ea; padding-bottom: 6px; }}
h3 {{ color: #3a4a60; margin-top: 20px; }}
.metrics {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 15px 0; }}
.metric {{ background: linear-gradient(135deg, #fff, #f7f9fc); border: 1px solid #dde3ed; border-radius: 10px; padding: 15px 20px; text-align: center; flex: 1; min-width: 150px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
.metric .value {{ font-size: 26px; font-weight: 800; color: #1a2a40; }}
.metric .label {{ font-size: 11px; color: #7a8a9e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
.metric.green .value {{ color: #28a745; }}
.metric.red .value {{ color: #dc3545; }}
.metric.blue .value {{ color: #4a90d9; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
th {{ background: #2a3a50; color: white; padding: 10px 12px; text-align: left; font-size: 12px; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #e0e4ea; }}
tr:nth-child(even) {{ background: #f8f9fb; }}
tr:hover {{ background: #eef1f5; }}
.box {{ background: white; border: 1px solid #dce2ea; border-radius: 10px; padding: 20px; margin: 15px 0; }}
.verdict {{ background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border-left: 5px solid #28a745; padding: 15px 20px; border-radius: 8px; margin: 15px 0; font-size: 15px; }}
.verdict.mixed {{ background: linear-gradient(135deg, #fff8e1, #fff3e0); border-left-color: #ff9800; }}
.verdict.push {{ background: linear-gradient(135deg, #fce4ec, #fff3e0); border-left-color: #dc3545; }}
img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; }}
.cols {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.col {{ flex: 1; min-width: 300px; }}
.timestamp {{ color: #999; font-size: 12px; }}
.param {{ display: inline-block; background: #eef1f5; padding: 3px 10px; border-radius: 12px; margin: 2px; font-size: 12px; }}
</style></head><body>

<h1>📊 SC Agility — Batch Simulation Results</h1>
<p class="timestamp">Generated: {time.strftime('%Y-%m-%d %H:%M')} · {len(df_full):,.0f} scenarios</p>

<h2>⚙️ Parameters</h2>
<div class="box">
<span class="param">💰 Price: €{price:,}</span>
<span class="param">🏭 Var cost: €{var_cost:,}/unit</span>
<span class="param">📦 Stock: {stock_levels}</span>
<span class="param">📈 Demand center: {center:.0%}</span>
<span class="param">📊 σ = {sigma}</span>
<span class="param">🏪 Splits A: {demand_splits}%</span>
<span class="param">📅 Weeks: {sim_weeks}</span>
<span class="param">📦 {len(stock_distribs)} stock distributions</span>
<span class="param">⏱️ {len(lt_combos)} LT combos</span>
<span class="param">💰 Fixed costs: {[f'{f:.0%}' for f in fixed_pcts]}</span>
</div>

<h2>📈 Demand Profile</h2>
<img src="data:image/png;base64,{dem_b64}" alt="Demand Profile">

<h2>📊 Key Metrics (demand-weighted)</h2>
<div class="metrics">
<div class="metric"><div class="value">{len(df_full):,.0f}</div><div class="label">Total Scenarios</div></div>
<div class="metric blue"><div class="value">{wm:.1%}</div><div class="label">Wtd Avg Net Margin</div></div>
<div class="metric"><div class="value">{ws:.1%}</div><div class="label">Wtd Avg Service Level</div></div>
<div class="metric {'green' if ppos > 0.7 else 'red'}"><div class="value">{ppos:.1%}</div><div class="label">% Profitable</div></div>
</div>

<h2>🌳 Regression Tree — Net Margin Drivers</h2>
<img src="data:image/png;base64,{tree_b64}" alt="Regression Tree">

<h2>🔑 Feature Importance + Direction</h2>
{df_to_html(imp_html_df, pct_cols=['Forest Importance', 'Tree Importance'])}

<h2>⚔️ Strategy Matrix: Stock Position × SC Speed</h2>
<p><em>Push to Store = ≥80% in stores · Fast = LT &lt; {int(median_lt)} wks · Same stock for all</em></p>
"""

    # Format strategy matrix for HTML
    sm_df = matrix_html_df.copy()
    pct_c = ['Weighted Avg Net Margin %', 'Median Net Margin %', 'Weighted Avg Service Level',
             '% Scenarios Profitable', 'Std Margin %']
    eur_c = ['Weighted Avg Net Profit €']
    int_c = ['Scenarios', 'Avg Initial Stock']
    html += df_to_html(sm_df, pct_cols=pct_c, eur_cols=eur_c, int_cols=int_c)

    # Head-to-head
    verdict_class = 'verdict' if dw > 55 else 'verdict mixed' if dw > 45 else 'verdict push'
    html += f"""
<h2>🏆 Head-to-Head: Push to Store+Slow vs Spread along SC+Fast</h2>
<div class="metrics">
<div class="metric"><div class="value">{pairs['pairs']:,.0f}</div><div class="label">Matched Pairs</div></div>
<div class="metric green"><div class="value">{pairs['pct_spread']:.1%}</div><div class="label">Spread along SC+Fast Wins</div></div>
<div class="metric red"><div class="value">{pairs['push_wins']:.1%}</div><div class="label">Push to Store+Slow Wins</div></div>
<div class="metric blue"><div class="value">{pairs['avg_delta']:+.1%}</div><div class="label">Wtd Δ Margin</div></div>
</div>

<h2>🧠 Smart Allocation vs Push 50/50</h2>
<div class="metrics">
<div class="metric"><div class="value">{smart['smart_margin']:.1%}</div><div class="label">Smart Avg Margin</div></div>
<div class="metric"><div class="value">{smart['push_margin']:.1%}</div><div class="label">Push Avg Margin</div></div>
<div class="metric"><div class="value">{smart['smart_svc']:.1%}</div><div class="label">Smart Avg Service</div></div>
<div class="metric"><div class="value">{smart['push_svc']:.1%}</div><div class="label">Push Avg Service</div></div>
</div>

<h2>📊 Breakdowns</h2>
<div class="cols"><div class="col">
<h3>By Stock Level</h3>
{df_to_html(bd_stock, pct_cols=['Wtd Avg Net Margin %', 'Wtd Avg Service Level'], eur_cols=['Wtd Avg Net Profit €'])}
</div><div class="col">
<h3>By % in Stores</h3>
{df_to_html(bd_store, pct_cols=['Wtd Avg Net Margin %', 'Wtd Avg Service Level'], eur_cols=['Wtd Avg Net Profit €'])}
</div></div>

<div class="cols"><div class="col">
<h3>By Total Lead Time</h3>
{df_to_html(bd_lt, pct_cols=['Wtd Avg Net Margin %', 'Wtd Avg Service Level'], eur_cols=['Wtd Avg Net Profit €'])}
</div><div class="col">
"""
    if bd_wh is not None:
        html += f"""<h3>By % in Warehouse</h3>
{df_to_html(bd_wh, pct_cols=['Wtd Avg Net Margin %', 'Wtd Avg Service Level'], eur_cols=['Wtd Avg Net Profit €'])}"""
    if bd_semi is not None:
        html += f"""<h3>By % in Semi-Finished</h3>
{df_to_html(bd_semi, pct_cols=['Wtd Avg Net Margin %', 'Wtd Avg Service Level'], eur_cols=['Wtd Avg Net Profit €'])}"""
    if bd_split is not None:
        html += f"""<h3>By Demand Split (Store A %)</h3>
{df_to_html(bd_split, pct_cols=['Wtd Avg Net Margin %', 'Wtd Avg Service Level'])}"""

    html += f"""</div></div>

<h2>📋 Executive Summary</h2>
<div class="box">

<h3>Setup</h3>
<p><strong>{len(df_full):,.0f}</strong> scenarios · €{price} price · €{var_cost} var cost · Stock {stock_levels} (absolute)
· Store A splits: {demand_splits}%<br>
Demand weighted (center={center:.0%}, σ={sigma}) · Initial stock cost included in variable costs</p>

<h3>Key Drivers (Random Forest R²={analysis['rf_r2']:.3f})</h3>
<table><thead><tr><th>Feature</th><th>Importance</th><th>Effect</th></tr></thead><tbody>"""

    for f, imp in t5:
        d = '📈 Increases' if analysis['correlations'][f]>0.01 else '📉 Decreases' if analysis['correlations'][f]<-0.01 else '↔️'
        html += f"<tr><td>{PRETTY_LABELS.get(f,f)}</td><td>{imp:.3f}</td><td>{d} margin</td></tr>"

    html += f"""</tbody></table>

<p><strong>Positive factors:</strong> {', '.join(PRETTY_LABELS.get(f,f) for f,_ in pos_f[:4]) or 'None'}</p>
<p><strong>Negative factors:</strong> {', '.join(PRETTY_LABELS.get(f,f) for f,_ in neg_f[:4]) or 'None'}</p>

<h3>Strategy Comparison — {pairs['pairs']:,.0f} matched pairs (equal stock)</h3>
<p><strong>Spread along SC+Fast wins {dw:.1f}%</strong> · Push to Store+Slow wins {cw:.1f}%<br>
Weighted Δ margin: <strong>{pairs['avg_delta']:+.1%}</strong></p>

<p>Smart allocation: Δ margin <strong>{sm_d:+.1%}</strong> vs push 50/50</p>

<div class="{verdict_class}">
<strong>Verdict:</strong>
{"Agility wins. Spreading stock + short LTs outperforms Push to Store." if dw>55 else "Mixed. Neither strategy dominates." if 45<dw<55 else "Push to Store holds. Pre-loading stores is competitive."}
</div>

</div>

<p class="timestamp" style="margin-top:40px; text-align:center;">
SC Agility Batch Simulator v4 · Generated {time.strftime('%Y-%m-%d %H:%M')}
</p>

</body></html>"""

    try:
        p1 = os.path.join(OUTPUT_DIR, 'sc_physical_results.csv')
        (df_phys.sample(min(1_000_000,len(df_phys)),random_state=42) if len(df_phys)>1_000_000 else df_phys).to_csv(p1, index=False)
        p2 = os.path.join(OUTPUT_DIR, 'sc_financial_results.csv')
        (df_full.sample(min(500_000,len(df_full)),random_state=42) if len(df_full)>500_000 else df_full).to_csv(p2, index=False)
        p3 = os.path.join(OUTPUT_DIR, 'sc_executive_summary.md')
        with open(p3,'w',encoding='utf-8') as f:
            f.write(f"# SC Agility — Executive Summary\n\n*{time.strftime('%Y-%m-%d %H:%M')}*\n\n{ex}")
        p4 = os.path.join(OUTPUT_DIR, 'regression_tree.png')
        analysis['tree_fig'].savefig(p4, dpi=150, bbox_inches='tight')
        p5 = os.path.join(OUTPUT_DIR, 'sc_results_report.html')
        with open(p5, 'w', encoding='utf-8') as f:
            f.write(html)
        st.success(f"✅ Exported to **{OUTPUT_DIR}/**")
        st.caption(f"📄 **HTML report:** {p5}")
        st.caption(f"📊 CSVs: {p1}, {p2}")
        st.caption(f"📋 Summary: {p3} · 🌳 Tree: {p4}")
    except Exception as e:
        st.error(f"Export failed: {e}")

else:
    st.markdown("---")
    st.markdown("👈 **Configure in sidebar**, then press **Run**.")
    st.markdown("""
    **v4:** Weighted demand · Absolute stock (fair comparison) · Store A/B demand split ·
    Visual regression tree · Clear 2×2 strategy matrix · Human-readable column names ·
    Initial stock cost in P&L
    """)
