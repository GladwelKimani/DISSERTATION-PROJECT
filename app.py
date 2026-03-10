import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_nse_data, clean_nse_data, engineer_features, reindex_to_business_days
from models import (load_lstm_results, load_lagllama_zeroshot, load_lagllama_finetuned,
                    get_best_model_per_ticker, get_prediction_for_date,
                    get_trade_signal, build_metrics_table, get_best_portfolio)
from charts import (plot_market_chart, plot_sector_rankings,
                    plot_price_volume, plot_sector_promise,
                    plot_portfolio_bar, plot_risk_horizon, plot_risk_return)

st.set_page_config(page_title="NSE Forecast Lab", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

html,body,[class*="css"]{ font-family:'Plus Jakarta Sans',sans-serif; background:#F0F4F8!important; }
.main .block-container{ padding:0.6rem 1.2rem 0.3rem 1.2rem!important; max-width:100%!important; }
.stApp{ background:#F0F4F8!important; }

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1A3A5C 0%,#152E4D 60%,#0F2035 100%)!important;
    border-right:1px solid rgba(0,160,120,0.2)!important;
}
[data-testid="stSidebar"] *{ color:#CBD5E1!important; }

/* Page header — compact */
.page-header{
    background:linear-gradient(135deg,#1A3A5C 0%,#1E4976 50%,#1A5C8A 100%);
    border:1px solid rgba(0,160,120,0.25); border-radius:12px;
    padding:0.55rem 1.1rem; margin-bottom:0.55rem;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 3px 14px rgba(0,0,0,0.1);
}
.page-title{ font-family:'Sora',sans-serif; font-size:1.2rem; font-weight:700; color:#F8FAFC; margin:0; }
.page-subtitle{ font-size:0.7rem; color:rgba(255,255,255,0.5); margin:0; }

/* Metric cards — compact */
[data-testid="stMetric"]{
    background:linear-gradient(135deg,#FFFFFF 0%,#EEF4FB 100%);
    border:1px solid rgba(0,120,100,0.15); border-radius:10px;
    padding:0.35rem 0.7rem!important;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMetric"] label{ color:#64748B!important; font-size:0.6rem!important; text-transform:uppercase; letter-spacing:0.7px; font-weight:600!important; }
[data-testid="stMetric"] [data-testid="stMetricValue"]{ color:#0F172A!important; font-family:'Sora',sans-serif!important; font-size:1rem!important; font-weight:700!important; }
/* Force green for positive delta, red for negative */
[data-testid="stMetricDelta"] svg{ display:none!important; }
[data-testid="stMetricDelta"]>div{ font-size:0.65rem!important; font-weight:600!important; }

/* Card */
.card{ background:#FFFFFF; border:1px solid rgba(0,0,0,0.07); border-radius:12px; box-shadow:0 3px 12px rgba(0,0,0,0.06); overflow:hidden; margin-bottom:0.35rem; }
.card-header{ font-family:'Sora',sans-serif; font-size:0.7rem; font-weight:700; color:#1E293B; text-transform:uppercase; letter-spacing:0.9px; background:linear-gradient(135deg,#F1F5F9,#E8F0F8); border-bottom:1px solid rgba(0,0,0,0.06); padding:0.35rem 0.9rem; display:flex; align-items:center; gap:5px; }
.card-body{ padding:0.45rem 0.9rem; }
.card-accent{ color:#0E7A5E; }

/* ── SIGNAL BADGE — vivid, compact ── */
.signal-buy{
    background:linear-gradient(160deg,#052e16 0%,#14532d 50%,#166534 100%);
    border:3px solid #4ade80; border-radius:14px;
    padding:0.9rem 0.8rem 0.7rem; text-align:center; width:100%;
    box-shadow:0 0 0 4px rgba(74,222,128,0.12), 0 0 28px rgba(22,163,74,0.4);
}
.signal-buy .si{ font-size:2rem; line-height:1; filter:drop-shadow(0 0 6px rgba(74,222,128,0.9)); }
.signal-buy .sl{ font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:800; color:#4ade80; letter-spacing:5px; text-shadow:0 0 14px rgba(74,222,128,0.5); margin-top:2px; }
.signal-buy .ss{ font-size:0.72rem; color:#86efac; font-weight:600; margin-top:4px; background:rgba(0,0,0,0.25); border-radius:20px; padding:2px 10px; display:inline-block; }

.signal-sell{
    background:linear-gradient(160deg,#450a0a 0%,#7f1d1d 50%,#991b1b 100%);
    border:3px solid #f87171; border-radius:14px;
    padding:0.9rem 0.8rem 0.7rem; text-align:center; width:100%;
    box-shadow:0 0 0 4px rgba(248,113,113,0.12), 0 0 28px rgba(220,38,38,0.4);
}
.signal-sell .si{ font-size:2rem; line-height:1; filter:drop-shadow(0 0 6px rgba(248,113,113,0.9)); }
.signal-sell .sl{ font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:800; color:#f87171; letter-spacing:5px; text-shadow:0 0 14px rgba(248,113,113,0.5); margin-top:2px; }
.signal-sell .ss{ font-size:0.72rem; color:#fca5a5; font-weight:600; margin-top:4px; background:rgba(0,0,0,0.25); border-radius:20px; padding:2px 10px; display:inline-block; }

.signal-hold{
    background:linear-gradient(160deg,#422006 0%,#78350f 50%,#92400e 100%);
    border:3px solid #fbbf24; border-radius:14px;
    padding:0.9rem 0.8rem 0.7rem; text-align:center; width:100%;
    box-shadow:0 0 0 4px rgba(251,191,36,0.12), 0 0 28px rgba(202,138,4,0.4);
}
.signal-hold .si{ font-size:2rem; line-height:1; filter:drop-shadow(0 0 6px rgba(251,191,36,0.9)); }
.signal-hold .sl{ font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:800; color:#fbbf24; letter-spacing:5px; text-shadow:0 0 14px rgba(251,191,36,0.5); margin-top:2px; }
.signal-hold .ss{ font-size:0.72rem; color:#fde68a; font-weight:600; margin-top:4px; background:rgba(0,0,0,0.25); border-radius:20px; padding:2px 10px; display:inline-block; }

/* CI pill */
.ci-pill{ background:rgba(180,83,9,0.08); border:1px solid rgba(180,83,9,0.25); border-radius:20px; padding:3px 10px; font-size:0.68rem; color:#92400E; font-weight:600; display:inline-block; margin-top:3px; }

/* Insight card */
.insight-card{ background:#FFFFFF; border:1px solid rgba(0,0,0,0.07); border-radius:10px; padding:0.55rem 0.8rem; box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.insight-label{ font-size:0.6rem; font-weight:700; text-transform:uppercase; letter-spacing:0.7px; color:#94A3B8; margin-bottom:1px; }
.insight-value{ font-family:'Sora',sans-serif; font-size:0.95rem; font-weight:700; color:#0F172A; }
.insight-sub{ font-size:0.68rem; color:#64748B; margin-top:1px; }

/* Directional row */
.dir-row{ display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid rgba(0,0,0,0.04); }
.dir-ticker{ font-family:'Sora',sans-serif; font-weight:700; font-size:0.88rem; color:#0F172A; min-width:46px; }
.dir-up{ font-size:1rem; color:#16a34a; font-weight:700; }
.dir-down{ font-size:1rem; color:#dc2626; font-weight:700; }
.dir-label-up{ font-size:0.72rem; color:#16a34a; font-weight:600; }
.dir-label-down{ font-size:0.72rem; color:#dc2626; font-weight:600; }

/* 52-week bar */
.wk52-bar{ background:#F8FAFC; border-radius:8px; padding:0.5rem 1rem; display:flex; gap:1.5rem; align-items:center; border:1px solid rgba(0,0,0,0.06); flex-wrap:wrap; }
.wk52-item{ text-align:center; }
.wk52-lbl{ font-size:0.6rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:#94A3B8; }
.wk52-val{ font-family:'Sora',sans-serif; font-size:0.9rem; font-weight:700; }
.wk52-low{ color:#DC2626; }
.wk52-high{ color:#0E7A5E; }
.wk52-cur{ color:#1E293B; }
.wk52-pred{ color:#B45309; }

/* Portfolio stock row */
.port-stock-card{ background:#F8FAFD; border:1px solid rgba(0,0,0,0.07); border-radius:10px; padding:0.6rem 0.9rem; margin-bottom:0.4rem; }

/* Inputs compact */
.stSelectbox>div>div,.stMultiSelect>div>div{ background:#FFFFFF!important; border:1px solid rgba(0,0,0,0.11)!important; border-radius:7px!important; }
.stSelectbox label,.stMultiSelect label,.stSlider label,.stRadio label,.stDateInput label,.stNumberInput label{ color:#64748B!important; font-size:0.68rem!important; font-weight:600!important; text-transform:uppercase; letter-spacing:0.4px; }
.stMultiSelect [data-baseweb="tag"]{ background:rgba(14,122,94,0.1)!important; border:1px solid rgba(14,122,94,0.3)!important; color:#0E7A5E!important; }
.stRadio [data-testid="stWidgetLabel"]{ display:none!important; }
.stRadio [role="radiogroup"]{ display:flex; gap:5px; flex-wrap:wrap; }
.stRadio label[data-baseweb="radio"] span:last-child{ background:rgba(0,0,0,0.04)!important; border:1px solid rgba(0,0,0,0.09)!important; border-radius:20px!important; padding:3px 12px!important; font-size:0.72rem!important; font-weight:500!important; color:#475569!important; }
.stRadio label[data-baseweb="radio"][aria-checked="true"] span:last-child{ background:rgba(14,122,94,0.1)!important; border-color:rgba(14,122,94,0.4)!important; color:#0E7A5E!important; font-weight:600!important; }
.stButton>button[kind="primary"]{ background:linear-gradient(135deg,#0E7A5E,#065F46)!important; border:none!important; color:#FFF!important; font-weight:700!important; border-radius:8px!important; font-size:0.8rem!important; box-shadow:0 3px 10px rgba(14,122,94,0.22)!important; }
.stButton>button[kind="secondary"]{ border:1.5px solid rgba(14,122,94,0.4)!important; color:#0E7A5E!important; font-weight:600!important; border-radius:8px!important; background:transparent!important; font-size:0.78rem!important; }

p,div{ color:#334155; }
[data-testid="stDataFrame"]{ border:1px solid rgba(0,0,0,0.07)!important; border-radius:8px!important; overflow:hidden; }
#MainMenu,footer,header{ visibility:hidden; }
.stDeployButton{ display:none; }
::-webkit-scrollbar{ width:3px; height:3px; }
::-webkit-scrollbar-thumb{ background:rgba(14,122,94,0.2); border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_price_at_date(all_data, ticker, target_date):
    """Return closing price nearest to target_date for a ticker."""
    td = all_data[all_data['Ticker'] == ticker].copy()
    td['Date'] = pd.to_datetime(td['Date'])
    target_ts = pd.to_datetime(target_date)
    if td.empty:
        return None
    idx = (td['Date'] - target_ts).abs().argsort()
    return float(td.iloc[idx.iloc[0]]['Close'])

def delta_color(val):
    return "normal" if val >= 0 else "inverse"

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    raw = load_nse_data(data_dir)
    return reindex_to_business_days(engineer_features(clean_nse_data(raw)))

@st.cache_data
def load_all_models():
    lstm = load_lstm_results()
    zs   = load_lagllama_zeroshot()
    ft   = load_lagllama_finetuned()
    best = get_best_model_per_ticker(lstm, zs, ft)
    met  = build_metrics_table(lstm, zs, ft)
    return lstm, zs, ft, best, met

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [('portfolio', []), ('last_pred', None), ('extra_stocks', [{},{},{}]), ('extra_slot_tickers', [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.4rem 0 0.8rem'>
        <div style='font-family:Sora,sans-serif;font-size:1.25rem;font-weight:700;color:#F8FAFC;text-transform:uppercase;letter-spacing:1px'>NSE Forecast Lab</div>
        <div style='font-size:0.65rem;color:#475569;margin-top:1px'>Transforming NSE data into forward-looking insights</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0.1rem 0 0.6rem'>", unsafe_allow_html=True)
    page = st.radio("NAV", ["📊 Market Overview","🔮 Stock Intelligence","💼 Portfolio Manager"],
                    label_visibility="collapsed")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:0.8rem 0'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(""):
    try:
        all_data = load_all_data()
        lstm_results, zs_results, ft_results, best_models, metrics_df = load_all_models()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False

if not data_loaded:
    st.stop()

all_tickers = sorted(all_data['Ticker'].unique().tolist())
all_sectors = sorted(all_data['Sector'].unique().tolist())
data_min_yr = int(all_data['Date'].min().year)
data_max_yr = int(all_data['Date'].max().year)

sector_ticker_map = (all_data[['Ticker','Sector']].drop_duplicates()
                     .groupby('Sector')['Ticker'].apply(sorted).to_dict())
ticker_sector_map = (all_data[['Ticker','Sector']].drop_duplicates()
                     .set_index('Ticker')['Sector'].to_dict())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    st.markdown("""<div class='page-header'><div>
        <div class='page-title'>📊 Market Overview</div>
        <div class='page-subtitle'>Nairobi Securities Exchange — Performance Intelligence</div>
    </div></div>""", unsafe_allow_html=True)

    # Year range defined FIRST so metric cards can use it
    yr_range = st.slider("Date Range", data_min_yr, data_max_yr,
                         (data_min_yr, data_max_yr), label_visibility="collapsed")
    sel_date_range = (pd.Timestamp(f'{yr_range[0]}-01-01'),
                      pd.Timestamp(f'{yr_range[1]}-12-31'))

    # Filter all_data by year range to compute best/worst within that window
    filtered_data = all_data[
        (pd.to_datetime(all_data['Date']) >= sel_date_range[0]) &
        (pd.to_datetime(all_data['Date']) <= sel_date_range[1])
    ]
    def compute_return(grp):
        grp = grp.sort_values('Date')
        if len(grp) < 2: return 0.0
        return (grp['Close'].iloc[-1] - grp['Close'].iloc[0]) / grp['Close'].iloc[0] * 100

    period_returns = filtered_data.groupby('Ticker').apply(compute_return)
    if not period_returns.empty:
        best_ticker  = period_returns.idxmax()
        best_return  = period_returns.max()
        worst_ticker = period_returns.idxmin()
        worst_return = period_returns.min()
    else:
        best_ticker  = metrics_df.loc[metrics_df['Cumulative_Return_%'].idxmax(), 'Ticker']
        best_return  = metrics_df['Cumulative_Return_%'].max()
        worst_ticker = metrics_df.loc[metrics_df['Cumulative_Return_%'].idxmin(), 'Ticker']
        worst_return = metrics_df['Cumulative_Return_%'].min()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Stocks", all_data['Ticker'].nunique())
    with c2: st.metric("Sectors", all_data['Sector'].nunique())
    with c3: st.metric("🏆 Best Stock", best_ticker, delta=f"+{best_return:.1f}%")
    with c4: st.metric("📉 Worst Stock", worst_ticker, delta=f"{worst_return:.1f}%",
                        delta_color="normal")

    st.markdown("<div style='margin:0.4rem 0'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 1.1], gap="medium")

    with left:
        st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> MARKET DATA</div><div class='card-body'>", unsafe_allow_html=True)
        fc1, fc2 = st.columns([2,1])
        with fc1:
            sel_sectors = st.multiselect("Sector", all_sectors, placeholder="Select sector(s)…")
        with fc2:
            st.markdown(f"<div style='font-size:0.68rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px'>Period: {yr_range[0]} – {yr_range[1]}</div>", unsafe_allow_html=True)
        avail = []
        if sel_sectors:
            for s in sel_sectors: avail += sector_ticker_map.get(s,[])
            avail = sorted(set(avail))
        else:
            avail = all_tickers
        sel_tickers = st.multiselect("Stocks", avail, placeholder="Select stock(s)…")
        sel_date_range = (pd.Timestamp(f'{yr_range[0]}-01-01'), pd.Timestamp(f'{yr_range[1]}-12-31'))
        chart_type = st.radio("View", ["Closing Prices","Cumulative Returns","Volatility (30-day)","Moving Averages"], horizontal=True, key="ct")
        fast_ma, slow_ma = 20, 50
        if chart_type == "Moving Averages":
            ma1,ma2,note = st.columns([1,1,3])
            with ma1: fast_ma = st.selectbox("Fast MA",[10,20,50],index=1)
            with ma2: slow_ma = st.selectbox("Slow MA",[50,100,200],index=2)
            with note:
                st.markdown(f"""<div style='background:rgba(217,119,6,0.06);border:1px solid rgba(217,119,6,0.18);border-radius:7px;padding:5px 10px;margin-top:5px;font-size:0.68rem;color:#334155'>
                ✨ <b style='color:#B45309'>Golden</b> SMA{fast_ma} > SMA{slow_ma} → <span style='color:#0E7A5E'>Bullish</span>
                &nbsp;|&nbsp; ☠️ <b style='color:#DC2626'>Death</b> SMA{fast_ma} < SMA{slow_ma} → <span style='color:#DC2626'>Bearish</span>
                </div>""", unsafe_allow_html=True)
        if bool(sel_tickers) or bool(sel_sectors):
            fig = plot_market_chart(all_data, chart_type=chart_type,
                                    tickers=sel_tickers or None,
                                    sectors=sel_sectors if not sel_tickers else None,
                                    date_range=sel_date_range, fast_ma=fast_ma, slow_ma=slow_ma)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        else:
            st.markdown("<div style='height:220px;display:flex;align-items:center;justify-content:center;color:#94A3B8;font-size:0.82rem'>Select a sector to get started</div>", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> TOP PERFORMERS</div><div class='card-body'>", unsafe_allow_html=True)
        # Build a filtered metrics df based on yr_range for the rankings chart
        filtered_metrics = metrics_df.copy()
        if not period_returns.empty:
            filtered_metrics = filtered_metrics.copy()
            filtered_metrics['Cumulative_Return_%'] = filtered_metrics['Ticker'].map(period_returns).fillna(0)
        st.plotly_chart(plot_sector_rankings(filtered_metrics), use_container_width=True, config={'displayModeBar':False})
        st.markdown("</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STOCK INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Stock Intelligence":
    st.markdown("""<div class='page-header'><div>
        <div class='page-title'>🔮 Stock Price Prediction</div>
        <div class='page-subtitle'>AI-powered price forecasting</div>
    </div></div>""", unsafe_allow_html=True)

    # ── Discover test date range from first available ticker ─────────────────
    @st.cache_data
    def get_test_date_range(_lstm_results, _zs_results, _ft_results, _best_models, tickers):
        """Find the min/max dates available across all test predictions."""
        all_mins, all_maxs = [], []
        for t in tickers[:20]:
            p = get_prediction_for_date(t, pd.Timestamp('2025-06-01'), _lstm_results,
                                        _zs_results, _ft_results, _best_models)
            if p and p.get('test_date_min') is not None:
                all_mins.append(p['test_date_min'])
                all_maxs.append(p['test_date_max'])
        if all_mins:
            return pd.to_datetime(min(all_mins)), pd.to_datetime(max(all_maxs))
        return pd.Timestamp('2025-01-01'), pd.Timestamp('2025-11-30')

    test_min, test_max = get_test_date_range(
        lstm_results, zs_results, ft_results, best_models, all_tickers)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1,ctrl2,ctrl3,ctrl4 = st.columns([2,1.2,1.2,0.9])
    with ctrl1:
        sel_ticker = st.selectbox("Stock", all_tickers)
    with ctrl2:
        start_date = st.date_input("Start Date",
                                   value=pd.Timestamp('2023-01-01').date(),
                                   min_value=pd.Timestamp('2015-01-01').date(),
                                   max_value=pd.Timestamp('2023-12-31').date())
    with ctrl3:
        pred_date  = st.date_input("Prediction Date",
                                   value=min(pd.Timestamp('2025-06-30').date(), test_max.date()),
                                   min_value=test_min.date(),
                                   max_value=test_max.date())
    with ctrl4:
        st.markdown("<div style='padding-top:1.45rem'>", unsafe_allow_html=True)
        run_btn = st.button("🔮 Analyse", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        with st.spinner("Running analysis…"):
            # Step 1: get model predicted_price at pred_date
            prediction = get_prediction_for_date(
                sel_ticker, pred_date, lstm_results, zs_results, ft_results,
                best_models, start_date=start_date)

            # Step 2: override current_price from FULL historical data at start_date
            # (test set only covers 2024-2025, so y_true lookup would fail for older dates)
            if prediction:
                actual_start = get_price_at_date(all_data, sel_ticker, start_date)
                if actual_start:
                    prediction['current_price']     = round(actual_start, 2)
                    prediction['expected_return_%'] = round(
                        (prediction['predicted_price'] - actual_start) / actual_start * 100, 2)

            # Market-wide predictions using same logic
            all_preds = {}
            for t in all_tickers[:30]:
                p = get_prediction_for_date(t, pred_date, lstm_results, zs_results,
                                            ft_results, best_models, start_date=start_date)
                if p:
                    actual_s = get_price_at_date(all_data, t, start_date)
                    if actual_s:
                        p['current_price']     = round(actual_s, 2)
                        p['expected_return_%'] = round(
                            (p['predicted_price'] - actual_s) / actual_s * 100, 2)
                    all_preds[t] = p

            st.session_state.last_pred = {
                'ticker': sel_ticker, 'pred_date': pred_date,
                'start_date': start_date, 'prediction': prediction,
                'all_preds': all_preds
            }
            st.session_state['_p2_ticker'] = sel_ticker
            st.session_state['_p2_pred']   = pred_date
            st.session_state['_p2_start']  = start_date

    lp = st.session_state.last_pred
    if lp:
        prediction  = lp['prediction']
        sel_ticker  = lp['ticker']
        pred_date   = lp['pred_date']
        start_date  = lp['start_date']
        all_preds   = lp.get('all_preds', {})
        model_used  = best_models.get(sel_ticker, 'LSTM')

        # Warn if current widget values differ from last analysis
        current_ticker = st.session_state.get('_p2_ticker', sel_ticker)
        current_pred   = st.session_state.get('_p2_pred', pred_date)
        current_start  = st.session_state.get('_p2_start', start_date)
        if (current_ticker != sel_ticker or current_pred != pred_date
                or current_start != start_date):
            st.warning("⚠️ Controls have changed — click **🔮 Analyse** to update all cards and charts.")

        # ── Compute insight data ──────────────────────────────────────────────
        # Sector scores — always live from metrics_df (no model needed)
        sector_scores = (
            metrics_df.loc[metrics_df.groupby('Ticker')['Cumulative_Return_%'].idxmax()]
            .groupby(metrics_df['Ticker'].map(ticker_sector_map))['Cumulative_Return_%']
            .mean().sort_values(ascending=False)
        )

        # Top 3 directional — always live from metrics_df
        dir_df = metrics_df.loc[metrics_df.groupby('Ticker')['Directional_Accuracy'].idxmax()]
        dir_df = dir_df.sort_values('Directional_Accuracy', ascending=False).head(3).copy()

        def get_direction(ticker):
            p = get_prediction_for_date(ticker, pred_date, lstm_results, zs_results,
                                        ft_results, best_models, start_date=start_date)
            if p: return p['expected_return_%']
            return 0.0

        dir_df['exp_ret'] = dir_df['Ticker'].apply(get_direction)
        best_sector = sector_scores.index[0] if not sector_scores.empty else 'N/A'

        try:
            best_t  = max(all_preds, key=lambda t: all_preds[t]['expected_return_%'])
            worst_t = min(all_preds, key=lambda t: all_preds[t]['expected_return_%'])
            best_r  = all_preds[best_t]['expected_return_%']
            worst_r = all_preds[worst_t]['expected_return_%']
        except Exception:
            best_t = worst_t = 'N/A'; best_r = worst_r = 0.0

        # ── 3 Insight cards (removed "Most Promising Sector" — in chart below) ─
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            arrow = "▲" if best_r >= 0 else "▼"
            col   = "#0E7A5E" if best_r >= 0 else "#DC2626"
            st.markdown(f"""<div class='insight-card'>
                <div class='insight-label'>🚀 Best Pick · {str(pred_date)}</div>
                <div class='insight-value'>{best_t}</div>
                <div class='insight-sub' style='color:{col};font-weight:600'>{arrow} {best_r:+.2f}% expected</div>
            </div>""", unsafe_allow_html=True)
        with ic2:
            arrow = "▼" if worst_r <= 0 else "▲"
            col   = "#DC2626" if worst_r <= 0 else "#0E7A5E"
            st.markdown(f"""<div class='insight-card'>
                <div class='insight-label'>⚠️ Worst Pick · {str(pred_date)}</div>
                <div class='insight-value'>{worst_t}</div>
                <div class='insight-sub' style='color:{col};font-weight:600'>{arrow} {worst_r:+.2f}% expected</div>
            </div>""", unsafe_allow_html=True)
        with ic3:
            # Directional top 3 — show as up/down labels
            dir_html = ""
            for _, row in dir_df.iterrows():
                if row['exp_ret'] >= 0:
                    dir_html += f"<div class='dir-row'><span class='dir-ticker'>{row['Ticker']}</span><span class='dir-up'>▲</span><span class='dir-label-up'>Strongly Bullish</span></div>"
                else:
                    dir_html += f"<div class='dir-row'><span class='dir-ticker'>{row['Ticker']}</span><span class='dir-down'>▼</span><span class='dir-label-down'>Strongly Bearish</span></div>"
            st.markdown(f"""<div class='insight-card'>
                <div class='insight-label'>🎯 Top Directional Picks</div>
                {dir_html}
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:0.4rem 0'></div>", unsafe_allow_html=True)

        # ── Main layout: chart | signal+stats | sector ──────────────────────
        chart_col, signal_col, sector_col = st.columns([2.4, 1, 1], gap="medium")

        with chart_col:
            st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> PRICE & VOLUME</div><div class='card-body'>", unsafe_allow_html=True)
            st.plotly_chart(plot_price_volume(all_data, sel_ticker, start_date, pred_date),
                            use_container_width=True, config={'displayModeBar':False})

            # 52-week strip
            td = all_data[all_data['Ticker']==sel_ticker].copy()
            td['Date'] = pd.to_datetime(td['Date'])
            wk52 = td[td['Date'] >= (pd.to_datetime(pred_date) - pd.DateOffset(weeks=52))]
            wk52_high = wk52['Close'].max() if not wk52.empty else 0
            wk52_low  = wk52['Close'].min() if not wk52.empty else 0
            cur_p  = prediction['current_price']   if prediction else 0
            pred_p = prediction['predicted_price'] if prediction else 0
            st.markdown(f"""<div class='wk52-bar'>
                <div class='wk52-item'><div class='wk52-lbl'>52-Wk Low</div><div class='wk52-val wk52-low'>KES {wk52_low:,.2f}</div></div>
                <div style='color:#E2E8F0;font-size:1rem'>|</div>
                <div class='wk52-item'><div class='wk52-lbl'>Current Price</div><div class='wk52-val wk52-cur'>KES {cur_p:,.2f}</div></div>
                <div style='color:#E2E8F0;font-size:1rem'>|</div>
                <div class='wk52-item'><div class='wk52-lbl'>Predicted Price</div><div class='wk52-val wk52-pred'>KES {pred_p:,.2f}</div></div>
                <div style='color:#E2E8F0;font-size:1rem'>|</div>
                <div class='wk52-item'><div class='wk52-lbl'>52-Wk High</div><div class='wk52-val wk52-high'>KES {wk52_high:,.2f}</div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with signal_col:
            if prediction:
                ret    = prediction['expected_return_%']
                signal = get_trade_signal(ret)
                if 'BUY' in signal:
                    cls, icon, lbl = 'signal-buy','📈','BUY'
                    sub = f"+{ret:.2f}% expected gain"
                elif 'SELL' in signal:
                    cls, icon, lbl = 'signal-sell','📉','SELL'
                    sub = f"{ret:.2f}% expected loss"
                else:
                    cls, icon, lbl = 'signal-hold','⏸️','HOLD'
                    sub = f"{ret:+.2f}% expected move"

                ci_lo = prediction.get('confidence_lower') or 0
                ci_hi = prediction.get('confidence_upper') or 0

                st.markdown(f"""<div class='card'>
                  <div class='card-header'><span class='card-accent'>◈</span> SIGNAL</div>
                  <div class='card-body'>
                    <div class='{cls}'>
                      <div class='si'>{icon}</div>
                      <div class='sl'>{lbl}</div>
                      <div class='ss'>{sub}</div>
                    </div>
                    <div style='margin-top:8px;font-size:0.68rem;color:#64748B;text-align:center'>
                      Best Model: <b style='color:#0E7A5E'>{model_used}</b>
                    </div>
                    <div style='text-align:center;margin-top:4px'>
                      <span class='ci-pill'>95% CI: {ci_lo:,.0f} – {ci_hi:,.0f}</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                ret_col = "normal" if ret >= 0 else "inverse"
                st.metric("Expected Return", f"{ret:+.2f}%", delta_color=ret_col,
                          delta=f"{'▲' if ret>=0 else '▼'} vs start date")
                price_diff = prediction['predicted_price'] - prediction['current_price']
                st.metric("Price Move",
                          f"KES {prediction['current_price']:,.2f} → {prediction['predicted_price']:,.2f}",
                          delta=f"{price_diff:+.2f}",
                          delta_color="normal" if price_diff >= 0 else "inverse")

        with sector_col:
            st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> SECTOR OUTLOOK</div><div class='card-body'>", unsafe_allow_html=True)
            st.plotly_chart(plot_sector_promise(sector_scores),
                            use_container_width=True, config={'displayModeBar':False})
            st.markdown("</div></div>", unsafe_allow_html=True)

    else:
        st.markdown("""<div style='height:260px;display:flex;flex-direction:column;align-items:center;
        justify-content:center;gap:0.6rem;color:#94A3B8'>
            <div style='font-size:2.2rem'>🔮</div>
            <div style='font-size:0.95rem;font-weight:600;color:#64748B'>Select a stock, set date range, click Analyse</div>
            <div style='font-size:0.78rem'>Start Date → Prediction Date defines your analysis window</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO MANAGER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio Manager":
    st.markdown("""<div class='page-header'><div>
        <div class='page-title'>💼 Portfolio Manager</div>
        <div class='page-subtitle'>Holdings tracker with predicted value and risk horizons</div>
    </div></div>""", unsafe_allow_html=True)

    lp = st.session_state.last_pred

    # ── ROW 1: Valuation Date + 3 summary metrics — 4 equal columns ──────────
    vd_col, sm1, sm2, sm3 = st.columns(4)
    with vd_col:
        port_pred_date = st.date_input("Valuation Date",
                                       value=pd.Timestamp('2025-06-30').date(),
                                       min_value=pd.Timestamp('2024-01-01').date(),
                                       max_value=pd.Timestamp('2025-12-31').date())

    # ── Compute portfolio rows FIRST so metrics are available ─────────────────
    port_df = None
    if st.session_state.portfolio:
        rows = []
        for h in st.session_state.portfolio:
            t        = h['ticker']
            buy_date = h.get('buy_date')
            pred = get_prediction_for_date(t, port_pred_date, lstm_results, zs_results,
                                           ft_results, best_models, start_date=buy_date)
            if pred:
                actual_buy = get_price_at_date(all_data, t, buy_date)
                if actual_buy:
                    pred['current_price']     = round(actual_buy, 2)
                    pred['expected_return_%'] = round(
                        (pred['predicted_price'] - actual_buy) / actual_buy * 100, 2)
            cur_p  = pred['current_price']   if pred else h['buy_price']
            pred_p = pred['predicted_price'] if pred else h['buy_price']
            exp_r  = pred['expected_return_%'] if pred else 0.0
            cur_v  = cur_p  * h['qty']
            pred_v = pred_p * h['qty']
            cost   = h['buy_price'] * h['qty']
            val_price = get_price_at_date(all_data, t, port_pred_date) or cur_p
            unreal    = round((val_price * h['qty']) - cost, 2)
            date_7d   = pd.to_datetime(port_pred_date) + pd.Timedelta(days=7)
            date_30d  = pd.to_datetime(port_pred_date) + pd.Timedelta(days=30)
            pred_7d   = get_prediction_for_date(t, date_7d,  lstm_results, zs_results, ft_results, best_models, start_date=buy_date)
            pred_30d  = get_prediction_for_date(t, date_30d, lstm_results, zs_results, ft_results, best_models, start_date=buy_date)
            actual_buy = get_price_at_date(all_data, t, buy_date) or h['buy_price']
            if pred_7d:  pred_7d['expected_return_%']  = round((pred_7d['predicted_price']  - actual_buy) / actual_buy * 100, 2)
            if pred_30d: pred_30d['expected_return_%'] = round((pred_30d['predicted_price'] - actual_buy) / actual_buy * 100, 2)
            ret_7d  = pred_7d['expected_return_%']  if pred_7d  else exp_r
            ret_30d = pred_30d['expected_return_%'] if pred_30d else exp_r
            rows.append({
                'Ticker': t, 'Qty': h['qty'], 'Buy Price': h['buy_price'],
                'Current Price': round(cur_p,2), 'Predicted Price': round(pred_p,2),
                'Cost Basis': round(cost,2), 'Current_Value': round(cur_v,2),
                'Predicted_Value': round(pred_v,2), 'Unrealised P&L': round(unreal,2),
                'Expected Return %': round(exp_r,2),
                'ret_pred': round(exp_r,2), 'ret_7d': round(ret_7d,2), 'ret_30d': round(ret_30d,2)
            })
        port_df     = pd.DataFrame(rows)
        total_cost  = port_df['Cost Basis'].sum()
        total_cur   = port_df['Current_Value'].sum()
        total_pred  = port_df['Predicted_Value'].sum()
        total_pnl   = total_cur - total_cost
        total_pnl_p = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        pred_up     = ((total_pred - total_cur) / total_cur * 100) if total_cur > 0 else 0

        with sm1: st.metric("Total Invested",  f"KES {total_cost:,.0f}")
        with sm2: st.metric("Current Value",   f"KES {total_cur:,.0f}",
                            delta=f"{total_pnl_p:+.1f}%", delta_color=delta_color(total_pnl_p))
        with sm3: st.metric("Predicted Value", f"KES {total_pred:,.0f}",
                            delta=f"{pred_up:+.1f}%", delta_color=delta_color(pred_up))
    else:
        with sm1: st.metric("Total Invested",  "KES 0")
        with sm2: st.metric("Current Value",   "KES 0")
        with sm3: st.metric("Predicted Value", "KES 0")

    st.markdown("<div style='margin:0.3rem 0'></div>", unsafe_allow_html=True)

    # ── ROW 2: LEFT (stock intel + holdings + bar chart) | RIGHT (add stocks + risk horizon) ──
    left_col, right_col = st.columns([1.55, 1], gap="medium")

    # ════════════ LEFT COLUMN ════════════
    with left_col:

        # ── FROM STOCK INTELLIGENCE card ──
        st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> FROM STOCK INTELLIGENCE</div><div class='card-body' style='padding-bottom:0.6rem'>", unsafe_allow_html=True)
        if lp and lp['prediction']:
            p     = lp['prediction']
            buy_p = get_price_at_date(all_data, lp['ticker'], lp['start_date'])
            exp_ret = p['expected_return_%']

            # TOP ROW: 3 equal info columns
            i1, i2, i3 = st.columns(3)
            with i1:
                st.markdown(f"<div style='font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#94A3B8;margin-bottom:2px'>Stock · Qty</div>", unsafe_allow_html=True)
                tc, qc = st.columns([1, 1])
                with tc: st.markdown(f"<div style='font-family:Sora,sans-serif;font-weight:700;font-size:1rem;color:#0F172A;padding-top:6px'>{lp['ticker']}</div>", unsafe_allow_html=True)
                with qc: auto_qty = st.number_input("Qty", min_value=1, value=100, step=10, key="auto_qty", label_visibility="collapsed")
            with i2:
                st.metric("Buy Price", f"KES {buy_p:,.2f}" if buy_p else "N/A")
            with i3:
                st.metric("Expected Return", f"{exp_ret:+.2f}%",
                          delta_color="normal" if exp_ret >= 0 else "inverse",
                          delta=f"{'▲' if exp_ret>=0 else '▼'}")

            # BOTTOM ROW: 2 equal columns — button | status (spans same width as 3 above)
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("<div style='padding-top:0.4rem'>", unsafe_allow_html=True)
                if st.button(f"⚡ Add {lp['ticker']} to Portfolio", type="primary", use_container_width=True):
                    bp = buy_p or p['current_price']
                    st.session_state.portfolio = [h for h in st.session_state.portfolio if h['ticker'] != lp['ticker']]
                    st.session_state.portfolio.append({'ticker': lp['ticker'], 'qty': auto_qty, 'buy_price': round(bp,2), 'buy_date': str(lp['start_date'])})
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with b2:
                st.markdown("<div style='padding-top:0.4rem'>", unsafe_allow_html=True)
                if lp['ticker'] in [h['ticker'] for h in st.session_state.portfolio]:
                    st.markdown(f"""<div style='background:rgba(14,122,94,0.08);border:1px solid
                    rgba(14,122,94,0.25);border-radius:8px;padding:7px 10px;font-size:0.75rem;
                    color:#065F46;font-weight:600;text-align:center;'>
                    ✅ {lp['ticker']} added to portfolio</div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.78rem;color:#94A3B8;padding:0.4rem 0'>Run an analysis on Stock Price Prediction first to auto-import a stock.</div>", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # ── HOLDINGS TABLE + BAR CHART (only when portfolio has data) ──
        if port_df is not None:
            st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> HOLDINGS</div><div class='card-body'>", unsafe_allow_html=True)
            disp = ['Ticker','Qty','Buy Price','Current Price','Predicted Price',
                    'Current_Value','Predicted_Value','Expected Return %']
            st.dataframe(
                port_df[disp].style.format({
                    'Buy Price':'KES {:,.2f}', 'Current Price':'KES {:,.2f}',
                    'Predicted Price':'KES {:,.2f}', 'Current_Value':'KES {:,.2f}',
                    'Predicted_Value':'KES {:,.2f}', 'Expected Return %':'{:+.2f}%',
                }),
                use_container_width=True, height=150
            )
            st.plotly_chart(plot_portfolio_bar(port_df), use_container_width=True,
                            config={'displayModeBar': False})
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ════════════ RIGHT COLUMN ════════════
    with right_col:

        # ── ADD UP TO 3 MORE STOCKS card ──
        st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> ADD UP TO 3 MORE STOCKS</div><div class='card-body'>", unsafe_allow_html=True)
        port_start = st.date_input("Start Date",
                                   value=pd.Timestamp('2024-01-01').date(),
                                   min_value=pd.Timestamp('2015-01-01').date(),
                                   max_value=pd.Timestamp('2025-11-30').date(),
                                   key="port_start")
        for i in range(3):
            rc1, rc2 = st.columns([2.2, 1])
            with rc1:
                t = st.selectbox(f"Stock {i+1}", ["— none —"] + all_tickers,
                                 key=f"es_ticker_{i}", label_visibility="collapsed")
            with rc2:
                q = st.number_input("Qty", min_value=1, value=100, step=10,
                                    key=f"es_qty_{i}", label_visibility="collapsed")
            st.session_state.extra_stocks[i] = {'ticker': t, 'qty': q}
        if st.button("＋ Add Stocks", type="primary", use_container_width=True):
            selected = [es.get('ticker','') for es in st.session_state.extra_stocks
                        if es.get('ticker','') not in ('','— none —')]
            prev_extra = st.session_state.get('extra_slot_tickers', [])
            removed = [tk for tk in prev_extra if tk not in selected]
            st.session_state.portfolio = [h for h in st.session_state.portfolio if h['ticker'] not in removed]
            added = 0
            for es in st.session_state.extra_stocks:
                t = es.get('ticker','')
                if t and t != '— none —':
                    bp = get_price_at_date(all_data, t, port_start) or 100.0
                    st.session_state.portfolio = [h for h in st.session_state.portfolio if h['ticker'] != t]
                    st.session_state.portfolio.append({'ticker':t,'qty':es['qty'],'buy_price':round(bp,2),'buy_date':str(port_start)})
                    added += 1
            st.session_state.extra_slot_tickers = selected
            if added: st.success(f"✅ {added} stock(s) added at {port_start} prices")
        st.markdown("</div></div>", unsafe_allow_html=True)

        # ── RISK HORIZON card ──
        if port_df is not None:
            st.markdown("<div class='card'><div class='card-header'><span class='card-accent'>◈</span> RISK HORIZON</div><div class='card-body'>", unsafe_allow_html=True)
            st.plotly_chart(plot_risk_horizon(port_df), use_container_width=True,
                            config={'displayModeBar': False})
            at_risk = port_df[port_df['ret_pred'] < -2]
            if not at_risk.empty:
                st.markdown(f"""<div style='background:rgba(220,38,38,0.05);border:1px solid rgba(220,38,38,0.18);
                border-radius:7px;padding:4px 8px;font-size:0.68rem;color:#991B1B;margin-top:4px'>
                ⚠️ <b>At risk:</b> {', '.join(at_risk['Ticker'].tolist())}</div>""", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # ── Clear button ──
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state.portfolio = []
            st.rerun()