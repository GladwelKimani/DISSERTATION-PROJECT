import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── PKL Paths ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

LSTM_PKL = os.path.join(BASE_DIR, 'notebooks', 'outputs', 'results', 'lstm_all_results.pkl')
ZS_PKL   = os.path.join(BASE_DIR, 'lag-llama', 'outputs', 'results', 'lagllama_zs_rolling.pkl')
FT_PKL   = os.path.join(BASE_DIR, 'lag-llama', 'outputs', 'results', 'lagllama_ft_rolling.pkl')

# ── Load Saved Results ────────────────────────────────────────────────────────

def load_lstm_results():
    if not os.path.exists(LSTM_PKL):
        raise FileNotFoundError(f"LSTM results not found at {LSTM_PKL}")
    results = joblib.load(LSTM_PKL)
    print(f"Loaded LSTM results: {len(results)} tickers")
    return results

def load_lagllama_zeroshot():
    if not os.path.exists(ZS_PKL):
        raise FileNotFoundError(f"LagLlama zero-shot results not found at {ZS_PKL}")
    results = joblib.load(ZS_PKL)
    print(f"Loaded LagLlama zero-shot results: {len(results)} tickers")
    return results

def load_lagllama_finetuned():
    if not os.path.exists(FT_PKL):
        raise FileNotFoundError(f"LagLlama fine-tuned results not found at {FT_PKL}")
    results = joblib.load(FT_PKL)
    print(f"Loaded LagLlama fine-tuned results: {len(results)} tickers")
    return results

# ── Helper: Extract LSTM actual/predicted safely ──────────────────────────────

def get_lstm_arrays(result):
    """
    Safely extract actual and predicted arrays from LSTM result dict.
    Handles both 'actual_kes'/'predicted_kes' and 'actual'/'predicted' keys.
    """
    p = result.get('predictions', {}).get('test', {})

    if 'actual_kes' in p and 'predicted_kes' in p:
        return np.array(p['actual_kes']), np.array(p['predicted_kes']), p.get('dates', [])

    if 'actual' in p and 'predicted' in p:
        return np.array(p['actual']), np.array(p['predicted']), p.get('dates', [])

    if 'actual' in result and 'predicted' in result:
        return np.array(result['actual']), np.array(result['predicted']), result.get('dates', [])

    raise KeyError(f"Cannot find actual/predicted arrays in LSTM result. Keys: {list(result.keys())}")

# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_statistical_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if len(y_true) > 1:
        da = np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0))
    else:
        da = np.nan
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'R2': round(r2, 4), 'Directional_Accuracy': round(da, 4)}

def calculate_financial_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    pred_returns   = np.diff(y_pred) / (y_pred[:-1] + 1e-8)
    actual_returns = np.diff(y_true) / (y_true[:-1] + 1e-8)
    signals          = np.sign(pred_returns)
    strategy_returns = signals * actual_returns
    cumulative_return = (np.prod(1 + strategy_returns) - 1) * 100
    volatility = np.std(strategy_returns) * np.sqrt(252) * 100
    sharpe = (np.mean(strategy_returns) * 252) / (np.std(strategy_returns) * np.sqrt(252) + 1e-8)
    return {
        'Sharpe_Ratio': round(sharpe, 4),
        'Cumulative_Return_%': round(cumulative_return, 4),
        'Volatility_%': round(volatility, 4),
    }

# ── Best Model Selection ──────────────────────────────────────────────────────

def get_best_model_per_ticker(lstm_results, zs_results, ft_results):
    best_models = {}
    all_tickers = set(lstm_results.keys()) | set(zs_results.keys()) | set(ft_results.keys())
    for ticker in all_tickers:
        scores = {}
        if ticker in lstm_results:
            try:
                y_true, y_pred, _ = get_lstm_arrays(lstm_results[ticker])
                stat = calculate_statistical_metrics(y_true, y_pred)
                fin  = calculate_financial_metrics(y_true, y_pred)
                scores['LSTM'] = (stat['R2'] + stat['Directional_Accuracy'] + max(fin['Sharpe_Ratio'], 0)) / 3
            except Exception:
                pass
        if ticker in zs_results:
            try:
                r    = zs_results[ticker]
                stat = calculate_statistical_metrics(r['actual'], r['predicted'])
                fin  = calculate_financial_metrics(r['actual'], r['predicted'])
                scores['LagLlama_ZeroShot'] = (stat['R2'] + stat['Directional_Accuracy'] + max(fin['Sharpe_Ratio'], 0)) / 3
            except Exception:
                pass
        if ticker in ft_results:
            try:
                r    = ft_results[ticker]
                stat = calculate_statistical_metrics(r['actual'], r['predicted'])
                fin  = calculate_financial_metrics(r['actual'], r['predicted'])
                scores['LagLlama_FineTuned'] = (stat['R2'] + stat['Directional_Accuracy'] + max(fin['Sharpe_Ratio'], 0)) / 3
            except Exception:
                pass
        if scores:
            best_models[ticker] = max(scores, key=scores.get)
    return best_models

# ── Prediction for a Target Date ──────────────────────────────────────────────

def get_prediction_for_date(ticker, target_date, lstm_results, zs_results, ft_results,
                            best_models, start_date=None):
    """
    Returns a prediction dict for ticker at target_date.

    current_price  = actual price at start_date (or earliest test date if start_date
                     is outside the test window)
    predicted_price = model's predicted price at target_date
    expected_return = (predicted_price - current_price) / current_price * 100

    This correctly measures the price appreciation from start_date to target_date
    as forecast by the model.
    """
    target_date = pd.to_datetime(target_date)
    start_ts    = pd.to_datetime(start_date) if start_date is not None else None
    model_name  = best_models.get(ticker, 'LSTM')
    try:
        if model_name == 'LSTM' and ticker in lstm_results:
            y_true, y_pred, dates = get_lstm_arrays(lstm_results[ticker])
            dates = pd.to_datetime(dates)
            if len(dates) == 0:
                return None

            # predicted_price: model output closest to target_date
            pred_idx        = np.argmin(np.abs(dates - target_date))
            predicted_price = float(y_pred[pred_idx])

            # current_price: actual price closest to start_date
            # if no start_date supplied, use actual at the pred date (old behaviour)
            if start_ts is not None:
                start_idx     = np.argmin(np.abs(dates - start_ts))
                current_price = float(y_true[start_idx])
            else:
                current_price = float(y_true[pred_idx])

            std_err = np.std(y_pred - y_true[:len(y_pred)])
            test_min = dates.min()
            test_max = dates.max()

        elif model_name in ('LagLlama_ZeroShot', 'LagLlama_FineTuned'):
            r = zs_results.get(ticker) if model_name == 'LagLlama_ZeroShot' \
                else ft_results.get(ticker)
            if not r:
                return None
            y_true = np.array(r['actual'])
            y_pred = np.array(r['predicted'])
            dates  = pd.to_datetime(r.get('dates', []))
            if len(dates) == 0:
                return None

            pred_idx        = np.argmin(np.abs(dates - target_date))
            predicted_price = float(y_pred[pred_idx])

            if start_ts is not None:
                start_idx     = np.argmin(np.abs(dates - start_ts))
                current_price = float(y_true[start_idx])
            else:
                current_price = float(y_true[pred_idx])

            std_err  = np.std(y_pred - y_true[:len(y_pred)])
            test_min = dates.min()
            test_max = dates.max()
        else:
            return None

        if current_price == 0:
            return None

        lower           = predicted_price - 1.96 * std_err
        upper           = predicted_price + 1.96 * std_err
        expected_return = ((predicted_price - current_price) / current_price) * 100

        return {
            'ticker':            ticker,
            'model_used':        model_name,
            'current_price':     round(current_price, 2),
            'predicted_price':   round(predicted_price, 2),
            'expected_return_%': round(expected_return, 2),
            'confidence_lower':  round(lower, 2),
            'confidence_upper':  round(upper, 2),
            'test_date_min':     test_min,
            'test_date_max':     test_max,
        }
    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None

# ── Trade Signal ──────────────────────────────────────────────────────────────

def get_trade_signal(expected_return, threshold=2.0):
    if expected_return >= threshold:
        return '🟢 BUY'
    elif expected_return <= -threshold:
        return '🔴 SELL'
    else:
        return '🟡 HOLD'

# ── Build Full Metrics Table ──────────────────────────────────────────────────

def build_metrics_table(lstm_results, zs_results, ft_results):
    rows = []
    for ticker, r in lstm_results.items():
        try:
            y_true, y_pred, _ = get_lstm_arrays(r)
            stat = calculate_statistical_metrics(y_true, y_pred)
            fin  = calculate_financial_metrics(y_true, y_pred)
            rows.append({'Ticker': ticker, 'Model': 'LSTM', **stat, **fin})
        except Exception as e:
            print(f"Skipping LSTM {ticker}: {e}")
    for ticker, r in zs_results.items():
        try:
            stat = calculate_statistical_metrics(r['actual'], r['predicted'])
            fin  = calculate_financial_metrics(r['actual'], r['predicted'])
            rows.append({'Ticker': ticker, 'Model': 'LagLlama_ZeroShot', **stat, **fin})
        except Exception as e:
            print(f"Skipping ZS {ticker}: {e}")
    for ticker, r in ft_results.items():
        try:
            stat = calculate_statistical_metrics(r['actual'], r['predicted'])
            fin  = calculate_financial_metrics(r['actual'], r['predicted'])
            rows.append({'Ticker': ticker, 'Model': 'LagLlama_FineTuned', **stat, **fin})
        except Exception as e:
            print(f"Skipping FT {ticker}: {e}")
    return pd.DataFrame(rows)

# ── Portfolio Optimizer ───────────────────────────────────────────────────────

def get_best_portfolio(metrics_df, top_n=5):
    best = metrics_df.loc[metrics_df.groupby('Ticker')['Sharpe_Ratio'].idxmax()].copy()
    best['Score'] = (
        best['Sharpe_Ratio'] * 0.4 +
        best['Cumulative_Return_%'] * 0.01 * 0.4 +
        best['Directional_Accuracy'] * 0.2
    )
    top = best.nlargest(top_n, 'Score')[
        ['Ticker', 'Model', 'Sharpe_Ratio', 'Cumulative_Return_%',
         'Directional_Accuracy', 'MAE', 'RMSE', 'Score']
    ].reset_index(drop=True)
    return top