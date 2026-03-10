import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STOCK_COLOURS = [
    '#0E7A5E','#B45309','#1D4ED8','#BE185D','#6D28D9',
    '#065F46','#C2410C','#0369A1','#86198F','#92400E',
    '#14532D','#991B1B','#3730A3','#0F766E','#9F1239',
    '#166534','#78350F','#1E40AF','#831843','#4C1D95',
]

TH = {
    'bg':'#FFFFFF','paper':'#FFFFFF',
    'grid':'rgba(0,0,0,0.05)','text':'#64748B','text_lt':'#1E293B',
    'border':'rgba(0,0,0,0.08)','zero':'rgba(0,0,0,0.15)',
    'accent':'#0E7A5E','gold':'#B45309','red':'#DC2626',
    'pos_bar':'#0E7A5E','neg_bar':'#DC2626',
    'ci_fill':'rgba(180,83,9,0.08)','bar_base':'#DBEAFE',
}

def _layout(title='', height=340, showlegend=True,
            xaxis_title='', yaxis_title='', hovermode='closest', **extra):
    base = dict(
        title=dict(text=title,
                   font=dict(size=12, color=TH['text_lt'], family='Sora, sans-serif'), x=0.01),
        template='plotly_white',
        font=dict(family='Plus Jakarta Sans, sans-serif', size=11, color=TH['text']),
        plot_bgcolor=TH['bg'], paper_bgcolor=TH['paper'],
        height=height, margin=dict(l=46, r=16, t=36, b=36),
        hovermode=hovermode,
        legend=dict(
            font=dict(size=9, color=TH['text']),
            bgcolor='rgba(0,0,0,0.03)', bordercolor=TH['border'], borderwidth=1,
            orientation='v', yanchor='top', y=1, xanchor='left', x=1.01,
        ) if showlegend else dict(visible=False),
        xaxis=dict(title=xaxis_title, gridcolor=TH['grid'], zeroline=False,
                   tickfont=dict(size=9, color=TH['text']), linecolor=TH['border']),
        yaxis=dict(title=yaxis_title, gridcolor=TH['grid'], zeroline=False,
                   tickfont=dict(size=9, color=TH['text']), linecolor=TH['border']),
    )
    base.update(extra)
    return base


def filter_data(all_data, tickers=None, sectors=None, date_range=None):
    df = all_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if tickers:   df = df[df['Ticker'].isin(tickers)]
    elif sectors: df = df[df['Sector'].isin(sectors)]
    if date_range:
        df = df[(df['Date'] >= pd.to_datetime(date_range[0])) &
                (df['Date'] <= pd.to_datetime(date_range[1]))]
    return df


# ── PAGE 1 ────────────────────────────────────────────────────────────────────
def plot_market_chart(all_data, chart_type='Closing Prices', tickers=None,
                      sectors=None, date_range=None, fast_ma=20, slow_ma=50):
    df   = filter_data(all_data, tickers=tickers, sectors=sectors, date_range=date_range)
    uniq = df['Ticker'].unique()
    cmap = {t: STOCK_COLOURS[i % len(STOCK_COLOURS)] for i, t in enumerate(uniq)}
    fig  = go.Figure()
    yaxis_title = title = ''

    for ticker in uniq:
        td = df[df['Ticker'] == ticker].sort_values('Date').copy()
        c  = cmap[ticker]

        if chart_type == 'Closing Prices':
            fig.add_trace(go.Scatter(x=td['Date'], y=td['Close'], mode='lines',
                name=ticker, line=dict(width=1.8, color=c),
                hovertemplate=f'<b>{ticker}</b><br>%{{x|%Y-%m-%d}}: KES %{{y:,.2f}}<extra></extra>'))
            yaxis_title, title = 'Price (KES)', 'Closing Prices'

        elif chart_type == 'Cumulative Returns':
            td['dr'] = td['Close'].pct_change()
            td['cr'] = (1 + td['dr']).cumprod() - 1
            fig.add_trace(go.Scatter(x=td['Date'], y=td['cr'] * 100, mode='lines',
                name=ticker, line=dict(width=1.8, color=c),
                hovertemplate=f'<b>{ticker}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>'))
            yaxis_title, title = 'Cumulative Return (%)', 'Cumulative Returns'

        elif chart_type == 'Volatility (30-day)':
            td['dr']  = td['Close'].pct_change()
            td['vol'] = td['dr'].rolling(30).std() * np.sqrt(252) * 100
            fig.add_trace(go.Scatter(x=td['Date'], y=td['vol'], mode='lines',
                name=ticker, line=dict(width=1.8, color=c),
                hovertemplate=f'<b>{ticker}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>'))
            yaxis_title, title = 'Annualised Volatility (%)', '30-Day Rolling Volatility'

        elif chart_type == 'Moving Averages':
            sw, lw = fast_ma, slow_ma
            td[f'sma{sw}'] = td['Close'].rolling(sw).mean()
            td[f'sma{lw}'] = td['Close'].rolling(lw).mean()
            fig.add_trace(go.Scatter(x=td['Date'], y=td['Close'], mode='lines',
                name=f'{ticker} Price', line=dict(width=1.2, color=c, dash='dot'), opacity=0.4))
            fig.add_trace(go.Scatter(x=td['Date'], y=td[f'sma{sw}'], mode='lines',
                name=f'{ticker} SMA{sw}', line=dict(width=2, color=c)))
            fig.add_trace(go.Scatter(x=td['Date'], y=td[f'sma{lw}'], mode='lines',
                name=f'{ticker} SMA{lw}', line=dict(width=2, color=c, dash='dash')))
            golden = td[(td[f'sma{sw}']>td[f'sma{lw}'])&(td[f'sma{sw}'].shift(1)<=td[f'sma{lw}'].shift(1))]
            death  = td[(td[f'sma{sw}']<td[f'sma{lw}'])&(td[f'sma{sw}'].shift(1)>=td[f'sma{lw}'].shift(1))]
            if not golden.empty:
                fig.add_trace(go.Scatter(x=golden['Date'], y=golden[f'sma{sw}'], mode='markers',
                    name=f'{ticker} ✨ Golden', marker=dict(symbol='star', size=12, color=TH['gold'])))
            if not death.empty:
                fig.add_trace(go.Scatter(x=death['Date'], y=death[f'sma{sw}'], mode='markers',
                    name=f'{ticker} ☠️ Death', marker=dict(symbol='x', size=10, color=TH['red'])))
            yaxis_title = 'Price (KES)'
            title = f'Moving Averages — SMA{sw} vs SMA{lw}'

    if chart_type == 'Cumulative Returns':
        fig.add_hline(y=0, line_dash='dash', line_color=TH['zero'], line_width=1)

    fig.update_layout(**_layout(title=title, height=360,
                                xaxis_title='Date', yaxis_title=yaxis_title,
                                hovermode='x unified'))
    return fig


def plot_sector_rankings(metrics_df):
    best = metrics_df.loc[metrics_df.groupby('Ticker')['Cumulative_Return_%'].idxmax()].copy()
    best = best.sort_values('Cumulative_Return_%', ascending=True).tail(12)
    colours = [TH['pos_bar'] if x >= 0 else TH['neg_bar'] for x in best['Cumulative_Return_%']]
    fig = go.Figure(go.Bar(
        x=best['Cumulative_Return_%'], y=best['Ticker'], orientation='h',
        marker=dict(color=colours, line=dict(color='rgba(0,0,0,0.06)', width=0.5)),
        text=[f"{v:.0f}%" for v in best['Cumulative_Return_%']],
        textposition='outside', textfont=dict(size=9, color=TH['text']),
        hovertemplate='<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>'
    ))
    layout = _layout(title='', height=360, xaxis_title='Return (%)', showlegend=False)
    layout['yaxis']['gridcolor'] = 'rgba(0,0,0,0)'
    layout['yaxis']['tickfont']  = dict(size=9, color=TH['text_lt'])
    layout['xaxis']['zeroline']      = True
    layout['xaxis']['zerolinecolor'] = TH['zero']
    fig.update_layout(**layout)
    return fig


# ── PAGE 2 ────────────────────────────────────────────────────────────────────
def plot_price_volume(all_data, ticker, start_date, pred_date):
    df = all_data[all_data['Ticker'] == ticker].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= pd.to_datetime(start_date)) &
            (df['Date'] <= pd.to_datetime(pred_date))].sort_values('Date')
    if df.empty:
        return go.Figure()

    vol_col = next((c for c in ['Volume','volume','Vol'] if c in df.columns), None)
    rows = 2 if vol_col else 1
    row_heights = [0.68, 0.32] if vol_col else [1]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], mode='lines', name='Close',
        line=dict(color=TH['accent'], width=2),
        hovertemplate='%{x|%d %b %Y}: KES %{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([df['Close'], pd.Series([df['Close'].min()]*len(df))]),
        fill='toself', fillcolor='rgba(14,122,94,0.07)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    pred_ts = pd.to_datetime(pred_date)
    nearest = df.iloc[(df['Date'] - pred_ts).abs().argsort()[:1]]
    if not nearest.empty:
        px_val = nearest['Close'].values[0]
        fig.add_vline(x=pred_ts, line_dash='dash',
                      line_color='rgba(180,83,9,0.5)', line_width=1.5)
        fig.add_trace(go.Scatter(
            x=[nearest['Date'].values[0]], y=[px_val],
            mode='markers+text', name='Pred Date',
            text=['◀'], textposition='top right',
            textfont=dict(size=10, color=TH['gold']),
            marker=dict(size=9, color=TH['gold'], line=dict(color='white', width=2)),
            hovertemplate=f'Prediction Date<br>KES {px_val:,.2f}<extra></extra>'
        ), row=1, col=1)

    if vol_col:
        vol_c = ['rgba(14,122,94,0.5)' if c >= o else 'rgba(220,38,38,0.5)'
                 for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df['Date'], y=df[vol_col], name='Volume',
            marker_color=vol_c,
            hovertemplate='%{x|%d %b %Y}: %{y:,.0f}<extra></extra>'
        ), row=2, col=1)

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Plus Jakarta Sans, sans-serif', size=10, color=TH['text']),
        plot_bgcolor=TH['bg'], paper_bgcolor=TH['paper'],
        height=300, margin=dict(l=46, r=16, t=20, b=36),
        hovermode='x unified', showlegend=True,
        legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0.02)',
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis2=dict(title='Date', gridcolor=TH['grid'],
                    tickfont=dict(size=9, color=TH['text'])),
        yaxis=dict(title='Price (KES)', gridcolor=TH['grid'],
                   tickfont=dict(size=9, color=TH['text'])),
        yaxis2=dict(title='Volume', gridcolor=TH['grid'],
                    tickfont=dict(size=8, color=TH['text'])),
    )
    return fig


def plot_sector_promise(sector_scores):
    sector_scores = sector_scores.sort_values(ascending=True)
    colours = [TH['pos_bar'] if v >= 0 else TH['neg_bar'] for v in sector_scores.values]
    fig = go.Figure(go.Bar(
        x=sector_scores.values, y=sector_scores.index, orientation='h',
        marker=dict(color=colours, line=dict(color='rgba(0,0,0,0.06)', width=0.5)),
        text=[f"{v:.1f}%" for v in sector_scores.values],
        textposition='outside', textfont=dict(size=9, color=TH['text']),
        hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'
    ))
    layout = _layout(title='', height=220, xaxis_title='Avg Cumulative Return (%)',
                     showlegend=False)
    layout['yaxis']['gridcolor'] = 'rgba(0,0,0,0)'
    layout['yaxis']['tickfont']  = dict(size=9, color=TH['text_lt'])
    layout['margin'] = dict(l=8, r=36, t=8, b=28)
    layout['xaxis']['zeroline']      = True
    layout['xaxis']['zerolinecolor'] = TH['zero']
    fig.update_layout(**layout)
    return fig


# ── PAGE 3 ────────────────────────────────────────────────────────────────────
def plot_portfolio_bar(portfolio_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Current Value', x=portfolio_df['Ticker'], y=portfolio_df['Current_Value'],
        marker_color=TH['bar_base'], marker_line=dict(color=TH['border'], width=1),
        hovertemplate='<b>%{x}</b><br>Current: KES %{y:,.2f}<extra></extra>'
    ))
    if 'Predicted_Value' in portfolio_df.columns:
        fig.add_trace(go.Bar(
            name='Predicted Value', x=portfolio_df['Ticker'], y=portfolio_df['Predicted_Value'],
            marker_color=[TH['pos_bar'] if p >= c else TH['neg_bar']
                          for p, c in zip(portfolio_df['Predicted_Value'], portfolio_df['Current_Value'])],
            marker_line=dict(color=TH['border'], width=1),
            hovertemplate='<b>%{x}</b><br>Predicted: KES %{y:,.2f}<extra></extra>'
        ))
    fig.update_layout(**_layout(title='Current vs Predicted Value', height=240,
                                xaxis_title='Stock', yaxis_title='Value (KES)',
                                hovermode='x unified', barmode='group'))
    return fig


def plot_risk_horizon(risk_df):
    fig = go.Figure()
    for i, row in risk_df.iterrows():
        colour = STOCK_COLOURS[i % len(STOCK_COLOURS)]
        fig.add_trace(go.Scatter(
            x=['Pred Date', '+7 Days', '+30 Days'],
            y=[row.get('ret_pred', 0), row.get('ret_7d', 0), row.get('ret_30d', 0)],
            mode='lines+markers', name=row['Ticker'],
            line=dict(width=2, color=colour),
            marker=dict(size=7, color=colour, line=dict(color='white', width=1.5)),
            hovertemplate=f'<b>{row["Ticker"]}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>'
        ))
    fig.add_hline(y=0, line_dash='dash', line_color=TH['zero'], line_width=1)
    fig.update_layout(**_layout(title='Expected Return by Horizon', height=240,
                                xaxis_title='Horizon', yaxis_title='Expected Return (%)',
                                hovermode='x unified'))
    return fig


def plot_risk_return(metrics_df):
    best = metrics_df.loc[metrics_df.groupby('Ticker')['Sharpe_Ratio'].idxmax()].copy()
    uniq = best['Ticker'].unique()
    cmap = {t: STOCK_COLOURS[i % len(STOCK_COLOURS)] for i, t in enumerate(uniq)}
    fig  = go.Figure()
    for _, row in best.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Volatility_%']], y=[row['Cumulative_Return_%']],
            mode='markers+text', name=row['Ticker'],
            text=[row['Ticker']], textposition='top center',
            textfont=dict(size=8, color=TH['text']),
            marker=dict(size=8, color=cmap[row['Ticker']],
                        line=dict(color='rgba(0,0,0,0.1)', width=1), opacity=0.9),
            showlegend=False,
            hovertemplate=f"<b>{row['Ticker']}</b><br>Vol: %{{x:.2f}}%<br>Ret: %{{y:.2f}}%<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash='dash', line_color=TH['zero'])
    fig.update_layout(**_layout(title='Risk vs Return', height=300,
                                xaxis_title='Volatility (%)', yaxis_title='Cumulative Return (%)'))
    return fig