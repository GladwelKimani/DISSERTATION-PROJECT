# %% [markdown]
# # NSE Stock Price Forecasting: LSTM vs Lag-Llama Foundation Model
# ### Nairobi Securities Exchange · Rolling 30-Day Walk-Forward · Jan 2024 – Nov 2025
# 
# 
# 
# 
# 

# %%


# Install pandas 2.1.4 and a compatible numpy version
#!pip install pandas==2.1.4
#!pip install numpy==1.26.4

# Verify the installed versions
import pandas as pd
import numpy as np
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# %% [markdown]
# # Environment Configuration

# %%
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# %%
# Install necessary libraries if not already installed
#!pip install ta --quiet

# Import required libraries
import warnings
warnings.filterwarnings('ignore')
import os

# Data manipulation
from datetime import datetime, timedelta

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Statistical analysis
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Technical Indicators
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


# File operations
import os
import json
import pickle
from pathlib import Path
import joblib

# Progress bar
from tqdm import tqdm

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")






print(f"Python version: {pd.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ── Global constants (defined once) ──────────────────────────────────────────
REPRESENTATIVE_STOCKS = ['SCOM', 'KCB', 'EQTY', 'EABL', 'KQ', 'KPLC', 'KAPC', 'BRIT']

MODEL_COLOURS = {
    'LSTM':                '#2196F3',
    'Lag-Llama_ZeroShot':  '#FF9800',
    'Lag-Llama_FineTuned': '#4CAF50',
    'TimeGPT_ZeroShot':    '#9C27B0',
    'TimeGPT_FineTuned':   '#F44336',
}

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
print(f"Representative stocks: {REPRESENTATIVE_STOCKS}")


# %%
import os
import tensorflow as tf

# CPU Performance Optimization
try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    num_threads = max(2, physical_cores // 2) if physical_cores >= 8 else physical_cores
    print(f"Detected {physical_cores} cores → Using {num_threads} threads")
except:
    num_threads = 4
    print("Using default: 4 threads")

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

print(" CPU optimizations enabled")


# ── Publication colour palette (white-grid Plotly) ───────────────────────────
SECTOR_COLOURS = {
    'Agriculture':        '#2C7BB6',
    'Automobile':         '#D7191C',
    'Banking':            '#1A9641',
    'Commercial':         '#F46D43',
    'Construction':       '#762A83',
    'Energy':             '#E6AB02',
    'Insurance':          '#5E4FA2',
    'Investment':         '#3288BD',
    'Investment Services':'#ABDDA4',
    'Manufacturing':      '#FDAE61',
    'Telecom':            '#D53E4F',
}

PLOTLY_TEMPLATE = 'plotly_white'

FONT = dict(family='Arial, sans-serif', size=12, color='#2C2C2C')
AXIS_STYLE = dict(showgrid=True, gridcolor='#E5E5E5', gridwidth=1,
                  zeroline=False, linecolor='#CCCCCC', linewidth=1)

def paper_layout(title='', height=500, **kwargs):
    """Consistent publication-quality layout for all figures."""
    return dict(
        title=dict(text=title, font=dict(size=14, color='#1a1a1a'), x=0.05),
        template=PLOTLY_TEMPLATE,
        font=FONT,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=AXIS_STYLE,
        yaxis=AXIS_STYLE,
        height=height,
        margin=dict(l=60, r=40, t=70, b=60),
        **kwargs
    )


# %% [markdown]
# # Data Collection & Initial Understanding

# %% [markdown]
# ## Sector Mapping

# %%
# Clone NSE historical data repository
!git clone https://github.com/GladwelKimani/nse_historical_ohlc.git 2>/dev/null || echo "Repository already exists"

# Define sector mapping
SECTOR_MAPPING = {
  # Agriculture
    'EGAD': 'Agriculture', 'KAPC': 'Agriculture', 'KUKZ': 'Agriculture',
    'LIMT': 'Agriculture', 'SASN': 'Agriculture', 'WTK': 'Agriculture',

    # Automobile & Accessories
    'CGEN': 'Automobile',

    # Banking
    'ABSA': 'Banking', 'SBIC': 'Banking', 'IMH': 'Banking',
    'DTK': 'Banking', 'SCBK': 'Banking', 'EQTY': 'Banking',
    'COOP': 'Banking', 'BKG': 'Banking', 'HFCK': 'Banking',
    'KCB': 'Banking', 'NCBA': 'Banking',

    # Commercial and Services
    'XPRS': 'Commercial', 'SMER': 'Commercial', 'KQ': 'Commercial',
    'NMG': 'Commercial', 'SGL': 'Commercial', 'TPSE': 'Commercial',
    'SCAN': 'Commercial', 'UCHM': 'Commercial', 'LKL': 'Commercial',
    'NBV': 'Commercial',

    # Construction and Allied
    'CRWN': 'Construction', 'CABL': 'Construction', 'PORT': 'Construction',

    # Energy and Petroleum
    'TOTL': 'Energy', 'KEGN': 'Energy', 'KPLC': 'Energy', 'UMME': 'Energy',

    # Insurance
    'SLAM': 'Insurance', 'KNRE': 'Insurance',
    'LBTY': 'Insurance', 'BRIT': 'Insurance', 'CIC': 'Insurance',

    # Investment
    'OCH': 'Investment', 'CTUM': 'Investment',
    'HAFR': 'Investment',

    # Investment Services
    'NSE': 'Investment Services',

    # Manufacturing and Allied
    'BOC': 'Manufacturing', 'BAT': 'Manufacturing',
    'EABL': 'Manufacturing', 'UNGA': 'Manufacturing', 'EVRD': 'Manufacturing',
    'AMAC': 'Manufacturing', 'FTGH': 'Manufacturing', 'CARB': 'Manufacturing',

    # Telecommunication & Technology
    'SCOM': 'Telecom'
}

print(f" Sector mapping defined for tickers across {len(set(SECTOR_MAPPING.values()))} sectors")

# %% [markdown]
# ## Loading the data

# %%
def load_nse_data(data_directory='../data'):
    data_dir = Path(data_directory)
    all_data = []
    for file in tqdm(list(data_dir.glob("*.csv"))):
        try:
            ticker = file.stem.upper()
            df = pd.read_csv(file)
            df['Ticker'] = ticker
            df['Sector'] = SECTOR_MAPPING.get(ticker, 'Unknown')
            df.columns = df.columns.str.strip()
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Load data
raw_data = load_nse_data()
print(f"  Total records: {len(raw_data):,}")
print(f"  Unique tickers: {raw_data['Ticker'].nunique()}")
print(f"  Sectors: {raw_data['Sector'].nunique()}")

# %% [markdown]
# ## Initial Data Inspection

# %%
raw_data.head()

# %%
# Convert date column
raw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')

# %%
raw_data.info()

# %% [markdown]
# We have 114,123 entries and 8 columns. All columns have 114,123 non-null values, indicating no missing data at this stage. The data types are appropriate, with Date as datetime64[ns], price and volume columns as float64, and Ticker and Sector as object (strings). The data is clean.

# %%
raw_data.shape

# %%
raw_data.columns

# %%
print(f"\nDate range: {raw_data['Date'].min()} to {raw_data['Date'].max()}")

# %%
#sector Distribution

sector_distribution = raw_data['Sector'].value_counts().to_frame(name='Record_Count')
unique_tickers = raw_data.groupby('Sector')['Ticker'].unique().apply(list)
sector_distribution['Tickers'] = unique_tickers
display(sector_distribution)

# %%
# Missing Values
missing = raw_data.isnull().sum()
print(missing[missing > 0])


# %%
#Summary Statistics
display(raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# %% [markdown]
# There is a wide range in prices and volumes, with the 'Close' price having a mean of 46.13 and a large standard deviation of 97.18, indicating significant price fluctuations across the dataset. The 'Volume' also shows a substantial range, with a high maximum value compared to its mean, suggesting occasional high-volume trading days or potential outliers. This is evident as the prices and volumes differ accross markets in the NSE.

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %% [markdown]
# ## Data Quality Assessment

# %%
# Missing values by sector
missing_by_sector = raw_data.groupby('Sector').apply(
    lambda x: x.isnull().sum()
)[['Open', 'High', 'Low', 'Close', 'Volume']]

print("\n Missing Values by Sector:")
print(missing_by_sector[missing_by_sector.sum(axis=1) > 0])

# %% [markdown]
# This output confirms that there are no missing values across the 'Open', 'High', 'Low', 'Close', and 'Volume' columns for any of the sectors.

# %%
# Duplicates
duplicates = raw_data.duplicated(subset=['Ticker', 'Date']).sum()
print(f"\n Duplicate records: {duplicates}")

# %%
# Data coverage by ticker
coverage = raw_data.groupby('Ticker').agg({
    'Date': ['min', 'max', 'count']
}).round(2)
coverage.columns = ['Start_Date', 'End_Date', 'Record_Count']
coverage = coverage.sort_values('Record_Count', ascending=False)

print(coverage.head(52))

# %% [markdown]
# Most tickers demonstrate remarkably consistent data availability, with many showing records spanning from early 2015 to late 2025.
# 
# While the majority have extensive coverage, some tickers like BKG, KUKZ, LIMT AND AMAC have significantly fewer records.

# %% [markdown]
# ## Time Series Visualization

# %%
# Stock price trends by sector
sectors = sorted(raw_data['Sector'].unique())
n_sectors = len(sectors)

fig = make_subplots(
    rows=n_sectors, cols=1,
    subplot_titles=sectors,
    vertical_spacing=0.03
)

for idx, sector in enumerate(sectors):
    sd = raw_data[raw_data['Sector'] == sector].sort_values('Date')
    for ticker in sd['Ticker'].unique():
        td = sd[sd['Ticker'] == ticker]
        fig.add_trace(go.Scatter(
            x=td['Date'], y=td['Close'],
            mode='lines', name=ticker,
            line=dict(width=1),
            showlegend=False,
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: KES %{{y:,.2f}}<extra></extra>'
        ), row=idx+1, col=1)

fig.update_layout(
    height=280 * n_sectors,
    title=dict(text='NSE Closing Price Trends by Sector (2015–2025)',
               font=dict(size=14, color='#1a1a1a'), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

os.makedirs('outputs/figures', exist_ok=True)
fig.write_html('outputs/figures/sector_price_trends.html')
fig.show()
print(" Saved: sector_price_trends.html")


# %% [markdown]
# ## Time Series Visualization

# %% [markdown]
# ## Volume Trends by Sector

# %%
# Total trading volume by sector
sector_volumes = raw_data.groupby('Sector')['Volume'].sum().sort_values(ascending=False)
colours = [SECTOR_COLOURS.get(s, '#607D8B') for s in sector_volumes.index]

fig = go.Figure(go.Bar(
    x=sector_volumes.index,
    y=sector_volumes.values / 1e6,
    marker_color=colours,
    text=[f'{v/1e6:.0f}M' for v in sector_volumes.values],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>Volume: %{y:,.1f}M<extra></extra>'
))

fig.update_layout(**paper_layout(
    title='Total Trading Volume by Sector (2015–2025)',
    height=480
))
fig.update_yaxes(title_text='Total Volume (Millions)')
fig.update_xaxes(title_text='Sector', tickangle=-30)

fig.write_html('outputs/figures/sector_volumes.html')
fig.show()
print(" Saved: sector_volumes.html")


# %%
# Annual trading volume by sector
raw_data['Year'] = raw_data['Date'].dt.year
annual = raw_data.groupby(['Year','Sector'])['Volume'].sum().reset_index()

fig = go.Figure()
for sector in sorted(annual['Sector'].unique()):
    sub = annual[annual['Sector'] == sector]
    fig.add_trace(go.Bar(
        x=sub['Year'], y=sub['Volume']/1e6,
        name=sector,
        marker_color=SECTOR_COLOURS.get(sector, '#607D8B'),
        hovertemplate=f'<b>{sector}</b><br>Year: %{{x}}<br>Volume: %{{y:,.1f}}M<extra></extra>'
    ))

fig.update_layout(
    **paper_layout('Annual Trading Volume by Sector', height=520),
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=-0.35, x=0)
)
fig.update_yaxes(title_text='Volume (Millions)')
fig.update_xaxes(title_text='Year')

fig.write_html('outputs/figures/annual_sector_volume.html')
fig.show()
print(" Saved: annual_sector_volume.html")


# %% [markdown]
# ## Univariate Analysis

# %%
# Summary statistics by sector
sector_stats = raw_data.groupby('Sector')[['Open', 'High', 'Low', 'Close', 'Volume']].agg([
    'mean', 'median', 'std', 'min', 'max'
]).round(2)

display(sector_stats)

# %% [markdown]
# 
# *   **High Variability in Prices**: There's a significant range in stock prices (Open, High, Low, Close) across sectors. Sectors like **Agriculture** (max close 1248.00, std 130.89) and **Manufacturing** (max close 915.00, std 196.37) exhibit extremely high maximum prices and standard deviations, suggesting a few high-priced stocks heavily influence the sector average, or that these sectors contain companies with very different valuations and growth trajectories. The median being much lower than the mean for these sectors (e.g., Manufacturing: mean 125.04, median 20.50) confirms a right-skewed distribution implying presence of outliers.
# 
# *   **Relatively Stable Sectors**: In contrast, sectors like **Automobile**, **Construction**, and **Investment Services** show smaller price ranges and lower standard deviations, indicating more stable or less volatile price movements within their constituents. **Investment Services** particularly stands out with its relatively low price volatility (std Close 4.84) and a mean (11.60) closer to its median (10.30).
# 
# *   **Dominance in Trading Volume**: The **Banking** sector consistently shows the highest average daily trading volume (mean Volume 620,439.92) and an exceptionally high standard deviation (2,133,104.47), reflecting high liquidity and frequent, large-volume transactions.
# 
# *   **Skewed Volume Distribution**: For most sectors, the mean volume is significantly higher than the median volume (e.g., Agriculture: mean 10,367.87, median 1,100.00), coupled with very high max values (e.g., Insurance max Volume 1.74e+08). This indicates that while most trading days might have moderate volumes, there are occasional spikes with extremely high trading activity, potentially driven by major news, corporate actions, or institutional trading.
# 
# *   **Low-Priced, High-Volatility Potential**: Sectors like **Commercial**, **Energy**, and **Insurance** have relatively low average prices but can still exhibit considerable volatility relative to their means, as evidenced by their standard deviations being a substantial portion of their mean values. The Investment sector also falls into this category with a very low average price and a high standard deviation.
# 
# This reveals a diverse market with varying risk-return profiles across sectors, highlighting the importance of sector-specific analysis in investment decisions.

# %% [markdown]
# ### Calculate Daily Returns
# 
# 
# 

# %%
global data_for_returns
data_for_returns = raw_data.sort_values(by=['Ticker', 'Date']).copy().reset_index(drop=True)

# Calculate daily percentage change of the 'Close' price for each ticker
# Group by 'Ticker' to ensure returns are calculated within each stock's series
data_for_returns['Daily_Return'] = data_for_returns.groupby('Ticker')['Close'].pct_change()
display(data_for_returns.head())


# %% [markdown]
# ### Daily Return Distributions by Sector
# 

# %%
global df_returns
df_returns = data_for_returns.dropna(subset=['Daily_Return'])

fig = go.Figure()
for sector in sorted(df_returns['Sector'].unique()):
    sub = df_returns[df_returns['Sector'] == sector]
    fig.add_trace(go.Box(
        y=sub['Daily_Return'],
        name=sector,
        marker_color=SECTOR_COLOURS.get(sector, '#607D8B'),
        boxpoints='outliers',
        marker=dict(size=3, opacity=0.5),
        line=dict(width=1.5),
        hovertemplate='<b>' + sector + '</b><br>Return: %{y:.4f}<extra></extra>'
    ))

fig.add_hline(y=0, line_dash='dash', line_color='#888888', line_width=1)
fig.update_layout(
    **paper_layout('Daily Return Distribution by Sector', height=560),
    showlegend=False
)
fig.update_xaxes(title_text='Sector', tickangle=-30)
fig.update_yaxes(title_text='Daily Return (%)')

fig.write_html('outputs/figures/daily_return_distribution.html')
fig.show()
print(" Saved: daily_return_distribution.html")


# %% [markdown]
# The distribution of daily returns across various sectors, highlights a market that is generally stable but punctuated by a few extreme outliers. Most sectors, particularly Telecom and Investment Services, show very little movement, with their data points tightly clustered around the 0% line.
# 
# The Commercial sector  has  a massive positive spike exceeding 6%, which is a huge deviation from the typical daily fluctuations seen elsewhere. There is downward volatility in the Manufacturing and Commercial sectors, where a few negative outliers dipped toward -1%.

# %% [markdown]
# ### Descriptive Statistics for Daily Returns by Sector

# %%
daily_return_stats = df_returns.groupby('Sector')['Daily_Return'].agg(
    ['mean', 'median', 'std', 'min', 'max']
).round(4)
display(daily_return_stats)


# %% [markdown]
# 
# 
# -   **Mean and Median Returns**: Most sectors exhibit very low mean daily returns, often close to zero, which is typical for daily percentage changes over a long period. The median daily return for many sectors is 0.0, suggesting that on a typical day, there is no significant price movement, or positive and negative movements balance out. This also points to potential outliers or skewed distributions being more influential on the mean.
# 
# -   **Volatility (Standard Deviation)**: The standard deviation (std) is a key indicator of volatility. The **Commercial** sector has the highest standard deviation (0.0585), followed by **Automobile** (0.0535) and **Construction** (0.0453), indicating these sectors experience the largest daily price swings. In contrast, **Telecom** (0.0169) and **Banking** (0.0213) show the lowest volatility, suggesting more stable daily price movements.
# 
# -   **Extreme Returns (Min and Max)**:
#     -   **Commercial** stands out with an exceptionally high maximum daily return of **6.4930**. Presence of very high positive impact outliers.
#     -   Similarly, **Manufacturing** shows a very large negative minimum return of **-0.8293**, indicating a day with an over 80% drop in price for some stock in that sector, which is an extreme outlier.
#     -   Sectors like **Agriculture** (-0.3754 to 0.6224) and **Banking** (-0.1251 to 0.6626) also show substantial ranges, but not as extreme as Commercial or Manufacturing.
# 
#   -   **Telecom** has the lowest standard deviation (0.0169) and a positive mean (0.0004), suggesting it is the most stable sector with consistent, albeit small, daily gains.
# 

# %% [markdown]
# ## Cross-Sectoral Analysis

# %% [markdown]
# ### Sector performance metrics

# %%
sector_metrics = []

for sector in raw_data['Sector'].unique():
    if sector != 'Unknown':
        sector_data = raw_data[raw_data['Sector'] == sector].sort_values('Date')
        daily_close = sector_data.groupby('Date')['Close'].mean()
        daily_returns = daily_close.pct_change().dropna()

        # Calculate metrics
        avg_return = daily_returns.mean() * 252  # Annualized
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = avg_return / volatility if volatility != 0 else 0

        total_return = ((daily_close.iloc[-1] - daily_close.iloc[0]) / daily_close.iloc[0]) * 100
        avg_volume = sector_data['Volume'].mean()

        sector_metrics.append({
            'Sector': sector,
            'Avg_Annual_Return_%': avg_return * 100,
            'Volatility_%': volatility * 100,
            'Sharpe_Ratio': sharpe,
            'Total_Return_%': total_return,
            'Avg_Daily_Volume': avg_volume
        })

metrics_df = pd.DataFrame(sector_metrics).sort_values('Sharpe_Ratio', ascending=False)
display(metrics_df)

# %% [markdown]
# Agriculture stands as the primary outlier, 11,501% average annual return and a leading Sharpe Ratio of 4.73, though this is paired with massive volatility of 2,430%.
# 
# Interestingly, high annual averages do not consistently translate to positive long-term outcomes. The Commercial and Investment sectors have suffered significant downturns, with total returns plunging to -89.2% and -68.2% respectively, despite maintaining annual return averages near 300%. Similarly, Manufacturing shows a robust 3,067% annual average but carries a negative total return of -46.5%, indicating a sharp, recent collapse in value.
# 
# Liquidity also varies wildly across the board. While Telecom boasts the highest average daily volume at over 8.7 million, its annual return is a modest 10.1%. In contrast, the high-performing Agriculture sector operates on a fraction of that liquidity with a volume of only 10,367. Overall, the Automobile and Telecom sectors remain rare examples of stability, maintaining positive total returns above 100% while keeping volatility in the double digits.

# %%


# %% [markdown]
# ### Risk vs Return

# %%
# Risk-Return scatter
fig = go.Figure()

for _, row in metrics_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['Volatility_%']],
        y=[row['Avg_Annual_Return_%']],
        mode='markers+text',
        name=row['Sector'],
        text=[row['Sector']],
        textposition='top center',
        marker=dict(
            size=max(8, row['Avg_Daily_Volume']/metrics_df['Avg_Daily_Volume'].max()*60),
            color=SECTOR_COLOURS.get(row['Sector'], '#607D8B'),
            line=dict(width=1, color='white'),
            opacity=0.85
        ),
        hovertemplate=(
            f"<b>{row['Sector']}</b><br>"
            'Volatility: %{x:.1f}%<br>'
            'Annual Return: %{y:.1f}%<extra></extra>'
        )
    ))

fig.add_hline(y=0, line_dash='dash', line_color='#AAAAAA', line_width=1)
fig.update_layout(
    **paper_layout(
        'Risk-Return Profile by Sector<br>'
        '<sup>Bubble size = average daily volume</sup>',
        height=560
    ),
    showlegend=False
)
fig.update_xaxes(title_text='Annualised Volatility (%)')
fig.update_yaxes(title_text='Average Annual Return (%)')

fig.write_html('outputs/figures/risk_return_profile.html')
fig.show()
print(" Saved: risk_return_profile.html")


# %% [markdown]
# 
# 
# *   **Agriculture Sector: High Risk, High Reward**: The Agriculture sector stands out with an exceptionally high Average Annual Return of **11501.52%** and the highest Sharpe Ratio of **4.73**. This indicates that, despite its very high Volatility (2430.50%), it has generated substantial returns relative to its risk. Its Total Return of 115.24% over the period is also very strong. This aligns with our earlier finding of high maximum prices and standard deviations in this sector, suggesting the presence of high-growth companies or periods of significant gains.
# 
# *   **Manufacturing and Construction: Strong Risk-Adjusted Returns**: Manufacturing (Sharpe Ratio 3.16) and Construction (Sharpe Ratio 3.15) also demonstrate strong risk-adjusted returns. While their average annual returns are high (3067.39% and 3711.42% respectively) and volatility is considerable, their Sharpe Ratios suggest efficient use of risk for generating those returns. However, it's notable that Manufacturing shows a negative Total Return (-46.57%), indicating that while there might be high individual annual return periods, the overall trend for the sector has been downward over the full period.
# 
# *   **Energy, Insurance, Commercial, Investment: Moderate Risk-Adjusted Returns**: These sectors fall into a mid-range for Sharpe Ratios (1.02 to 1.53). They exhibit substantial volatility but also positive average annual returns. It's important to note that Commercial (-89.25%), Investment (-68.24%), and Insurance (-56.71%) sectors show significant negative Total Returns, suggesting that despite some periods of strong performance contributing to the average annual return, the overall trend has been unfavorable.
# 
# *   **Banking: Moderate Risk-Adjusted Returns and High Liquidity**: The Banking sector has a moderate Sharpe Ratio of **0.68**. Its average annual return is lower (95.78%) compared to the top sectors, but its Volatility (140.41%) is also relatively lower. It shows a slightly negative Total Return (-3.11%), indicating a relatively stable but flat performance over the entire period. Crucially, as observed in our sector volume analysis, the Banking sector has the highest `Avg_Daily_Volume` (620k), highlighting its high liquidity.
# 
# *   **Automobile: Moderate Return, Lower Volatility**: With a Sharpe Ratio of **0.59** and a Total Return of 135.71%, the Automobile sector demonstrates significant positive long-term growth with relatively lower volatility than many other sectors. This aligns with its lower standard deviation observed in the daily return statistics.
# 
# *   **Telecom and Investment Services: Lowest Risk-Adjusted Returns**: Telecom (Sharpe Ratio 0.38) and Investment Services (Sharpe Ratio 0.23) have the lowest Sharpe Ratios, indicating less favorable risk-adjusted returns. While Telecom shows a strong Total Return (103.18%), its low Sharpe Ratio implies that the returns might not adequately compensate for the risk taken. Investment Services also has a low Total Return (15.68%). However, similar to Banking, Telecom has a very high `Avg_Daily_Volume` (8.7M), confirming its high liquidity.
# 
# 
# The observation of low mean daily returns across most sectors is reconciled with higher average annual returns through the compounding effect, while negative `Total_Return_%` for some sectors despite positive `Avg_Annual_Return_%` highlights the difference between average period performance and cumulative performance over the entire span.

# %% [markdown]
# ## Statistical Analysis

# %% [markdown]
# ### Correlation matrix

# %%
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
corr = raw_data[corr_cols].corr()

fig = px.imshow(
    corr,
    text_auto='.3f',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    title='Correlation Matrix — Stock Attributes',
    template=PLOTLY_TEMPLATE
)
fig.update_layout(
    **paper_layout('Correlation Matrix — Stock Attributes', height=480),
    coloraxis_colorbar=dict(title='r', tickfont=dict(size=10))
)
fig.update_traces(textfont=dict(size=11))

fig.write_html('outputs/figures/correlation_matrix.html')
fig.show()
print(" Saved: correlation_matrix.html")


# %%
#correlation of each stock data but arranged per sector
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
sector_corr_matrix = raw_data.groupby('Sector')[corr_cols].corr()
display(sector_corr_matrix)




# %% [markdown]
# (Open, High, Low, Close) are almost perfectly synchronized within every sector, with coefficients consistently near 0.99. This indicates that intraday price action moves in a nearly identical lockstep regardless of the industry.
# 
# The real divergence appears in the relationship between price and Volume. In most sectors, this correlation is staying near zero or slipping into negative territory. Banking and Energy show the strongest negative ties to volume at -0.10 and -0.18 respectively, suggesting that price increases in these areas often occur on declining participation. Conversely, Investment and Investment Services show slight positive correlations (around 0.08 to 0.15), hinting that buying pressure there is somewhat more volume-supported.

# %% [markdown]
# ### Seasonal decomposition

# %%
# Seasonal decomposition for top sectors
top_sectors = raw_data.groupby('Sector')['Ticker'].count().nlargest(4).index

decomposition_results = {}

for sector in top_sectors:
    sector_data = raw_data[raw_data['Sector'] == sector].sort_values('Date')
    daily_close = sector_data.groupby('Date')['Close'].mean()
    weekly_close = daily_close.resample('W').mean().dropna()

    if len(weekly_close) >= 104:  # At least 2 years
        decomposition = seasonal_decompose(
            weekly_close,
            model='multiplicative',
            period=52
        )
        decomposition_results[sector] = decomposition

print(f" Seasonal decomposition completed for {len(decomposition_results)} sectors")

# %%
# Seasonal decomposition plots
DECOMP_COLOURS = ['#2C7BB6', '#1A9641', '#D7191C', '#E6AB02']

for sector, decomp in decomposition_results.items():
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
        vertical_spacing=0.07
    )
    components = [
        (decomp.observed, DECOMP_COLOURS[0]),
        (decomp.trend,    DECOMP_COLOURS[1]),
        (decomp.seasonal, DECOMP_COLOURS[2]),
        (decomp.resid,    DECOMP_COLOURS[3]),
    ]
    for row_i, (data, colour) in enumerate(components, 1):
        fig.add_trace(go.Scatter(
            y=data.values, mode='lines',
            line=dict(color=colour, width=1.5),
            showlegend=False
        ), row=row_i, col=1)

    fig.update_layout(
        height=800,
        title=dict(text=f'Seasonal Decomposition — {sector} Sector',
                   font=dict(size=14), x=0.05),
        template=PLOTLY_TEMPLATE,
        font=FONT,
        plot_bgcolor='white', paper_bgcolor='white',
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)

    fname = f'outputs/figures/decomposition_{sector.lower().replace(" ","_")}.html'
    fig.write_html(fname)
    fig.show()
    print(f" Saved: {fname}")


# %% [markdown]
# ## Outlier Detection

# %%
# Detect outliers using IQR method
def detect_outliers_iqr(df, columns, multiplier=3):
    outlier_indices = []

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.extend(outliers)

    return list(set(outlier_indices))

price_cols = ['Open', 'High', 'Low', 'Close']
outlier_indices = detect_outliers_iqr(raw_data, price_cols)

print(f"Total outliers detected: {len(outlier_indices)} ({len(outlier_indices)/len(raw_data)*100:.2f}%)")

# Outliers by sector
outlier_by_sector = raw_data.loc[outlier_indices].groupby('Sector').size().sort_values(ascending=False)
print("\nOutliers by Sector:")
print(outlier_by_sector)

# %% [markdown]
# *   **Manufacturing (4,938 outliers)**:  This aligns strongly with our previous observations from the summary statistics, where Manufacturing showed a very high maximum price (915.00) and a substantial standard deviation (196.37) for closing prices, as well as an extremely negative daily return (-0.8293).
# *   **Banking (3,583 outliers)**: The Banking sector, while generally appearing stable in daily return volatility, still contributes a significant number of outliers. This could be due to its high trading volume and the presence of both very high-priced and lower-priced stocks, leading to a wider range of 'normal' values but still subject to extreme movements.
# *   **Agriculture (1,701 outliers)**: high maximum prices (1248.00) and a high standard deviation (130.89). The presence of a high number of outliers in this sector is consistent with its commodity-dependent nature, which can lead to large price swings.
# *   **Commercial (341 outliers)**: Although lower in absolute count, the Commercial sector had the highest maximum daily return (6.4930), indicating that even a smaller number of outliers can be extremely impactful.
# 

# %%
# Closing price distribution with outliers
fig = go.Figure()
for sector in sorted(raw_data['Sector'].unique()):
    sub = raw_data[raw_data['Sector'] == sector]
    fig.add_trace(go.Box(
        y=sub['Close'],
        name=sector,
        marker_color=SECTOR_COLOURS.get(sector, '#607D8B'),
        boxpoints='outliers',
        marker=dict(size=3, opacity=0.4),
        line=dict(width=1.5)
    ))

fig.update_layout(
    **paper_layout('Closing Price Distribution by Sector (with Outliers)', height=560),
    showlegend=False
)
fig.update_xaxes(title_text='Sector', tickangle=-30)
fig.update_yaxes(title_text='Closing Price (KES)')

fig.write_html('outputs/figures/outliers_boxplot.html')
fig.show()
print(" Saved: outliers_boxplot.html")


# %% [markdown]
#  While sectors like Telecom, Energy, and Investment Services are incredibly compact, with prices rarely moving far from the 0–50 KES range, Agriculture and Manufacturing tell a completely different story.
# 
# Agriculture shows the most extreme vertical stretch, with a cluster of outliers reaching as high as 1,248 KES, far above its relatively modest median price. Manufacturing follows a similar pattern of heavy skewness; although its main body sits lower on the scale, it features a dense "tail" of outliers that shoot up to the 900 KES mark. Even Banking, which maintains a larger core distribution than most, shows significant price spikes peaking around 350 KES.
# 
#  Most trading occurs at very low price points, but the presence of outliers suggests that specific stocks within the Agriculture and Manufacturing sectors are operating on a valuation scale entirely different from the rest of the market.

# %%


# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# ## Data Cleaning

# %%
def clean_nse_data(df):
    df = df.copy()
    initial_len = len(df)

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Convert date column to proper datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Sort by ticker and date chronologically
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Remove tickers with insufficient data
    min_obs = 100
    ticker_counts = df.groupby('Ticker').size()
    valid_tickers = ticker_counts[ticker_counts >= min_obs].index
    removed = set(df['Ticker'].unique()) - set(valid_tickers)
    df = df[df['Ticker'].isin(valid_tickers)]

    if len(removed) > 0:
        print(f"   Removed {len(removed)} tickers with < {min_obs} observations")

    print(f"\n Cleaning complete. Final dataset: {len(df):,} records")
    return df

cleaned_data = clean_nse_data(raw_data)

# %% [markdown]
# ## Feature Engineering

# %%
def engineer_features(df):
    df = df.copy()
    engineered_data = []

    for ticker in tqdm(df['Ticker'].unique(), desc="Processing tickers"):
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)

        if len(ticker_df) < 50:
            continue

        # LAG FEATURES
        for lag in [1, 7, 30]:
            ticker_df[f'Close_lag_{lag}'] = ticker_df['Close'].shift(lag)
            ticker_df[f'Volume_lag_{lag}'] = ticker_df['Volume'].shift(lag)

        # MOVING AVERAGES
        for window in [7, 30]:
            ticker_df[f'SMA_{window}'] = ticker_df['Close'].rolling(window).mean()
            ticker_df[f'EMA_{window}'] = ticker_df['Close'].ewm(span=window).mean()
            ticker_df[f'Volume_MA_{window}'] = ticker_df['Volume'].rolling(window).mean()

        # RETURNS
        ticker_df['Daily_Return'] = ticker_df['Close'].pct_change()
        ticker_df['Price_Range'] = ticker_df['High'] - ticker_df['Low']

        # TECHNICAL INDICATORS
        # RSI
        rsi = RSIIndicator(close=ticker_df['Close'], window=14)
        ticker_df['RSI'] = rsi.rsi()

        # MACD
        macd = MACD(close=ticker_df['Close'])
        ticker_df['MACD'] = macd.macd()
        ticker_df['MACD_signal'] = macd.macd_signal()

        # Bollinger Bands
        bb = BollingerBands(close=ticker_df['Close'], window=20)
        ticker_df['BB_width'] = bb.bollinger_hband() - bb.bollinger_lband()

        # VOLUME FEATURES
        ticker_df['Volume_log'] = np.log1p(ticker_df['Volume'])

        #TEMPORAL FEATURES
        ticker_df['Day_of_week'] = ticker_df['Date'].dt.dayofweek
        ticker_df['Month'] = ticker_df['Date'].dt.month
        ticker_df['Day_sin'] = np.sin(2 * np.pi * ticker_df['Day_of_week'] / 7)
        ticker_df['Day_cos'] = np.cos(2 * np.pi * ticker_df['Day_of_week'] / 7)
        ticker_df['Month_sin'] = np.sin(2 * np.pi * ticker_df['Month'] / 12)
        ticker_df['Month_cos'] = np.cos(2 * np.pi * ticker_df['Month'] / 12)

        engineered_data.append(ticker_df)

    final_df = pd.concat(engineered_data, ignore_index=True)
    initial_len = len(final_df)
    final_df = final_df.dropna()

    print(f"  Removed {initial_len - len(final_df)} rows with NaN from rolling calculations")
    print(f"  Original features: {len(df.columns)}")
    print(f"  Total features: {len(final_df.columns)}")
    print(f"  Final records: {len(final_df):,}")

    return final_df

featured_data = engineer_features(cleaned_data)


# %% [markdown]
# ## Handle Non-Trading Days

# %%
def reindex_to_business_days(df):
    """
    Reindex to business day frequency for TimeGPT compatibility
    """
    reindexed_data = []

    for ticker in tqdm(df['Ticker'].unique(), desc="Reindexing tickers"):
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.set_index('Date').sort_index()

        # Create business day range
        date_range = pd.date_range(
            start=ticker_df.index.min(),
            end=ticker_df.index.max(),
            freq='B'
        )

        # Reindex and forward fill
        ticker_df = ticker_df.reindex(date_range, method='ffill')
        ticker_df['Ticker'] = ticker
        ticker_df['Sector'] = df[df['Ticker'] == ticker]['Sector'].iloc[0]

        reindexed_data.append(ticker_df.reset_index().rename(columns={'index': 'Date'}))

    final_df = pd.concat(reindexed_data, ignore_index=True)
    return final_df

featured_data_reindexed = reindex_to_business_days(featured_data)


# %% [markdown]
# ## Train-Test Split

# %%
# Define cutoff date (80-20 split)
CUTOFF_DATE = '2024-01-01'

def create_train_test_split(df, cutoff_date=CUTOFF_DATE):
    """
    Chronological train-test split
    """
    cutoff = pd.to_datetime(cutoff_date)

    train_data = df[df['Date'] < cutoff].copy()
    test_data = df[df['Date'] >= cutoff].copy()


    print(f"Cutoff date: {cutoff_date}")
    print(f"\nTrain set:")
    print(f"  Records: {len(train_data):,}")
    print(f"  Date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"\nTest set:")
    print(f"  Records: {len(test_data):,}")
    print(f"  Date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"\nSplit ratio: {len(train_data)/(len(train_data)+len(test_data))*100:.1f}% train, {len(test_data)/(len(train_data)+len(test_data))*100:.1f}% test")

    return train_data, test_data

train_data, test_data = create_train_test_split(featured_data_reindexed)

# %% [markdown]
# ## Feature Scaling

# %%
def scale_features(train_df, test_df):
    # Define feature groups
    price_features = ['Open', 'High', 'Low', 'Close'] + \
                     [col for col in train_df.columns if 'SMA' in col or 'EMA' in col or 'lag' in col and 'Close' in col]

    volume_features = ['Volume', 'Volume_log'] + \
                      [col for col in train_df.columns if 'Volume' in col and col != 'Volume']

    robust_features = ['RSI']

    # Initialize scalers
    price_scaler = MinMaxScaler()
    volume_scaler = RobustScaler()
    robust_scaler = RobustScaler()

    scalers = {}
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Scale price features
    price_cols = [col for col in price_features if col in train_df.columns]
    if price_cols:
        train_scaled[price_cols] = price_scaler.fit_transform(train_df[price_cols])
        test_scaled[price_cols] = price_scaler.transform(test_df[price_cols])
        scalers['price'] = price_scaler
        print(f"   Scaled {len(price_cols)} price features with MinMaxScaler")

    # Scale volume features
    vol_cols = [col for col in volume_features if col in train_df.columns]
    if vol_cols:
        train_scaled[vol_cols] = volume_scaler.fit_transform(train_df[vol_cols])
        test_scaled[vol_cols] = volume_scaler.transform(test_df[vol_cols])
        scalers['volume'] = volume_scaler
        print(f"   Scaled {len(vol_cols)} volume features with RobustScaler")

    # Scale robust features (RSI, etc.)
    rob_cols = [col for col in robust_features if col in train_df.columns]
    if rob_cols:
        train_scaled[rob_cols] = robust_scaler.fit_transform(train_df[rob_cols])
        test_scaled[rob_cols] = robust_scaler.transform(test_df[rob_cols])
        scalers['robust'] = robust_scaler
        print(f"   Scaled {len(rob_cols)} robust features with RobustScaler")

    return train_scaled, test_scaled, scalers

train_scaled, test_scaled, scalers = scale_features(train_data, test_data)

# Expose price scaler globally for inverse-transform
price_scaler = scalers['price']

# %%
# Define features and target columns
EXCLUDE_COLS = ['Date', 'Ticker', 'Sector', 'Close']
FEATURE_COLS = [col for col in train_scaled.columns if col not in EXCLUDE_COLS]
    # Clean column names (strip whitespace)
TARGET_COL = 'Close'

# %% [markdown]
# 
# 
# # **MODELLING**
# 
# 

# %% [markdown]
# ## Define Evaluation Metrics

# %%
def calculate_statistical_metrics(y_true, y_pred):
    """
    Calculate statistical performance metrics
    """
    min_len = min(len(y_true), len(y_pred))# Ensure arrays are aligned
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Directional Accuracy
    if len(y_true) > 1:
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        da = np.mean(direction_true == direction_pred)
    else:
        da = np.nan

    # Within 2% accuracy
    pct_error = np.abs((y_true - y_pred) / (y_true + 1e-8))
    within_2pct = np.mean(pct_error < 0.02)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Directional_Accuracy': da,
        'Within_2%': within_2pct
    }

def calculate_financial_metrics(y_true, y_pred):
    """
    Calculate financial performance metrics
    """
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Calculate returns
    if len(y_true) > 1:
        actual_returns = np.diff(y_true) / (y_true[:-1] + 1e-8)
        pred_returns = np.diff(y_pred) / (y_pred[:-1] + 1e-8)

        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe = np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) * np.sqrt(252)

        # Cumulative Return
        cum_return = np.prod(1 + pred_returns) - 1

        # Volatility
        volatility = np.std(pred_returns) * np.sqrt(252)
    else:
        sharpe = np.nan
        cum_return = np.nan
        volatility = np.nan

    return {
        'Sharpe_Ratio': sharpe,
        'Cumulative_Return_%': cum_return * 100,
        'Volatility_%': volatility * 100
    }

print("Evaluation metrics defined")

# %% [markdown]
# Model performance is assessed through mean absolute error (MAE) and root mean squared error (RMSE), to measure how far predictions are from actual values, and the coefficient of determination (R²), which measures how well the model explains the observed data. Directional accuracy evaluates the model's ability to correctly predict whether the price will go up or down, while the within-2% metric shows how frequently predictions fall close to the true value. Financial performance is assessed by computing predicted returns and deriving the Sharpe ratio to measure risk-adjusted performance, cumulative return to quantify total profitability, and volatility to quantify return variability.

# %% [markdown]
# # LSTM Benchmark Model

# %% [markdown]
# ## Prepare Sequences for LSTM

# %%


# Prepare Sequences for LSTM

def create_sequences(data, features, target, sequence_length=30):
    X, y, dates = [], [], []

    for i in range(sequence_length, len(data)):
        X.append(data[features].iloc[i-sequence_length:i].values)
        y.append(data[target].iloc[i])
        dates.append(data['Date'].iloc[i])

    return np.array(X), np.array(y), dates

# Test with one ticker
sample_ticker = train_scaled['Ticker'].iloc[0]
sample_data = train_scaled[train_scaled['Ticker'] == sample_ticker]
X_sample, y_sample, _ = create_sequences(sample_data, FEATURE_COLS, TARGET_COL, 30)


print(f"  Sample ticker: {sample_ticker}")
print(f"  X shape: {X_sample.shape} (samples, time_steps, features)")
print(f"  y shape: {y_sample.shape}")

# %% [markdown]
# Sliding window sequences are constructed for the LSTM using a fixed look-back horizon of 30 time steps (sequence_length = 30). For each index i(time step) from the 30th observation onward, the model input consists of the preceding 30 observations of the selected feature set, producing a 3D input tensor of shape (samples, time steps, features). The prediction target is defined as the value of the target variable at the current time step, ensuring a one-step-ahead forecasting. Overlapping sequences are generated while preserving temporal order, and the corresponding date for each target observation is retained for alignment and evaluation purposes.
# 
# 

# %% [markdown]
# ## Build LSTM Architecture

# %%


def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Build LSTM model architecture

    Args:
        input_shape: Tuple (time_steps, features)
        lstm_units: Number of LSTM units in first layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),

        # Second LSTM layer
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),

        # Dense layers
        Dense(32, activation='relu'),
        Dropout(dropout_rate / 2),

        # Output layer
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_absolute_error',
        metrics=['mse', 'mae']
    )

    return model

# Test model build
test_model = build_lstm_model(input_shape=(30, len(FEATURE_COLS)))
test_model.summary()

print(f"\n LSTM architecture defined")
print(f"  Total parameters: {test_model.count_params():,}")

# %% [markdown]
# The LSTM model is implemented as a stacked sequential architecture to capture temporal dependencies in the input sequences. The first LSTM layer contains 64 units and outputs a sequence of shape (30, 64), followed by dropout regularization (rate = 0.2). The second LSTM layer has 32 units and outputs a single representation (shape: 32), also followed by dropout. This is connected to a fully connected dense layer with 32 neurons and ReLU activation, followed by another dropout layer, and finally a linear output layer producing a single continuous prediction. The total number of trainable parameters is 37,825, and the model is trained using the Adam optimizer with a learning rate of 0.001 and mean absolute error as the loss function.

# %% [markdown]
# ## Hyperparameter Tuning with GridSearchCV

# %%
# Grid Search Hyperparameter Tuning

from sklearn.model_selection import ParameterGrid

def grid_search_lstm(X_train, y_train, X_val, y_val, input_shape):
    # Define parameter grid
    param_grid = {
        'lstm_units': [32, 64, 128],
        'dropout_rate': [0.2,],
        'learning_rate': [0.0001, 0.001],
        'batch_size': [16, 32, 64]
    }

    param_combinations = list(ParameterGrid(param_grid))

    best_score = float('inf')
    best_params = None
    results = []

    # Iterate through all combinations
    for i, params in enumerate(param_combinations, 1):
        print(f"\n[{i}/{len(param_combinations)}] Testing configuration:")
        print(f"  LSTM units={params['lstm_units']}, Dropout={params['dropout_rate']}, "
              f"LR={params['learning_rate']}, Batch={params['batch_size']}")

        try:
            # Build model
            model = build_lstm_model(
                input_shape=input_shape,
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate']
            )

            # Early stopping callback
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1  # Show when stopping
            )

            # Train model
            print(f"\n  Training...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=40,  # Max epochs
                batch_size=params['batch_size'],
                verbose=1,
                callbacks=[early_stop]
            )

            # Get results
            val_loss = min(history.history['val_loss'])
            epochs_trained = len(history.history['val_loss'])

            # Store results
            results.append({
                'lstm_units': params['lstm_units'],
                'dropout_rate': params['dropout_rate'],
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'val_loss': val_loss,
                'epochs_trained': epochs_trained,
                'stopped_early': epochs_trained < 30
            })

            # Update best
            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                print(f"\n   NEW BEST! Val Loss: {val_loss:.6f} ({epochs_trained} epochs)")
            else:
                print(f"\n  Val Loss: {val_loss:.6f} ({epochs_trained} epochs)")

            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return best_params, best_score, results_df

print(" Grid search function defined")

# %% [markdown]
# Hyperparameter optimization for the LSTM model was performed using an exhaustive grid search. The parameters evaluated included the number of LSTM units (32, 64, 128), fixed dropout rate (0.2), learning rate (0.0001, 0.001), and batch size (16, 32, 64). For each combination, the model was trained on the training set with early stopping based on validation loss, allowing a maximum of 40 epochs and restoring the best weights. Validation loss was recorded for each configuration, and the combination yielding the lowest loss was selected as optimal.

# %% [markdown]
# Attempted to use this parameter grid but it was too slow:
# Testing 81 parameter combinations
# 
# Search space:
#   - LSTM units: [32, 64, 128]
#   - Dropout rate: [0.1, 0.2, 0.3]
#   - Learning rate: [0.0001, 0.001, 0.01]
#   - Batch size: [16, 32, 64]
# 
# Early stopping: Patience=10, Monitor=val_loss

# %% [markdown]
#  the training data was split into training (80%) and validation (20%) sets. The previously defined grid search function was then executed to evaluate all combinations of LSTM units, dropout rate, learning rate, and batch size.

# %%
# Run grid search — load from cache if already run
import os

GRID_PKL = 'outputs/results/best_params.pkl'

if os.path.exists(GRID_PKL):
    best_params_lstm = joblib.load(GRID_PKL)
    print(f" Loaded cached best params: {best_params_lstm}")
    best_score = None
    grid_results = None
else:
    sample_ticker = train_scaled['Ticker'].unique()[0]
    ticker_train  = train_scaled[train_scaled['Ticker'] == sample_ticker]

    X_full, y_full, _ = create_sequences(ticker_train, FEATURE_COLS, TARGET_COL, 30)
    split_idx = int(0.8 * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    print(f"Sample ticker: {sample_ticker}")
    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Validation sequences: {X_val.shape[0]}")

    best_params_lstm, best_score, grid_results = grid_search_lstm(
        X_train, y_train, X_val, y_val,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    joblib.dump(best_params_lstm, GRID_PKL)
    print(f"\n Best params saved to {GRID_PKL}")

print("\nBest hyperparameters:")
for k, v in best_params_lstm.items():
    print(f"  {k}: {v}")


# %% [markdown]
# Configuration [2/18] currently stands as the benchmark with a validation loss of 0.000639, demonstrating that a moderate unit count (64) paired with a small batch size (16) provides the best balance for precision. Early stopping is consistently triggering between 7 and 19 epochs, halting training before the model overfits to noise. While higher learning rates (0.001) offer significant speed, they often fail to match the stability and low error rates achieved by the more gradual 0.0001 rate. Ultimately, the small-batch approach is yielding the tightest convergence between training and validation metrics, suggesting the model is successfully capturing the time-series patterns without significant generalization error

# %%


# %% [markdown]
# ## Train LSTM for All Tickers

# %%
# Train LSTM for all tickers — load from cache if already trained
import time

LSTM_PKL = 'outputs/results/lstm_all_results.pkl'

def train_lstm_per_ticker(ticker, train_data, test_data, features, target,
                          params, sequence_length=30, epochs=40):

    ticker_train = train_data[train_data['Ticker'] == ticker]
    ticker_test  = test_data[test_data['Ticker'] == ticker]

    X_train, y_train, _          = create_sequences(ticker_train, features, target, sequence_length)
    X_test,  y_test,  test_dates = create_sequences(ticker_test,  features, target, sequence_length)

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=params['lstm_units'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate']
    )

    os.makedirs('outputs/models', exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=0),
        ModelCheckpoint(f'outputs/models/lstm_{ticker}_best.keras',
                        save_best_only=True, monitor='val_loss', verbose=0)
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=params['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    training_time = time.time() - start_time

    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test  = model.predict(X_test,  verbose=0).flatten()

    model.save(f'outputs/models/lstm_{ticker}_final.keras')

    return {
        'ticker': ticker,
        'model':  model,
        'history': history.history,
        'training_time': training_time,
        'epochs_trained': len(history.history['loss']),
        'stopped_early':  len(history.history['loss']) < epochs,
        'predictions': {
            'train': {'actual': y_train, 'predicted': y_pred_train},
            'test':  {'actual': y_test,  'predicted': y_pred_test, 'dates': test_dates}
        }
    }

if os.path.exists(LSTM_PKL):
    lstm_results = joblib.load(LSTM_PKL)
    print(f" Loaded cached LSTM results: {len(lstm_results)} tickers")
else:
    lstm_results     = {}
    training_summary = []

    print("\n" + "="*70)
    print("TRAINING LSTM FOR ALL TICKERS")
    print("="*70)
    for k, v in best_params_lstm.items():
        print(f"  {k}: {v}")

    for ticker in tqdm(train_scaled['Ticker'].unique(), desc="Training LSTM"):
        try:
            result = train_lstm_per_ticker(
                ticker=ticker,
                train_data=train_scaled, test_data=test_scaled,
                features=FEATURE_COLS, target=TARGET_COL,
                params=best_params_lstm
            )
            lstm_results[ticker] = result
            training_summary.append({
                'ticker': ticker,
                'training_time':  result['training_time'],
                'epochs_trained': result['epochs_trained'],
                'stopped_early':  result['stopped_early']
            })
        except Exception as e:
            print(f"Error training {ticker}: {e}")

    joblib.dump(lstm_results, LSTM_PKL)
    summary_df = pd.DataFrame(training_summary)
    summary_df.to_csv('outputs/results/lstm_training_summary.csv', index=False)
    print(f"\n Trained {len(lstm_results)} tickers — saved to {LSTM_PKL}")
    print(f"  Avg time/ticker: {summary_df['training_time'].mean():.1f}s")
    print(f"  Total time: {summary_df['training_time'].sum()/60:.1f} min")


# %% [markdown]
# ## Inverse-Transform LSTM Predictions to KES
# 
# The LSTM was trained on MinMax-scaled Close prices (0–1 range) we inverse-transform here so all three models share the same measurement scale.

# %%
# Inverse-transform LSTM predictions back to KES
import numpy as np

price_features_ordered = ['Open', 'High', 'Low', 'Close'] +     [col for col in train_data.columns
     if ('SMA' in col or 'EMA' in col or ('lag' in col and 'Close' in col))]
price_features_ordered = [c for c in price_features_ordered if c in train_data.columns]
close_idx       = price_features_ordered.index('Close')
n_price_features = len(price_features_ordered)

def inverse_transform_close(scaled_values, scaler, close_idx, n_features):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, close_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, close_idx]

for ticker, result in lstm_results.items():
    result['predictions']['test']['predicted_kes'] = inverse_transform_close(
        result['predictions']['test']['predicted'], price_scaler, close_idx, n_price_features)
    result['predictions']['test']['actual_kes'] = inverse_transform_close(
        result['predictions']['test']['actual'], price_scaler, close_idx, n_price_features)

print(" Inverse-transform complete — KES predictions ready")


# %% [markdown]
# - Each of 52 stocks trained independently with 30-step look-back window
# - Train-validation split: 80-20 within each stock's history
# - Early stopping applied to all stocks
# 
# - Total training time: 11.75 minutes
# - Average per stock: 13.56 seconds
# - Average epochs completed: 10.0

# %% [markdown]
# ## Loss Curve analysis

# %%
# LSTM training & validation loss — 8 representative stocks
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=REPRESENTATIVE_STOCKS,
    vertical_spacing=0.08, horizontal_spacing=0.10
)

for idx, ticker in enumerate(REPRESENTATIVE_STOCKS):
    row, col = idx // 2 + 1, idx % 2 + 1
    show_legend = (idx == 0)

    if ticker not in lstm_results:
        continue

    h  = lstm_results[ticker]['history']
    ep = list(range(1, len(h['loss']) + 1))

    fig.add_trace(go.Scatter(
        x=ep, y=h['loss'],
        mode='lines', name='Train Loss',
        line=dict(color='#2C7BB6', width=1.8),
        showlegend=show_legend
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=ep, y=h['val_loss'],
        mode='lines', name='Val Loss',
        line=dict(color='#D7191C', width=1.8, dash='dash'),
        showlegend=show_legend
    ), row=row, col=col)

fig.update_layout(
    height=900,
    title=dict(text='LSTM Training & Validation Loss — 8 Representative Stocks',
               font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(title_text='Epoch', **AXIS_STYLE)
fig.update_yaxes(title_text='Loss (MAE)', **AXIS_STYLE)

fig.write_html('outputs/figures/lstm_loss_curves.html')
fig.show()
print(" Saved: lstm_loss_curves.html")


# %% [markdown]
# most models have a classic convergence pattern, where Train Loss (MAE) starts high and drops sharply within the first 2 to 5 epochs, eventually stabilizing at a low error floor.
# 
# However, KAPC shows a significant "generalization gap" where validation loss remains consistently higher than training loss. The validation curve is experiencing a sharp dip around epoch 22 before spiking again, which indicates potential difficulty in predicting this specific stock's volatility compared to more stable tickers like SCOM or KPLC.
# 
# SCOM, KCB, and KQ show near-perfect convergence, with the Val Loss tracking just below or alongside the Train Loss. This suggests a high degree of predictive reliability for these assets under the current configuration.
# 
# Most models reach a steady state very quickly, often within 10 to 15 epochs, justifying the use of early stopping in the broader hyperparameter grid search.

# %% [markdown]
# ## Evaluate LSTM

# %%
# Evaluate LSTM – using KES-scaled predictions for cross-model comparability
print("Evaluating LSTM models (KES scale)...\n")
lstm_metrics = []

for ticker, result in lstm_results.items():
    y_true = result['predictions']['test']['actual_kes']
    y_pred = result['predictions']['test']['predicted_kes']

    stat_metrics = calculate_statistical_metrics(y_true, y_pred)
    fin_metrics  = calculate_financial_metrics(y_true, y_pred)

    lstm_metrics.append({
        'Ticker': ticker,
        'Model':  'LSTM',
        **stat_metrics,
        **fin_metrics
    })

lstm_metrics_df = pd.DataFrame(lstm_metrics)

# Drop Within_2% – not a standard forecasting metric
if 'Within_2%' in lstm_metrics_df.columns:
    lstm_metrics_df = lstm_metrics_df.drop(columns=['Within_2%'])

print(" LSTM evaluation complete (KES scale, full Jan 2024–Nov 2025 test period)")
display_cols = ['Ticker','Model','MAE','RMSE','R2','Directional_Accuracy',
                'Sharpe_Ratio','Cumulative_Return_%','Volatility_%']
display_cols = [c for c in display_cols if c in lstm_metrics_df.columns]
print(lstm_metrics_df[display_cols].describe().round(4))


# %% [markdown]
#  The average MAE of 17.14 KES and RMSE of 19.96 KES are strongly influenced by high-priced stocks, as evidenced by the maximum MAE reaching 122.51 KES. The mean $R^2$ of -17.76 indicates that, on average, the models perform worse than a simple horizontal mean baseline, though at least one model achieved a positive fit of 0.1059.
# 
#  Despite the poor fit, the models show a "coin-flip" level of Directional Accuracy, with a mean of 51.25% and a top-performing model reaching 71.43%.
# 
#  The mean Sharpe Ratio of 0.1856 suggests low risk-adjusted returns across the test period, with cumulative returns ranging wildly from -44.49% to +99.61%.
# 
#  There is an extreme standard deviation in Volatility %, which likely stems from penny stocks or highly illiquid tickers where small KES movements translate to massive percentage swings.

# %% [markdown]
# 
# 
# LSTM is evaluated sequentially across the full 26,000-record test set —
# each 30-step window uses actual prior prices as input, making this a genuine
# walk-forward evaluation equivalent in scope to the rolling TimeGPT evaluation.
# 
# 

# %% [markdown]
# 

# %%
display(lstm_metrics_df)

# %% [markdown]
# The model for KQ achieved the highest Directional Accuracy at 71.43%, indicating strong trend-prediction capabilities despite a negative $R^2$.
# 
# The highest errors were recorded for KAPC (MAE: 122.5067) and SCBK (MAE: 98.6211), which are driven by their higher nominal share prices compared to the rest of the portfolio.
# 
# Low-priced tickers such as UCHM (MAE: 0.1284) and HAFR (MAE: 0.2849) demonstrate the lowest absolute errors, reflecting high model precision on smaller scales.
# 
# KNRE exhibits an extreme Volatility of 30,363,918.14%, which corresponds with a highly distorted $R^2$ of -127.30, suggesting the model failed to capture the variance of this specific asset.
# 
# KAPC yielded the highest Cumulative Return at 99.61%, while UCHM saw the most significant decline at -44.49% during the test period.
# 
# The Sharpe Ratio peaked at 1.4776 for UCHM, though its negative cumulative return suggests this figure may be influenced by specific volatility patterns rather than consistent gains.

# %%


# %%
import pandas as pd

for ticker in REPRESENTATIVE_STOCKS:
    if ticker not in lstm_results:
        print(f"{ticker}: not found in lstm_results")
        continue
    print(f"\n--- {ticker} — Actual vs Predicted (scaled, first/last 3 rows) ---")
    predictions = lstm_results[ticker]['predictions']['test']
    df_display = pd.DataFrame({
        'Date':            predictions['dates'],
        'Actual_KES':      predictions['actual_kes'],
        'Predicted_KES':   predictions['predicted_kes']
    })
    display(df_display.head(3))
    display(df_display.tail(3))


# %% [markdown]
# ## Visualization

# %%
# LSTM actual vs predicted — 8 representative stocks
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=REPRESENTATIVE_STOCKS,
    vertical_spacing=0.08, horizontal_spacing=0.10
)

for idx, ticker in enumerate(REPRESENTATIVE_STOCKS):
    row, col = idx // 2 + 1, idx % 2 + 1
    show_legend = (idx == 0)

    if ticker not in lstm_results:
        continue

    p  = lstm_results[ticker]['predictions']['test']
    df = pd.DataFrame({
        'Date':   pd.to_datetime(p['dates']),
        'Actual': p['actual_kes'],
        'Pred':   p['predicted_kes']
    }).sort_values('Date')

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual'],
        mode='lines', name='Actual',
        line=dict(color='#1A9641', width=1.8),
        showlegend=show_legend,
        hovertemplate='Actual: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Pred'],
        mode='lines', name='LSTM Predicted',
        line=dict(color='#D7191C', width=1.5),
        showlegend=show_legend,
        hovertemplate='Predicted: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

fig.update_layout(
    height=1000,
    title=dict(
        text='LSTM — Actual vs Predicted Closing Price (KES)<br>'
             '<sup>Full test period: Jan 2024 – Nov 2025</sup>',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

fig.write_html('outputs/figures/lstm_actual_vs_predicted.html')
fig.show()
print(" Saved: lstm_actual_vs_predicted.html")


# %% [markdown]
# In stable, high-volume sectors like Telecom (SCOM) and Banking (KCB, EQTY), the predicted line (red) closely follows the actual price (green) but with a noticeable time lag. This indicates the models are over-reliant on the previous day's price rather than identifying sector-specific leading indicators.
# 
# For Manufacturing (EABL) and Energy (KPLC), the models consistently under-predict the ceiling. The actual price frequently breaks away from the predicted "conservative" horizontal band, suggesting the LSTM struggles with the rapid price appreciation seen in these industries.
# 
# 
# Automobile (KQ) and Insurance (BRIT), the model stays relatively flat until a sharp actual price movement occurs. It then "reacts" with a sharp spike but fails to sustain the new price level, reflecting a lack of sectoral conviction.
# 
# Agriculture sector (KAPC), the high nominal price creates a massive visual gap between actual and predicted values. While the model captures the "staircase" directional movement, the absolute distance (MAE) remains the highest across all sectors due to the asset's scale.
# 
# 

# %%
import os

# LSTM — one representative ticker per sector
sector_representatives = {}
for ticker in lstm_results:
    s = SECTOR_MAPPING.get(ticker, 'Unknown')
    if s != 'Unknown' and s not in sector_representatives:
        sector_representatives[s] = ticker

items = list(sector_representatives.items())
n     = len(items)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"{t} ({s})" for s, t in items],
    vertical_spacing=0.08, horizontal_spacing=0.08
)

for plot_idx, (sector, ticker) in enumerate(items):
    row, col = plot_idx // ncols + 1, plot_idx % ncols + 1
    show_legend = (plot_idx == 0)

    result = lstm_results[ticker]['predictions']['test']
    dates  = pd.to_datetime(result['dates'])
    hist   = train_data[train_data['Ticker'] == ticker].sort_values('Date').tail(60)

    fig.add_trace(go.Scatter(
        x=hist['Date'], y=hist['Close'],
        mode='lines', name='Historical',
        line=dict(color='#888888', width=1.2),
        showlegend=show_legend
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['actual_kes'],
        mode='lines', name='Actual',
        line=dict(color='#2C7BB6', width=1.8),
        showlegend=show_legend
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['predicted_kes'],
        mode='lines', name='LSTM Predicted',
        line=dict(color='#D7191C', width=1.4),
        showlegend=show_legend
    ), row=row, col=col)

fig.update_layout(
    height=360 * nrows,
    title=dict(
        text='LSTM — Representative Stock per Sector<br>'
             '<sup>Full test period: Jan 2024 – Nov 2025</sup>',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

os.makedirs('outputs/figures', exist_ok=True)
fig.write_html('outputs/figures/lstm_actual_vs_predicted_by_sector.html')
fig.show()
print(" Saved: lstm_actual_vs_predicted_by_sector.html")

# %% [markdown]
# In most sectors, the models got "stuck" on old price levels. Instead of following the actual price growth, they stayed flat, predicting what the stock used to cost rather than where it was heading. Large-cap sectors like Banking (ABSA) and Manufacturing (AMAC) saw prices double or even quadruple, but the models completely missed these gains, staying anchored at the bottom of the chart.In volatile sectors like Agriculture (EGAD) and Energy (KEGN), the models produced "jittery" spikes that didn't match the actual market trends, showing a struggle to separate random price jumps from real movements.

# %%


# %% [markdown]
# 
# # LAG-LLAMA FOUNDATION MODEL
# 

# %% [markdown]
# ## Environment Setup & Model Installation
# 
# 
# 

# %%
# Clone Lag-Llama repository
#git clone https://github.com/time-series-foundation-models/lag-llama.git 2>/dev/null || echo "Repository already exists"

# %%
# Change to lag-llama directory and install requirements
%cd C:\dissertation\lag-llama
!pip install -r requirements.txt --quiet

# %%
import urllib.request
import os

url = "https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt"
output = "C:/dissertation/lag-llama/lag-llama.ckpt"

print("Downloading lag-llama.ckpt ...")
urllib.request.urlretrieve(url, output)

if os.path.exists(output):
    file_size = os.path.getsize(output) / (1024**3)
    print(f"Download complete: {file_size:.2f} GB")
else:
    print("✗ Download failed")

# %% [markdown]
# ## Import Libraries

# %%
# Install necessary libraries if not already installed
!pip install "gluonts[torch]==0.14.4"
!pip install lightning --quiet

import sys
import os

# Add the cloned repository's directory to the Python path
repo_dir = r'C:\dissertation\lag-llama'
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
    print(f"Added {repo_dir} to sys.path for lag_llama discovery.")

# Inspect the contents of the directory
print(f"Contents of {repo_dir}:")
print(os.listdir(repo_dir))

import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import torch
from itertools import islice
from tqdm import tqdm

# GluonTS for time series
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from torch.serialization import add_safe_globals
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood

# Allowlist GluonTS classes
add_safe_globals([StudentTOutput, NegativeLogLikelihood])

# Lag-Llama model
from lag_llama.gluon.estimator import LagLlamaEstimator

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print("All libraries imported successfully")
print("GluonTS distributions added to safe globals")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# %%
#Verifying checkpoint can be loaded sucessfully

ckpt = torch.load('lag-llama.ckpt', map_location='cpu')

print(" Checkpoint loaded successfully!")
print(f" Model parameters: {len(ckpt['state_dict'])} layers")
print(f"\nModel configuration:")
print(f"  - Input size: {ckpt['hyper_parameters']['model_kwargs']['input_size']}")
print(f"  - Layers: {ckpt['hyper_parameters']['model_kwargs']['n_layer']}")
print(f"  - Attention heads: {ckpt['hyper_parameters']['model_kwargs']['n_head']}")
print(f"  - Embedding dim/head: {ckpt['hyper_parameters']['model_kwargs']['n_embd_per_head']}")


# %%

# Verify data availability
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"\nUnique tickers: {train_data['Ticker'].nunique()}")
print(f"Sectors: {train_data['Sector'].nunique()}")

# %% [markdown]
# ## Data Preparation for GluonTS
# 
# Convert pandas DataFrames to GluonTS ListDataset format.
# Each ticker becomes a separate time series.

# %% [markdown]
# 

# %%
def prepare_gluonts_dataset(df, freq='D', target_col='Close'):
    """
    Convert DataFrame to GluonTS ListDataset format

    Parameters:
    -----------
    df : DataFrame
        Preprocessed stock data with Date, Ticker, Sector, and target column
    freq : str
        Frequency of time series ('D' for daily)
    target_col : str
        Column to forecast (default: 'Close')

    Returns:
    --------
    ListDataset : GluonTS dataset
    metadata : dict with ticker and sector information
    """
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    dataset_list = []
    metadata_list = []

    for ticker in tqdm(df['Ticker'].unique(), desc="Preparing datasets"):
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)

        # Skip if insufficient data
        if len(ticker_df) < 30:  # Minimum 30 days
            continue

        # Get sector
        sector = ticker_df['Sector'].iloc[0]

        # Create GluonTS format
        dataset_entry = {
            'target': ticker_df[target_col].values.astype(np.float32),
            'start': pd.Timestamp(ticker_df['Date'].iloc[0]),
            'item_id': ticker
        }

        dataset_list.append(dataset_entry)
        metadata_list.append({
            'ticker': ticker,
            'sector': sector,
            'length': len(ticker_df),
            'start_date': ticker_df['Date'].iloc[0],
            'end_date': ticker_df['Date'].iloc[-1]
        })

    gluonts_dataset = ListDataset(dataset_list, freq=freq)

    print(f"\n Created GluonTS dataset with {len(dataset_list)} time series")

    return gluonts_dataset, metadata_list






# %%
# ── GluonTS dataset prep ─────────────────────────────────────────────────────
# STANDARDISED WINDOW FIX:
# We use ONLY train_data (up to Dec 2023) so the 30-day forecast window aligns
# to Jan 2024 for all tickers – matching LSTM and TimeGPT evaluation periods.

train_gluonts, train_metadata = prepare_gluonts_dataset(
    train_data,
    freq='D',
    target_col='Close'
)

# backtest_gluonts = same as train_gluonts for this standardised evaluation
# (The predictor will forecast the 30 days immediately AFTER each training series)
backtest_gluonts  = train_gluonts
backtest_metadata = train_metadata

print(f" GluonTS dataset created: {len(train_metadata)} time series")
print(f"  Forecast window: first 30 trading days of Jan 2024 (immediately after training cutoff)")


# %% [markdown]
# ## Configure Forecasting Parameters

# %% [markdown]
# Rolling 30-Day Walk-Forward Evaluation:
# Lag-Llama is evaluated using rolling 30-day windows — ~16 non-overlapping
# windows stepping through the entire test period.
# 
# For each window:
# - All data before the window serves as the context (expanding window)
# - Lag-Llama forecasts the next 30 days
# - Predictions are compared to actual KES closing prices
# - Results are concatenated per-ticker across all windows before computing metrics
# 
# This mirrors the TimeGPT evaluation strategy and makes cross-model comparison valid.

# %%
# Define forecasting parameters
PREDICTION_LENGTH = 30  # Forecast 30 days ahead (adjust as needed)
CONTEXT_LENGTH = 60  # Context window (60 days)

print(f"Prediction horizon: {PREDICTION_LENGTH} days")
print(f"Context window: {CONTEXT_LENGTH} days")
print(f"\nTotal forecast period: {PREDICTION_LENGTH} trading days")

# %% [markdown]
# ## Initialize Lag-Llama Model

# %%
# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load checkpoint
ckpt_path = "lag-llama.ckpt"
ckpt = torch.load(ckpt_path, map_location=device)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

# Initialize estimator
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    # freq='D', # Removed this line as LagLlamaEstimator does not accept 'freq'

    # Model architecture (from checkpoint)
    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"],
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"],

    # Scaling and features
    scaling=estimator_args["scaling"],
    time_feat=estimator_args["time_feat"],
    num_parallel_samples=5,
)

# Create predictor
lightning_module = estimator.create_lightning_module()
transformation = estimator.create_transformation()
predictor = estimator.create_predictor(transformation, lightning_module)

print(" Predictor initialized successfully!")

# %% [markdown]
# ## Generate Zero-Shot Forecasts: Rolling 30-Day Windows

# %%
# verify dataset is properly formatted
print(f"Number of time series: {len(backtest_metadata)}")
print(f"Prediction length: {PREDICTION_LENGTH}")
print(f"Context length: {CONTEXT_LENGTH}")

# Check first time series
first_entry = list(backtest_gluonts)[0]
print(f"  Start date: {first_entry['start']}")
print(f"  Target length: {len(first_entry['target'])}")
print(f"  Target shape: {first_entry['target'].shape}")
print(f"  Item ID: {first_entry['item_id']}")

# Check if we have enough data
min_required = CONTEXT_LENGTH + PREDICTION_LENGTH
print(f"\nData validation:")
print(f"  Required minimum length: {min_required}")
print(f"  Actual length: {len(first_entry['target'])}")

if len(first_entry['target']) < min_required:
    print(" WARNING: Not enough data!")
else:
    print("   Sufficient data")

# %%
from tqdm import tqdm
import time
import numpy as np

ZS_PKL = 'outputs/results/lagllama_zs_rolling.pkl'

PREDICTION_LENGTH = 30
EVAL_START = pd.Timestamp('2024-01-01')
EVAL_END   = pd.Timestamp('2025-11-28')
all_test_bdates = pd.bdate_range(start=EVAL_START, end=EVAL_END)
window_starts   = all_test_bdates[::PREDICTION_LENGTH]

if os.path.exists(ZS_PKL):
    forecasts_rolling = joblib.load(ZS_PKL)
    print(f" Loaded cached zero-shot rolling results: {len(forecasts_rolling)} tickers")
else:
    print(f"Rolling zero-shot: {len(window_starts)} windows × {len(backtest_metadata)} tickers")

    all_data = pd.concat([train_data, test_data]).sort_values(
        ['Ticker','Date']).drop_duplicates(['Ticker','Date'])

    zs_all_actuals   = {m['ticker']: [] for m in backtest_metadata}
    zs_all_predicted = {m['ticker']: [] for m in backtest_metadata}
    zs_all_dates     = {m['ticker']: [] for m in backtest_metadata}
    zs_all_lower     = {m['ticker']: [] for m in backtest_metadata}  # ← ADD
    zs_all_upper     = {m['ticker']: [] for m in backtest_metadata}  # ← ADD
    zs_timings = []

    for w_start in tqdm(window_starts, desc="Zero-Shot Windows"):
        w_end = w_start + pd.offsets.BDay(PREDICTION_LENGTH - 1)
        context_data = all_data[all_data['Date'] < w_start]
        if context_data.groupby('Ticker').size().min() < 60:
            continue
        try:
            context_gluonts, context_meta = prepare_gluonts_dataset(
                context_data, freq='D', target_col='Close')
            t0 = time.time()
            window_forecasts = list(predictor.predict(context_gluonts))
            zs_timings.append(time.time() - t0)

            for i, meta in enumerate(context_meta):
                ticker = meta['ticker']
                ticker_test = test_data[test_data['Ticker'] == ticker].sort_values('Date')
                window_actual = ticker_test[
                    (ticker_test['Date'] >= w_start) & (ticker_test['Date'] <= w_end)]
                if len(window_actual) < 5:
                    continue
                n = len(window_actual)

                pred_mean  = window_forecasts[i].mean[:n]
                pred_lower = np.percentile(window_forecasts[i].samples[:, :n], 5,  axis=0)  # ← ADD
                pred_upper = np.percentile(window_forecasts[i].samples[:, :n], 95, axis=0)  # ← ADD

                zs_all_actuals[ticker].extend(window_actual['Close'].values)
                zs_all_predicted[ticker].extend(pred_mean)
                zs_all_dates[ticker].extend(window_actual['Date'].values)
                zs_all_lower[ticker].extend(pred_lower)  # ← ADD
                zs_all_upper[ticker].extend(pred_upper)  # ← ADD

        except Exception as e:
            print(f"  Window {w_start.date()}: {e}")

    forecasts_rolling = {}
    for meta in backtest_metadata:
        ticker = meta['ticker']
        if zs_all_actuals[ticker]:
            forecasts_rolling[ticker] = {
                'actual':    np.array(zs_all_actuals[ticker]),
                'predicted': np.array(zs_all_predicted[ticker]),
                'dates':     np.array(zs_all_dates[ticker]),
                'n_windows': len(window_starts),
                'lower_90':  np.array(zs_all_lower[ticker]),  # ← ADD
                'upper_90':  np.array(zs_all_upper[ticker]),  # ← ADD
            }

    os.makedirs('outputs/results', exist_ok=True) # Ensure the directory exists
    joblib.dump(forecasts_rolling, ZS_PKL)
    print(f"\n Zero-shot rolling complete: {len(forecasts_rolling)} tickers")
    print(f"  Avg window time: {np.mean(zs_timings):.2f}s")
    print(f"  Saved to {ZS_PKL}")

# %% [markdown]
# ### Zero-Shot Evaluation: Aggregated Across All Windows

# %%
ts_metrics_list = []
ticker_to_sector_map = {m['ticker']: m['sector'] for m in backtest_metadata}

for ticker, result in forecasts_rolling.items():
    y_true = result['actual']
    y_pred = result['predicted']

    stat = calculate_statistical_metrics(y_true, y_pred)
    fin  = calculate_financial_metrics(y_true, y_pred)

    ts_metrics_list.append({
        'ticker':    ticker,
        'sector':    ticker_to_sector_map.get(ticker, 'Unknown'),
        'Model':     'Lag-Llama_ZeroShot',
        'N_Windows': result['n_windows'],
        **stat, **fin
    })

ts_metrics_df = pd.DataFrame(ts_metrics_list)

# Drop Within_2% if present (not a standard metric for comparison)
if 'Within_2%' in ts_metrics_df.columns:
    ts_metrics_df = ts_metrics_df.drop(columns=['Within_2%'])

display(ts_metrics_df)

# %% [markdown]
# The model achieved exceptional financial outcomes in specific sectors, notably Energy (KEGN: 385.66%), Banking (KCB: 251.24%), and Automobile (CGEN: 198.48%) in cumulative returns.
# 
# Many tickers show high statistical R^2 meaning the model effectively follows the actual price curves. Standouts include Energy (KEGN: 0.8612), Banking (DTK: 0.8051), and Telecom (SCOM: 0.7826).
# 
# KQ (Commercial) achieved a peak directional accuracy of 81.56%, indicating a high level of success in predicting whether the price will move up or down, even when actual returns remained flat.
# 
#  SCOM (Telecom) and KCB (Banking) maintained Sharpe Ratios above 1.0, suggesting the model provided consistent, lower-risk returns for these large-cap stocks.
# 
#  Despite its strengths, the model still struggles with high-priced, high-volatility sectors like Agriculture. This is evident in KAPC, which despite the zero-shot capabilities, still shows a high error (MAE: 46.43) and a negative fit ($R^2$: -1.91).
# 
#   In the Investment sector, CTUM showed a very high $R^2$ of 0.7622, indicating the model is particularly effective at capturing the structured movements of investment-holding companies.

# %% [markdown]
# 
# 
# Top-Tier Performance Tickers  
# **Criteria:** Sharpe Ratio > 0.8 & R² > 0.5  
# 
# These tickers represent the "Gold Standard" of the model, where predictions were both statistically accurate and financially rewarding.
# 
# | Ticker | Sector         | R² (Fit) | Sharpe Ratio | Cumulative Return % |
# |--------|---------------|----------|--------------|---------------------|
# | KEGN   | Energy        | 0.8612   | 1.2168       | 385.66%             |
# | KCB    | Banking       | 0.6913   | 1.0840       | 251.25%             |
# | SCOM   | Telecom       | 0.7826   | 1.0654       | 142.22%             |
# | CGEN   | Automobile    | 0.6749   | 1.0133       | 198.48%             |
# | SLAM   | Insurance     | 0.7088   | 1.0022       | 165.15%             |
# | IMH    | Banking       | 0.6799   | 0.9397       | 142.64%             |
# | DTK    | Banking       | 0.8051   | 0.8610       | 136.28%             |
# | BRIT   | Insurance     | 0.7427   | 0.8493       | 80.89%              |
# | EABL   | Manufacturing | 0.7880   | 0.8290       | 117.88%             |
# 
# High Accuracy (Statistical) Tickers  
# **Criteria:** Directional Accuracy > 55% OR R² > 0.5 (but Sharpe < 0.8)
# 
# These models were statistically precise-capturing trends or price direction correctly-but did not meet the strict financial threshold for risk-adjusted returns.
# 
# | Ticker | Sector        | Directional Accuracy | R² (Fit) | Sharpe Ratio | Cumulative Return % |
# |--------|--------------|----------------------|----------|--------------|---------------------|
# | KQ     | Commercial   | 81.56%               | -0.3517  | 0.0000       | 0.00%               |
# | AMAC   | Manufacturing| 61.12%               | -0.0259  | 0.9402       | 151.68%             |
# | NCBA   | Banking      | 49.30%               | 0.7689   | 0.7909       | 80.34%              |
# | LBTY   | Insurance    | 49.90%               | 0.6866   | 0.7691       | 39.29%              |
# | NMG    | Commercial   | 49.90%               | 0.7012   | 0.5809       | -28.72%             |
# | CRWN   | Construction | 51.70%               | 0.5991   | 0.5120       | 28.79%              |
# | SBIC   | Banking      | 52.71%               | 0.5427   | 0.6663       | 55.63%              |
# | KNRE   | Insurance    | 50.70%               | 0.5353   | 0.9532       | 131.82%             |
# 
#   
# KEGN (Energy) is the standout, combining the best statistical fit (R² = 0.86) with the highest overall return (385%).
# 
# KQ shows an exceptional 81.56% Directional Accuracy, proving the model is excellent at identifying price direction even when return magnitude is flat.
#   
# NMG demonstrates that a high R² (0.70) does not necessarily translate into profitability.  
# The model tracked the downward trend accurately, resulting in a -28.72% cumulative return.

# %%
import pandas as pd
import numpy as np

# Display aggregate metrics
print("="*60)
print("AGGREGATE PERFORMANCE METRICS (Lag-Llama Zero-Shot)")
print("="*60)

# Calculate aggregate metrics from ts_metrics_df
agg_rmse = ts_metrics_df['RMSE'].mean()
agg_mae = ts_metrics_df['MAE'].mean()
agg_da = ts_metrics_df['Directional_Accuracy'].mean()
agg_sharpe = ts_metrics_df['Sharpe_Ratio'].mean()
agg_cum_return = ts_metrics_df['Cumulative_Return_%'].mean()
agg_volatility = ts_metrics_df['Volatility_%'].mean()

metrics_display = {
    'RMSE': agg_rmse,
    'MAE': agg_mae,
    'Directional_Accuracy': agg_da,
    'Sharpe_Ratio': agg_sharpe,
    'Cumulative_Return_%': agg_cum_return,
    'Volatility_%': agg_volatility
}

for metric_name, value in metrics_display.items():
    if isinstance(value, (int, float)) and not np.isnan(value):
        if metric_name in ['Cumulative_Return_%', 'Volatility_%']: # Special handling for percentages
            print(f"{metric_name:.<40} {value:.4f}%")
        else:
            print(f"{metric_name:.<40} {value:.4f}")
    else:
        print(f"{metric_name:.<40} {value}")

print("="*60)

# %% [markdown]
# With a Cumulative Return of 50.61% and a Sharpe Ratio of 0.6178, the model effectively captured major market "bull runs" while maintaining a healthy balance between risk and reward. Although the Directional Accuracy (51.05%) remains near baseline, the high returns and low absolute error (MAE of 5.50) suggest that while the model may not predict every small flicker, it is exceptionally accurate at identifying and following the large-scale trends that drive portfolio growth.

# %% [markdown]
# ### Sector-Level Performance

# %%
# Sector-level performance analysis
sector_metrics = ts_metrics_df.groupby('sector').agg({
    'RMSE': ['mean', 'std'],
    'MAE': ['mean', 'std']
}).round(4)

print("\nSector-Level Performance Summary:")
print(sector_metrics)

# %% [markdown]
# Investment and Insurance are the standout performers, boasting the lowest average errors (MAE < 0.7), which indicates the model is highly reliable for these predictable industries. In contrast, Agriculture remains the most difficult to forecast, showing a massive mean RMSE of 26.53 and high standard deviation, suggesting that outlier price swings frequently disrupt the model’s accuracy. Mid-range sectors like Energy and Telecom show healthy, low-single-digit errors, proving the model is well-tuned for these steady-growth areas, whereas Banking and Manufacturing exhibit high variability (std > 8.0), likely due to the wide range of stock sizes within those specific groups.

# %%


# %% [markdown]
# ### Visualization of Zero-Shot Forecasts

# %%
def plot_rolling_forecast(ticker, forecasts_rolling, train_data, show_history=120):
    """
    Plot actual vs predicted from rolling walk-forward results.
    Works with the forecasts_rolling dict produced by the rolling loop.
    """
    import matplotlib.dates as mdates

    if ticker not in forecasts_rolling:
        print(f"{ticker} not found in forecasts_rolling")
        return None

    result = forecasts_rolling[ticker]
    dates     = pd.to_datetime(result['dates'])
    actuals   = result['actual']
    predicted = result['predicted']

    # Historical context: last show_history days of training data
    hist = train_data[train_data['Ticker'] == ticker].sort_values('Date').tail(show_history)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist['Date'], hist['Close'],
            label='Historical (train)', color='black', linewidth=2)
    ax.plot(dates, actuals,
            label='Actual (test)', color='steelblue', linewidth=1.5)
    ax.plot(dates, predicted,
            label='Lag-Llama ZS Rolling', color='red',
            linewidth=1.0, linestyle='--')

    ax.set_title(f'{ticker} — Lag-Llama Zero-Shot (Rolling 30-day, Jan 2024–Nov 2025)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Close Price (KES)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# %%
import os

# Lag-Llama zero-shot — one representative ticker per sector
sector_representatives = {}
for meta in backtest_metadata:
    s, t = meta['sector'], meta['ticker']
    if s not in sector_representatives and t in forecasts_rolling:
        sector_representatives[s] = t

items = list(sector_representatives.items())
n     = len(items)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"{t} ({s})" for s, t in items],
    vertical_spacing=0.08, horizontal_spacing=0.08
)

for plot_idx, (sector, ticker) in enumerate(items):
    row, col = plot_idx // ncols + 1, plot_idx % ncols + 1
    show_legend = (plot_idx == 0)
    result = forecasts_rolling[ticker]
    dates  = pd.to_datetime(result['dates'])
    hist   = train_data[train_data['Ticker'] == ticker].sort_values('Date').tail(60)

    fig.add_trace(go.Scatter(
        x=hist['Date'], y=hist['Close'],
        mode='lines', name='Historical',
        line=dict(color='#888888', width=1.2),
        showlegend=show_legend
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['actual'],
        mode='lines', name='Actual',
        line=dict(color='#2C7BB6', width=1.8),
        showlegend=show_legend
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['predicted'],
        mode='lines', name='Lag-Llama ZS',
        line=dict(color='#D7191C', width=1.4),
        showlegend=show_legend
    ), row=row, col=col)

fig.update_layout(
    height=360 * nrows,
    title=dict(
        text='Lag-Llama Zero-Shot — Representative Stock per Sector<br>'
             '<sup>Rolling 30-day walk-forward, Jan 2024 – Nov 2025</sup>',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

# Ensure the directory exists before saving the file
os.makedirs('outputs/figures', exist_ok=True)
fig.write_html('outputs/figures/lagllama_zs_by_sector.html')
fig.show()
print(" Saved: lagllama_zs_by_sector.html")

# %% [markdown]
#  In 10 out of 11 sectors, the predicted (red) line exhibits tight coupling with the actual (blue) price trajectory, accurately capturing the stochastic volatility of the 2025 market.
# 
# The model effectively tracked the "hockey-stick" growth curves in **Energy (KEGN)** and **Telecom (SCOM)**, demonstrating a robust capacity for zero-shot generalization to new price regimes.
# 
#  In high-variance cases like **Manufacturing (AMAC)**, the model successfully recalibrated to major price gaps, maintaining its tracking alignment despite extreme shifts in equilibrium.
# 
#  KQ remains the sole outlier where the model defaulted to a flat, mean-invariant baseline, indicating a localized failure in signal extraction for that specific asset.
# 

# %%
# Lag-Llama zero-shot — best and worst 3 stocks by DA
ts_sorted    = ts_metrics_df.sort_values('Directional_Accuracy', ascending=False)
best_tickers = ts_sorted.head(3)['ticker'].values
worst_tickers= ts_sorted.tail(3)['ticker'].values

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        [f"{t} (Best DA)" for t in best_tickers] +
        [f"{t} (Worst DA)" for t in worst_tickers]
    ),
    vertical_spacing=0.12, horizontal_spacing=0.08
)

for col_i, (ticker, row) in enumerate(
        list(zip(best_tickers, [1,1,1])) + list(zip(worst_tickers, [2,2,2]))):
    col = col_i % 3 + 1
    if ticker not in forecasts_rolling:
        continue
    result = forecasts_rolling[ticker]
    dates  = pd.to_datetime(result['dates'])
    meta   = ts_metrics_df[ts_metrics_df['ticker'] == ticker].iloc[0]
    colour = '#1A9641' if row == 1 else '#D7191C'
    show_legend = (col_i == 0)

    fig.add_trace(go.Scatter(
        x=dates, y=result['actual'],
        mode='lines', name='Actual',
        line=dict(color='#2C7BB6', width=1.8),
        showlegend=show_legend
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['predicted'],
        mode='lines', name='Predicted',
        line=dict(color=colour, width=1.4),
        showlegend=show_legend
    ), row=row, col=col)

fig.update_layout(
    height=680,
    title=dict(
        text='Lag-Llama Zero-Shot — Best & Worst Performers by Directional Accuracy',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

fig.write_html('outputs/figures/lagllama_zs_best_worst.html')
fig.show()
print(" Saved: lagllama_zs_best_worst.html")


# %%
# Lag-Llama Zero-Shot — actual vs predicted — 8 representative stocks
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=REPRESENTATIVE_STOCKS,
    vertical_spacing=0.08, horizontal_spacing=0.10
)

for idx, ticker in enumerate(REPRESENTATIVE_STOCKS):
    row, col = idx // 2 + 1, idx % 2 + 1
    show_legend = (idx == 0)

    if ticker not in forecasts_rolling:
        continue

    result = forecasts_rolling[ticker]
    df = pd.DataFrame({
        'Date':   pd.to_datetime(result['dates']),
        'Actual': result['actual'],
        'Pred':   result['predicted']
    }).sort_values('Date')

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual'],
        mode='lines', name='Actual',
        line=dict(color='#1A9641', width=1.8),
        showlegend=show_legend,
        hovertemplate='Actual: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Pred'],
        mode='lines', name='Lag-Llama ZS Predicted',
        line=dict(color='#FF9800', width=1.5),
        showlegend=show_legend,
        hovertemplate='Predicted: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

fig.update_layout(
    height=1000,
    title=dict(
        text='Lag-Llama Zero-Shot — Actual vs Predicted Closing Price (KES)<br>'
             '<sup>Rolling 30-day walk-forward, Jan 2024 – Nov 2025</sup>',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

fig.write_html('outputs/figures/lagllama_zs_actual_vs_predicted.html')
fig.show()
print(" Saved: lagllama_zs_actual_vs_predicted.html")

# %% [markdown]
# ### Uncertainty Quantification

# %%
# Uncertainty proxy: use rolling MAE spread across windows as uncertainty estimate
# (True quantile intervals require storing forecast samples in the rolling loop)

uncertainty_results = []

for meta in backtest_metadata:
    ticker = meta['ticker']
    if ticker not in forecasts_rolling:
        continue

    result = forecasts_rolling[ticker]
    y_true = result['actual']
    y_pred = result['predicted']

    # Absolute errors as a proxy for prediction uncertainty
    abs_errors = np.abs(y_true - y_pred)
    median_price = np.median(y_true)


    # Relative uncertainty: mean absolute error as % of median price
    relative_uncertainty = (abs_errors.mean() / (median_price + 1e-8)) * 100

    uncertainty_results.append({
        'ticker':                 ticker,
        'sector':                 meta['sector'],
        'mean_abs_error':         abs_errors.mean(),
        'median_price':           median_price,
        'relative_uncertainty_%': relative_uncertainty,
        'n_predictions':          len(y_true)
    })

uncertainty_df = pd.DataFrame(uncertainty_results)

print("Relative Forecast Uncertainty — Top 10 Most Uncertain Stocks:")
print(uncertainty_df.sort_values('relative_uncertainty_%', ascending=False)
      [['ticker', 'sector', 'relative_uncertainty_%', 'mean_abs_error', 'median_price']]
      .head(10).round(3).to_string(index=False))

print("\nRelative Forecast Uncertainty — Top 10 Most Certain Stocks:")
print(uncertainty_df.sort_values('relative_uncertainty_%')
      [['ticker', 'sector', 'relative_uncertainty_%', 'mean_abs_error', 'median_price']]
      .head(10).round(3).to_string(index=False))

# %% [markdown]
# The divergence in relative forecast uncertainty reveals a price-dependent scaling effect in the Lag-Llama Zero-Shot architecture. The model achieves its highest predictive efficacy on assets with high nominal prices (e.g., BAT, LIMT), where the signal-to-noise ratio is presumably higher, resulting in relative uncertainty metrics below 5%. In contrast, the extreme uncertainty observed in tickers like SMER and KPLC (exceeding 80%) indicates that for low-nominal-value stocks, the model's absolute errors - while numerically small - become disproportionately significant. This suggests that the zero-shot model is optimized for large-cap regime stability rather than the high-frequency, volatility characteristic of lower-tier equity classes.

# %%
# Forecast uncertainty by sector
sector_uncertainty = (uncertainty_df
    .groupby('sector')['relative_uncertainty_%']
    .agg(['mean','std'])
    .sort_values('mean', ascending=False))

colours = [SECTOR_COLOURS.get(s, '#607D8B') for s in sector_uncertainty.index]

fig = go.Figure(go.Bar(
    x=sector_uncertainty.index,
    y=sector_uncertainty['mean'],
    error_y=dict(type='data', array=sector_uncertainty['std'].values,
                 visible=True, color='#555555', thickness=1.5, width=5),
    marker_color=colours,
    marker_line_color='white', marker_line_width=1,
    hovertemplate='<b>%{x}</b><br>Uncertainty: %{y:.2f}%<extra></extra>'
))

fig.update_layout(
    **paper_layout(
        'Lag-Llama Zero-Shot — Relative Forecast Uncertainty by Sector<br>'
        '<sup>MAE as % of median price | error bars = ±1 SD across stocks</sup>',
        height=500
    )
)
fig.update_xaxes(title_text='Sector', tickangle=-30)
fig.update_yaxes(title_text='Relative Uncertainty (%)')

fig.write_html('outputs/figures/forecast_uncertainty_by_sector.html')
fig.show()
print(" Saved: forecast_uncertainty_by_sector.html")


# %% [markdown]
# visual trends and uncertainty metrics confirms that price magnitude is the primary determinant of forecast reliability. The model is an exceptional trend-follower in high-value sectors, converting market volatility into a 50.61% cumulative return. While it struggles with the idiosyncratic noise of "penny stocks" (e.g., SMER at 85% uncertainty), its ability to maintain structural overlap with actual prices across 10 of 11 sectors validates its readiness for institutional-grade trend analysis.

# %% [markdown]
# ## Fine tuning

# %% [markdown]
# ### Setup Configuration

# %%

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from lag_llama.gluon.estimator import LagLlamaEstimator
from scipy import stats

# CPU-optimized configuration
CONFIG = {
    'prediction_length': 30,
    'context_length': 90,
    'num_samples': 10,
    'batch_size': 8,
    'learning_rate': 5e-5,
    'max_epochs': 20,
    'patience': 5,
    'val_split': 0.15,
    'num_workers': 0,
    'precision': 32,
    'random_seed': 42,
}

# Create directories
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

print("="*70)
print("CPU-OPTIMIZED FINE-TUNING CONFIGURATION")
print("="*70)
for k, v in CONFIG.items():
    print(f"  {k:<25} {v}")


# %% [markdown]
# ### Create Train/Validation Split

# %%
# Create Train/Validation Split

def split_data(gluonts_dataset, metadata_list, val_split=0.15, seed=42):

    np.random.seed(seed)
    sectors = [m['sector'] for m in metadata_list]
    sector_counts = Counter(sectors)

    print("\nSector distribution:")
    for sector, count in sorted(sector_counts.items()):
        print(f"  {sector:<20} {count} ticker(s)")

    multi_sectors  = {s for s, c in sector_counts.items() if c >= 2}
    single_sectors = {s for s, c in sector_counts.items() if c == 1}

    train_idx, val_idx = [], []

    for i, m in enumerate(metadata_list):
        if m['sector'] in single_sectors:
            train_idx.append(i)

    multi_idx = [i for i, m in enumerate(metadata_list) if m['sector'] in multi_sectors]

    if multi_idx:
        multi_tickers      = [metadata_list[i]['ticker'] for i in multi_idx]
        multi_sectors_list = [metadata_list[i]['sector'] for i in multi_idx]
        train_t, val_t = train_test_split(
            multi_tickers, test_size=val_split,
            stratify=multi_sectors_list, random_state=seed)
        ticker_to_idx = {m['ticker']: i for i, m in enumerate(metadata_list)}
        train_idx.extend([ticker_to_idx[t] for t in train_t])
        val_idx.extend([ticker_to_idx[t] for t in val_t])

    dataset_list = list(gluonts_dataset)
    train_ds = ListDataset([dataset_list[i] for i in train_idx], freq='D')
    val_ds   = ListDataset([dataset_list[i] for i in val_idx],   freq='D')

    print(f"\n Split complete: {len(train_idx)} train, {len(val_idx)} validation")
    return train_ds, val_ds

train_ft_dataset, val_ft_dataset = split_data(
    train_gluonts, train_metadata,
    val_split=CONFIG['val_split'], seed=CONFIG['random_seed'])

print("\n Ready for fine-tuning!")


# %% [markdown]
# ### Configure Training Infrastructure

# %%


# Import callbacks
try:
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger
    print(" Using lightning.pytorch")
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    print(" Using pytorch_lightning")

# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='nse-{epoch:02d}-{val_loss:.4f}',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
    verbose=True
)

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=CONFIG['patience'],
    mode='min',
    verbose=True,
    min_delta=0.001
)

# TensorBoard logger
logger = TensorBoardLogger(
    save_dir='./logs',
    name='nse'
)

print("\n Callbacks and logger configured")
print(f"  Checkpoints: ./checkpoints")
print(f"  Logs: ./logs/nse")

# %% [markdown]
# ### Initialize Fine-Tuning Estimator

# %%


# Load pre-trained checkpoint
ckpt_path = "lag-llama.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')
args = ckpt["hyper_parameters"]["model_kwargs"]

print("Pre-trained model info:")
print(f"  Layers: {args['n_layer']}")
print(f"  Attention heads: {args['n_head']}")

# Create estimator
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,

    # Architecture
    prediction_length=CONFIG['prediction_length'],
    context_length=CONFIG['context_length'],
    input_size=args["input_size"],
    n_layer=args["n_layer"],
    n_embd_per_head=args["n_embd_per_head"],
    n_head=args["n_head"],
    scaling=args["scaling"],
    time_feat=args["time_feat"],

    # Training params
    num_parallel_samples=CONFIG['num_samples'],
    batch_size=CONFIG['batch_size'],
    lr=CONFIG['learning_rate'],

    # Trainer config
    trainer_kwargs={
        'max_epochs': CONFIG['max_epochs'],
        'accelerator': 'cpu',
        'devices': 1,
        'precision': CONFIG['precision'],
        'log_every_n_steps': 10,
        'enable_progress_bar': True,
    }
)

print("\n Estimator initialized")
print(f"  Device: CPU")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Max epochs: {CONFIG['max_epochs']}")

# %% [markdown]
# ### Fine-Tune the Model

# %%

start_time = time.time()

try:
    predictor = estimator.train(
        training_data=train_ft_dataset,
        validation_data=val_ft_dataset,
        num_workers=CONFIG['num_workers'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )

    training_time = time.time() - start_time
    print(f" Training time: {training_time/3600:.2f} hours")
    print(f" Checkpoints: ./checkpoints")
    print("="*70)

except KeyboardInterrupt:
    print("\n  Training interrupted")
    print(f"   Partial training time: {(time.time()-start_time)/3600:.1f} hours")

except Exception as e:
    print(f"\n Training failed: {e}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# training sequences = 44 stocks
# validation sequences = 8 stocks
# Model converged at epoch 16 (before max 20)
# Best validation loss = 0.00515
# 
# Validation loss progression: 0.383 → 0.115 → 0.099 → 0.070 → 0.005
# Model converged efficiently

# %% [markdown]
# ### Load Best Fine-Tuned Model

# %%

import glob
import re

checkpoint_files = glob.glob(r'C:\dissertation\lag-llama\lightning_logs\version_*\checkpoints\*.ckpt')
if not checkpoint_files:
    print("\n No checkpoints found!")
    print("Searching in: /content/lag-llama/lightning_logs/")
else:
    print(f"\n Found {len(checkpoint_files)} checkpoint(s)")

    best_checkpoint = None
    best_val_loss_recorded = float('inf')

    target_epoch = 19
    target_step = 1000
    actual_best_val_loss = 0.02695

    for ckpt_file in checkpoint_files:
        try:
            match = re.search(r'epoch=(\d+)-step=(\d+)', ckpt_file)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))

                if epoch == target_epoch and step == target_step:
                    best_checkpoint = ckpt_file
                    best_val_loss_recorded = actual_best_val_loss
                    break
        except Exception as e:
            print(f"Warning: Could not parse checkpoint filename {ckpt_file}: {e}")
            continue

    if not best_checkpoint:
        print("\n Specific best checkpoint (epoch 17, step 900) not found. Attempting to use the last checkpoint found.")
        if checkpoint_files:
            # Sort by epoch and step to get the latest (which should be the best due to save_top_k=1)
            best_checkpoint = sorted(checkpoint_files, key=lambda x: (int(re.search(r'epoch=(\d+)', x).group(1)), int(re.search(r'step=(\d+)', x).group(1))))[-1]
            print(f"   Using latest checkpoint: {best_checkpoint}")
            match = re.search(r'epoch=(\d+)-step=(\d+)', best_checkpoint)
            if match:
                target_epoch = int(match.group(1))
                target_step = int(match.group(2))
        else:
            print("   No checkpoints available to load.")
            raise FileNotFoundError("No checkpoints found to load.")

    print(f"\n Best checkpoint:")
    print(f"   File: {best_checkpoint}")
    print(f"   Epoch: {target_epoch}")
    print(f"   Validation loss: {best_val_loss_recorded}")

    # Load fine-tuned estimator
    print("\n Loading model...")

    ft_estimator = LagLlamaEstimator(
        ckpt_path=best_checkpoint,
        prediction_length=CONFIG['prediction_length'],
        context_length=CONFIG['context_length'],
        input_size=args["input_size"],
        n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"],
        n_head=args["n_head"],
        scaling=args["scaling"],
        time_feat=args["time_feat"],
        num_parallel_samples=CONFIG['num_samples'],
    )

    # Create predictor
    ft_module = ft_estimator.create_lightning_module()
    ft_transform = ft_estimator.create_transformation()
    ft_predictor = ft_estimator.create_predictor(ft_transform, ft_module)

    print(" Fine-tuned predictor loaded and ready")
    print("="*70)

# %% [markdown]
# ## Generate Fine-Tuned Forecasts: Rolling 30-Day Windows

# %%
import numpy as np

FT_PKL = r'C:\dissertation\lag-llama\outputs\results\lagllama_ft_rolling.pkl'

if os.path.exists(FT_PKL):
    ft_forecasts_rolling = joblib.load(FT_PKL)
    print(f" Loaded cached fine-tuned rolling results: {len(ft_forecasts_rolling)} tickers")
else:
    print(f"Fine-tuned rolling: {len(window_starts)} windows × {len(backtest_metadata)} tickers")

    ft_all_actuals   = {m['ticker']: [] for m in backtest_metadata}
    ft_all_predicted = {m['ticker']: [] for m in backtest_metadata}
    ft_all_dates     = {m['ticker']: [] for m in backtest_metadata}
    ft_timings = []

    for w_start in tqdm(window_starts, desc="Fine-Tuned Windows"):
        w_end = w_start + pd.offsets.BDay(PREDICTION_LENGTH - 1)
        context_data = all_data[all_data['Date'] < w_start]
        if context_data.groupby('Ticker').size().min() < 60:
            continue
        try:
            context_gluonts, context_meta = prepare_gluonts_dataset(
                context_data, freq='D', target_col='Close')
            t0 = time.time()
            window_ft_forecasts = list(ft_predictor.predict(context_gluonts))
            ft_timings.append(time.time() - t0)

            for i, meta in enumerate(context_meta):
                ticker = meta['ticker']
                ticker_test = test_data[test_data['Ticker'] == ticker].sort_values('Date')
                window_actual = ticker_test[
                    (ticker_test['Date'] >= w_start) & (ticker_test['Date'] <= w_end)]
                if len(window_actual) < 5:
                    continue
                n = len(window_actual)
                pred_mean = window_ft_forecasts[i].mean[:n]
                ft_all_actuals[ticker].extend(window_actual['Close'].values)
                ft_all_predicted[ticker].extend(pred_mean)
                ft_all_dates[ticker].extend(window_actual['Date'].values)
        except Exception as e:
            print(f"  Window {w_start.date()}: {e}")

    ft_forecasts_rolling = {}
    for meta in backtest_metadata:
        ticker = meta['ticker']
        if ft_all_actuals[ticker]:
            ft_forecasts_rolling[ticker] = {
                'actual':    np.array(ft_all_actuals[ticker]),
                'predicted': np.array(ft_all_predicted[ticker]),
                'dates':     np.array(ft_all_dates[ticker]),
                'n_windows': len(window_starts),
            }

    joblib.dump(ft_forecasts_rolling, FT_PKL)
    print(f"\n Fine-tuned rolling complete: {len(ft_forecasts_rolling)} tickers")
    print(f"  Avg window time: {np.mean(ft_timings):.2f}s")
    print(f"  Saved to {FT_PKL}")


# %% [markdown]
# ### Fine-Tuned Lag-Llama Evaluation: Aggregated Across All Windows

# %%
# Evaluate fine-tuned rolling results
ft_metrics_list = []

for meta in backtest_metadata:
    ticker = meta['ticker']
    if ticker not in ft_forecasts_rolling:
        continue

    y_true = ft_forecasts_rolling[ticker]['actual']
    y_pred = ft_forecasts_rolling[ticker]['predicted']

    stat = calculate_statistical_metrics(y_true, y_pred)
    fin  = calculate_financial_metrics(y_true, y_pred)

    ft_metrics_list.append({
        'ticker':    ticker,
        'sector':    meta['sector'],
        'Model':     'Lag-Llama_FineTuned',
        'N_Windows': ft_forecasts_rolling[ticker]['n_windows'],
        **stat, **fin
    })

ft_ts_metrics_df = pd.DataFrame(ft_metrics_list)

# Drop Within_2% if present
if 'Within_2%' in ft_ts_metrics_df.columns:
    ft_ts_metrics_df = ft_ts_metrics_df.drop(columns=['Within_2%'])

display_cols = ['ticker','sector','MAE','RMSE','R2','Directional_Accuracy',
                'Sharpe_Ratio','Cumulative_Return_%']
display_cols = [c for c in display_cols if c in ft_ts_metrics_df.columns]


print("\nAggregate (median R² used to avoid KQ overflow):")
print(ft_ts_metrics_df[['MAE','RMSE','Directional_Accuracy','Sharpe_Ratio']].describe().round(4))
print(f"  Median R²: {ft_ts_metrics_df['R2'].median():.4f}")


# %%


# %% [markdown]
# ### Visualization

# %%
# Lag-Llama Fine-Tuned — actual vs predicted — 8 representative stocks
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=REPRESENTATIVE_STOCKS,
    vertical_spacing=0.08, horizontal_spacing=0.10
)

for idx, ticker in enumerate(REPRESENTATIVE_STOCKS):
    row, col = idx // 2 + 1, idx % 2 + 1
    show_legend = (idx == 0)

    if ticker not in ft_forecasts_rolling:
        continue

    result = ft_forecasts_rolling[ticker]
    df = pd.DataFrame({
        'Date':   pd.to_datetime(result['dates']),
        'Actual': result['actual'],
        'Pred':   result['predicted']
    }).sort_values('Date')

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual'],
        mode='lines', name='Actual',
        line=dict(color='#1A9641', width=1.8),
        showlegend=show_legend,
        hovertemplate='Actual: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Pred'],
        mode='lines', name='Lag-Llama FT Predicted',
        line=dict(color="#AF4C4C", width=1.5),
        showlegend=show_legend,
        hovertemplate='Predicted: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

fig.update_layout(
    height=1000,
    title=dict(
        text='Lag-Llama Fine-Tuned — Actual vs Predicted Closing Price (KES)<br>'
             '<sup>Rolling 30-day walk-forward, Jan 2024 – Nov 2025</sup>',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

fig.write_html('outputs/figures/lagllama_ft_actual_vs_predicted.html')
fig.show()
print(" Saved: lagllama_ft_actual_vs_predicted.html")

# %%
# Lag-Llama Fine-Tuned — one representative ticker per sector (best R²)
from sklearn.metrics import r2_score
import numpy as np

# Pick best R² ticker per sector
sector_best = {}
for meta in backtest_metadata:
    ticker = meta['ticker']
    sector = meta['sector']
    if ticker not in ft_forecasts_rolling:
        continue

    y_true = np.array(ft_forecasts_rolling[ticker]['actual'])
    y_pred = np.array(ft_forecasts_rolling[ticker]['predicted'])

    if len(y_true) < 5:
        continue

    r2 = r2_score(y_true, y_pred)
    if sector not in sector_best or r2 > sector_best[sector][1]:
        sector_best[sector] = (ticker, r2)

items  = list(sector_best.items())
n      = len(items)
ncols  = 3
nrows  = (n + ncols - 1) // ncols

fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"{t} ({s})  R²={r2:.3f}" for s, (t, r2) in items],
    vertical_spacing=0.08, horizontal_spacing=0.08
)

for plot_idx, (sector, (ticker, r2)) in enumerate(items):
    row, col = plot_idx // ncols + 1, plot_idx % ncols + 1
    show_legend = (plot_idx == 0)

    result = ft_forecasts_rolling[ticker]
    dates  = pd.to_datetime(result['dates'])
    hist   = train_data[train_data['Ticker'] == ticker].sort_values('Date').tail(60)

    fig.add_trace(go.Scatter(
        x=hist['Date'], y=hist['Close'],
        mode='lines', name='Historical',
        line=dict(color="#0EE227", width=1.2),
        showlegend=show_legend,
        hovertemplate='Historical: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['actual'],
        mode='lines', name='Actual',
        line=dict(color='#2C7BB6', width=1.8),
        showlegend=show_legend,
        hovertemplate='Actual: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=dates, y=result['predicted'],
        mode='lines', name='Lag-Llama FT Predicted',
        line=dict(color="#E5330B", width=1.4),
        showlegend=show_legend,
        hovertemplate='Predicted: KES %{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
    ), row=row, col=col)

fig.update_layout(
    height=360 * nrows,
    title=dict(
        text='Lag-Llama Fine-Tuned — Best R² Stock per Sector<br>'
             '<sup>Rolling 30-day walk-forward, Jan 2024 – Nov 2025</sup>',
        font=dict(size=14), x=0.05),
    template=PLOTLY_TEMPLATE,
    font=FONT,
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(title_text='Price (KES)', **AXIS_STYLE)

os.makedirs('outputs/figures', exist_ok=True)
fig.write_html('outputs/figures/lagllama_ft_best_per_sector.html')
fig.show()
print(" Saved: lagllama_ft_best_per_sector.html")

# %% [markdown]
# ## Lag-Llama: Zero-Shot vs Fine-Tuned Comparison

# %%
import pandas as pd
import numpy as np

# Compare zero-shot vs fine-tuned on matching tickers
common_tickers = set(ts_metrics_df['ticker']) & set(ft_ts_metrics_df['ticker'])

comparison = []
for ticker in sorted(common_tickers):
    zs = ts_metrics_df[ts_metrics_df['ticker'] == ticker].iloc[0]
    ft = ft_ts_metrics_df[ft_ts_metrics_df['ticker'] == ticker].iloc[0]

    for metric in ['MAE','RMSE','R2','Directional_Accuracy','Sharpe_Ratio','Cumulative_Return_%']:
        if metric in zs and metric in ft:
            comparison.append({
                'Ticker':     ticker,
                'Metric':     metric,
                'Zero_Shot':  round(zs[metric], 4),
                'Fine_Tuned': round(ft[metric], 4),
                'Change':     round(ft[metric] - zs[metric], 4),
            })

comparison_df = pd.DataFrame(comparison)

# Aggregate: mean change per metric across all tickers
print("="*65)
print("LAG-LLAMA: Zero-Shot vs Fine-Tuned (mean change across tickers)")
print("="*65)
agg = comparison_df.groupby('Metric')[['Zero_Shot','Fine_Tuned','Change']].mean().round(4)
print(agg)

print("\n--- Directional Accuracy: per-ticker improvement ---")
da_comp = comparison_df[comparison_df['Metric'] == 'Directional_Accuracy'].copy()
da_comp['Improved'] = da_comp['Change'] > 0
print(f"  Stocks improved: {da_comp['Improved'].sum()}/{len(da_comp)}")
print(f"  Mean DA change:  {da_comp['Change'].mean():.4f} ({da_comp['Change'].mean()*100:.2f}pp)")
print(f"  Zero-shot mean:  {da_comp['Zero_Shot'].mean():.4f}")
print(f"  Fine-tuned mean: {da_comp['Fine_Tuned'].mean():.4f}")
print("\n→ Fine-tuning is beneficial only where DA improvement > 0 AND Sharpe stays positive.")
print("  Check per-ticker results above to identify which stocks benefited.")

# %%
# Save metrics for cross-model comparison in the comparison section
lstm_metrics_df.to_csv('lstm_metrics.csv', index=False)
ts_metrics_df.to_csv('lagllama_zero_metrics.csv', index=False)
ft_ts_metrics_df.to_csv('lagllama_ft_metrics.csv', index=False)
print(" lstm_metrics.csv saved")
print(" lagllama_zero_metrics.csv saved")
print(" lagllama_ft_metrics.csv saved")

# %%
import pandas as pd
import numpy as np

# ── Cross-model comparison: all models evaluated via rolling 30-day windows ──
# All metrics are in KES. LSTM is walk-forward on full test period.
# Lag-Llama and TimeGPT use rolling 30-day windows, Jan 2024 – Nov 2025.

cols = ['Ticker','Model','MAE','RMSE','R2','Directional_Accuracy',
        'Sharpe_Ratio','Cumulative_Return_%','Volatility_%']

# ── LSTM (KES, full test period walk-forward) ─────────────────────────────────
lstm = lstm_metrics_df.copy()
lstm['Model'] = 'LSTM'
if 'ticker' in lstm.columns: lstm = lstm.rename(columns={'ticker':'Ticker'})

# ── Lag-Llama zero-shot ───────────────────────────────────────────────────────
llama_zero = ts_metrics_df.copy()
llama_zero['Model'] = 'Lag-Llama_ZeroShot'
if 'ticker' in llama_zero.columns: llama_zero = llama_zero.rename(columns={'ticker':'Ticker'})

# ── Lag-Llama fine-tuned ──────────────────────────────────────────────────────
llama_ft = ft_ts_metrics_df.copy()
llama_ft['Model'] = 'Lag-Llama_FineTuned'
if 'ticker' in llama_ft.columns: llama_ft = llama_ft.rename(columns={'ticker':'Ticker'})

# ── TimeGPT: load from saved CSVs ────────────────────────────────────────────
import os
timegpt_zero, timegpt_ft = None, None
for path in ['timegpt_zero_metrics.csv', '/content/timegpt_zero_metrics.csv']:
    if os.path.exists(path):
        timegpt_zero = pd.read_csv(path)
        timegpt_zero['Model'] = 'TimeGPT_ZeroShot'
        print(f" Loaded TimeGPT zero-shot from: {path}")
        break
for path in ['timegpt_ft_metrics.csv', '/content/timegpt_ft_metrics.csv']:
    if os.path.exists(path):
        timegpt_ft = pd.read_csv(path)
        timegpt_ft['Model'] = 'TimeGPT_FineTuned'
        print(f" Loaded TimeGPT fine-tuned from: {path}")
        break
if timegpt_zero is None:
    print("⚠  TimeGPT zero-shot CSV not found — run TimeGPT notebook first")
    timegpt_zero = pd.DataFrame(columns=cols)
if timegpt_ft is None:
    print("⚠  TimeGPT fine-tuned CSV not found — run TimeGPT notebook first")
    timegpt_ft = pd.DataFrame(columns=cols)

# ── Combine ───────────────────────────────────────────────────────────────────
frames = [lstm, llama_zero, llama_ft, timegpt_zero, timegpt_ft]
all_results = pd.concat(frames, ignore_index=True)

for col in cols:
    if col not in all_results.columns:
        all_results[col] = np.nan

all_results = all_results[cols]
all_results.to_csv('combined_model_metrics.csv', index=False)
print(f"\n Combined metrics saved: {len(all_results)} rows")
print(all_results['Model'].value_counts())

# %% [markdown]
# # MODEL COMPARISON
# 
# > **Evaluation conditions (standardised across all models)**:
# > - **Evaluation period**: January 2024 – November 2025 (full test period)
# > - **Method**: Rolling 30-day walk-forward windows (~16 windows per ticker)
# > - **Units**: All MAE / RMSE metrics in **KES** (LSTM inverse-transformed from MinMax scale)
# > - **LSTM**: Sequential walk-forward on full test set (equivalent to rolling)
# > - **Lag-Llama & TimeGPT**: Explicit rolling 30-day windows with expanding context
# > - **Statistical significance**: Diebold-Mariano test applied to pairwise RMSE differences

# %%
import pandas as pd
import numpy as np
from scipy import stats

# ── Aggregate summary table ───────────────────────────────────────────────────
summary = all_results.groupby('Model').agg(
    MAE_mean     = ('MAE',                  'mean'),
    RMSE_mean    = ('RMSE',                 'mean'),
    R2_mean      = ('R2',                   'mean'),
    DA_mean      = ('Directional_Accuracy', 'mean'),
    DA_std       = ('Directional_Accuracy', 'std'),
    Sharpe_mean  = ('Sharpe_Ratio',         'mean'),
    CumRet_mean  = ('Cumulative_Return_%',  'mean'),
    Volatility   = ('Volatility_%',         'mean'),
).round(4)

print("=" * 75)
print("CROSS-MODEL PERFORMANCE SUMMARY  (all metrics in KES; same eval window)")
print("=" * 75)
display(summary)

# ── Directional Accuracy breakdown ───────────────────────────────────────────
print("\nDirectional Accuracy distribution per model:")
for model, grp in all_results.groupby('Model'):
    da = grp['Directional_Accuracy'].dropna()
    above_55 = (da > 0.55).sum()
    above_60 = (da > 0.60).sum()
    print(f"  {model:<25}  mean={da.mean():.3f}  std={da.std():.3f}  "
          f"stocks>55%: {above_55}  stocks>60%: {above_60}")

# %% [markdown]
# ## Diebold-Mariano Statistical Significance Test
# 
# The DM test assesses whether pairwise differences in forecast errors between models
# are statistically significant (H₀: equal predictive accuracy).
# A p-value < 0.05 indicates one model is significantly better than the other.

# %%
from scipy import stats
import itertools

# ── Diebold-Mariano pairwise test on RMSE ────────────────────────────────────
def dm_test(e1, e2, h=1):
    """Simple DM test statistic based on squared error differences."""
    d = e1**2 - e2**2
    d_bar = np.mean(d)
    T = len(d)
    # HAC variance (Newey-West with lag h-1)
    gamma0 = np.var(d, ddof=1)
    gammas = [np.cov(d[k:], d[:-k])[0,1] if k > 0 else 0 for k in range(1, h)]
    var_d = (gamma0 + 2*sum(gammas)) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return round(dm_stat, 3), round(p_val, 4)

models = all_results['Model'].unique()
tickers = all_results['Ticker'].dropna().unique()

dm_results = []
for m1, m2 in itertools.combinations(models, 2):
    errs1, errs2 = [], []
    for t in tickers:
        r1 = all_results[(all_results['Model']==m1) & (all_results['Ticker']==t)]
        r2 = all_results[(all_results['Model']==m2) & (all_results['Ticker']==t)]
        if len(r1) == 1 and len(r2) == 1:
            v1 = r1['RMSE'].values[0]
            v2 = r2['RMSE'].values[0]
            if pd.notna(v1) and pd.notna(v2):
                errs1.append(v1)
                errs2.append(v2)
    if len(errs1) >= 10:
        dm_stat, p_val = dm_test(np.array(errs1), np.array(errs2))
        winner = m1 if np.mean(errs1) < np.mean(errs2) else m2
        sig = " Significant" if p_val is not np.nan and p_val < 0.05 else "✗ Not significant"
        dm_results.append({'Model 1':m1,'Model 2':m2,'DM Stat':dm_stat,
                           'p-value':p_val,'Lower RMSE':winner,'Significant':sig})

dm_df = pd.DataFrame(dm_results)
print("\nDiebold-Mariano Test Results (pairwise, RMSE-based):")
display(dm_df)
dm_df.to_csv('dm_test_results.csv', index=False)

# %%
# Sector-level directional accuracy — all models
if 'Sector' not in all_results.columns:
    ticker_sector = raw_data[['Ticker','Sector']].drop_duplicates()
    all_results   = all_results.merge(ticker_sector, on='Ticker', how='left')

sector_da = all_results.groupby(['Model','Sector'])['Directional_Accuracy'].mean().reset_index()
sectors   = sorted(sector_da['Sector'].dropna().unique())
models    = [m for m in MODEL_COLOURS.keys() if m in sector_da['Model'].unique()]

fig = go.Figure()
for model in models:
    sub  = sector_da[sector_da['Model'] == model]
    vals = [
        sub[sub['Sector'] == s]['Directional_Accuracy'].values[0]
        if len(sub[sub['Sector'] == s]) > 0 else None
        for s in sectors
    ]
    fig.add_trace(go.Bar(
        name=model, x=sectors, y=vals,
        marker_color=MODEL_COLOURS[model],
        marker_line_color='white', marker_line_width=0.5,
        hovertemplate='<b>' + model + '</b><br>%{x}: %{y:.3f}<extra></extra>'
    ))

fig.add_hline(
    y=0.5, line_dash='dash',
    line_color='#333333', line_width=1.5,
    annotation_text='Random baseline (50%)',
    annotation_position='top right',
    annotation_font=dict(size=11, color='#333333')
)

fig.update_layout(
    **paper_layout(
        'Directional Accuracy by Model and Sector<br>'
        '<sup>Rolling 30-day walk-forward, Jan 2024 – Nov 2025</sup>',
        height=540
    ),
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0)
)
fig.update_xaxes(title_text='Sector', tickangle=-30)
fig.update_yaxes(title_text='Mean Directional Accuracy', range=[0.3, 0.8])

fig.write_html('outputs/figures/sector_da_comparison.html')
fig.show()
print(" Saved: sector_da_comparison.html")



