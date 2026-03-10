import os
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

CUTOFF_DATE = '2024-01-01'

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SECTOR_MAPPING = {
    # Agriculture
    'EGAD': 'Agriculture', 'KAPC': 'Agriculture', 'KUKZ': 'Agriculture',
    'LIMT': 'Agriculture', 'SASN': 'Agriculture', 'WTK': 'Agriculture',
    # Automobile
    'CGEN': 'Automobile',
    # Banking
    'ABSA': 'Banking', 'SBIC': 'Banking', 'IMH': 'Banking',
    'DTK': 'Banking', 'SCBK': 'Banking', 'EQTY': 'Banking',
    'COOP': 'Banking', 'BKG': 'Banking', 'HFCK': 'Banking',
    'KCB': 'Banking', 'NCBA': 'Banking',
    # Commercial
    'XPRS': 'Commercial', 'SMER': 'Commercial', 'KQ': 'Commercial',
    'NMG': 'Commercial', 'SGL': 'Commercial', 'TPSE': 'Commercial',
    'SCAN': 'Commercial', 'UCHM': 'Commercial', 'LKL': 'Commercial',
    'NBV': 'Commercial',
    # Construction
    'CRWN': 'Construction', 'CABL': 'Construction', 'PORT': 'Construction',
    # Energy
    'TOTL': 'Energy', 'KEGN': 'Energy', 'KPLC': 'Energy', 'UMME': 'Energy',
    # Insurance
    'SLAM': 'Insurance', 'KNRE': 'Insurance',
    'LBTY': 'Insurance', 'BRIT': 'Insurance', 'CIC': 'Insurance',
    # Investment
    'OCH': 'Investment', 'CTUM': 'Investment', 'HAFR': 'Investment',
    # Investment Services
    'NSE': 'Investment Services',
    # Manufacturing
    'BOC': 'Manufacturing', 'BAT': 'Manufacturing',
    'EABL': 'Manufacturing', 'UNGA': 'Manufacturing', 'EVRD': 'Manufacturing',
    'AMAC': 'Manufacturing', 'FTGH': 'Manufacturing', 'CARB': 'Manufacturing',
    # Telecom
    'SCOM': 'Telecom'
}

SECTOR_COLOURS = {
    'Agriculture': '#2C7BB6', 'Automobile': '#D7191C',
    'Banking': '#1A9641', 'Commercial': '#F46D43',
    'Construction': '#762A83', 'Energy': '#E6AB02',
    'Insurance': '#5E4FA2', 'Investment': '#3288BD',
    'Investment Services': '#ABDDA4', 'Manufacturing': '#FDAE61',
    'Telecom': '#D53E4F',
}

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_nse_data(data_directory=None):
    """Load all NSE stock CSV files from the data directory."""
    if data_directory is None:
        data_directory = DATA_DIR
    data_dir = Path(data_directory)
    all_data = []
    for file in tqdm(list(data_dir.glob("*.csv")), desc="Loading CSVs"):
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
    print(f"Loaded {len(combined_df):,} records | {combined_df['Ticker'].nunique()} tickers")
    return combined_df

# ── Data Cleaning ─────────────────────────────────────────────────────────────

def clean_nse_data(df):
    """Clean raw NSE data."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    min_obs = 100
    ticker_counts = df.groupby('Ticker').size()
    valid_tickers = ticker_counts[ticker_counts >= min_obs].index
    removed = set(df['Ticker'].unique()) - set(valid_tickers)
    df = df[df['Ticker'].isin(valid_tickers)]
    if removed:
        print(f"Removed {len(removed)} tickers with < {min_obs} observations")
    print(f"Cleaning complete. Final dataset: {len(df):,} records")
    return df

# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    """Add technical indicators and lag features."""
    df = df.copy()
    engineered_data = []
    for ticker in tqdm(df['Ticker'].unique(), desc="Engineering features"):
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
        if len(ticker_df) < 50:
            continue
        for lag in [1, 7, 30]:
            ticker_df[f'Close_lag_{lag}'] = ticker_df['Close'].shift(lag)
            ticker_df[f'Volume_lag_{lag}'] = ticker_df['Volume'].shift(lag)
        for window in [7, 30]:
            ticker_df[f'SMA_{window}'] = ticker_df['Close'].rolling(window).mean()
            ticker_df[f'EMA_{window}'] = ticker_df['Close'].ewm(span=window).mean()
            ticker_df[f'Volume_MA_{window}'] = ticker_df['Volume'].rolling(window).mean()
        ticker_df['Daily_Return'] = ticker_df['Close'].pct_change()
        ticker_df['Price_Range'] = ticker_df['High'] - ticker_df['Low']
        rsi = RSIIndicator(close=ticker_df['Close'], window=14)
        ticker_df['RSI'] = rsi.rsi()
        macd = MACD(close=ticker_df['Close'])
        ticker_df['MACD'] = macd.macd()
        ticker_df['MACD_signal'] = macd.macd_signal()
        bb = BollingerBands(close=ticker_df['Close'], window=20)
        ticker_df['BB_width'] = bb.bollinger_hband() - bb.bollinger_lband()
        ticker_df['Volume_log'] = np.log1p(ticker_df['Volume'])
        ticker_df['Day_of_week'] = ticker_df['Date'].dt.dayofweek
        ticker_df['Month'] = ticker_df['Date'].dt.month
        ticker_df['Day_sin'] = np.sin(2 * np.pi * ticker_df['Day_of_week'] / 7)
        ticker_df['Day_cos'] = np.cos(2 * np.pi * ticker_df['Day_of_week'] / 7)
        ticker_df['Month_sin'] = np.sin(2 * np.pi * ticker_df['Month'] / 12)
        ticker_df['Month_cos'] = np.cos(2 * np.pi * ticker_df['Month'] / 12)
        engineered_data.append(ticker_df)
    final_df = pd.concat(engineered_data, ignore_index=True).dropna()
    return final_df

# ── Business Day Reindex ──────────────────────────────────────────────────────

def reindex_to_business_days(df):
    """Reindex to business day frequency."""
    reindexed_data = []
    for ticker in tqdm(df['Ticker'].unique(), desc="Reindexing"):
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.set_index('Date').sort_index()
        date_range = pd.date_range(
            start=ticker_df.index.min(),
            end=ticker_df.index.max(),
            freq='B'
        )
        ticker_df = ticker_df.reindex(date_range, method='ffill')
        ticker_df['Ticker'] = ticker
        ticker_df['Sector'] = df[df['Ticker'] == ticker]['Sector'].iloc[0]
        reindexed_data.append(ticker_df.reset_index().rename(columns={'index': 'Date'}))
    return pd.concat(reindexed_data, ignore_index=True)

# ── Train/Test Split ──────────────────────────────────────────────────────────

def create_train_test_split(df, cutoff_date=CUTOFF_DATE):
    """Chronological train-test split."""
    cutoff = pd.to_datetime(cutoff_date)
    train_data = df[df['Date'] < cutoff].copy()
    test_data = df[df['Date'] >= cutoff].copy()
    print(f"Train: {len(train_data):,} records | Test: {len(test_data):,} records")
    return train_data, test_data

# ── Feature Scaling ───────────────────────────────────────────────────────────

def scale_features(train_df, test_df):
    """Scale features using MinMax and Robust scalers."""
    price_features = ['Open', 'High', 'Low', 'Close'] + \
                     [col for col in train_df.columns if 'SMA' in col or 'EMA' in col
                      or ('lag' in col and 'Close' in col)]
    volume_features = ['Volume', 'Volume_log'] + \
                      [col for col in train_df.columns if 'Volume' in col and col != 'Volume']
    robust_features = ['RSI']

    price_scaler = MinMaxScaler()
    volume_scaler = RobustScaler()
    robust_scaler = RobustScaler()
    scalers = {}
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    price_cols = [col for col in price_features if col in train_df.columns]
    if price_cols:
        train_scaled[price_cols] = price_scaler.fit_transform(train_df[price_cols])
        test_scaled[price_cols] = price_scaler.transform(test_df[price_cols])
        scalers['price'] = price_scaler

    vol_cols = [col for col in volume_features if col in train_df.columns]
    if vol_cols:
        train_scaled[vol_cols] = volume_scaler.fit_transform(train_df[vol_cols])
        test_scaled[vol_cols] = volume_scaler.transform(test_df[vol_cols])
        scalers['volume'] = volume_scaler

    rob_cols = [col for col in robust_features if col in train_df.columns]
    if rob_cols:
        train_scaled[rob_cols] = robust_scaler.fit_transform(train_df[rob_cols])
        test_scaled[rob_cols] = robust_scaler.transform(test_df[rob_cols])
        scalers['robust'] = robust_scaler

    return train_scaled, test_scaled, scalers

# ── Full Pipeline ─────────────────────────────────────────────────────────────

def load_and_prepare_data(data_directory=None):
    """Run the full data pipeline and return all outputs."""
    raw = load_nse_data(data_directory)
    cleaned = clean_nse_data(raw)
    featured = engineer_features(cleaned)
    reindexed = reindex_to_business_days(featured)
    train, test = create_train_test_split(reindexed)
    train_scaled, test_scaled, scalers = scale_features(train, test)
    return {
        'raw': raw,
        'cleaned': cleaned,
        'featured': featured,
        'all_data': reindexed,
        'train': train,
        'test': test,
        'train_scaled': train_scaled,
        'test_scaled': test_scaled,
        'scalers': scalers,
    }