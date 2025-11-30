# import pandas as pd

# def add_calendar_features(df, date_col='ds'):
#     df['day_of_week'] = df[date_col].dt.dayofweek
#     df['week'] = df[date_col].dt.isocalendar().week.astype(int)
#     df['month'] = df[date_col].dt.month
#     return df

"""
Feature engineering for holiday sales forecasting.
- calendar features (dow, month, week-of-year)
- holiday flags (Thanksgiving, Christmas, Black Friday window)
- lag features (t-1, t-7, t-14) and rolling stats
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["dow"] = dt.dt.dayofweek        # 0=Mon
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    return df

def add_us_holiday_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    # Thanksgiving: 4th Thursday of Nov
    df["is_thanksgiving"] = ((dt.dt.month==11) & (dt.dt.weekday==3) & ((dt.dt.day-1)//7==3)).astype(int)
    # Christmas fixed date
    df["is_christmas"] = ((dt.dt.month==12) & (dt.dt.day==25)).astype(int)
    # Black Friday: day after Thanksgiving
    df["is_black_friday"] = df["is_thanksgiving"].shift(1, fill_value=0)
    # Holiday windows
    df["win_thanks_7"] = df["is_thanksgiving"].rolling(7, min_periods=1, center=True).max()
    df["win_xmas_7"] = df["is_christmas"].rolling(7, min_periods=1, center=True).max()
    return df

def add_lags(df: pd.DataFrame, target: str, lags=(1,7,14,28)) -> pd.DataFrame:
    df = df.copy()
    for l in lags:
        df[f"{target}_lag{l}"] = df[target].shift(l)
    return df

def add_rollups(df: pd.DataFrame, target: str, windows=(7,14,28)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{target}_rollmean{w}"] = df[target].shift(1).rolling(w).mean()
        df[f"{target}_rollstd{w}"] = df[target].shift(1).rolling(w).std()
    return df
