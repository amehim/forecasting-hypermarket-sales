import pandas as pd

def add_calendar_features(df, date_col='ds'):
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week'] = df[date_col].dt.isocalendar().week.astype(int)
    df['month'] = df[date_col].dt.month
    return df
