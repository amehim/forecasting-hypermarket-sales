# import pandas as pd

# def build_holidays(start_year, end_year):
#     holidays = []

#     for year in range(start_year, end_year + 1):
#         thanksgiving = pd.Timestamp(f"{year}-11-01") + pd.offsets.Week(week=3, weekday=3)
#         christmas = pd.Timestamp(f"{year}-12-25")

#         holidays.append({'ds': thanksgiving, 'holiday': 'thanksgiving'})
#         holidays.append({'ds': thanksgiving + pd.Timedelta(days=1), 'holiday': 'black_friday'})
#         holidays.append({'ds': christmas, 'holiday': 'christmas'})

#     return pd.DataFrame(holidays)

"""
Utilities to create a Prophet-compatible holiday table for U.S. Thanksgiving,
Black Friday, and Christmas across a range of years.

Returns a DataFrame with columns: ['holiday','ds','lower_window','upper_window'].
"""
from __future__ import annotations
import pandas as pd
from datetime import date, timedelta

def thanksgiving_date(year: int) -> date:
    """U.S. Thanksgiving is the 4th Thursday in November."""
    # Find first day of November
    d = date(year, 11, 1)
    # weekday(): Monday=0 ... Sunday=6 ; we want Thursday=3
    days_to_thu = (3 - d.weekday()) % 7
    first_thu = d + timedelta(days=days_to_thu)
    fourth_thu = first_thu + timedelta(weeks=3)
    return fourth_thu

def build_holidays(start_year: int, end_year: int,
                   thanksg_win=(-2, 3),
                   blackfri_win=(0, 2),
                   xmas_win=(-3, 3)) -> pd.DataFrame:
    """
    Build Prophet holiday table. Windows are tuples (lower_window, upper_window)
    in days relative to the holiday date.
    """
    records = []
    for y in range(start_year, end_year + 1):
        th = thanksgiving_date(y)
        bf = th + timedelta(days=1)
        xmas = date(y, 12, 25)
        records += [
            {"holiday": "thanksgiving", "ds": pd.Timestamp(th), "lower_window": thanksg_win[0], "upper_window": thanksg_win[1]},
            {"holiday": "black_friday", "ds": pd.Timestamp(bf), "lower_window": blackfri_win[0], "upper_window": blackfri_win[1]},
            {"holiday": "christmas", "ds": pd.Timestamp(xmas), "lower_window": xmas_win[0], "upper_window": xmas_win[1]},
        ]
    return pd.DataFrame.from_records(records)

if __name__ == "__main__":
    df = build_holidays(2011, 2025)
    print(df.head())
    print(df.tail())
