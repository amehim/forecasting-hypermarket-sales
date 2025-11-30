import pandas as pd

def build_holidays(start_year, end_year):
    holidays = []

    for year in range(start_year, end_year + 1):
        thanksgiving = pd.Timestamp(f"{year}-11-01") + pd.offsets.Week(week=3, weekday=3)
        christmas = pd.Timestamp(f"{year}-12-25")

        holidays.append({'ds': thanksgiving, 'holiday': 'thanksgiving'})
        holidays.append({'ds': thanksgiving + pd.Timedelta(days=1), 'holiday': 'black_friday'})
        holidays.append({'ds': christmas, 'holiday': 'christmas'})

    return pd.DataFrame(holidays)
