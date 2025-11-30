
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from prophet import Prophet

from .holidays_us import build_holidays
from .train import evaluate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--date", default="Date")
    p.add_argument("--target", default="Weekly_Sales")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--freq", default="W-FRI")
    p.add_argument("--aggregate", choices=["yes","no"], default="yes")
    p.add_argument("--out_fig", default="../figures")
    p.add_argument("--out_rep", default="../reports")
    args = p.parse_args()

    out_fig = Path(args.out_fig); out_fig.mkdir(parents=True, exist_ok=True)
    out_rep = Path(args.out_rep); out_rep.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.date not in df.columns or args.target not in df.columns:
        raise SystemExit(f"Missing columns: {args.date} or {args.target}")
    df[args.date] = pd.to_datetime(df[args.date])

    if args.aggregate == "yes":
        ts = df.groupby(args.date, as_index=False)[args.target].sum().sort_values(args.date)
        ts = ts.rename(columns={args.date:'ds', args.target:'y'})
    else:
        ts = df.rename(columns={args.date:'ds', args.target:'y'})[['ds','y']].sort_values('ds')

    holidays = build_holidays(int(ts['ds'].dt.year.min()), int(ts['ds'].dt.year.max()))

    m = Prophet(holidays=holidays, weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(ts)

    future = m.make_future_dataframe(periods=args.horizon, freq=args.freq)
    forecast = m.predict(future)

    if len(ts) > args.horizon:
        valid = ts.iloc[-args.horizon:].copy()
        pred_valid = forecast.set_index('ds').loc[valid['ds']][['yhat']].reset_index()
        metrics = evaluate(valid['y'].values, pred_valid['yhat'].values)
    else:
        metrics = {"MAE": None, "RMSE": None, "MAPE%": None}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig1 = m.plot(forecast); fig1.suptitle("Prophet Forecast with Holiday Effects", y=1.02)
    fig1.savefig(out_fig / f"prophet_forecast_{stamp}.png", bbox_inches="tight")
    fig2 = m.plot_components(forecast); fig2.suptitle("Prophet Components", y=1.02)
    fig2.savefig(out_fig / f"prophet_components_{stamp}.png", bbox_inches="tight")

    (out_rep / f"forecast_{stamp}.csv").write_text(forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False))
    import pandas as pd
    pd.DataFrame([metrics]).to_csv(out_rep / f"metrics_{stamp}.csv", index=False)
    print("Done. Metrics:", metrics)

if __name__ == "__main__":
    main()
