# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np

# def evaluate(y_true, y_pred):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = mean_squared_error(y_true, y_pred, squared=False)
#     mape = (np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))).mean()*100
#     return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

# src/train.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(mse ** 0.5)  # manual RMSE (no 'squared' kwarg)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float((np.abs((y_true - y_pred) / denom)).mean() * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}
