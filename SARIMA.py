## SARIMAX model with explicit training step and conditional log-transform fallback.
# Model will trained first and then forecast.

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from itertools import product
import warnings

warnings.filterwarnings("ignore")

# === USER INPUTS ===
input_file = r'C:\Users\PC\OneDrive\Thesis\Data\Input_Forecast_Data\All_Company_Automotive_Source.xlsx'
output_file = r'C:\Users\PC\OneDrive\Thesis\Data\Output_Forecast_Data\All_Company_Automotive_Source_Result.xlsx'
forecast_horizon = 4
rmse_threshold = 25

# === PARAMETER GRIDS ===
p_values = range(0, 4)
d_values = range(0, 3)
q_values = range(0, 4)
P_values = range(0, 3)
D_values = range(0, 2)
Q_values = range(0, 3)
m_values = [16]  # Quarterly seasonality

# === FUNCTION TO FILL MISSING VALUES USING DECOMPOSITION ===
def fill_missing_with_decomposition(series):
    series = series.asfreq('Q')
    temp_series = series.interpolate(method='linear')
    decomposition = seasonal_decompose(temp_series, model='additive', period=4)
    reconstructed = decomposition.trend + decomposition.seasonal
    filled_series = series.copy()
    filled_series[series.isna()] = reconstructed[series.isna()]
    return filled_series

# === READ ALL SHEETS ===
all_sheets = pd.read_excel(input_file, sheet_name=None, parse_dates=True)

# === PREPARE OUTPUT WRITER ===
with pd.ExcelWriter(output_file) as writer:

    for sheet_name, df in all_sheets.items():
        print(f"\nProcessing sheet: {sheet_name}")

        df.columns = df.columns.str.strip()
        df.index = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.drop(df.columns[0], axis=1)
        df = df[~df.index.isna()]

        if df.index.empty:
            print(f"Skipping sheet: {sheet_name}, no valid datetime index.")
            continue

        try:
            forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.QuarterEnd(), periods=forecast_horizon, freq='Q')
        except Exception as e:
            print(f"Skipping sheet: {sheet_name}, invalid forecast start date: {e}")
            continue

        forecast_df = pd.DataFrame(index=forecast_index)
        rmse_results = {}

        for column in df.columns:
            raw_series = df[column]

            if raw_series.isna().all():
                print(f"Skipping {column}, all values are missing.")
                continue

            try:
                series = fill_missing_with_decomposition(raw_series)
            except Exception as e:
                print(f"Skipping {column}, decomposition failed: {e}")
                continue

            series = series.dropna()

            if len(series) < forecast_horizon + 12:
                print(f"Skipping {column}, not enough data after filling.")
                continue

            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

            best_rmse = float('inf')
            best_forecast = None

            # === TRAIN MODEL ON FULL SERIES ===
            for order in product(p_values, d_values, q_values):
                for seasonal_order in product(P_values, D_values, Q_values, m_values):
                    try:
                        model = SARIMAX(
                            scaled_series,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        trained_model = model.fit(disp=False)
                        forecast_scaled = trained_model.get_forecast(steps=forecast_horizon).predicted_mean
                        forecast_inverse = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

                        recent_actual = series[-forecast_horizon:]
                        rmse = sqrt(np.mean((forecast_inverse - recent_actual) ** 2))
                        percentage_rmse = (rmse / recent_actual.mean()) * 100

                        if percentage_rmse < best_rmse:
                            best_rmse = percentage_rmse
                            best_forecast = forecast_inverse

                    except Exception:
                        continue

            # === FALLBACK: LOG-TRANSFORM IF RMSE TOO HIGH ===
            if best_rmse > rmse_threshold and (series > 0).all():
                print(f"{column}: RMSE {best_rmse:.2f}% too high, retrying with log-transform...")

                log_series = np.log(series)
                log_scaler = MinMaxScaler()
                scaled_log_series = log_scaler.fit_transform(log_series.values.reshape(-1, 1)).flatten()

                for order in product(p_values, d_values, q_values):
                    for seasonal_order in product(P_values, D_values, Q_values, m_values):
                        try:
                            model = SARIMAX(
                                scaled_log_series,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            trained_model = model.fit(disp=False)
                            forecast_scaled = trained_model.get_forecast(steps=forecast_horizon).predicted_mean
                            forecast_log_inverse = log_scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                            forecast_final = np.exp(forecast_log_inverse)

                            recent_actual = series[-forecast_horizon:]
                            rmse = sqrt(np.mean((forecast_final - recent_actual) ** 2))
                            percentage_rmse = (rmse / recent_actual.mean()) * 100

                            if percentage_rmse < best_rmse:
                                best_rmse = percentage_rmse
                                best_forecast = forecast_final

                        except Exception:
                            continue

            if best_forecast is None:
                print(f"{column}: No suitable model found.")
                continue

            forecast_df[column] = best_forecast
            rmse_results[column] = round(best_rmse, 2)
            print(f"{column}: Final RMSE% = {best_rmse:.2f}")

        combined_df = pd.concat([df, forecast_df])
        combined_df.index.name = 'Date'

        combined_df.to_excel(writer, sheet_name=sheet_name)
        pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE (%)']).to_excel(writer, sheet_name=f'RMSE_{sheet_name}')

print("\n✅ Forecasting complete. Output saved to:", output_file)