#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data
DATA_FILE = "DataPod_Hourly.csv"   
STEP = pd.Timedelta(minutes=30)
MAX_GAP = pd.Timedelta(hours=1)
N_PAST = 144                       
N_FUTURE = 48                     
FEATURES = ["DO", "ORP", "PH", "COND", "TEMP"]
N_FEATURES = len(FEATURES)


DO_WEIGHT = 4.0
LOSS_WEIGHTS = np.array([DO_WEIGHT, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

## Clean
df = pd.read_csv(DATA_FILE)
df["DATE"] = pd.to_datetime(df["DATE"], format="%H:%M %m/%d/%Y")
df = df.rename(columns={
    "DISSOLVED OXYGEN": "DO",
    "CONDUCTIVITY": "COND",
    "TEMPERATURE": "TEMP"
})
df = (df.sort_values("DATE")
        .drop_duplicates(subset="DATE")
        .set_index("DATE"))[FEATURES]

raw_rows = len(df)
df = df[
    (df["DO"] > 0) & (df["DO"] < 20) &
    (df["PH"] >= 6.0) & (df["PH"] <= 10.0) &
    (df["TEMP"] > 40) & (df["TEMP"] < 100) &
    (df["COND"] > 20000) & (df["COND"] < 70000)
]
print(f"Rows: {raw_rows} raw -> {len(df)} after cleaning "
      f"({len(df)/raw_rows*100:.0f}% retained)")


gap_break = df.index.to_series().diff() > MAX_GAP
segment_id = gap_break.cumsum()
seg_sizes = df.groupby(segment_id).size()
usable = seg_sizes[seg_sizes >= N_PAST + N_FUTURE]
print(f"Contiguous segments: {len(seg_sizes)} total, "
      f"{len(usable)} long enough to use")

## Scale
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=FEATURES)
scaled["segment"] = segment_id.values

## Windows
X_list, y_list = [], []
for seg, g in scaled.groupby("segment"):
    vals = g[FEATURES].values
    if len(vals) < N_PAST + N_FUTURE:
        continue
    for i in range(N_PAST, len(vals) - N_FUTURE + 1):
        X_list.append(vals[i - N_PAST:i])
        y_list.append(vals[i:i + N_FUTURE])

X = np.array(X_list)
y = np.array(y_list)
print(f"Training windows: {len(X)}")

## Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train
from keras.models import Sequential
from keras.layers import Input, LSTM, Dropout, Dense, Reshape
from keras.callbacks import EarlyStopping

_w = tf.constant(LOSS_WEIGHTS, dtype=tf.float32)  

def weighted_mse(y_true, y_pred):
    se = tf.square(y_true - y_pred)
    return tf.reduce_mean(se * _w)

model = Sequential([
    Input(shape=(N_PAST, N_FEATURES)),
    LSTM(128, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(N_FUTURE * N_FEATURES),
    Reshape((N_FUTURE, N_FEATURES))
])
model.compile(optimizer='adam', loss=weighted_mse)

early_stop = EarlyStopping(monitor='val_loss', patience=5,
                           restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=1)


forecast_df = None
for seg in sorted(scaled["segment"].unique(), reverse=True):
    g = scaled[scaled["segment"] == seg]
    if len(g) >= N_PAST:
        window = g[FEATURES].values[-N_PAST:].reshape(1, N_PAST, N_FEATURES)
        last_ts = g.index[-1]
        pred_scaled = model.predict(window)
        forecast = scaler.inverse_transform(pred_scaled[0])
        forecast_df = pd.DataFrame(forecast, columns=FEATURES)
        forecast_df.index = pd.date_range(start=last_ts + STEP,
                                          periods=N_FUTURE, freq=STEP)
        print(f"\n24-Hour Forecast (from data ending {last_ts})")
        print(forecast_df.head(10).round(3))
        break


y_pred_scaled = model.predict(X_test)
y_test_inv = scaler.inverse_transform(
    y_test.reshape(-1, N_FEATURES)).reshape(y_test.shape)
y_pred_inv = scaler.inverse_transform(
    y_pred_scaled.reshape(-1, N_FEATURES)).reshape(y_pred_scaled.shape)


last_input_scaled = X_test[:, -1, :]                       
persist_scaled = np.repeat(last_input_scaled[:, None, :], N_FUTURE, axis=1)
persist_inv = scaler.inverse_transform(
    persist_scaled.reshape(-1, N_FEATURES)).reshape(y_test.shape)

print("\nPerformance Metrics (24h horizon, cleaned data)")
print(f"{'Variable':<10}{'MAE':>10}{'RMSE':>10}{'Persist MAE':>14}")
for i, col in enumerate(FEATURES):
    actual = y_test_inv[:, :, i].flatten()
    pred = y_pred_inv[:, :, i].flatten()
    persist = persist_inv[:, :, i].flatten()
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    pmae = mean_absolute_error(actual, persist)
    verdict = "BEATS" if mae < pmae else "WORSE"
    print(f"{col:<10}{mae:>10.4f}{rmse:>10.4f}{pmae:>14.4f}{verdict:>12}")


print("\nVariability check (std of predictions vs std of actuals)")
print(f"{'Variable':<10}{'Actual std':>12}{'Pred std':>12}{'Ratio':>8}")
for i, col in enumerate(FEATURES):
    a_std = y_test_inv[:, :, i].std()
    p_std = y_pred_inv[:, :, i].std()
    ratio = p_std / a_std if a_std else float('nan')
    print(f"{col:<10}{a_std:>12.4f}{p_std:>12.4f}{ratio:>8.2f}")
print("Ratio near 1.0 = healthy. Near 0 = flat-lining (predicting the mean).")


y_pred_24h = y_pred_inv[:, -1, :]
results = pd.DataFrame(y_pred_24h, columns=FEATURES)


# Results
## Classification
def classify_hab_risk(row):
    s = 0
    if row['TEMP'] > 77.0: s += 40
    if row['PH'] > 8.5:    s += 30
    if row['DO'] < 4.0:    s += 30
    return s

def assign_HAB_level(s):
    return "HIGH" if s >= 70 else "MEDIUM" if s >= 40 else "LOW"

def classify_fish_risk(row):
    s = 0
    if row['DO'] < 3.0:   s += 50
    elif row['DO'] < 5.0: s += 25
    if row['TEMP'] > 80.0:   s += 30
    elif row['TEMP'] > 75.0: s += 15
    if row['PH'] > 9.0 or row['PH'] < 6.0: s += 20
    return s

def assign_fish_level(s):
    return "CRITICAL" if s >= 70 else "MODERATE" if s >= 40 else "SAFE"

results['HAB_Risk_score'] = results.apply(classify_hab_risk, axis=1)
results['fish_risk_score'] = results.apply(classify_fish_risk, axis=1)
results['HAB_Warning'] = results['HAB_Risk_score'].apply(assign_HAB_level)
results['Fish_Status'] = results['fish_risk_score'].apply(assign_fish_level)
results['HAB_Risk_%'] = results['HAB_Risk_score'].astype(str) + '%'
results['fish_risk_%'] = results['fish_risk_score'].astype(str) + '%'

cols = ["DO", "PH", "TEMP", "HAB_Risk_%", "HAB_Warning",
        "fish_risk_%", "Fish_Status"]
print("\nRisk Assessment (first 15 test samples)")
print(results[cols].head(15).round(3))
print("\nRisk level distribution across all test samples:")
print(results['HAB_Warning'].value_counts().to_string())
print(results['Fish_Status'].value_counts().to_string())

## Plotting
def plot_indicator(variable_name, sequence_idx):
    var_idx = FEATURES.index(variable_name)
    actual_values = y_test_inv[sequence_idx, :, var_idx]
    predicted_values = y_pred_inv[sequence_idx, :, var_idx]
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    plt.plot(actual_values, label='Actual', marker='o', color='gray', alpha=0.7)
    plt.plot(predicted_values, label='Forecasted', marker='o', color='cornflowerblue')
    max_val = max(np.max(actual_values), np.max(predicted_values))
    plt.ylim(0, max_val * 1.15)
    plt.title(f"24-Hour Forecast: {variable_name} (Test Sample {sequence_idx})")
    plt.xlabel("Steps (30 min each)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


plot_indicator("DO", 1)