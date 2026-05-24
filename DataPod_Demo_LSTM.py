#!/usr/bin/env python
# coding: utf-8

# ## DATA

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("DataPod_Hourly.csv")
df


# In[ ]:


df = df[["DATE", "DISSOLVED OXYGEN", "ORP", "PH", "CONDUCTIVITY","TEMPERATURE"]].copy()


# In[ ]:


df["DATE"] = pd.to_datetime(df["DATE"])
df.set_index("DATE", inplace=True)
df = df.sort_index(ascending=True)


# In[ ]:


df.columns = ["DO", "ORP", "PH", "COND", "TEMP"]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_vals = scaler.fit_transform(df)

n_past = 72   
n_future = 24  
n_features = 5 

def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        y.append(data[i:i + n_future, :])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_vals, n_past, n_future)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape

model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_past, n_features)),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    
    Dense(n_future * n_features),
    Reshape((n_future, n_features)) 
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=1)


# In[ ]:


latest_window = scaled_vals[-n_past:] 
latest_window = latest_window.reshape(1, n_past, n_features)

prediction_scaled = model.predict(latest_window)

forecast = scaler.inverse_transform(prediction_scaled[0])

forecast_df = pd.DataFrame(forecast, columns=df.columns)
forecast_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), 
                                  periods=24, freq='H')

print("24 Hour Forecast")
print(forecast_df.head())


# ## TESTING

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred_scaled = model.predict(X_test)


y_test_reshaped = y_test.reshape(-1, n_features)
y_pred_reshaped = y_pred_scaled.reshape(-1, n_features)

y_test_inv = scaler.inverse_transform(y_test_reshaped).reshape(y_test.shape)
y_pred_inv = scaler.inverse_transform(y_pred_reshaped).reshape(y_pred_scaled.shape)

print("Perofrmance Metrics")

for i, col_name in enumerate(df.columns):
    actual = y_test_inv[:, :, i].flatten()
    predicted = y_pred_inv[:, :, i].flatten()
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    print(f"{col_name:15} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")


# In[ ]:


def plot_indicator(variable_name, sequence_idx):
    column_list = list(df.columns)

    var_idx = column_list.index(variable_name)
    
    actual_values = y_test_inv[sequence_idx, :, var_idx]
    predicted_values = y_pred_inv[sequence_idx, :, var_idx]
    
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    plt.plot(actual_values, label='Actual', marker='o', color='gray', alpha=0.7)
    plt.plot(predicted_values, label='Forecasted', marker='o', linestyle='--', color='cornflowerblue')
    
    max_val = max(np.max(actual_values), np.max(predicted_values))
    plt.ylim(0, max_val * 1.15)
    
    plt.title(f"24-Hour Forecast: {variable_name} (Test Sample {sequence_idx})")
    plt.xlabel("Hours")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# In[ ]:


plot_indicator("TEMP", 1)


# In[ ]:




