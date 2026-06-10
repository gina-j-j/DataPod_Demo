#!/usr/bin/env python
# coding: utf-8

# ## DATA

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("DataPod_Hourly.csv", parse_dates=["DATE"])
df = df.rename(columns={
    "DISSOLVED OXYGEN": "DO", 
    "CONDUCTIVITY": "COND", 
    "TEMPERATURE": "TEMP"
}).set_index("DATE").sort_index()

df = df[["DO", "ORP", "PH", "COND", "TEMP"]]
print("Data shape:", df.shape)

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_vals = scaler.fit_transform(df)

n_past = 72   
n_future = 24  
n_features = 5 

X = np.array([scaled_vals[i-n_past:i] for i in range(n_past, len(scaled_vals) - n_future + 1)])
y = np.array([scaled_vals[i:i+n_future] for i in range(n_past, len(scaled_vals) - n_future + 1)])

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


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

# In[26]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

metrics = []

for i, col_name in enumerate(df.columns):
    actual = y_test_inv[:, :, i].flatten()
    predicted = y_pred_inv[:, :, i].flatten()
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    metrics.append({
        "Variable": col_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.set_index("Variable", inplace=True)
display(metrics_df)


# In[18]:


y_pred_24h_lstm = y_pred_inv[:, 23, :]

results = pd.DataFrame(y_pred_24h_lstm, columns=df.columns)

def classify_hab_risk(row):
    HAB_risk_score = 0
    if row['TEMP'] > 77.0:
        HAB_risk_score += 40   
    if row['PH'] > 8.5:
        HAB_risk_score += 30  
    if row['DO'] < 4.0:
        HAB_risk_score += 30
    return HAB_risk_score

results['HAB_Risk_score'] = results.apply(classify_hab_risk, axis=1)

def assign_HAB_level(HAB_risk_score):
    if HAB_risk_score >= 70:
        return "HIGH"
    elif HAB_risk_score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


# In[22]:


def classify_fish_risk(row):
    fish_risk_score = 0
    if row['DO'] < 3.0:      
        fish_risk_score += 50
    elif row['DO'] < 5.0:    
        fish_risk_score += 25
     
    if row['TEMP'] > 80.0:    
        fish_risk_score += 30
    elif row['TEMP'] > 75.0:
        fish_risk_score += 15
        
    if row['PH'] > 9.0 or row['PH'] < 6.0:
        fish_risk_score += 20
        
    return fish_risk_score

results['fish_risk_score'] = results.apply(classify_fish_risk, axis=1)

def assign_fish_level(fish_risk_score):
    if fish_risk_score >= 70:
        return "CRITICAL"
    elif fish_risk_score >= 40:
        return "MODERATE"
    else:
        return "SAFE"


# In[23]:


results['HAB_Warning'] = results['HAB_Risk_score'].apply(assign_HAB_level)
results['Fish_Status'] = results['fish_risk_score'].apply(assign_fish_level)

results['HAB_Risk_%'] = results['HAB_Risk_score'].astype(str) + '%'
results['fish_risk_%'] = results['fish_risk_score'].astype(str) + '%'

cols_to_display = ["DO", "PH", "TEMP", "HAB_Risk_%", "HAB_Warning", "fish_risk_%", "Fish_Status"]
display(results[cols_to_display].head(15))


# In[24]:


def plot_indicator(variable_name, sequence_idx):
    column_list = list(df.columns)

    var_idx = column_list.index(variable_name)
    
    actual_values = y_test_inv[sequence_idx, :, var_idx]
    predicted_values = y_pred_inv[sequence_idx, :, var_idx]
    
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    plt.plot(actual_values, label='Actual', marker='o', color='gray', alpha=0.7)
    plt.plot(predicted_values, label='Forecasted', marker='o', color='cornflowerblue')
    
    max_val = max(np.max(actual_values), np.max(predicted_values))
    plt.ylim(0, max_val * 1.15)
    
    plt.title(f"24-Hour Forecast: {variable_name} (Test Sample {sequence_idx})")
    plt.xlabel("Hours")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# In[25]:

# input any of the variables here
plot_indicator("TEMP", 1)


# In[ ]:




