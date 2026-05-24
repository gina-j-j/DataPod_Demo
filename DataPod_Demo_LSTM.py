{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b91e2f0f-5843-453d-886b-a286e12bac7e",
   "metadata": {},
   "source": [
    "## DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89920914-c4ea-42f0-a3f7-ea7b29456700",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c477343b-7873-4500-8692-69e0b268dfeb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"DataPod_Hourly.csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "673e00a2-cd0f-4c7e-aeea-bbdad65b8ab8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[[\"DATE\", \"DISSOLVED OXYGEN\", \"ORP\", \"PH\", \"CONDUCTIVITY\",\"TEMPERATURE\"]].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ab55d5a-2a9e-4eb6-b4a6-7c43176c308b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"DATE\"] = pd.to_datetime(df[\"DATE\"])\n",
    "df.set_index(\"DATE\", inplace=True)\n",
    "df = df.sort_index(ascending=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b39842d6-d884-4cfa-af44-6ae5e3eb7fc8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns = [\"DO\", \"ORP\", \"PH\", \"COND\", \"TEMP\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8eb1ebfb-3a01-4599-862e-05e124e392ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "scaler = MinMaxScaler()\n",
    "scaled_vals = scaler.fit_transform(df)\n",
    "\n",
    "n_past = 72   \n",
    "n_future = 24  \n",
    "n_features = 5 \n",
    "\n",
    "def create_sequences(data, n_past, n_future):\n",
    "    X, y = [], []\n",
    "    for i in range(n_past, len(data) - n_future + 1):\n",
    "        X.append(data[i - n_past:i, :])\n",
    "        y.append(data[i:i + n_future, :])\n",
    "    return np.array(X), np.array(y)\n",
    "\n",
    "X, y = create_sequences(scaled_vals, n_past, n_future)\n",
    "\n",
    "split = int(0.8 * len(X))\n",
    "X_train, X_test = X[:split], X[split:]\n",
    "y_train, y_test = y[:split], y[split:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f66f59a-1f51-4967-9b8a-28f9e17d0c0a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape\n",
    "\n",
    "model = Sequential([\n",
    "    LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_past, n_features)),\n",
    "    Dropout(0.2),\n",
    "    LSTM(64, activation='tanh', return_sequences=False),\n",
    "    Dropout(0.2),\n",
    "    \n",
    "    Dense(n_future * n_features),\n",
    "    Reshape((n_future, n_features)) \n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='mse')\n",
    "\n",
    "history = model.fit(X_train, y_train, epochs=30, batch_size=32, \n",
    "                    validation_data=(X_test, y_test), verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4cc105de-6df9-45ec-bc20-49273a9142a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "latest_window = scaled_vals[-n_past:] \n",
    "latest_window = latest_window.reshape(1, n_past, n_features)\n",
    "\n",
    "prediction_scaled = model.predict(latest_window)\n",
    "\n",
    "forecast = scaler.inverse_transform(prediction_scaled[0])\n",
    "\n",
    "forecast_df = pd.DataFrame(forecast, columns=df.columns)\n",
    "forecast_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), \n",
    "                                  periods=24, freq='H')\n",
    "\n",
    "print(\"--- 24 Hour Water Quality Forecast ---\")\n",
    "print(forecast_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dde584a4-a6d2-4cdc-89db-793baa6a7b41",
   "metadata": {},
   "source": [
    "## TESTING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b04c2672-215b-49e8-aca1-3e2fbad67e26",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
    "\n",
    "y_pred_scaled = model.predict(X_test)\n",
    "\n",
    "\n",
    "y_test_reshaped = y_test.reshape(-1, n_features)\n",
    "y_pred_reshaped = y_pred_scaled.reshape(-1, n_features)\n",
    "\n",
    "y_test_inv = scaler.inverse_transform(y_test_reshaped).reshape(y_test.shape)\n",
    "y_pred_inv = scaler.inverse_transform(y_pred_reshaped).reshape(y_pred_scaled.shape)\n",
    "\n",
    "print(\"Perofrmance Metrics\")\n",
    "\n",
    "for i, col_name in enumerate(df.columns):\n",
    "    actual = y_test_inv[:, :, i].flatten()\n",
    "    predicted = y_pred_inv[:, :, i].flatten()\n",
    "    \n",
    "    mae = mean_absolute_error(actual, predicted)\n",
    "    rmse = np.sqrt(mean_squared_error(actual, predicted))\n",
    "    \n",
    "    print(f\"{col_name:15} | MAE: {mae:.4f} | RMSE: {rmse:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "088cb7bc-b189-42a4-b73c-633490f2f721",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_indicator(variable_name, sequence_idx):\n",
    "    column_list = list(df.columns)\n",
    "\n",
    "    var_idx = column_list.index(variable_name)\n",
    "    \n",
    "    actual_values = y_test_inv[sequence_idx, :, var_idx]\n",
    "    predicted_values = y_pred_inv[sequence_idx, :, var_idx]\n",
    "    \n",
    "    plt.figure(figsize=(10, 5))\n",
    "    plt.style.use('dark_background')\n",
    "    plt.plot(actual_values, label='Actual', marker='o', color='gray', alpha=0.7)\n",
    "    plt.plot(predicted_values, label='Forecasted', marker='o', linestyle='--', color='cornflowerblue')\n",
    "    \n",
    "    max_val = max(np.max(actual_values), np.max(predicted_values))\n",
    "    plt.ylim(0, max_val * 1.15)\n",
    "    \n",
    "    plt.title(f\"24-Hour Forecast: {variable_name} (Test Sample {sequence_idx})\")\n",
    "    plt.xlabel(\"Hours\")\n",
    "    plt.ylabel(\"Value\")\n",
    "    plt.legend()\n",
    "    plt.grid(True, alpha=0.3)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0e065d2-b450-484a-b34a-ab58c8b65787",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_indicator(\"TEMP\", 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5397d9b-c2a8-44e8-b6c6-f9d02cd09bdf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
