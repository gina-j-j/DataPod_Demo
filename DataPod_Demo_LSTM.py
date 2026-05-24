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
   "execution_count": 1,
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
   "execution_count": 2,
   "id": "c477343b-7873-4500-8692-69e0b268dfeb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>DEVICE</th>\n",
       "      <th>DATE</th>\n",
       "      <th>DISSOLVED OXYGEN</th>\n",
       "      <th>ORP</th>\n",
       "      <th>PH</th>\n",
       "      <th>CONDUCTIVITY</th>\n",
       "      <th>TEMPERATURE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>04:21 09/05/2025</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-204.399994</td>\n",
       "      <td>1.941</td>\n",
       "      <td>24440.000000</td>\n",
       "      <td>77.876599</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>01:29 08/07/2025</td>\n",
       "      <td>4.060000</td>\n",
       "      <td>-99.599998</td>\n",
       "      <td>5.873</td>\n",
       "      <td>46210.000000</td>\n",
       "      <td>76.321399</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>00:59 08/07/2025</td>\n",
       "      <td>3.920000</td>\n",
       "      <td>-107.400002</td>\n",
       "      <td>6.225</td>\n",
       "      <td>46750.000000</td>\n",
       "      <td>76.382599</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>00:30 08/07/2025</td>\n",
       "      <td>4.110000</td>\n",
       "      <td>-101.900002</td>\n",
       "      <td>6.227</td>\n",
       "      <td>46910.000000</td>\n",
       "      <td>76.353801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>00:00 08/07/2025</td>\n",
       "      <td>4.190000</td>\n",
       "      <td>-99.500000</td>\n",
       "      <td>6.342</td>\n",
       "      <td>46980.000000</td>\n",
       "      <td>76.332199</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16856</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>23:51 05/13/2024</td>\n",
       "      <td>2.360000</td>\n",
       "      <td>238.399994</td>\n",
       "      <td>8.896</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>72.316401</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16857</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>18:18 05/10/2024</td>\n",
       "      <td>7.250000</td>\n",
       "      <td>132.500000</td>\n",
       "      <td>7.351</td>\n",
       "      <td>116.099998</td>\n",
       "      <td>73.918400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16858</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>14:28 05/10/2024</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>74.500000</td>\n",
       "      <td>8.204</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>70.464199</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16859</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>20:43 05/09/2024</td>\n",
       "      <td>1.400000</td>\n",
       "      <td>70.500000</td>\n",
       "      <td>8.113</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>76.859602</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16860</th>\n",
       "      <td>Lifeguard Dock DataPod™</td>\n",
       "      <td>20:32 03/05/2024</td>\n",
       "      <td>34.889999</td>\n",
       "      <td>-858.099976</td>\n",
       "      <td>14.000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1809.400000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>16861 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                        DEVICE              DATE  DISSOLVED OXYGEN  \\\n",
       "0      Lifeguard Dock DataPod™  04:21 09/05/2025          0.000000   \n",
       "1      Lifeguard Dock DataPod™  01:29 08/07/2025          4.060000   \n",
       "2      Lifeguard Dock DataPod™  00:59 08/07/2025          3.920000   \n",
       "3      Lifeguard Dock DataPod™  00:30 08/07/2025          4.110000   \n",
       "4      Lifeguard Dock DataPod™  00:00 08/07/2025          4.190000   \n",
       "...                        ...               ...               ...   \n",
       "16856  Lifeguard Dock DataPod™  23:51 05/13/2024          2.360000   \n",
       "16857  Lifeguard Dock DataPod™  18:18 05/10/2024          7.250000   \n",
       "16858  Lifeguard Dock DataPod™  14:28 05/10/2024          1.000000   \n",
       "16859  Lifeguard Dock DataPod™  20:43 05/09/2024          1.400000   \n",
       "16860  Lifeguard Dock DataPod™  20:32 03/05/2024         34.889999   \n",
       "\n",
       "              ORP      PH  CONDUCTIVITY  TEMPERATURE  \n",
       "0     -204.399994   1.941  24440.000000    77.876599  \n",
       "1      -99.599998   5.873  46210.000000    76.321399  \n",
       "2     -107.400002   6.225  46750.000000    76.382599  \n",
       "3     -101.900002   6.227  46910.000000    76.353801  \n",
       "4      -99.500000   6.342  46980.000000    76.332199  \n",
       "...           ...     ...           ...          ...  \n",
       "16856  238.399994   8.896      0.000000    72.316401  \n",
       "16857  132.500000   7.351    116.099998    73.918400  \n",
       "16858   74.500000   8.204      0.000000    70.464199  \n",
       "16859   70.500000   8.113      0.000000    76.859602  \n",
       "16860 -858.099976  14.000      0.000000 -1809.400000  \n",
       "\n",
       "[16861 rows x 7 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"DataPod_Hourly.csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "673e00a2-cd0f-4c7e-aeea-bbdad65b8ab8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[[\"DATE\", \"DISSOLVED OXYGEN\", \"ORP\", \"PH\", \"CONDUCTIVITY\",\"TEMPERATURE\"]].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
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
   "execution_count": 5,
   "id": "b39842d6-d884-4cfa-af44-6ae5e3eb7fc8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns = [\"DO\", \"ORP\", \"PH\", \"COND\", \"TEMP\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
   "execution_count": 7,
   "id": "3f66f59a-1f51-4967-9b8a-28f9e17d0c0a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ginaj\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\layers\\rnn\\rnn.py:200: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(**kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m74s\u001b[0m 162ms/step - loss: 0.0636 - val_loss: 0.0021\n",
      "Epoch 2/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 163ms/step - loss: 0.0081 - val_loss: 0.0016\n",
      "Epoch 3/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m67s\u001b[0m 160ms/step - loss: 0.0060 - val_loss: 0.0015\n",
      "Epoch 4/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m68s\u001b[0m 161ms/step - loss: 0.0050 - val_loss: 0.0013\n",
      "Epoch 5/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m68s\u001b[0m 163ms/step - loss: 0.0044 - val_loss: 0.0011\n",
      "Epoch 6/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m61s\u001b[0m 146ms/step - loss: 0.0042 - val_loss: 0.0011\n",
      "Epoch 7/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m54s\u001b[0m 129ms/step - loss: 0.0036 - val_loss: 9.2330e-04\n",
      "Epoch 8/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m56s\u001b[0m 134ms/step - loss: 0.0034 - val_loss: 9.6099e-04\n",
      "Epoch 9/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m61s\u001b[0m 146ms/step - loss: 0.0031 - val_loss: 0.0011\n",
      "Epoch 10/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m57s\u001b[0m 136ms/step - loss: 0.0029 - val_loss: 9.7961e-04\n",
      "Epoch 11/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m68s\u001b[0m 162ms/step - loss: 0.0028 - val_loss: 0.0011\n",
      "Epoch 12/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m63s\u001b[0m 150ms/step - loss: 0.0026 - val_loss: 0.0015\n",
      "Epoch 13/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m68s\u001b[0m 163ms/step - loss: 0.0025 - val_loss: 0.0016\n",
      "Epoch 14/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m68s\u001b[0m 163ms/step - loss: 0.0024 - val_loss: 0.0010\n",
      "Epoch 15/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 165ms/step - loss: 0.0022 - val_loss: 0.0014\n",
      "Epoch 16/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 164ms/step - loss: 0.0020 - val_loss: 0.0019\n",
      "Epoch 17/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m70s\u001b[0m 166ms/step - loss: 0.0020 - val_loss: 0.0016\n",
      "Epoch 18/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 163ms/step - loss: 0.0019 - val_loss: 0.0030\n",
      "Epoch 19/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 165ms/step - loss: 0.0018 - val_loss: 0.0031\n",
      "Epoch 20/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 165ms/step - loss: 0.0018 - val_loss: 0.0021\n",
      "Epoch 21/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m68s\u001b[0m 163ms/step - loss: 0.0017 - val_loss: 0.0028\n",
      "Epoch 22/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m70s\u001b[0m 166ms/step - loss: 0.0018 - val_loss: 0.0020\n",
      "Epoch 23/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 164ms/step - loss: 0.0017 - val_loss: 0.0026\n",
      "Epoch 24/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 164ms/step - loss: 0.0017 - val_loss: 0.0041\n",
      "Epoch 25/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 164ms/step - loss: 0.0016 - val_loss: 0.0026\n",
      "Epoch 26/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 165ms/step - loss: 0.0015 - val_loss: 0.0020\n",
      "Epoch 27/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m70s\u001b[0m 166ms/step - loss: 0.0016 - val_loss: 0.0023\n",
      "Epoch 28/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 164ms/step - loss: 0.0015 - val_loss: 0.0024\n",
      "Epoch 29/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 163ms/step - loss: 0.0015 - val_loss: 0.0020\n",
      "Epoch 30/30\n",
      "\u001b[1m420/420\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 164ms/step - loss: 0.0015 - val_loss: 0.0028\n"
     ]
    }
   ],
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
   "execution_count": 8,
   "id": "4cc105de-6df9-45ec-bc20-49273a9142a6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 508ms/step\n",
      "--- 24 Hour Water Quality Forecast ---\n",
      "                           DO        ORP        PH          COND       TEMP\n",
      "2025-09-05 05:21:00  2.413461 -30.036409  3.564281  50582.167969  77.262589\n",
      "2025-09-05 06:21:00  2.408238 -30.650810  3.411238  49054.414062  79.872879\n",
      "2025-09-05 07:21:00  2.291712 -24.652990  3.315942  48575.648438  79.370979\n",
      "2025-09-05 08:21:00  2.462757 -27.285341  3.160418  52727.925781  69.413536\n",
      "2025-09-05 09:21:00  2.199865 -21.717329  3.061657  50271.804688  77.283272\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ginaj\\AppData\\Local\\Temp\\ipykernel_19812\\1523131549.py:9: FutureWarning: 'H' is deprecated and will be removed in a future version, please use 'h' instead.\n",
      "  forecast_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1),\n"
     ]
    }
   ],
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
   "execution_count": 34,
   "id": "b04c2672-215b-49e8-aca1-3e2fbad67e26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 59ms/step\n",
      "Perofrmance Metrics\n",
      "DO              | MAE: 0.8288 | RMSE: 0.9171\n",
      "ORP             | MAE: 45.3884 | RMSE: 60.4967\n",
      "PH              | MAE: 0.7497 | RMSE: 1.5689\n",
      "COND            | MAE: 5851.1986 | RMSE: 6985.4424\n",
      "TEMP            | MAE: 8.3257 | RMSE: 9.7507\n"
     ]
    }
   ],
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
   "execution_count": 29,
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
   "execution_count": 33,
   "id": "e0e065d2-b450-484a-b34a-ab58c8b65787",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0kAAAHWCAYAAACi1sL/AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAB31klEQVR4nO3dd3wUdf7H8feWZNMTAqQgELoNFUFERAFFlLOcegq2U8F2ini2OxveAZ4eViyIv7MgihUUUCx0EKWKSBGlEwKEJCQkpCeb7H5/f8SsuySBBLLZEF7Px+P7yO7Md+b7nd3vTuYz853vWCQZAQAAAAAkSdZAVwAAAAAAGhOCJAAAAADwQpAEAAAAAF4IkgAAAADAC0ESAAAAAHghSAIAAAAALwRJAAAAAOCFIAkAAAAAvBAkAQAAAIAXgiQAABqJCRMmaO7cuYGuRpM3adIkJScnN3i5drtdu3bt0j333NPgZQOoG4IkAH5x1llnafz48dqwYYMKCgqUkpKiKVOmqHPnzodczm6369dff5UxRg8//HCtyrr11ltljFGPHj2qnb9o0SL98ssvdd4Gf0pKSpIxptq0fPnyQFfP73r37q1Ro0YpOjr6iJYfNWpUjZ+fd1q0aJGkioPimvIUFxd71tuvXz/P9JtuuqnaspcsWSJjTJU2lZyc7LPejIwMff/997rqqqtqtU3t2rXTHXfcof/+97+SKtptbbZx1KhRR/AJVnXPPffo1ltvrXX+8PBwjR49Wr/88osKCgqUlZWlNWvW6JVXXlFiYmK91OlYkZCQoLFjx2rhwoXKy8uTMUb9+vWrkq+8vFzjxo3TyJEj5XA4AlBTALVlD3QFADRNjz76qPr06aPPPvtM69evV0JCgkaMGKGff/5Z55xzjn799ddql7vvvvvUtm3bBq5t4Hz88cf69ttvfaZlZmYGqDYN59xzz9Xo0aP13nvvKTc3t87LT58+Xdu2bfO8j4iI0P/+9z9Nnz5d06dP90zPyMjwvC4pKdEdd9xRZV0ul6vKtOLiYt1444366KOPfKYnJSWpT58+PoGVtzVr1uill16SJLVq1Up/+9vfNGPGDN1999168803D7lN999/v5KTk/Xdd99Jkp555hm98847nvk9e/bU/fffr2eeeUYbN270TF+/fv0h11tbw4cPV1ZWlt5///3D5rXb7fr+++910kkn6f3339f48eMVERGhU089VTfeeKNmzJihtLS0eqnXseDEE0/UY489pi1btuiXX37RueeeW2PeSZMm6dlnn9WNN96oSZMmNWAtAdSVIZFIpPpOvXv3NkFBQT7TOnXqZIqLi80HH3xQ7TItW7Y0OTk55sknnzTGGPPwww/Xqqxbb73VGGNMjx49qp2/aNEi88svvzT4ZxAWFlbjvKSkpDptY12Tw+EwFosl4O2gpvTwww8bY4xJSkqql/U1b97cGGPMqFGjqp0/adIkk5+ff9j19OvXzxhjzOeff26cTqdp3ry5z/zHH3/cpKWlme+//75Km0pOTjZfffWVz7T4+HiTn59vNm3adMhy7Xa72bdvn3nqqadqzHPNNdcYY4zp16+fX76TX375xSxatKhWea+99lpjjDE33HBDtW0vMjIy4G3sUGnSpEkmOTm53tYXERFhmjVrVuvvaebMmWbx4sUB/xxIJFLNie52APxi+fLlKisr85m2bds2/frrrzr55JOrXebZZ5/V5s2b9eGHH/q9fjabTU8++aS2bdumkpISJScn65lnnlFwcLBPvpq6MyUnJ/ucBa7s8te3b19NmDBBGRkZ2rNnz1HXs3379po6dar279+vwsJCLV++XJdeeqlPnsouYtddd53+85//aM+ePSoqKlJUVJQk6eyzz9asWbN04MABFRYW6rvvvqv2THerVq30zjvvKDU1VSUlJdqxY4feeOMNBQUFSZKaNWumF154QevXr1d+fr5yc3P17bff6vTTT6+yrhEjRmjDhg0qLCxUdna2Vq1apRtuuEFSRVe5F198UZK0c+dOT7expKQkSVLz5s114oknKjQ09Kg/vyP15ZdfqrS0VIMHD/aZfuONN2rq1KnVXn2qTkZGhjZu3Kj27dsfMt95552nli1bav78+XWu66BBg/T999+roKBAeXl5+vrrr3XKKaf45ImPj9e7776r3bt3q6SkRHv37tUXX3zh+cyTk5PVtWtX9e/fv0pXxep07NhRkrR06dIq80pLS5Wfn+95f9ppp2nSpEnavn27iouLlZaWpokTJyo2NtZnucoulJ07d9YHH3ygAwcOaN++fXrqqackSa1bt9YXX3yh3NxcpaWl6aGHHvJZvvJ3MGTIED3zzDNKS0tTQUGBvvzyS7Vu3fqwn6PFYtH999+vDRs2qLi4WOnp6frf//6nmJiYwy5bUFCgnJycw+arNG/ePJ133nlq1qxZrZcB0LDobgegQcXHx1fb1a5nz5669dZbdd5558kYc0Trjo6OVvPmzatMrzzI9/bOO+9o6NCh+uyzz/TSSy+pV69eeuKJJ3TyySfrL3/5yxGVL0lvvPGGMjMz9dRTTyk8PPyw+cPCwqrUOTc3V+Xl5YqLi9OyZcsUFham1157Tfv379ett96qmTNn6tprr9UXX3zhs9y//vUvOZ1Ovfjii3I4HHI6nbrgggs0a9YsrV69WmPGjJHb7dawYcO0cOFCnX/++Vq1apUkKTExUT/++KNiYmL01ltvadOmTTrhhBN07bXXKiwsTLm5uerQoYOuuuoqffbZZ0pOTlZ8fLz+9re/afHixTrllFM83avuuOMOjR8/Xp999pleffVVhYSE6PTTT1evXr30ySefaPr06erSpYtuvPFGPfDAA8rKypL0RzfDESNGaPTo0erfv78WL158xN9FdaprH06n0+egXpKKior05Zdf6oYbbtD//vc/SdLpp5+url276o477qg2MKyO3W5XmzZttH///kPmO/fcc+V2u7VmzZpabkmFv/71r3r//fc1Z84cPfroowoLC9M999yjJUuW6Mwzz1RKSookadq0aTr11FM1fvx47dy5U3FxcRo4cKDatm2rlJQUPfDAAxo/frwKCgr0zDPPSPLtqniwyvXecsstevrppw9Zx4EDB6pDhw6aNGmS0tPTdeqpp+quu+7SqaeeqnPOOadK/ilTpmjjxo167LHHdNlll+lf//qXsrOz9be//U0LFy7Uo48+qptuukkvvfSSVq1apR9++MFn+ZEjR8oYo+eee05xcXF64IEHNH/+fHXr1k0lJSU11vPNN9/U0KFDNWnSJL322mtq3769RowYoTPPPFN9+vRReXn5IbezLlavXi2r1apzzz1X33zzTb2tF0D9CvjlLBKJdHykm266yRhjzLBhw6rMW7Fihfnoo4+MVPeuaJXd7Q7Fu2vU6aefbowx5q233vJZz/PPP2+MMaZ///6eaTV14UpOTjaTJk2qUofvv//eWK3Ww9a5churU9lNZ9y4ccYYY/r06eNZLjw83Gzfvt3s2LHD052usovYtm3bTEhIiE85mzdvNrNmzfKZFhISYrZv327mzJnjmfbee++Z8vLyGrssSjLBwcFVuvAlJSWZ4uJi8+STT3qmzZgx47DdGw/V3W7UqFF17lZWm+52NfH+fCo/y2uuucZceumlxuVymdatWxtJ5rnnnjPbtm0zUvVdOJOTk83s2bNN8+bNTfPmzc1pp51mPv74Y2OMMa+++uoh6z958mSTmZl5yDwHd+MKDw832dnZ5s033/TJFxcXZ3JycjzTo6Oja/V7qkt3u5CQELNx40ZjjDHJycnm3XffNcOGDTMtW7asNu/B06677jpjjDHnnXdele/9f//7n2ea1Wo1u3btMi6XyzzyyCOe6dHR0aawsNDnN1j53e3evdtERER4pld2Dbzvvvt82oN3d7s+ffoYY6p2H7z44ournV6X76m6lJCQYIwx5p///Get10sikRo20d0OQIM48cQTNWHCBC1btqzKjeFDhw7VaaedpkcfffSoyhg+fLguuuiiKmndunU++Sq7q40bN85neuUN95dddtkR1+Htt9+W2+2udf4333yzxvpeeumlWrlypU+XpsLCQr311ltq3759lS5V77//vs+Z8m7duqlLly76+OOP1bx5c08KDw/XggUL1LdvX1ksFlksFl111VX66quvtHr16hrr6nQ6PVf5rFarYmNjVVBQoM2bN6t79+6efAcOHFDr1q111lln1fpz8DZmzBhZLJZ6v4pUXFxcbft47LHHqs0/d+5cZWdn6/rrr5ckXX/99frkk08OWcYll1yirKwsZWVlaf369Ro8eLAmT5582LbdvHnzOnXXkiqu0DRr1kyffPKJz/frcrm0cuVKXXDBBZ7tLi0tVf/+/WvVdaw2SkpK1KtXLz3//POSpGHDhundd99VWlqaXnvtNZ9uq95t0uFwqHnz5lqxYoUk+bSbSt6DVbjdbv3000+yWq2aOHGiZ3pubq42b96sDh06VFl+8uTJKigo8Lz//PPPtXfv3irdVL0NHjxYBw4c0Lx583w+y9WrVys/P9/zWdaXyu+6RYsW9bpeAPWH7nYA/C4+Pl7ffPONcnNzde211/oEEZGRkRo7dqxeeOGFQ97DY7Va1bJlS59p2dnZPvc9/fjjj9Ue5Ofk5PgcjCQlJcnlcvmMjiZVdC/Kycnx3KdxJOr67JWtW7dqwYIF1c5LSkrSypUrq0yvHNksKSnJp+viwWVXDrc+efLkGsuPjo5WcHCwoqOjtWHDhkPWtfKejeHDh6t9+/ay2//4F+Ldney5557TRRddpFWrVmnr1q2aO3euPv74Yy1btuyQ6/c3l8tV42ddnfLycn322We68cYb9eOPP6pt27b6+OOPD7nMihUr9OSTT8oYo6KiIm3cuLHWo/dZLJZa10364/ut6d6hynKdTqceffRRvfTSS8rIyNCKFSv09ddfa/LkyYfsUnc4eXl5evTRR/Xoo4+qbdu2GjBggP7xj3/ovvvuU25urv71r39JqriXbdSoUbr++usVHx/vs47qhoDftWtXle0oLi6u0mUxNze32u6TW7durTJt27ZtateuXY3b0rlzZ8XExNQ4smRcXFyNyx6Jyu/aHGHXYgD+R5AEwK+ioqI0a9YsxcTE6Pzzz68yLPA//vEPBQcHa8qUKZ7gpPIm62bNmikpKUl79+5Vq1attHPnTp9lj/aelaM5QLHZbNVOr2lo6IZwcNlWa0VngX/84x9au3ZttcsUFBRUuYG+Jk888YSefvppTZw40XOfiNvt1iuvvOIpS5I2bdqkE088UZdffrkGDRqka665Rvfee6/GjBmj0aNHH9G2BcrHH3+se+65R6NHj9batWt9ht6uTlZWVp0CsUr79++v8038lZ/5X//6V6Wnp1eZ730PzauvvqqvvvpKV111lS655BL95z//0eOPP64LL7ywxrZRF7t27dKkSZM0Y8YM7dixQzfddJMnSJo6darOPfdcvfDCC1q7dq0KCgpktVo1Z84cn3ZTqbpBMWoaKKOugWVNrFarMjIyanw2Vn0Py1/5XVfejweg8SFIAuA3DodDX331lbp06aKLLrqo2gPMtm3bKjY2Vr/99luVeSNHjtTIkSPVrVs3bdq0SRdddJHP/IO70dVWSkqKbDabOnfurE2bNnmmx8XFqVmzZp6b0qWKq1UHd1EKCgpqkIdlpqSk6MQTT6wy/aSTTvLMP5Tt27dLqjjjf6gD98zMTOXm5qpr166HXN+1116rhQsXVnnWUExMTJWDvaKiIk2dOlVTp05VUFCQpk+frpEjR2rs2LEqLS09Zs6gL1myRCkpKbrgggv0yCOP+K2cTZs26aabblJUVJTy8vJqtUzl97tv375aBWY7duzQuHHjNG7cOHXq1Elr167Vww8/rJtvvllS/VzVOHDggLZv3+5pSzExMbrooov073//W//5z388+Tp16nTUZdWkugdWd+rU6ZDPk9q+fbsuuugiLV269JCDO9SXytEODxd0Awgc7kkC4BdWq1VTpkxR7969NXjwYM89CAd77bXXdNVVV/mku+66S1LFQxevuuoqJScnq7S0VAsWLPBJBw4cOKK6VT689YEHHvCZXjmksPdoU9u3b1ffvn198t11110+Xc385dtvv1WvXr18RgALCwvTXXfdpeTk5GoDS2+rV6/Wtm3b9I9//KPakfYquyAaY/TFF1/oiiuuUI8ePWpcn8vlqnLm/tprr60yvPLBV6bKysr022+/yWKxeEYaLCwslKRq75FpDEOAe/v73/+u0aNH64MPPvBbGcuXL5fVaj3k53+wOXPmKDc3V0888US17bHy+w0NDZXD4fCZt337duXn5/tMLywsrPU9S6effnq1Xd3atm2rU045RZs3b5b0xxWgg9vNwb+9+nTLLbcoIiLC8/7aa69Vq1atNGvWrBqXmTp1qux2u+fqlzebzVZtt8Cj0aNHD7ndbi1fvrxe1wug/nAlCYBfvPTSS7ryyis1c+ZMxcbGVunG8tFHH0mS1qxZU2XY48pud7/++qu+/PLLeq/b+vXr9d577+lvf/ubYmJitHjxYp199tkaOnSoZsyYoe+++86T95133tGbb76pzz//XPPmzdMZZ5yhSy65pN6731Tn2Wef1Q033KBZs2bptddeU3Z2tm699Va1b99e11xzzWHP/BtjdMcdd2jWrFn69ddfNWnSJKWmpuqEE07QBRdcoLy8PP35z3+WVNGV7uKLL9bixYv11ltvaePGjUpMTNTgwYN13nnnKTc3V19//bVGjRqld999V8uWLdNpp52mm266yXNFo9LcuXOVnp6upUuXKiMjQyeffLJGjBihb775xnNDfeW9Y88884w+/fRTlZWV6auvvlJRUZHfhgC32+01dqeaMWOGioqKqp03c+ZMzZw5s97qUZ0lS5YoKytLF1100SGfT+QtPz9f99xzjz744AP9/PPP+vTTT5WZmam2bdvqsssu09KlS3XfffepS5cuWrBggaZOnarffvtN5eXluvrqq5WQkKBPP/3Us77Vq1frnnvu0ciRI7Vt2zbt27evxroMHDhQY8aM0cyZM7VixQoVFBSoQ4cOuu222+RwODzdKvPz87V48WI98sgjCgoKUmpqqi6++OLDPjfqaGRnZ2vJkiWaNGmS4uPj9cADD2jr1q16++23a1zm+++/1//+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6/XFF19Uu86IiAiVl5erR48ecrlcPvMqu+MBAAAA9Y2BG46QMW6t+OwJz+uD50nSis9HVpnnbxs3blTv3r19pvXp00d5eXnas2dPjcv9/PPPOvXUU7Vz505t377dJxUVFWnr1q0qKiry3J90sHPPPVcpKSn673//q9WrV2vbtm1KSkqqkm/r1q165ZVXdMkll2j69OkaNmyYJMnpdPoMLiFJa9askd1uV1xcXJU6VXbL27hxo3r16uWz3DnnnHP4DwoAAACoAUHSUdi57msteHuYCg+k+UwvzNmrBW8P0851Xzd4nd544w21adNG48eP14knnqg///nPGjNmjMaNG+e5H6k6EyZMUGxsrD755BOdddZZ6tChgy6++GK9++67slqtKi0t1XPPPafnn39eN998szp06KBevXrptttuk1QR/LRt21bXXXedOnTooPvuu09XX321Z/0hISEaP368+vXrp7Zt2+rcc89Vz549tXHjRknSzp07FRkZqQsvvFDNmzdXaGiotm7dqg8//FCTJ0/W1VdfrXbt2qlnz5567LHHdOmll0qSXnvtNQ0aNEgPP/ywOnXqpHvvvZeudgAAADhqpimnyMhIY4wxkZGRVeYlJSWZyZMnm6SkpKMqw2KxmsTOfUyHHn8xiZ37GIvFWmPe6OjoetmuSZMmmRkzZlQ7r2/fvmblypWmpKTE7N2714wdO9bYbDbP/EWLFpmXX365ynKdOnUy06ZNM9nZ2aawsND89ttvZty4cV7baTFPPPGESU5ONqWlpWbnzp3mscce88x/7rnnTGZmpsnLyzOffPKJuf/++01OTo6RZIKCgszHH39sUlJSTElJidmzZ4957bXXjMPh8Cz/xhtvmMzMTGOMMaNGjTKSjN1uN6NHjzY7duwwpaWlJjU11UybNs107drVs9ywYcPMrl27TGFhofnyyy/NQw895Cm3ulRf33tjTDabzVx++eU+3zeJVJ+JNkZqiEQ7I/k70caO33So2OCgFPjKBuqDCMTBcn0FSaQjTwRJJNKRJ9oYqSES7Yzk73S8tbG6nNBv6qm2QRIDNwAAcAgWi1UJnXorNCpexXkZSt+2vEHuNw1UuYEuG0D9anfG5Tpn8H99nu9ZkJOqFZ89EZBbQ44VBEkAANQgUAcXgTyo4YAKaDranXG5Btw5qcr08JhEDbhzUoPcQ3+snnQh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4OBgn3nXXHNNvVQMAAAAAAKhzleSrrvuOi1btkwnn3yyrr76agUFBenUU0/VhRdeqNzcXH/UEQAAAAAaTJ2DpCeeeEIPPvig/vznP8vpdOr+++/XSSedpKlTp2rXrl3+qCMAAAAANJg6B0kdO3bUN998I0lyOp0KDw+XJL388su666676rd2AAAAANDA6hwk5eTkKDIyUpKUmpqqrl27SpJiYmIUFhZWv7UDAAAAgAZW6yDp1FNPlSR9//33GjhwoCTps88+06uvvqq33npLn3zyiRYsWOCfWgIAAABAA6n16Hbr16/XqlWr9MUXX+izzz6TJD3zzDMqKyvTueeeq2nTpunpp5/2W0UBAAAAoCHUOkjq16+fhg0bpscff1wjR47UtGnT9M477+i5557zZ/0AAAAAoEHVurvdkiVLdPvttysxMVH33Xef2rVrp8WLF2vz5s165JFHFB8f7896AgAAAECDqPPADUVFRXrvvffUv39/denSRZ999pnuvfde7dq1S19++aU/6ggAAAAADabOQZK37du367///a+efvpp5efn67LLLquvegEAAABAQNT6nqSDnX/++brtttt0zTXXyO12a+rUqZo4cWJ91g0AAAAAGlydgqTExEQNHTpUQ4cOVadOnbRs2TL9/e9/19SpU1VUVOSvOgIAAABAg6l1kPTtt9/qoosuUlZWliZPnqx3331XW7Zs8WfdAAAAAKDB1TpIKisr07XXXquvv/5abrfbn3UCAAAAgICpdZB05ZVX+rMeAAAAANAoHNXodgAAAADQ1BAkAQAAAIAXgiQAAAAA8EKQBAAAAABeCJIAAAAAwAtBEgAAAAB4IUgCAAAAAC8ESQAAAADghSAJAAAAALwQJAEAAACAF4IkAAAAAPBCkAQAAAAAXgiSAAAAAMALQRIAAAAAeAlokHT33Xdr3bp1ys3NVW5urpYtW6ZBgwZ55jscDr3++uvKyspSfn6+Pv/8c8XFxQWwxgAAAACauoAGSXv27NFjjz2mHj166KyzztLChQv15Zdf6pRTTpEkvfzyy7riiis0ePBg9evXT61atdL06dMDWWUAAAAATZw9kIV//fXXPu+ffPJJ3XPPPTrnnHO0Z88e3X777brxxhu1aNEiSdKwYcO0adMm9erVSytXrqx2ncHBwXI4HJ73kZGRkiSbzSabzeanLakdm80m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",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
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
