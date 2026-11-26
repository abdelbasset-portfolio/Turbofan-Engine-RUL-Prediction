import sys
import subprocess

# Auto-install essential dependencies
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

for pkg, imp in {'torch': 'torch', 'numpy': 'numpy', 'pandas': 'pandas', 'scikit-learn': 'sklearn', 'matplotlib': 'matplotlib'}.items():
    install_and_import(pkg, imp)

# Import required standard libraries
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Data
file_path_csv = "data/train_FD001.csv"
file_path_txt = "data/train_FD001.txt"

if os.path.exists(file_path_csv):
    df = pd.read_csv(file_path_csv)
elif os.path.exists(file_path_txt):
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    df = pd.read_csv(file_path_txt, sep=r'\s+', header=None, names=col_names)
else:
    raise FileNotFoundError("Data file missing from 'data/' directory.")

if 'unit_number' in df.columns:
    df.rename(columns={'unit_number': 'unit_nr'}, inplace=True)

if 'RUL' not in df.columns:
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_time_cycles']
    df = df.merge(max_cycles, on='unit_nr', how='left')
    df['RUL'] = df['max_time_cycles'] - df['time_cycles']
    df.drop(columns=['max_time_cycles'], inplace=True)

df['RUL_clipped'] = df['RUL'].clip(upper=125)

setting_names = [c for c in ['setting_1', 'setting_2', 'setting_3'] if c in df.columns]
informative_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                       's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
features = setting_names + [s for s in informative_sensors if s in df.columns]

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

def create_sequences(data, sequence_length, feature_cols, target_col):
    x_seq, y_target = [], []
    for unit in data['unit_nr'].unique():
        unit_data = data[data['unit_nr'] == unit]
        unit_features = unit_data[feature_cols].values
        unit_targets = unit_data[target_col].values
        for i in range(len(unit_data) - sequence_length + 1):
            x_seq.append(unit_features[i : i + sequence_length])
            y_target.append(unit_targets[i + sequence_length - 1])
    return np.array(x_seq), np.array(y_target)

X, y = create_sequences(df, 50, features, 'RUL_clipped')

# 2. Define LSTM Architecture
class RUL_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(RUL_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# 3. Evaluate Model
model_path = "models/lstm_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run 3_model_training.py first.")

model = RUL_LSTM(input_dim=len(features))
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    predictions = model(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()

rmse = np.sqrt(mean_squared_error(y, predictions))
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

print("\n--- Phase 4: Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
print(f"Mean Absolute Error (MAE): {mae:.2f} cycles")
print(f"R-squared Score (R2): {r2:.4f}")

# 4. Plot Results
plt.figure(figsize=(12, 5))
plt.plot(y[:200], label='Actual RUL', color='blue', linewidth=2)
plt.plot(predictions[:200], label='Predicted RUL', color='red', linestyle='--', linewidth=2)
plt.title('RUL Estimation - Actual vs Predicted (First 200 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.legend()
plt.grid(True)
plt.show()