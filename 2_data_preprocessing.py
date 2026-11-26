import sys
import subprocess


def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


for pkg, imp in {'pandas': 'pandas', 'numpy': 'numpy', 'scikit-learn': 'sklearn', 'joblib': 'joblib'}.items():
    install_and_import(pkg, imp)

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

os.makedirs("models", exist_ok=True)

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

MAX_RUL_CLIP = 125
df['RUL_clipped'] = df['RUL'].clip(upper=MAX_RUL_CLIP)

# Select variables with high variance
setting_names = [c for c in ['setting_1', 'setting_2', 'setting_3'] if c in df.columns]
informative_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                       's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
features = setting_names + [s for s in informative_sensors if s in df.columns]

# Scale features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

scaler_path = "models/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"MinMaxScaler successfully saved to '{scaler_path}'")


# Generate 3D sequences (samples, sequence_length, feature_dim)
def create_sequences(data, sequence_length, feature_cols, target_col):
    x_seq, y_target = [], []
    for unit in data['unit_nr'].unique():
        unit_data = data[data['unit_nr'] == unit]
        unit_features = unit_data[feature_cols].values
        unit_targets = unit_data[target_col].values

        for i in range(len(unit_data) - sequence_length + 1):
            x_seq.append(unit_features[i: i + sequence_length])
            y_target.append(unit_targets[i + sequence_length - 1])

    return np.array(x_seq), np.array(y_target)


SEQUENCE_LENGTH = 50
X, y = create_sequences(df, SEQUENCE_LENGTH, features, 'RUL_clipped')

print("\n--- Phase 2: Data Preprocessing Summary ---")
print(f"3D Sequence Input Matrix (X) Shape: {X.shape}")
print(f"Target Array (y) Shape: {y.shape}")