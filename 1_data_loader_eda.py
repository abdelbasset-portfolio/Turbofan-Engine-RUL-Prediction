import sys
import subprocess

# Auto-installation mechanism for missing packages
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Installing automatically...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'torch': 'torch',
    'joblib': 'joblib',
    'streamlit': 'streamlit'
}

for pkg, imp in required_packages.items():
    install_and_import(pkg, imp)

import os
import pandas as pd
import numpy as np

# 1. Verify Local Data Path
file_path_csv = "data/train_FD001.csv"
file_path_txt = "data/train_FD001.txt"

if os.path.exists(file_path_csv):
    file_path = file_path_csv
    df = pd.read_csv(file_path)
elif os.path.exists(file_path_txt):
    file_path = file_path_txt
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
else:
    raise FileNotFoundError("Data file not found. Place 'train_FD001.csv' or 'train_FD001.txt' inside 'data/' folder.")

# Ensure consistent naming schema
if 'unit_number' in df.columns:
    df.rename(columns={'unit_number': 'unit_nr'}, inplace=True)

# 2. Compute RUL and Clipped RUL Target
if 'RUL' not in df.columns:
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_time_cycles']
    df = df.merge(max_cycles, on='unit_nr', how='left')
    df['RUL'] = df['max_time_cycles'] - df['time_cycles']
    df.drop(columns=['max_time_cycles'], inplace=True)

MAX_RUL_CLIP = 125
df['RUL_clipped'] = df['RUL'].clip(upper=MAX_RUL_CLIP)

print("\n--- Phase 1: Data Loading & EDA Summary ---")
print(f"Dataset File Used: {file_path}")
print(f"Dataset Dimensions: {df.shape}")
print(f"Total Unique Jet Engines: {df['unit_nr'].nunique()}")
print("\nFirst 5 Sample Rows:")
print(df.head())