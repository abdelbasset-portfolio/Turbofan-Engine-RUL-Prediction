import sys
import subprocess


def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


for pkg, imp in {'streamlit': 'streamlit', 'torch': 'torch', 'pandas': 'pandas', 'joblib': 'joblib',
                 'matplotlib': 'matplotlib'}.items():
    install_and_import(pkg, imp)

import os
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("✈️ Predictive Maintenance & Interactive RUL Estimation Dashboard")
st.write("Real-time Prognostics and Health Management for Turbofan Jet Engines")


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


@st.cache_resource
def load_resources():
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
        st.error("❌ Data file missing! Make sure 'train_FD001.csv' or 'train_FD001.txt' is in 'data/' folder.")
        st.stop()

    if 'unit_number' in df.columns:
        df.rename(columns={'unit_number': 'unit_nr'}, inplace=True)

    # Calculate actual RUL for reference display
    if 'RUL' not in df.columns:
        max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_nr', 'max_time_cycles']
        df = df.merge(max_cycles, on='unit_nr', how='left')
        df['RUL'] = df['max_time_cycles'] - df['time_cycles']
        df.drop(columns=['max_time_cycles'], inplace=True)

    setting_names = [c for c in ['setting_1', 'setting_2', 'setting_3'] if c in df.columns]
    informative_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                           's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    features = setting_names + [s for s in informative_sensors if s in df.columns]

    scaler_path = "models/scaler.pkl"
    model_path = "models/lstm_model.pth"

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.error("❌ Trained model or scaler missing! Run 2_data_preprocessing.py and 3_model_training.py first.")
        st.stop()

    scaler = joblib.load(scaler_path)

    model = RUL_LSTM(input_dim=len(features))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return df, features, scaler, model


df, features, scaler, model = load_resources()

# Sidebar Engine Selector
st.sidebar.header("⚙️ Control Panel")
available_units = sorted(df['unit_nr'].unique())
selected_engine = st.sidebar.selectbox("Choose Engine Unit Number:", available_units)

# Filter Data for Selected Engine
unit_data = df[df['unit_nr'] == selected_engine].copy().sort_values('time_cycles')
unit_data_scaled = unit_data.copy()
unit_data_scaled[features] = scaler.transform(unit_data[features])

max_cycle = int(unit_data['time_cycles'].max())

# Operational Cycle Time Slider
st.sidebar.markdown("---")
selected_cycle = st.sidebar.slider(
    "Select Operational Cycle (Time Step):",
    min_value=50,
    max_value=max_cycle,
    value=max_cycle,
    step=1
)

st.markdown("---")
st.subheader(f"📊 Engine Unit #{selected_engine} Analysis at Cycle {selected_cycle} / {max_cycle}")

# Sensor Chart up to Selected Cycle
history_data = unit_data[unit_data['time_cycles'] <= selected_cycle]
st.line_chart(history_data[['time_cycles', 's_2', 's_3', 's_4', 's_11']].set_index('time_cycles'))

# Extract sequence of 50 cycles ending at selected_cycle
sub_data_scaled = unit_data_scaled[unit_data_scaled['time_cycles'] <= selected_cycle]

if len(sub_data_scaled) >= 50:
    sequence_input = sub_data_scaled[features].values[-50:]
    input_tensor = torch.tensor(sequence_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_rul = model(input_tensor).item()

    actual_rul = unit_data[unit_data['time_cycles'] == selected_cycle]['RUL'].values[0]

    st.markdown("---")
    st.subheader("🔮 Predictive Maintenance Diagnosis")
    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted RUL", f"{int(predicted_rul)} Cycles")
    col2.metric("Actual Ground Truth RUL", f"{int(actual_rul)} Cycles")

    if predicted_rul < 20:
        col3.error("🚨 CRITICAL: Maintenance Required Immediately!")
    elif predicted_rul < 50:
        col3.warning("⚠️ WARNING: Schedule Inspection Soon")
    else:
        col3.success("✅ HEALTHY: Engine Operating Normally")
else:
    st.info("Insufficient sequence length for prediction at this point (Requires at least 50 operational cycles).")
######(C:\Users\hp\AppData\Local\Python\bin\python.exe -m streamlit run 5_dashboard_app.py)