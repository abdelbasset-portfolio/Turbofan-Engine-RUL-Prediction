import os
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(
    page_title="AeroPredict | Jet Engine RUL Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Modern Dark Aviation Theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .header-box {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #38bdf8;
        margin-bottom: 25px;
    }
    .header-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 5px;
    }
    .status-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Architecture
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

# 4. Load Data & Pretrained Model
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
        st.error("❌ Dataset missing in 'data/' folder.")
        st.stop()

    if 'unit_number' in df.columns:
        df.rename(columns={'unit_number': 'unit_nr'}, inplace=True)

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
        st.error("❌ Trained model or scaler missing in 'models/' folder.")
        st.stop()

    scaler = joblib.load(scaler_path)
    model = RUL_LSTM(input_dim=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return df, features, scaler, model

df, features, scaler, model = load_resources()

# 5. Header Banner
st.markdown("""
<div class="header-box">
    <p class="header-title">✈️ AeroPredict | Turbofan Jet Engine Health Monitor</p>
    <p class="header-subtitle">Real-Time Remaining Useful Life (RUL) Prognostics powered by PyTorch LSTM & NASA C-MAPSS Telemetry</p>
</div>
""", unsafe_allow_html=True)

# 6. Sidebar Control Center
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/jet-engine.png", width=70)
    st.markdown("### 🎛️ Flight Telemetry Controls")
    st.markdown("---")
    
    available_units = sorted(df['unit_nr'].unique())
    selected_engine = st.selectbox("Select Turbofan Engine ID:", available_units, format_func=lambda x: f"Engine Unit #{x:03d}")

    unit_data = df[df['unit_nr'] == selected_engine].copy().sort_values('time_cycles')
    unit_data_scaled = unit_data.copy()
    unit_data_scaled[features] = scaler.transform(unit_data[features])
    max_cycle = int(unit_data['time_cycles'].max())

    selected_cycle = st.slider(
        "Simulate Flight Cycle (Time Step):",
        min_value=50,
        max_value=max_cycle,
        value=min(140, max_cycle),
        step=1,
        help="Slide to travel back and forth in the engine's operational history."
    )

    st.markdown("---")
    st.markdown("### 📡 Sensor Telemetry Select")
    sensor_labels = {
        's_2': 'LPC Outlet Temp (s_2)',
        's_3': 'HPC Outlet Temp (s_3)',
        's_4': 'LPT Outlet Temp (s_4)',
        's_11': 'HPC Static Pressure (s_11)',
        's_14': 'Core Speed (s_14)',
        's_15': 'Bypass Ratio (s_15)'
    }
    selected_sensors = st.multiselect(
        "Active Sensors to Plot:",
        options=list(sensor_labels.keys()),
        default=['s_3', 's_4', 's_11'],
        format_func=lambda x: sensor_labels[x]
    )

    st.markdown("---")
    st.caption("Model Architecture: 2-Layer LSTM (Window: 50 Cycles) | Dataset: NASA FD001")

# 7. Inference Calculation
sub_data_scaled = unit_data_scaled[unit_data_scaled['time_cycles'] <= selected_cycle]
sequence_input = sub_data_scaled[features].values[-50:]
input_tensor = torch.tensor(sequence_input, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    predicted_rul = max(0, int(round(model(input_tensor).item())))

actual_rul = int(unit_data[unit_data['time_cycles'] == selected_cycle]['RUL'].values[0])
error_diff = predicted_rul - actual_rul
health_percentage = min(100, int((predicted_rul / 125) * 100))

# 8. Top KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Flight Cycle", f"{selected_cycle} / {max_cycle}", f"{max_cycle - selected_cycle} to Failure")
col2.metric("🤖 AI Predicted RUL", f"{predicted_rul} Cycles", f"{error_diff:+d} vs Actual")
col3.metric("🎯 Ground Truth RUL", f"{actual_rul} Cycles")
col4.metric("❤️ Estimated Engine Health", f"{health_percentage}%")

st.markdown("<br>", unsafe_allow_html=True)

# 9. Main Dashboard Layout: Gauge + Status on Left, Interactive Charts on Right
left_col, right_col = st.columns([1.1, 2])

with left_col:
    st.markdown("#### 🧭 RUL Cockpit Gauge")
    
    # Plotly Aviation Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Remaining Cycles", 'font': {'size': 18, 'color': '#cbd5e1'}},
        number={'font': {'size': 46, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 130], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#38bdf8"},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(239, 68, 68, 0.45)'},
                {'range': [20, 50], 'color': 'rgba(245, 158, 11, 0.45)'},
                {'range': [50, 130], 'color': 'rgba(16, 185, 129, 0.35)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=290,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Dynamic Status Alert Card
    if predicted_rul < 20:
        st.markdown("""
        <div class="status-card" style="background-color: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #fca5a5;">
            🚨 CRITICAL ALERT<br><span style="font-size:14px; font-weight:normal;">Ground aircraft immediately & schedule overhaul.</span>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_rul < 50:
        st.markdown("""
        <div class="status-card" style="background-color: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #fcd34d;">
            ⚠️ MAINTENANCE WARNING<br><span style="font-size:14px; font-weight:normal;">Degradation detected. Prepare spare parts.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="background-color: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7;">
            ✅ OPTIMAL CONDITION<br><span style="font-size:14px; font-weight:normal;">All turbine parameters within nominal limits.</span>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown(f"#### 📈 Live Sensor Degradation Telemetry (Engine #{selected_engine:03d})")
    
    history_data = unit_data[unit_data['time_cycles'] <= selected_cycle]
    
    if selected_sensors:
        fig_sensors = make_subplots(rows=len(selected_sensors), cols=1, shared_xaxes=True, vertical_spacing=0.06)
        colors = ['#38bdf8', '#f43f5e', '#fbbf24', '#a855f7', '#10b981', '#fb923c']
        
        for idx, sensor in enumerate(selected_sensors):
            fig_sensors.add_trace(
                go.Scatter(
                    x=history_data['time_cycles'],
                    y=history_data[sensor],
                    mode='lines',
                    name=sensor_labels[sensor],
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(int(colors[idx % len(colors)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.08,)}"
                ),
                row=idx+1, col=1
            )
            fig_sensors.update_yaxes(title_text=sensor, gridcolor="#1e293b", row=idx+1, col=1)
        
        fig_sensors.update_xaxes(title_text="Operational Flight Cycles", gridcolor="#1e293b", row=len(selected_sensors), col=1)
        fig_sensors.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            font=dict(color="#cbd5e1"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("Please select at least one sensor from the sidebar to visualize telemetry.")
