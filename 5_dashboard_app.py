import os
import time
from datetime import datetime, timedelta
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
    page_title="AeroPredict | Live Engine Telemetry",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Modern Aviation Command Center
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
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        padding: 22px 28px;
        border-radius: 15px;
        border-left: 5px solid #38bdf8;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 4px;
    }
    .live-badge {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: monospace;
        font-size: 14px;
        font-weight: bold;
    }
    .status-card {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 17px;
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

# 5. Sidebar Control Center & Time-Travel State
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/jet-engine.png", width=65)
    st.markdown("### 🎛️ Flight Telemetry Controls")
    st.markdown("---")
    
    available_units = sorted(df['unit_nr'].unique())
    selected_engine = st.selectbox(
        "Select Turbofan Engine ID:",
        available_units,
        format_func=lambda x: f"Engine Unit #{x:03d}"
    )

    unit_data = df[df['unit_nr'] == selected_engine].copy().sort_values('time_cycles')
    max_cycle = int(unit_data['time_cycles'].max())

    # Generate realistic timestamps (1 flight cycle every 6 hours ending at current date)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    unit_data['timestamp'] = [now - timedelta(hours=6 * (max_cycle - c)) for c in unit_data['time_cycles']]

    # Initialize session state for cycle navigation
    if 'current_cycle' not in st.session_state or st.session_state.get('last_engine') != selected_engine:
        st.session_state.current_cycle = min(100, max_cycle)
        st.session_state.last_engine = selected_engine
        st.session_state.is_playing = False

    st.markdown("### ⏱️ Time-Travel & Live Simulation")
    
    # Rewind / Play-Pause / Fast-Forward Buttons
    b1, b2, b3 = st.columns(3)
    if b1.button("⏪ -10"):
        st.session_state.current_cycle = max(50, st.session_state.current_cycle - 10)
        st.session_state.is_playing = False
    if b2.button("⏯️ Live" if not st.session_state.is_playing else "⏸️ Pause"):
        st.session_state.is_playing = not st.session_state.is_playing
    if b3.button("⏩ +10"):
        st.session_state.current_cycle = min(max_cycle, st.session_state.current_cycle + 10)
        st.session_state.is_playing = False

    # Time Slider synced with session state
    selected_cycle = st.slider(
        "Flight Timeline (Cycle):",
        min_value=50,
        max_value=max_cycle,
        value=st.session_state.current_cycle,
        step=1
    )
    st.session_state.current_cycle = selected_cycle

    st.markdown("---")
    st.markdown("### 📡 Active Telemetry Sensors")
    sensor_labels = {
        's_2': 'LPC Outlet Temp (s_2)',
        's_3': 'HPC Outlet Temp (s_3)',
        's_4': 'LPT Outlet Temp (s_4)',
        's_11': 'HPC Static Pressure (s_11)',
        's_14': 'Core Speed (s_14)',
        's_15': 'Bypass Ratio (s_15)'
    }
    selected_sensors = st.multiselect(
        "Select Sensors to Monitor:",
        options=list(sensor_labels.keys()),
        default=['s_3', 's_4', 's_11'],
        format_func=lambda x: sensor_labels[x]
    )

# 6. Current Timestamp & Header Banner
current_row = unit_data[unit_data['time_cycles'] == selected_cycle].iloc[0]
current_time_str = current_row['timestamp'].strftime("%Y-%m-%d | %H:%M UTC")
status_indicator = "🟢 LIVE TELEMETRY STREAM" if st.session_state.is_playing else "🔵 HISTORICAL REPLAY MODE"

st.markdown(f"""
<div class="header-box">
    <div>
        <p class="header-title">✈️ AeroPredict | Turbofan Engine #{selected_engine:03d}</p>
        <p class="header-subtitle">Real-Time Remaining Useful Life (RUL) Prognostics • NASA C-MAPSS Telemetry</p>
    </div>
    <div class="live-badge">
        {status_indicator}<br>🕒 {current_time_str}
    </div>
</div>
""", unsafe_allow_html=True)

# 7. Inference Calculation
unit_data_scaled = unit_data.copy()
unit_data_scaled[features] = scaler.transform(unit_data[features])

sub_data_scaled = unit_data_scaled[unit_data_scaled['time_cycles'] <= selected_cycle]
sequence_input = sub_data_scaled[features].values[-50:]
input_tensor = torch.tensor(sequence_input, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    predicted_rul = max(0, int(round(model(input_tensor).item())))

actual_rul = int(current_row['RUL'])
error_diff = predicted_rul - actual_rul
health_percentage = min(100, int((predicted_rul / 125) * 100))
est_failure_date = (current_row['timestamp'] + timedelta(hours=6 * predicted_rul)).strftime("%Y-%m-%d %H:%M")

# 8. Top KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Flight Cycle & Time", f"Cycle {selected_cycle} / {max_cycle}", current_row['timestamp'].strftime("%b %d, %H:%M"))
col2.metric("🤖 AI Predicted RUL", f"{predicted_rul} Cycles", f"{error_diff:+d} vs Actual")
col3.metric("📅 Est. Failure Date", est_failure_date, f"Actual RUL: {actual_rul} cycles")
col4.metric("❤️ Engine Health Index", f"{health_percentage}%")

st.markdown("<br>", unsafe_allow_html=True)

# 9. Main Dashboard Layout
left_col, right_col = st.columns([1.1, 2])

with left_col:
    st.markdown("#### 🧭 RUL Cockpit Gauge (Max: 130 Cycles)")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Remaining Cycles", 'font': {'size': 16, 'color': '#cbd5e1'}},
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
        height=280,
        margin=dict(l=20, r=20, t=35, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Dynamic Status Alert Card
    if predicted_rul < 20:
        st.markdown("""
        <div class="status-card" style="background-color: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #fca5a5;">
            🚨 CRITICAL ALERT<br><span style="font-size:13px; font-weight:normal;">Ground aircraft immediately & schedule overhaul.</span>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_rul < 50:
        st.markdown("""
        <div class="status-card" style="background-color: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #fcd34d;">
            ⚠️ MAINTENANCE WARNING<br><span style="font-size:13px; font-weight:normal;">Degradation detected. Prepare spare parts.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="background-color: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7;">
            ✅ OPTIMAL CONDITION<br><span style="font-size:13px; font-weight:normal;">All turbine parameters within nominal limits.</span>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown(f"#### 📈 Real-Time Sensor Telemetry Stream (Up to {current_time_str})")
    
    history_data = unit_data[unit_data['time_cycles'] <= selected_cycle]
    
    if selected_sensors:
        fig_sensors = make_subplots(rows=len(selected_sensors), cols=1, shared_xaxes=True, vertical_spacing=0.06)
        colors = ['#38bdf8', '#f43f5e', '#fbbf24', '#a855f7', '#10b981', '#fb923c']
        
        for idx, sensor in enumerate(selected_sensors):
            fig_sensors.add_trace(
                go.Scatter(
                    x=history_data['timestamp'],
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
        
        fig_sensors.update_xaxes(title_text="Flight Timeline (Date & Time)", gridcolor="#1e293b", row=len(selected_sensors), col=1)
        fig_sensors.update_layout(
            height=420,
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

# 10. Auto-Play Live Stream Loop
if st.session_state.is_playing:
    if st.session_state.current_cycle < max_cycle:
        time.sleep(0.6)
        st.session_state.current_cycle += 1
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.rerun()
