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
    page_title="AeroPredict | Turbofan RUL Prognostics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Industrial CSS with Instant Custom Hover Popovers
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .stMetric {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 16px 20px;
        border-radius: 8px;
        transition: border-color 0.2s ease;
    }
    .stMetric:hover {
        border-color: #0284c7;
    }
    .header-container {
        background-color: #111827;
        padding: 24px 28px;
        border-radius: 8px;
        border-left: 4px solid #0284c7;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        color: #f9fafb;
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 6px;
        font-family: monospace;
    }
    .telemetry-badge {
        background-color: #0f172a;
        border: 1px solid #334155;
        color: #38bdf8;
        padding: 10px 16px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 13px;
        text-align: right;
    }
    .status-banner {
        padding: 16px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        margin-top: 12px;
        font-family: monospace;
    }

    /* Instant Custom CSS Tooltip Styling */
    .tooltip-wrapper {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip-trigger {
        color: #e5e7eb;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.8px;
        border-bottom: 1px dashed #38bdf8;
        padding-bottom: 3px;
    }
    .tooltip-info-tag {
        background-color: #1e293b;
        color: #38bdf8;
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 6px;
        border: 1px solid #334155;
        font-family: monospace;
    }
    .tooltip-box {
        visibility: hidden;
        opacity: 0;
        width: 340px;
        background-color: #1e293b;
        color: #f1f5f9;
        text-align: left;
        border-radius: 8px;
        padding: 12px 14px;
        position: absolute;
        z-index: 9999;
        top: 135%;
        left: 0;
        border: 1px solid #0284c7;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
        font-size: 12.5px;
        line-height: 1.5;
        font-weight: normal;
        transition: opacity 0.15s ease-in-out;
        pointer-events: none;
    }
    .tooltip-wrapper:hover .tooltip-box {
        visibility: visible;
        opacity: 1;
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

# 4. Load Data & Pretrained Weights
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
        st.error("Dataset file missing in 'data/' directory.")
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
        st.error("Trained model or scaler missing in 'models/' directory.")
        st.stop()

    scaler = joblib.load(scaler_path)
    model = RUL_LSTM(input_dim=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return df, features, scaler, model

df, features, scaler, model = load_resources()

# Sensor Metadata Dictionary (Names, Units, and Physical Explanations)
SENSOR_INFO = {
    's_2': {
        'name': 'S2 - LPC Outlet Temperature',
        'unit': '°R',
        'desc': 'Low-Pressure Compressor outlet temperature. Increases gradually as compressor efficiency drops due to blade fouling.'
    },
    's_3': {
        'name': 'S3 - HPC Outlet Temperature',
        'unit': '°R',
        'desc': 'High-Pressure Compressor outlet temperature. Rising values indicate thermal stress and internal airflow degradation.'
    },
    's_4': {
        'name': 'S4 - LPT Outlet Temperature',
        'unit': '°R',
        'desc': 'Low-Pressure Turbine exhaust gas temperature (EGT). One of the strongest indicators of turbine wear near end-of-life.'
    },
    's_11': {
        'name': 'S11 - HPC Static Pressure',
        'unit': 'psia',
        'desc': 'Static pressure at High-Pressure Compressor outlet. Drops or fluctuates as internal seals wear out.'
    },
    's_14': {
        'name': 'S14 - Core Rotor Speed',
        'unit': 'rpm',
        'desc': 'Physical rotational speed of the engine core shaft. Shifts as the control system compensates for power loss.'
    },
    's_15': {
        'name': 'S15 - Bypass Ratio',
        'unit': 'ratio',
        'desc': 'Ratio of air bypassing the engine core to air passing through it. Changes reflect aerodynamic imbalance.'
    }
}

# 5. Sidebar Controls
with st.sidebar:
    st.markdown("### SYSTEM PARAMETERS")
    st.markdown("---")
    
    available_units = sorted(df['unit_nr'].unique())
    selected_engine = st.selectbox(
        "Engine Unit ID",
        available_units,
        format_func=lambda x: f"UNIT-{x:03d}",
        help="Select an independent turbofan engine unit from the NASA fleet (100 units total). Each unit has a unique operational lifespan."
    )

    unit_data = df[df['unit_nr'] == selected_engine].copy().sort_values('time_cycles')
    max_cycle = int(unit_data['time_cycles'].max())

    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    unit_data['timestamp'] = [base_time - timedelta(hours=6 * (max_cycle - c)) for c in unit_data['time_cycles']]

    if 'current_cycle' not in st.session_state or st.session_state.get('last_engine') != selected_engine:
        st.session_state.current_cycle = min(100, max_cycle)
        st.session_state.last_engine = selected_engine
        st.session_state.is_playing = False

    st.markdown("### TIMELINE SIMULATION")
    
    b1, b2, b3 = st.columns(3)
    if b1.button("-10 Cyc", help="Step backward 10 flight cycles to inspect earlier healthy engine states."):
        st.session_state.current_cycle = max(50, st.session_state.current_cycle - 10)
        st.session_state.is_playing = False
    if b2.button("Play" if not st.session_state.is_playing else "Pause", help="Automatically advance flight cycles in real time to watch sensor degradation and RUL drop."):
        st.session_state.is_playing = not st.session_state.is_playing
    if b3.button("+10 Cyc", help="Step forward 10 flight cycles toward engine end-of-life."):
        st.session_state.current_cycle = min(max_cycle, st.session_state.current_cycle + 10)
        st.session_state.is_playing = False

    selected_cycle = st.slider(
        "Operational Cycle",
        min_value=50,
        max_value=max_cycle,
        value=st.session_state.current_cycle,
        step=1,
        help="Select a specific flight cycle. Starts at 50 because the LSTM network requires a 50-cycle sliding window of past readings."
    )
    st.session_state.current_cycle = selected_cycle

    sim_speed = st.select_slider(
        "Simulation Step Interval (sec)",
        options=[0.5, 1.0, 1.5, 2.0],
        value=1.0,
        help="Delay in seconds between each flight cycle when Live Play mode is active."
    )

    st.markdown("---")
    st.markdown("### TELEMETRY CHANNELS")
    selected_sensors = st.multiselect(
        "Active Sensors",
        options=list(SENSOR_INFO.keys()),
        default=['s_3', 's_4', 's_11'],
        format_func=lambda x: SENSOR_INFO[x]['name'],
        help="Choose which physical engine sensors to display on the main telemetry chart."
    )

    with st.expander("Sensor Glossary & Guide"):
        for code, info in SENSOR_INFO.items():
            st.markdown(f"**{info['name']}** (`{info['unit']}`)")
            st.caption(info['desc'])

    st.markdown("---")
    st.caption("ARCH: 2-Layer PyTorch LSTM | SEQ: 50 | DATA: NASA C-MAPSS FD001")

# 6. Header Section with Instant Hover Explanation
current_row = unit_data[unit_data['time_cycles'] == selected_cycle].iloc[0]
current_time_str = current_row['timestamp'].strftime("%Y-%m-%d %H:%M UTC")
mode_label = "MODE: LIVE STREAM" if st.session_state.is_playing else "MODE: MANUAL INSPECTION"

st.markdown(f"""
<div class="header-container">
    <div class="tooltip-wrapper">
        <p class="header-title">AeroPredict | Turbofan Prognostics & Health Management <span class="tooltip-info-tag">?</span></p>
        <p class="header-subtitle">ASSET: ENGINE UNIT-{selected_engine:03d} | MODEL: PYTORCH LSTM | DATASET: NASA FD001</p>
        <div class="tooltip-box">
            <b>System Overview:</b><br>
            This dashboard monitors a commercial turbofan jet engine using 14 active telemetry sensors. A 2-layer LSTM neural network analyzes the last 50 flight cycles to predict Remaining Useful Life (RUL) before failure.
        </div>
    </div>
    <div class="tooltip-wrapper">
        <div class="telemetry-badge">
            {mode_label}<br>SIMULATED LOG: {current_time_str}
        </div>
        <div class="tooltip-box" style="left: auto; right: 0;">
            <b>Flight Timeline Log:</b><br>
            Maps NASA sequential flight cycles to a realistic flight schedule (1 flight cycle = 6 operational hours). Use Play/Pause in the sidebar to simulate live flight progression.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 7. Model Inference
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

# 8. KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Operational Cycle",
    f"{selected_cycle} / {max_cycle}",
    f"{max_cycle - selected_cycle} cycles remaining",
    help="Current completed flight cycle out of the engine's total lifespan."
)
col2.metric(
    "Predicted RUL",
    f"{predicted_rul} Cycles",
    f"{error_diff:+d} vs Ground Truth",
    help="AI-estimated remaining cycles before failure, computed from the 50-step sensor sequence."
)
col3.metric(
    "Ground Truth RUL",
    f"{actual_rul} Cycles",
    f"Est. EOL: {est_failure_date}",
    help="True remaining cycles from the NASA dataset, plus the projected End-of-Life timestamp."
)
col4.metric(
    "Health Index",
    f"{health_percentage}%",
    help="Overall engine health score (0-100%) normalized against the 125-cycle healthy baseline."
)

st.markdown("<br>", unsafe_allow_html=True)

# 9. Main Analytics Grid
left_col, right_col = st.columns([1.1, 2])

with left_col:
    st.markdown("""
    <div class="tooltip-wrapper">
        <span class="tooltip-trigger">REMAINING USEFUL LIFE GAUGE <span class="tooltip-info-tag">?</span></span>
        <div class="tooltip-box">
            <b>Cockpit RUL Dial (Max 130 Cycles):</b><br>
            • <b>Green (50–130):</b> Healthy engine operation.<br>
            • <b>Amber (20–50):</b> Early wear detected; plan maintenance.<br>
            • <b>Red (0–20):</b> Critical failure threshold; ground aircraft immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cycles Remaining (Cap: 130)", 'font': {'size': 14, 'color': '#9ca3af'}},
        number={'font': {'size': 44, 'color': '#f9fafb'}},
        gauge={
            'axis': {'range': [0, 130], 'tickwidth': 1, 'tickcolor': "#6b7280"},
            'bar': {'color': "#0284c7"},
            'bgcolor': "#111827",
            'borderwidth': 1,
            'bordercolor': "#374151",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(220, 38, 38, 0.35)'},
                {'range': [20, 50], 'color': 'rgba(217, 119, 6, 0.35)'},
                {'range': [50, 130], 'color': 'rgba(5, 150, 105, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 3},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin=dict(l=20, r=20, t=35, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Clean Industrial Status Banner with Instant Hover Card
    if predicted_rul < 20:
        st.markdown("""
        <div class="tooltip-wrapper" style="width: 100%;">
            <div class="status-banner" style="background-color: rgba(220, 38, 38, 0.15); border: 1px solid #dc2626; color: #fca5a5;">
                STATUS: CRITICAL<br><span style="font-size:12px; font-weight:normal;">Immediate maintenance required. Ground asset.</span>
            </div>
            <div class="tooltip-box" style="width: 100%;">
                <b>Action Protocol (Critical):</b><br>
                Predicted RUL has dropped below 20 cycles. High probability of turbine/compressor failure. Remove engine from active flight duty immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_rul < 50:
        st.markdown("""
        <div class="tooltip-wrapper" style="width: 100%;">
            <div class="status-banner" style="background-color: rgba(217, 119, 6, 0.15); border: 1px solid #d97706; color: #fcd34d;">
                STATUS: ADVISORY WARNING<br><span style="font-size:12px; font-weight:normal;">Degradation trend confirmed. Schedule inspection.</span>
            </div>
            <div class="tooltip-box" style="width: 100%;">
                <b>Action Protocol (Advisory):</b><br>
                Predicted RUL is between 20 and 50 cycles. Sensor drift confirms progressive wear. Order replacement modules and schedule hangar slot.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tooltip-wrapper" style="width: 100%;">
            <div class="status-banner" style="background-color: rgba(5, 150, 105, 0.15); border: 1px solid #059669; color: #6ee7b7;">
                STATUS: NOMINAL<br><span style="font-size:12px; font-weight:normal;">All sensor parameters operating within limits.</span>
            </div>
            <div class="tooltip-box" style="width: 100%;">
                <b>Action Protocol (Nominal):</b><br>
                Predicted RUL exceeds 50 cycles. Thermal and pressure readings are stable. Engine cleared for normal commercial operation.
            </div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown(f"""
    <div class="tooltip-wrapper">
        <span class="tooltip-trigger">SENSOR DEGRADATION TELEMETRY — UNIT-{selected_engine:03d} <span class="tooltip-info-tag">?</span></span>
        <div class="tooltip-box" style="width: 400px;">
            <b>How to Read This Chart:</b><br>
            • <b>Horizontal Axis (X):</b> Flight cycles elapsed since engine installation.<br>
            • <b>Vertical Axis (Y):</b> Physical sensor measurement (Temperature in °R or Pressure in psia).<br>
            • <b>Hover over any curve</b> to inspect exact readings and physical degradation behavior at that cycle.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    history_data = unit_data[unit_data['time_cycles'] <= selected_cycle]
    
    if selected_sensors:
        fig_sensors = make_subplots(rows=len(selected_sensors), cols=1, shared_xaxes=True, vertical_spacing=0.06)
        colors = ['#0284c7', '#e11d48', '#d97706', '#7c3aed', '#059669', '#ea580c']
        
        for idx, sensor in enumerate(selected_sensors):
            s_info = SENSOR_INFO[sensor]
            fig_sensors.add_trace(
                go.Scatter(
                    x=history_data['time_cycles'],
                    y=history_data[sensor],
                    mode='lines',
                    name=s_info['name'],
                    hovertemplate=(
                        f"<b>{s_info['name']}</b><br>"
                        f"Cycle: %{{x}} | Value: %{{y:.2f}} {s_info['unit']}<br>"
                        f"<i>{s_info['desc']}</i><extra></extra>"
                    ),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(int(colors[idx % len(colors)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.06,)}"
                ),
                row=idx+1, col=1
            )
            fig_sensors.update_yaxes(title_text=f"{sensor.upper()} ({s_info['unit']})", gridcolor="#1f2937", row=idx+1, col=1)
        
        fig_sensors.update_xaxes(title_text="Operational Flight Cycles", gridcolor="#1f2937", row=len(selected_sensors), col=1)
        fig_sensors.update_layout(
            height=410,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font=dict(color="#9ca3af", size=12),
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("Select at least one telemetry channel from the sidebar to display sensor trends.")

# 10. Controlled Simulation Loop
if st.session_state.is_playing:
    if st.session_state.current_cycle < max_cycle:
        time.sleep(sim_speed)
        st.session_state.current_cycle += 1
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.rerun()
