# AeroPredict - Turbofan Engine RUL Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://turbofan-engine-rul-prediction-abdelbasset.streamlit.app/)

A predictive maintenance and remaining useful life (RUL) estimation dashboard for commercial turbofan engines. Built with PyTorch (LSTM) and Streamlit using the NASA C-MAPSS (`FD001`) dataset.

**Live Application:** [https://turbofan-engine-rul-prediction-abdelbasset.streamlit.app/](https://turbofan-engine-rul-prediction-abdelbasset.streamlit.app/)

## Overview

The goal of this project is to estimate how many flight cycles a turbofan engine has left before requiring maintenance. Using multivariate time-series sensor data (temperatures, pressures, rotor speeds, and bypass ratio), the model tracks engine degradation over time and visualizes operational risk in an interactive dashboard.

Main features:
- **Deep Learning Prognostics:** A 2-layer LSTM network trained on 50-cycle sliding windows, achieving an average RMSE of ~12–15 cycles on unseen engines.
- **Piecewise Linear Degradation Target:** Caps maximum RUL at 125 cycles during training to reflect realistic mechanical wear (stable health in early cycles followed by linear degradation).
- **Interactive Dashboard:** Dark-themed Streamlit interface featuring timeline playback (Play, Stop, Step), a cockpit RUL gauge (Nominal, Warning, Critical zones), and multi-sensor degradation trends.
- **Automated Pipeline:** Includes `run_all.py` to run preprocessing, training, evaluation, and dashboard deployment in a single step.

## Model & Data Pipeline

1. **Feature Selection:** Removes constant/zero-variance channels from the 21 raw sensors, retaining 14 informative physical sensors plus 3 operational settings (17 features total).
2. **Sequence Framing:** Scales sensor readings with `MinMaxScaler` and structures them into overlapping 3D windows of shape `(samples, 50, 17)`.
3. **LSTM Architecture:** 2 stacked LSTM layers (`hidden_dim=64`, `dropout=0.2`) followed by a fully connected regression head (`64 -> 32 -> 1` with ReLU) to output remaining cycles.

## Project Structure

```text
Turbofan-Engine-RUL-Prediction/
├── data/
│   └── train_FD001.csv          # NASA C-MAPSS FD001 sensor dataset
├── models/
│   ├── lstm_model.pth           # Pre-trained PyTorch LSTM weights
│   └── scaler.pkl               # Fitted MinMaxScaler
├── 1_data_loader_eda.py         # Data loading and RUL target calculation
├── 2_data_preprocessing.py      # Feature selection, scaling, and 3D windowing
├── 3_model_training.py          # PyTorch LSTM model and training loop
├── 4_evaluation.py              # Performance metrics (RMSE, MAE, R²)
├── 5_dashboard_app.py           # Main Streamlit & Plotly dashboard
├── run_all.py                   # End-to-end pipeline script
├── requirements.txt
└── README.md
```

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/abdelbasset-portfolio/Turbofan-Engine-RUL-Prediction.git
   cd Turbofan-Engine-RUL-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard (using pre-trained weights in `models/`):
   ```bash
   streamlit run 5_dashboard_app.py
   ```

4. *(Optional)* Re-run the full training and evaluation pipeline from scratch:
   ```bash
   python run_all.py
   ```

## Dataset Reference

- **Source:** NASA Turbofan Jet Engine Simulation Data (`C-MAPSS FD001`), NASA Ames Prognostics Data Repository.
- **Scope:** 100 independent engine units run from healthy initial operation until failure under sea-level conditions.
