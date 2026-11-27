# AeroPredict: Turbofan Engine Remaining Useful Life (RUL) Prediction

An end-to-end Predictive Maintenance and Prognostics & Health Management (PHM) system built with **PyTorch (LSTM)** and **Streamlit** to estimate the Remaining Useful Life (RUL) of commercial turbofan jet engines using the **NASA C-MAPSS (FD001)** dataset.

---

## Project Overview

Unplanned jet engine failures lead to severe safety risks and costly operational delays. **AeroPredict** analyzes multivariate time-series telemetry from 14 informative engine sensors (temperature, pressure, rotor speed, and bypass ratio) to predict how many operational flight cycles remain before an engine requires maintenance.

### Core Capabilities
* **Deep Learning Prognostics:** 2-layer LSTM neural network trained on 50-cycle sliding window sequences, achieving an average error (RMSE) of ~12–15 cycles on unseen engine units.
* **Piecewise Linear RUL Target:** Caps maximum RUL at 125 cycles during training to reflect realistic mechanical degradation behavior (constant health in early life followed by linear wear).
* **Interactive Command Center (`AeroPredict`):** A dark-themed, aviation-inspired Streamlit dashboard featuring:
  * **Live Timeline Simulation:** Play, pause, rewind, or fast-forward through an engine's operational flight history with simulated timestamps.
  * **Cockpit RUL Gauge:** Visual Plotly gauge divided into Nominal (Green), Advisory Warning (Amber), and Critical (Red) operational zones.
  * **Multi-Sensor Degradation Telemetry:** Interactive time-series plots with physical sensor definitions, units, and instant hover tooltips.
* **Automated ML Pipeline:** Single-command execution (`run_all.py`) covering data ingestion, feature scaling, model training, evaluation, and dashboard launch.

---

## Repository Structure

    Turbofan-Engine-RUL-Prediction/
    ├── data/
    │   └── train_FD001.csv          # NASA C-MAPSS FD001 multivariate sensor dataset
    ├── models/
    │   ├── lstm_model.pth           # Pre-trained PyTorch 2-layer LSTM weights
    │   └── scaler.pkl               # Fitted MinMaxScaler for feature normalization
    ├── 1_data_loader_eda.py         # Data ingestion, schema validation, and RUL calculation
    ├── 2_data_preprocessing.py      # Feature selection, MinMax scaling, and 3D sequence generation
    ├── 3_model_training.py          # PyTorch LSTM architecture definition and training loop
    ├── 4_evaluation.py              # Model performance evaluation (RMSE, MAE, R²) and diagnostics
    ├── 5_dashboard_app.py           # Interactive AeroPredict Streamlit & Plotly web application
    ├── run_all.py                   # End-to-end pipeline automation script
    ├── requirements.txt             # Lightweight CPU dependencies for cloud deployment
    └── README.md                    # Project documentation

---

## Model Architecture & Methodology

1. **Feature Selection:** Out of 21 raw sensors, constant/zero-variance channels are removed, retaining 14 high-variance physical sensors alongside 3 operational settings.
2. **Sequence Framing:** Time-series readings for each engine unit are normalized using `MinMaxScaler` and structured into overlapping 3D windows of shape `(samples, 50, features)`.
3. **Neural Network (`RUL_LSTM`):**
   * **Input Layer:** 17 normalized telemetry features over 50 time steps.
   * **Recurrent Layers:** 2 stacked LSTM layers (`hidden_dim=64`, `dropout=0.2`).
   * **Regression Head:** Fully connected layer (`64 -> 32` with ReLU activation) followed by a linear output layer (`32 -> 1`) predicting remaining cycles.

---

## Quickstart & Local Execution

### 1. Clone the Repository

    git clone https://github.com/abdelbasset-portfolio/Turbofan-Engine-RUL-Prediction.git
    cd Turbofan-Engine-RUL-Prediction

### 2. Install Dependencies

    pip install -r requirements.txt

### 3. Launch the Interactive Dashboard
To run the web application immediately using the pre-trained model in `models/`:

    streamlit run 5_dashboard_app.py

### 4. Run the Full Training Pipeline (Optional)
To re-run preprocessing, model training, and evaluation from scratch before launching the app:

    python run_all.py

---

## Dataset Reference
* **Dataset:** NASA Turbofan Jet Engine Simulation Data (`C-MAPSS FD001`), provided by the NASA Ames Prognostics Data Repository.
* **Fleet Scope:** 100 independent turbofan engine units monitored from initial healthy operation until system failure under a single operating condition (sea level).
