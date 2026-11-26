import sys
import subprocess
import os

python_exe = sys.executable

scripts = [
    ("1. Data Loading & EDA", "1_data_loader_eda.py"),
    ("2. Data Preprocessing & Scaling", "2_data_preprocessing.py"),
    ("3. Training PyTorch LSTM Model", "3_model_training.py"),
    ("4. Model Evaluation & Plotting", "4_evaluation.py"),
    ("5. External Unseen Data Testing", "run_test_eval.py")
]

print("==================================================")
print("🚀 Starting Complete Predictive Maintenance Pipeline")
print("==================================================\n")

for title, script in scripts:
    if os.path.exists(script):
        print(f"\n---> Executing Step: {title} ({script})...")
        result = subprocess.run([python_exe, script])
        if result.returncode != 0:
            print(f"❌ Error encountered in {script}. Pipeline stopped.")
            sys.exit(1)
    else:
        print(f"⚠️ Script {script} not found, skipping...")

print("\n==================================================")
print("🎉 All Phases Completed Successfully!")
print("🌐 Launching Streamlit Interactive Dashboard...")
print("==================================================\n")

# Launch Streamlit app automatically
subprocess.run([python_exe, "-m", "streamlit", "run", "5_dashboard_app.py"])


#(C:\Users\hp\AppData\Local\Python\bin\python.exe run_all.py)