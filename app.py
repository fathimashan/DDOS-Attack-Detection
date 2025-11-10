# app.py

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import datetime
import os
import pandas as pd
from flask import Flask, render_template, request

# --- DDoSDetector CLASS ---
class DDoSDetector:
    def __init__(self, z_score_threshold=3.0):
        self.threshold = z_score_threshold
        self.mu = None
        self.sigma = None

    def train_baseline(self, normal_rates):
        """Calculates the mean (mu) and standard deviation (sigma) from normal rates."""
        data = np.array(normal_rates)
        self.mu = np.mean(data, axis=0)
        # Add a small epsilon to prevent division by zero
        self.sigma = np.std(data, axis=0) + 1e-6 

    def check_anomaly(self, current_rates):
        """Calculates the Z-score for the primary rate (index 0) and checks against the threshold."""
        if self.mu is None or self.sigma is None:
            return False, 0, 0
            
        Z_p = (current_rates[0] - self.mu[0]) / self.sigma[0]
        # Z-score for secondary metric (not used for detection)
        Z_b = (current_rates[1] - self.mu[1]) / self.sigma[1] 
        
        is_anomaly = (Z_p > self.threshold)
        return is_anomaly, Z_p, Z_b

# --- FLASK APP SETUP ---

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- CORE UTILITY FUNCTION ---

def process_data(file_path, z_threshold):
    
    # Create a new detector instance for each analysis
    detector = DDoSDetector(z_score_threshold=z_threshold)
    
    # Simulate a time window of 10 seconds per row for log timestamps
    TIME_INCREMENT_SECONDS = 10 
    
    # 1. READ AND PREPARE DATA
    try:
        df = pd.read_csv(file_path)
        
        # --- AUTOMATIC COLUMN SELECTION LOGIC ---
        rate_column = None
        # Columns in CIC-IDS datasets often have a leading space, so check for both.
        if ' Flow Packets/s' in df.columns:
            rate_column = ' Flow Packets/s'
        elif ' Flow Bytes/s' in df.columns:
            rate_column = ' Flow Bytes/s'
        
        if rate_column is None:
            # Check without leading space as a fallback
            if 'Flow Packets/s' in df.columns:
                rate_column = 'Flow Packets/s'
            elif 'Flow Bytes/s' in df.columns:
                rate_column = 'Flow Bytes/s'
            else:
                 # Fallback to first numerical column if preferred columns not found
                 numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                 if numerical_cols:
                     rate_column = numerical_cols[0]
                 else:
                     raise ValueError("No numerical columns found in the CSV file.")
        
        # --- PERFORMANCE FIX: LIMIT ROWS ---
        MAX_ROWS_TO_USE = 10000 
        initial_total_rows = len(df)
        if initial_total_rows > MAX_ROWS_TO_USE:
             df = df.head(MAX_ROWS_TO_USE)
        
        # Placeholder column
        df['secondary_metric'] = 0.0 
        
        TRAINING_ROWS = min(50, len(df) // 2)
        
        training_data = df.iloc[:TRAINING_ROWS]
        monitoring_data = df.iloc[TRAINING_ROWS:].reset_index(drop=True)
        
        if len(training_data) < 10:
             raise ValueError("Insufficient data for training (requires at least 10 rows).")

        normal_rates = training_data[[rate_column, 'secondary_metric']].values.tolist()

    except Exception as e:
        return f"ERROR: Could not process data. {e}", "placeholder_plot", []

    # === A. TRAINING PHASE (Phase 1) ===
    detector.mu = None
    detector.sigma = None
    detector.train_baseline(normal_rates)
    
    # Store key stats for the summary
    training_summary = {
        'rate_column': rate_column.strip(),
        'z_threshold': z_threshold,
        'training_rows': len(training_data),
        'mean': float(detector.mu[0]),  # Convert to Python float
        'std': float(detector.sigma[0]),  # Convert to Python float
        'total_rows_analyzed': len(df)
    }

    # === B. MONITORING PHASE (Phase 2) ===
    TOTAL_WINDOWS = len(monitoring_data)
    rates_history = []
    z_scores_history = []
    anomaly_flags_history = []
    detection_log = []
    start_time = datetime.datetime.now()

    for i, row in monitoring_data.iterrows():
        current_timestamp = (start_time + datetime.timedelta(seconds=i * TIME_INCREMENT_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")

        p_rate = float(row[rate_column])  # Ensure float type
        b_rate = float(row['secondary_metric'])
        current_rates = (p_rate, b_rate)
        
        is_anomaly, Z_p, Z_b = detector.check_anomaly(current_rates)

        rates_history.append(p_rate)
        z_scores_history.append(float(Z_p))  # Ensure float type
        anomaly_flags_history.append('red' if is_anomaly else 'green') 

        # Add entry to the log list
        log_entry = {
            'timestamp': current_timestamp,
            'rate': p_rate,
            'z_score': float(Z_p),  # Ensure float type
            'status': 'ANOMALY DETECTED' if is_anomaly else 'Traffic Normal',
            'is_anomaly': is_anomaly
        }
        detection_log.append(log_entry)

    # === C. PLOTTING PHASE (Phase 3) ===
    
    time_steps = list(range(1, TOTAL_WINDOWS + 1))
    packet_rate_threshold_value = detector.mu[0] + (detector.threshold * detector.sigma[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Rate over time with better colors
    ax1.scatter(time_steps, rates_history, c=anomaly_flags_history, s=50, alpha=0.7)
    ax1.axhline(float(detector.mu[0]), color='blue', linestyle='-', 
               label=f'Mean Rate (μ={detector.mu[0]:.2f})', linewidth=2)
    ax1.axhline(float(packet_rate_threshold_value), color='red', linestyle='--', 
               linewidth=2, label=f'Threshold (μ + {z_threshold}σ = {packet_rate_threshold_value:.2f})')
    ax1.set_title(f'DDoS Anomaly Detection: {rate_column.strip()} over Time\n(Z-Score Threshold: {z_threshold})', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel(rate_column.strip(), fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Z-scores over time with better colors
    ax2.scatter(time_steps, z_scores_history, c=anomaly_flags_history, s=50, alpha=0.7)
    ax2.axhline(float(detector.threshold), color='red', linestyle='--', 
               linewidth=2, label=f'Z-Score Threshold ({detector.threshold})')
    ax2.set_title('Z-Score over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Window', fontsize=12)
    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return training_summary, plot_data, detection_log 

# --- FLASK ROUTES (Endpoints) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    z_threshold = 3.0
    plot_data = "placeholder_plot" 
    status_message = "Upload a CSV file and set the Sensitivity Threshold to begin analysis."
    summary = None
    detection_log = []

    if request.method == 'POST':
        if 'data_file' in request.files and request.files['data_file'].filename:
            file = request.files['data_file']
            
            if file.filename.endswith('.csv'):
                try:
                    filename = 'temp_traffic_data.csv'
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Get z_threshold with validation
                    try:
                        z_threshold = float(request.form.get('z_threshold', 3.0))
                        # Ensure it's within the valid range
                        z_threshold = max(2.0, min(5.0, z_threshold))
                    except (ValueError, TypeError):
                        z_threshold = 3.0  # Default if invalid

                    summary, plot_data, detection_log = process_data(file_path, z_threshold) 
                    
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    if isinstance(summary, str) and "ERROR" in summary:
                        status_message = summary
                        summary = None
                    else:
                        status_message = "Analysis Complete!"
                        
                except Exception as e:
                    status_message = f"ERROR: {str(e)}"
                    
    return render_template('index_file.html', 
                           z_threshold=z_threshold,
                           status_message=status_message,
                           plot_data=plot_data,
                           summary=summary,
                           detection_log=detection_log)

@app.route('/health')
def health_check():
    return "DDoS Detection Service is running!"

if __name__ == '__main__':
    print("Starting DDoS Anomaly Detection Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)