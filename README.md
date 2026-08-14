# 🛡️ DDoS Attack Detection System

A Flask-based web application designed to identify abnormal network traffic patterns that may indicate Distributed Denial-of-Service (DDoS) activity using statistical **Z-Score anomaly detection**.

The application provides a web-based dashboard where users can upload network traffic datasets, configure detection sensitivity, analyze traffic patterns, visualize anomalies, and review detailed detection logs.

---

## 📌 Project Overview

Distributed Denial-of-Service (DDoS) attacks can generate unusually high volumes of network traffic that differ significantly from normal traffic behavior.

This project uses a statistical baseline approach to identify such abnormal traffic patterns.

The system establishes a baseline from the initial portion of the uploaded network traffic dataset by calculating the **mean** and **standard deviation**. It then evaluates subsequent traffic records using the Z-Score.

When the calculated Z-Score exceeds the configured threshold, the traffic is classified as a potential anomaly.

The results are presented through a web dashboard with visualizations, statistical summaries, and a detailed detection log.

---

## 🎯 Objectives

The main objectives of this project are:

- Detect abnormal network traffic patterns.
- Identify potential DDoS-related traffic anomalies.
- Apply statistical anomaly detection using Z-Score.
- Provide configurable detection sensitivity.
- Visualize network traffic and detected anomalies.
- Maintain a detailed detection log.
- Provide a simple web interface for cybersecurity analysis.

---

## ✨ Key Features

### 📂 CSV Network Traffic Upload

Users can upload network traffic datasets in CSV format directly through the web interface.

### 🔍 Automatic Traffic Feature Detection

The application automatically searches for commonly used traffic-rate columns:
If neither column is available, the application automatically falls back to the first numerical column in the dataset.

### 🎚️ Adjustable Detection Sensitivity

Users can configure the detection sensitivity by selecting a Z-Score threshold between **2.0 and 5.0**.

| Threshold | Sensitivity | Description |
|-----------|-------------|-------------|
| **2.0** | High | Detects more potential anomalies |
| **3.0** | Balanced | Default detection level |
| **5.0** | Low | Detects only significant deviations |

Lower threshold values increase sensitivity and may identify more potential anomalies, while higher values focus on more significant traffic deviations.

### 📊 Statistical Anomaly Detection

The system uses the **Z-Score** method to identify traffic records that significantly differ from the established baseline.

The Z-Score is calculated using:

```text
Z = (X - μ) / σ
```

### 🔢 Z-Score Interpretation

The calculated Z-Score represents how far a traffic value deviates from the established baseline.

| Z-Score | Interpretation |
|---------|----------------|
| Below threshold | Normal traffic |
| Above threshold | Potential anomaly |
| Higher Z-Score | Greater deviation from baseline |

The system uses the configured threshold to classify traffic records as normal or potentially anomalous.

### 🧠 Baseline Training

The application creates a baseline using the initial portion of the uploaded dataset.

The baseline is calculated using:

- Mean traffic rate
- Standard deviation
- Configured Z-Score threshold

The first available training records are used to establish the expected traffic behavior. Subsequent records are then evaluated against this baseline.

### 🔄 Detection Process

The detection process consists of three main stages:

**1. Baseline Training**

Normal traffic data is analyzed to calculate the mean and standard deviation.

**2. Traffic Monitoring**

The remaining traffic records are evaluated individually using their Z-Scores.

**3. Anomaly Classification**

If the calculated Z-Score exceeds the configured threshold, the record is classified as a potential anomaly.

### 📊 Analysis Dashboard

The application provides a web-based dashboard for performing the complete analysis.

The interface includes:

- 📂 CSV network traffic upload
- 🎚️ Detection sensitivity control
- 🔍 Automatic traffic feature selection
- ▶️ Analyze Data & Detect Anomalies button
- 📋 Detection Log
- 📈 Traffic and Z-Score visualizations
- 📑 Analysis summary
- ❤️ Application health check

### 🖥️ Dashboard Interface

The main dashboard is divided into two primary sections:

#### Upload & Analyze

Users can upload a CSV network traffic dataset and configure the Z-Score detection threshold.

The interface provides three sensitivity levels:

- **2.0** → High sensitivity
- **3.0** → Balanced sensitivity
- **5.0** → Low sensitivity

After selecting the dataset and threshold, the user can click **Analyze Data & Detect Anomalies** to start the analysis.

#### Detection Log

The Detection Log displays the results of the analysis in a structured table containing:

- Timestamp
- Detection status
- Traffic rate value
- Z-Score

Normal traffic is reported as:

```text
Traffic Normal
```

### 📈 Traffic & Z-Score Visualization

After the analysis is completed, the application generates visualizations to help identify abnormal traffic behavior.

The dashboard displays two main graphs:

#### Traffic Rate Over Time

This graph shows the selected network traffic rate across the analyzed time windows.

It includes:

- Normal traffic points
- Detected anomaly points
- Mean traffic rate
- Calculated anomaly threshold

#### Z-Score Over Time

This graph displays the calculated Z-Score for each traffic record and compares it against the configured detection threshold.

Traffic records exceeding the threshold are classified as potential anomalies.

### 📋 Analysis Summary

The application provides a summary of the analysis, including:

| Metric | Description |
|--------|-------------|
| **Traffic Feature** | Network traffic column selected for analysis |
| **Z-Score Threshold** | Detection sensitivity used during analysis |
| **Training Rows** | Records used to establish the baseline |
| **Mean Rate** | Average traffic rate in the baseline |
| **Standard Deviation** | Variation in the baseline traffic |
| **Total Rows Analyzed** | Number of records processed |

### 🔄 Detection Workflow

```text
CSV Network Dataset
        │
        ▼
Upload Dataset
        │
        ▼
Automatic Feature Detection
        │
        ▼
Baseline Training
        │
        ├── Mean
        └── Standard Deviation
        │
        ▼
Z-Score Calculation
        │
        ▼
Compare Z-Score with Threshold
        │
        ├── Normal Traffic
        │
        └── Potential Anomaly
        │
        ▼
Visualization
        │
        ▼
Detection Log & Analysis Summary
```

### 🖥️ Project Interface

The application provides a web-based dashboard that combines data upload, detection configuration, analysis results, and monitoring information in a single interface.

#### Upload & Analyze

Users can:

- Upload a CSV network traffic dataset.
- Configure the Z-Score detection threshold.
- Start the anomaly detection process.
- View the analysis status.

#### Detection Log

The Detection Log provides a structured view of analyzed traffic records, including timestamps, traffic-rate values, Z-Scores, and detection status.

#### Analysis Results

After processing the dataset, the dashboard presents the statistical analysis and graphical results to help users understand detected traffic anomalies.

### 📸 Dashboard Screenshot

![DDoS Anomaly Detection Dashboard](docs/dashboard.png)
