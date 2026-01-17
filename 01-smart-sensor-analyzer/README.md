# 🏭 Smart Sensor Data Analyzer

A Python-based toolkit for simulating and analyzing industrial sensor data. Generates synthetic readings for temperature, humidity, and vibration with realistic noise patterns and anomalies, then performs statistical analysis and visualization—demonstrating a foundational predictive maintenance pipeline.

**Author:** Sadek Dhokar  
**Institution:** Industrial Computer Engineering (GII), ENET'Com, University of Sfax  
**Tech Stack:** Python, Pandas, Matplotlib

---

## 📁 Project Structure
```
smart-sensor-analyzer/
├── sensor_simulator.py      # Generates synthetic sensor data with anomalies
├── data_analyzer.py         # Statistical analysis and visualization
├── simple_dashboard.py      # Console-based metrics dashboard
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Sadek-Dhokar/smart-sensor-analyzer.git
cd smart-sensor-analyzer
pip install -r requirements.txt
```

### Usage
```bash
# 1. Generate synthetic sensor data
python sensor_simulator.py

# 2. Analyze data and create visualizations
python data_analyzer.py

# 3. View summary dashboard
python simple_dashboard.py
```

---

## 🔍 Features

### Realistic Data Simulation
Generates timestamped sensor readings with:
- Normal operational patterns (temperature, humidity, vibration)
- Injected anomalies (random spikes, sensor drift)
- Configurable noise and sampling rates

### Statistical Analysis
- Time-series visualization with Matplotlib
- Anomaly highlighting based on statistical thresholds
- Summary statistics (mean, std dev, min/max)

### Console Dashboard
At-a-glance view of:
- Latest sensor readings
- System status alerts
- Key metrics and trends

---

## 🎯 Learning Objectives

This project was built to develop foundational skills in:
- Industrial data pipeline architecture
- Python for data analysis (Pandas, Matplotlib)
- Modular code design and separation of concerns
- Simulating realistic sensor behavior

**Context:** Preparation for Intelligent & Interconnected Systems (SII) specialization, focusing on predictive maintenance and industrial IoT applications.

---

## 🔮 Potential Enhancements

- [ ] Integrate machine learning models (Isolation Forest, LSTM) for automated anomaly detection
- [ ] Add real-time data streaming simulation
- [ ] Create interactive web dashboard (Streamlit or Dash)
- [ ] Expand to multi-sensor environments with correlated failures
- [ ] Deploy as microservice with REST API

---

## 📊 Sample Output

**Generated Data:**
- 1000+ timestamped readings per sensor
- CSV format for easy integration
- Configurable anomaly injection rate

**Visualizations:**
- Time-series plots with anomaly markers
- Distribution histograms
- Correlation analysis

---

## 🛠️ Technical Details

**Dependencies:**
- Python 3.7+
- Pandas (data manipulation)
- Matplotlib (visualization)
- NumPy (numerical operations)

**Design Principles:**
- Modular architecture (separate generation, analysis, display)
- Reusable components for future projects
- Clear separation between data layer and presentation

---

## 📝 License

This project is available under the MIT License.

---

## 🔗 Related Work

Part of my [Industrial IoT & ML Portfolio](https://github.com/Sadek-Dhokar/Industrial-IoT-ML-Portfolio) exploring AI and automation in industrial contexts.


*Built as a foundation for learning industrial data analysis and predictive maintenance systems.*
```

---
