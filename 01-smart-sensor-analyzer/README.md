## 🏭 Smart Sensor Data Analyzer

A Python-based simulation and analysis toolkit for industrial sensor data. This project generates synthetic readings for temperature, humidity, and vibration, performs statistical analysis, and provides visual insights—mimicking a foundational pipeline for predictive maintenance systems.

**Author:** Sadek Dhokar | Industrial Computer Engineering (GII) Student | National School of Electronics and Telecommunications of Sfax (ENET'Com), University of Sfax
**Tech Stack:** Python, Pandas, Matplotlib

---

### 📁 Project Structure

The project consists of the following main files:
- **`sensor_simulator.py`** - Generates synthetic sensor data and saves it to a CSV file.
- **`data_analyzer.py`** - Loads the data, performs statistical analysis, and creates visualizations.
- **`simple_dashboard.py`** - Displays a console-based summary dashboard with key metrics.
- **`requirements.txt`** - Lists the Python dependencies (Pandas, Matplotlib).
- **`README.md`** - This documentation file.

### 🚀 Getting Started

1.  **Clone & Setup**
    ```bash
    git clone https://github.com/Sadek-Dhokar/smart-sensor-analyzer.git
    cd smart-sensor-analyzer
    pip install -r requirements.txt
    ```

2.  **Run the Simulation & Analysis Pipeline**
    ```bash
    # 1. Generate synthetic sensor data
    python sensor_simulator.py
    # 2. Analyze data and generate plots
    python data_analyzer.py
    # 3. View the console dashboard
    python simple_dashboard.py
    ```

### 🔍 Key Features & Implementation

- **Realistic Data Simulation:** The `sensor_simulator.py` script creates timestamped sensor readings with injected anomalies (e.g., random high-vibration events) to simulate real industrial equipment behavior.
- **Statistical Analysis & Visualization:** Using **Pandas** for data manipulation and **Matplotlib** for plotting, `data_analyzer.py` produces time-series graphs and highlights potential anomalies.
- **Console Dashboard:** `simple_dashboard.py` provides an at-a-glance summary of key metrics, including the latest readings and system status alerts.

### 🎯 Project Context & Learning Goals

As an Industrial Computer Engineering (GII) student, this project served as a foundational hands-on exercise to build practical skills in data analysis and system simulation—core competencies for future specializations like Intelligent & Interconnected Systems (SII). The goal was to:
- Bridge theoretical programming concepts with hands-on Python implementation.
- Understand the fundamentals of data pipelines relevant to industrial applications.
- Build a modular codebase that separates data generation, analysis, and presentation.

### 🔮 Future Enhancements

Potential directions to expand this project include:
- Integrating a simple machine learning model (e.g., Isolation Forest) for automated anomaly detection.
- Converting the console dashboard into an interactive web application using **Streamlit**.
- Simulating data streams from multiple sensor types to model a more complex industrial environment.
