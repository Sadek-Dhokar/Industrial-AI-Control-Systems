# Industrial AI & Control Systems Portfolio

> Bridging Advanced Artificial Intelligence with Industrial Automation and Control

[![GitHub](https://img.shields.io/badge/GitHub-Sadek--Dhokar-181717?logo=github)](https://github.com/Sadek-Dhokar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/sadek-dhokar-318342326)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail)](mailto:dhokarsadek.enetcom@gmail.com)

---

## About Me

**Sadek Dhokar** | Industrial ML & Control Systems Student
**Location:** National School of Electronics and Telecommunications of Sfax (ENET'Com), Tunisia  
**Specialization Focus:** Intelligent & Interconnected Systems (SII)

I am an engineering student specializing at the intersection of machine learning, advanced control systems, and industrial automation. My work focuses on transitioning theoretical AI and control principles into robust, reproducible systems for physical industrial environments.

### Career Objectives

- **Short-term:** Architect predictive frameworks and reinforcement learning controllers that respect the mechanical and electrical constraints of real-world industrial systems.
- **Mid-term:** Transition into a role as an ML Engineer that works with physical systems, such as Edge ML or Industrial ML, bridging the gap between classical automation engineering and modern ML operations (MLOps).
- **Long-term:** Lead technical innovations in Industry 4.0, developing scalable AI-driven solutions that modernize manufacturing paradigms.

---

## Engineering Projects

This portfolio demonstrates practical implementations of control theory, deep reinforcement learning, and advanced data analytics applied to industrial challenges.

### [03 - Smart Grid Energy Management System with EV and RL](./03-smart-grid-energy-management-ev-rl)
*An intelligent, AI-driven Home Energy Management System (HEMS).*

**Stack:** Python, scikit-learn, Gymnasium, Stable-Baselines3  
**Domain:** Smart Grids x Deep Reinforcement Learning  

**Key Achievements:**
- Developed a dynamic PPO agent to optimize solar self-consumption and battery/EV dispatch against time-of-use energy pricing.
- Orchestrated Vehicle-to-Home (V2H) energy transfers, demonstrating the capacity of ML to minimize household energy costs securely.
- Integrated accurate physical constraints for battery degradation and electrical limits within a custom Gymnasium environment.

---

### [02 - AI-Driven DC Motor Control Suite](./02-dc-motor-rl-control)
*Benchmarking Reinforcement Learning against Classical PID Control.*

**Stack:** Python, Gymnasium, Stable-Baselines3, NumPy, Filtering (Kalman)  
**Domain:** Control Systems x Reinforcement Learning  

**Key Achievements:**
- Replaced a traditional, hand-tuned PID controller with an autonomous PPO agent that discovered a near time-optimal control strategy without prior system knowledge.
- Designed a custom Continuous-Time simulation via Forward-Euler discretization (dt = 0.01 s) of a DC motor with mechanical/electrical parameters (J, b, K, R, L).
- Handled critical industrial constraints via a **Delta action space**, enforcing a voltage slew rate to prevent physically impossible bang-bang actuator damage.
- Implemented state estimation and denoising using an algorithmically pure **Discrete Kalman Filter** and Moving-Average tracking.

---

### [01 - Smart Sensor Data Analyzer](./01-smart-sensor-analyzer)
*Simulated IoT telemetry for predictive maintenance.*

**Stack:** Python, Pandas, Matplotlib  
**Domain:** Predictive Maintenance x IoT  

**Key Achievements:**
- Constructed a data ingestion pipeline simulating typical industrial telemetry (vibration, temperature, and humidity).
- Deployed statistical anomaly detection models to identify equipment degradation.
- Established a foundational framework for future integration into real-time monitoring streams.

---

## Technical Competencies

### Programming & AI Frameworks
- **Languages:** Python (Advanced), C/C++ (Embedded), Java
- **ML/DS Libraries:** scikit-learn, PyTorch, Pandas, NumPy
- **RL Frameworks:** Gymnasium, Stable-Baselines3

### Engineering & Operations
- **Control Theory:** PID formulation, discrete-time systems, transfer functions, Kalman filtering.
- **Embedded Systems:** Microcontrollers (ESP32, Arduino, STM32 prep), hardware-software integration.
- **MLOps/DevOps:** Linux, Docker, Git, MLflow (Introductory), CI/CD fundamentals.

---

## Collaborative Opportunities

I am actively seeking:
- **Summer Internships (2026):** Roles focused on Applied Machine Learning, MLOps, or AI-Driven Control Systems.
- **Academic/Industrial Research:** Collaborative work concerning predictive maintenance, signal processing, or intelligent robotics architectures.

**Contact:**
- **Email:** dhokarsadek.enetcom@gmail.com
- **LinkedIn:** [linkedin.com/in/sadek-dhokar-318342326](https://linkedin.com/in/sadek-dhokar-318342326)

---

## Licensing & Usage

- Sub-projects contain highly customized AI/RL architectures. The motor and smart-grid controllers are heavily restricted to protect their technical viability for commercial transition. 
- Please consult the individual LICENSE files in each sub-directory for specifics. Generally, viewing for academic or recruitment evaluation is permitted; commercial use or reproduction is forbidden.

---
_Last Updated: April 2026_
