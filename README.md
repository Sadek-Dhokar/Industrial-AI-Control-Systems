# 🏭 Industrial IoT & ML Portfolio

> Building intelligent systems at the intersection of hardware and software

[![GitHub](https://img.shields.io/badge/GitHub-Sadek--Dhokar-181717?logo=github)](https://github.com/Sadek-Dhokar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/sadek-dhokar-318342326)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail)](mailto:dhokarsadek.enetcom@gmail.com)

---

## 👨‍🎓 About Me

**Sadek Dhokar** | Industrial Computer Engineering (GII) Student  
📍 National School of Electronics and Telecommunications of Sfax (ENET'Com), University of Sfax  
🎯 Target Specialization: Intelligent & Interconnected Systems (SII)

I'm building a career at the intersection of **industrial systems, embedded hardware, and machine learning** creating smart solutions for real-world manufacturing, automation, and IoT challenges.

### 🎓 Academic Background
- **Current:** Semester 2, Industrial Computer Engineering (GII) @ ENET'Com
- **Target Specialization:** Intelligent & Interconnected Systems (SII) - Selection after Semester 4
- **Core Focus Areas:**
  - Embedded Systems (C/C++, Microcontrollers)
  - Machine Learning & Predictive Analytics (Python)
  - Industrial Automation (PLCs, Sensors, Robotics)
  - Hardware-Software Integration

### 🚀 Career Vision

**Short-term:** Build technical foundation in industrial IoT, embedded systems, and edge ML  
**Mid-term:** Field Application Engineer or Robotics Integration Engineer bridging hardware and software  
**Long-term:** Technical Consultant or Founder in smart manufacturing / Industry 4.0 space

---

## 🗂️ Projects

### 🔹 [01 - Smart Sensor Data Analyzer](./01-smart-sensor-analyzer)
**Stack:** Python • Pandas • Matplotlib  
**Domain:** Predictive Maintenance  
**Status:** ✅ Completed (January 2026)

A foundational pipeline simulating industrial sensor data (temperature, humidity, vibration) with anomaly detection and statistical analysis. This project demonstrates data generation, processing, and visualization—core skills for predictive maintenance systems.

**Key Features:**
- Synthetic sensor data generation with realistic noise and anomalies
- Statistical analysis and time-series visualization
- Console dashboard for real-time monitoring

**Skills Demonstrated:**
- Data pipeline architecture
- Statistical analysis and anomaly pattern recognition
- Modular Python development
- Foundation for ML integration in industrial contexts

[View Project →](./01-smart-sensor-analyzer)

---

### 🔹 [02 - AI-Driven DC Motor Control Suite](./02-dc-motor-rl-control)
**Stack:** Python • Gymnasium • Stable-Baselines3 • NumPy • Matplotlib  
**Domain:** Control Systems × Reinforcement Learning  
**Status:** ✅ Completed (March 2026)

A final academic project combining classical **Control Theory** (Automatique/Discretization) with **Reinforcement Learning**. A PPO agent is trained to autonomously control the speed of a DC motor, replacing a hand-tuned PID controller  and discovering a near time-optimal control strategy with no prior knowledge of the physics.

**The Motor Model (TP3 Parameters):**
- Physical system: J=0.01 kg·m², b=0.1 N·m·s, K=0.01 N·m/A, R=1Ω, L=0.5H
- Simulated via Forward-Euler discretization (dt = 0.01 s)
- Target: ω\* = 1.0 rad/s | Supply limit: V_max = 12V

**Three Project Pillars:**
- **Régulation de vitesse** — speed step-response control from 0 to 1.0 rad/s
- **Variateur** — voltage saturation handling, anti-windup, slew-rate limiting
- **Lissage / Échantillonnage** — Gaussian encoder noise + Kalman & Moving-Average filters

**Key Engineering Decisions:**
- **Delta action space** (v4 architecture): agent outputs voltage *increments* (max 60 V/s slew), making bang-bang control physically impossible a constraint-based fix rather than a reward-shaping hack
- **5D observation vector** includes current voltage level so the agent can plan increments
- **Discrete Kalman Filter** (Q=1e-5, R_n=4e-4) feeds clean speed estimates to both controllers
- **PID baseline** (Kp=5, Ki=12, Kd=0.8) tuned from open-loop transfer function poles

**Results (1M training steps):**

| Metric | PID Controller | PPO Agent | Winner |
|--------|---------------|-----------|--------|
| Rise Time (10%→90%) | 1.450 s | **0.770 s** | PPO — 2× faster |
| Settling Time (±2%) | **4.460 s** | 4.780 s | Comparable |
| Overshoot | **2.24 %** | 3.23 % | Comparable |
| Steady-State Error | **0.004 rad/s** | 0.013 rad/s | PID — integral guarantee |
| Voltage ΔV smoothness | 0.479 V/step | **0.087 V/step** | PPO — 5.5× smoother |

The PPO agent discovered autonomously that saturating at 12V during the transient then dropping to ~10V at steady state is near time-optimal  the same strategy a control engineer would design deliberately, found through 1M steps of trial and error.

**Skills Demonstrated:**
- Custom Gymnasium environment design (physics simulation + sensor noise)
- PPO reinforcement learning with Stable-Baselines3 (VecNormalize, EvalCallback)
- Discrete-time signal filtering (Kalman filter from scratch, Moving Average)
- Classical PID controller design from transfer function analysis
- Publication-quality scientific plotting and quantitative performance benchmarking

[View Project →](./02-dc-motor-rl-control)

---

## 🛠️ Technical Skills

### Programming & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![C](https://img.shields.io/badge/C-A8B9CC?logo=c&logoColor=black)
![Java](https://img.shields.io/badge/Java-007396?logo=java&logoColor=white)

### Embedded & Hardware
![ESP32](https://img.shields.io/badge/ESP32-000000?logo=espressif&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-00979D?logo=arduino&logoColor=white)
![Microcontrollers](https://img.shields.io/badge/Microcontrollers-03234B?logo=stmicroelectronics&logoColor=white)

### Data & ML
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

### Tools & Platforms
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?logo=visualstudiocode&logoColor=white)

---

## 📚 Relevant Coursework

**Completed (Semester 1):**
- Probability & Statistics
- Fluid Mechanics
- Operations Research & Optimization
- Computer Architecture
- Operating Systems (LPI Certification Prep)
- Algorithms & C Programming
- Analog & Digital Electronics
- Continuous Control Systems
- Electrotechnics & Electrical Schematics

**Current (Semester 2):**
- Numerical Modeling of Systems
- Numerical Analysis
- Python for Engineering
- Power Electronics
- Electrical Machines & Variable Speed Drives
- Object-Oriented Programming (Java)
- Databases & DBMS (Oracle Certification Prep)
- Microcontroller Architecture & Programming
- Modular Electronics
- Industrial Instrumentation Sensors

**Upcoming (Semester 3-4):**
- Networks & Data Transmission (CCNA Prep)
- Signal Processing
- Programmable Logic Controllers (PLCs)
- STM32 Microcontrollers
- FPGA Programming
- Artificial Intelligence
- Real-Time Systems
- Robotics Introduction

**Future (Semester 5 - SII Track Goal):**
- Deep Learning & Neural Networks
- Robotics & Mechatronics
- Industrial Vision Systems
- IoT & Industry 4.0
- Smart Sensors & Industrial Communication Networks

---

## 🎯 Current Learning Focus

- ✅ **DataCamp ML Engineer Track** (5/12 courses completed + 2 projects)
- 🔄 **ESP32 Hands-on Projects** (bridging embedded systems + IoT)
- 🔄 **Semester 2 Coursework** (Microcontrollers, Java OOP, Databases, Python)
- 📅 **Upcoming:** Predictive maintenance system with edge ML deployment
- 📅 **Upcoming:** TensorFlow Lite for microcontrollers

---

## 📈 Portfolio Progress

**Current Status:**
- ✅ 2 projects completed (Smart Sensor Analyzer, DC Motor RL Control Suite)
- 📚 5/12 DataCamp ML courses completed + 2 projects
- 🎯 Target: 6+ hands-on projects by Summer 2026

**Next Milestone:** ESP32 Weather Station with cloud integration (Target: June 2026)

---

## 💼 Experience & Opportunities

**Current:**
- Building portfolio projects in industrial IoT and predictive maintenance
- Exploring internship opportunities in embedded systems and automation

**Seeking:**
- Summer 2026 internship in industrial IoT, embedded systems, or automation
- Collaborative projects in smart manufacturing and Industry 4.0
- Mentorship from professionals in robotics and industrial automation

**Open to:**
- Predictive maintenance implementations
- Edge computing and sensor integration projects
- Industrial automation challenges

---

## 📫 Let's Connect

I'm always interested in discussing:
- Industrial IoT and smart manufacturing
- Embedded systems and microcontroller projects
- Machine learning deployment on edge devices
- Internship or collaboration opportunities

**Reach out:**
- 📧 Email: [dhokarsadek.enetcom@gmail.com](mailto:dhokarsadek.enetcom@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/sadek-dhokar-318342326](https://linkedin.com/in/sadek-dhokar-318342326)
- 🐙 GitHub: [github.com/Sadek-Dhokar](https://github.com/Sadek-Dhokar)

---

## 📝 License

This portfolio and all projects are available under the MIT License unless otherwise specified in individual project folders.

---

*Last Updated: March 2026 | Currently in Semester 2 | 2 Projects Completed*
