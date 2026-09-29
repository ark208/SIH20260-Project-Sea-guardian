# 🌊 Sea Guardian (Bot Guardian)
> **AI-Powered Maritime Oil Spill Detection, Hydrodynamic Backward Dispersion & Dark Vessel Culprit Attribution Engine**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![Model Version](https://img.shields.io/badge/Bot_Guardian-v2.0-purple.svg)]()

---

## 📌 Overview

**Sea Guardian** (featuring the **Bot Guardian** AI Analyst) is a cutting-edge maritime surveillance and intelligence platform designed to address illegal bilge dumping and oil spill incidents across critical Sea Lines of Communication (SLOC). 

Traditional surveillance models rely on naive spatial proximity—wrongly blaming whichever ship is closest to the oil slick at detection time. **Sea Guardian** breaks this flaw by combining:
1. **Sentinel-1 SAR Computer Vision**: Multi-harmonic organic oil slick boundary detection (Sheen + Dense Core layers).
2. **Physics-Guided Fay Hydrodynamic Rewind**: Inverting wind drift, ocean surface currents, and Stokes wave drift to calculate the exact historical discharge origin cone.
3. **ST-GNN & Dark Vessel Attribution**: Reconstructing dark transponder gaps (turned-off AIS) to identify the true culprit ship with high statistical confidence.
4. **Bot Guardian Analyst**: Real-time LLM-powered forensic analyst providing automated NTRO audit dossiers and interactive query answering.

---

## 🏗️ System Architecture

```
sea_guardian/
├── core/
│   ├── sar_vision/           # SAR Calibration, Speckle Filtering & Natural Slick Geometry
│   │   ├── preprocessor.py   # Sentinel-1 SAR Calibration & Refined Lee filter
│   │   ├── detector.py       # YOLO / DeepLabV3+ Segmentation Pipeline
│   │   ├── slick_geometry.py # Natural organic polygon & rewind cone projection
│   │   └── lookalike_filter.py # Low-wind & biogenic false-alarm filter
│   │
│   ├── hydrodynamics/        # Ocean Transport & Fay 3-Phase Spreading
│   │   ├── fay_spread.py     # Fay spreading law & spill age T_spill inversion
│   │   ├── stokes_drift.py   # Wave-induced Stokes surface drift vector
│   │   ├── ocean_service.py  # MetOcean & ERA5 10m wind fields with Coriolis
│   │   └── lagrangian_rewind.py # Monte Carlo backward particle tracking (P_origin)
│   │
│   ├── maritime_graph/       # AIS Stream & Dark Vessel Tracking
│   │   ├── ais_generator.py  # AIS fleet simulator (compliant + dark transponder gaps)
│   │   ├── spline_interpolator.py # Hermite cubic spline trajectory smoother
│   │   └── st_gnn_model.py   # Spatio-temporal graph neural network gap filler
│   │
│   ├── attribution/          # Multi-Factor Scoring & Intelligence Relay
│   │   ├── spatial_scorer.py # Hydrodynamic-trajectory overlap integral (35%)
│   │   ├── dark_scorer.py    # Blackout anomaly & zone proximity scorer (30%)
│   │   ├── behavior_risk.py  # Speed drops, loitering & vessel risk profiler (20% + 15%)
│   │   ├── attribution_engine.py # Composite normalized scoring & ranking
│   │   └── llm_intelligence.py # Bot Guardian Groq Llama-3 Analyst Relay
│   │
│   └── pipeline.py           # End-to-End Maritime Pipeline Orchestrator
│
├── web/                      # Interactive Web Dashboard
│   ├── server.py             # REST API Server & Dynamic Routing
│   └── templates/
│       └── index.html        # High-tech Dark Monochrome Dashboard
│
├── weights/                  # Pre-trained Neural Network Weights
├── data/                     # Maritime Hotspot Scenarios & Synthetic AIS Data
├── .env.example              # Environment Configuration Template
├── MODELS_AND_DATASETS.md   # AI Models & Dataset Documentation
├── requirements.txt          # Dependencies
└── README.md                 # Project Documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ark208/SIH20260-Project-Sea-guardian.git
cd SIH20260-Project-Sea-guardian
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (optional, for Bot Guardian LLM chat)
```

### 3. Run Pipeline CLI Test
```bash
python -m core.pipeline
```

### 4. Launch Web Surveillance Dashboard
```bash
python web/server.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your web browser.

---

## 🤖 AI Models & Datasets

For full details regarding network architectures, mathematical formulations, training hyperparameters, and dataset sources, please refer to:
👉 **[MODELS_AND_DATASETS.md](MODELS_AND_DATASETS.md)**

---

## 🧮 Mathematical Formulations

- **Total Ocean Drift Velocity**:
  $$\vec{U}_{\text{drift}}(x, y, t) = \vec{u}_{\text{current}}(x, y, t) + \alpha_{\text{wind}} \cdot \mathbf{R}(\theta_{\text{coriolis}}) \vec{u}_{10\text{m}}(x, y, t) + \vec{u}_{\text{Stokes}}(x, y, t)$$

- **Fay's 3-Phase Spreading**:
  $$R_1(t) = k_1 (\Delta g V t^2)^{1/4}, \quad R_2(t) = k_2 \left(\frac{\Delta g V^2 t^{3/2}}{\nu^{1/2}}\right)^{1/6}, \quad R_3(t) = k_3 \left(\frac{\sigma_{\text{net}}^2 t^3}{\rho^2 \nu}\right)^{1/4}$$

- **Composite Suspect Score**:
  $$S_{\text{culprit}}(i) = \frac{1}{\sqrt{M}} \left[ 0.35 \cdot S_{\text{spatial}} + 0.30 \cdot S_{\text{dark}} + 0.20 \cdot S_{\text{behavior}} + 0.15 \cdot S_{\text{risk}} \right]$$

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for details.
