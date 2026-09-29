# 🌊 Sea Guardian
### Autonomous Satellite SAR Oil Spill Detection, Hydrodynamic Backward Rewind & Dark Vessel Attribution Engine

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![SAR Processing](https://img.shields.io/badge/SAR-Sentinel--1_C--Band-orange.svg)]()
[![Physics Engine](https://img.shields.io/badge/Physics-Fay_Spreading_%2B_Lagrangian_Rewind-teal.svg)]()
[![Graph Neural Network](https://img.shields.io/badge/Attribution-ST--GNN_Dark_Vessel-purple.svg)]()
[![AI Intelligence](https://img.shields.io/badge/Analyst-Bot_Guardian_LLM-success.svg)]()

---

## 📌 Executive Summary & The Operational Challenge

Every day across critical Sea Lines of Communication (SLOC)—such as the Strait of Malacca, Gulf of Kutch, and the Mumbai High offshore corridor—commercial vessels deliberately discharge untreated oily bilge water under cover of night. By international maritime law (MARPOL Annex I), oily discharges must never exceed 15 parts per million. In practice, rogue operators routinely bypass oily water separators (OWS) using illegal "magic pipes" while transiting open waters to save tens of thousands of dollars in port reception facility fees.

When maritime security agencies or satellite constellations detect an oil slick hours later, conventional monitoring systems commit a **fatal forensic error**: they flag whichever ship happens to be geographically closest to the slick at the instant of satellite acquisition ($t_0$).

This naive proximity approach regularly breaks down:
1. **Dynamic Hydrodynamic Drift**: Ocean surface currents, tidal rips, and wind-induced drag displace the slick kilometers away from where it was actually dumped.
2. **Intentional AIS Blackouts**: Malicious vessels intentionally deactivate their Automatic Identification System (AIS) transponders ("go dark") before discharging, then accelerate away to blend back into high-density traffic corridors.
3. **Innocent Bystander False Accusations**: Law-abiding merchant ships operating on standard shipping lanes often pass near the aged, drifting slick hours later—leading to wrongful enforcement actions while the true culprit escapes with zero liability.

**Sea Guardian** solves this fundamental enforcement problem. By unifying **Sentinel-1 SAR computer vision**, **Fay hydrodynamic spreading inversion**, **Lagrangian backward particle tracking**, and **Spatio-Temporal Graph Neural Networks (ST-GNN)**, Sea Guardian rewinds ocean time to locate the original discharge origin cone and statistically unmasks the culprit vessel—even when its transponder was disabled.

---

## 🔬 How Sea Guardian Works: The 5-Stage Intelligence Pipeline

```
 🛰️ Sentinel-1 Synthetic Aperture Radar (SAR) Acquisition
                    │
                    ▼
 [STAGE 1]  SAR Computer Vision & Dual-Layer Morphology
            ├── Gamma-0 Decibel Calibration & Refined Lee Despeckle Filter (5x5)
            ├── DeepLabV3+ / U-Net Ensemble with Wave Contrast Analysis
            └── Multi-Harmonic Fractal Boundary Extraction (Sheen vs Dense Core)
                    │
                    ▼
 [STAGE 2]  Hydrodynamic Discharge Age Inversion & Backward Rewind
            ├── Fay 3-Phase Spreading Mechanics (Inverting T_spill from observed area)
            ├── MetOcean Vector Drag (ERA5 10m Wind + Coriolis + CMEMS Currents)
            ├── Wave-Induced Stokes Drift Integration
            └── Backward Monte Carlo Lagrangian Particle Dispersion (Origin Cone P_origin)
                    │
                    ▼
 [STAGE 3]  AIS Fleet Processing & Dark Vessel Trajectory In-Filling
            ├── High-density SLOC AIS Telemetry Stream Processing
            ├── Automated Transponder Anomaly & Gap Detection (Δt > 45 min)
            └── ST-GNN (Spatio-Temporal Graph Neural Network) Trajectory Reconstruction
                    │
                    ▼
 [STAGE 4]  Multi-Factor Forensic Attribution Engine
            ├── Hydrodynamic Overlap Integral (Spatial-Temporal Coincidence) [35%]
            ├── Transponder Blackout Anomaly Scorer [30%]
            ├── Kinematic Deviations (Speed Dips & Loitering Profiler) [20%]
            └── Flag State & Historical Port State Control (PSC) Risk Index [15%]
                    │
                    ▼
 [STAGE 5]  Bot Guardian Autonomous Maritime Intelligence Analyst
            ├── Groq High-Speed Llama-3 Inference Engine
            ├── Forensic Case Attribution & Comparative Proof Explanations
            └── Automated NTRO / Indian Coast Guard Evidentiary Dossier Relay
```

---

## 🧠 Algorithmic & Mathematical Breakdown

### 1. Natural Slick Vision & Look-Alike Discrimination
In Synthetic Aperture Radar (SAR) imagery, oil slicks dampen capillary and short gravity waves on the sea surface, lowering surface roughness and causing reduced radar backscatter (appearing as dark patches). However, low-wind calms, biogenic natural films, and internal oceanic waves mimic this phenomenon.
- **Preprocessing**: The raw Level-1 Ground Range Detected (GRD) amplitude is calibrated to Sigma-Nought ($\sigma_0$) and Gamma-Nought ($\gamma_0$):
  $$\gamma_0 = 10 \log_{10} \left( \frac{DN^2 + A_i}{K} \right) - 10 \log_{10}(\cos \theta_{\text{inc}})$$
- **Adaptive Despeckling**: A $5 \times 5$ Refined Lee filter preserves edge boundaries between the slick and open sea while suppressing coherent speckle noise.
- **Fractal Slick Morphology**: Slicks rarely spread in uniform circles. Sea Guardian implements multi-harmonic fractal perturbation equations simulating turbulent ocean shear:
  $$r(\theta) = R_0 \left[ 1 + \sum_{k=1}^N A_k \cos(k\theta + \phi_k) \right]$$
  separating the outer iridescent **sheen polygon** from the viscous **dense core**.

### 2. Fay 3-Phase Spreading Law & Discharge Age Inversion
An oil discharge on open water evolves through three distinct physical regimes dominated by competing gravitational, inertial, viscous, and interfacial surface-tension forces:
1. **Gravity-Inertial Phase**:
   $$R_1(t) = k_1 \left( \Delta g V t^2 \right)^{1/4}$$
2. **Gravity-Viscous Phase**:
   $$R_2(t) = k_2 \left( \frac{\Delta g V^2 t^{3/2}}{\nu_w^{1/2}} \right)^{1/6}$$
3. **Surface Tension-Viscous Phase**:
   $$R_3(t) = k_3 \left( \frac{\sigma_{\text{net}}^2 t^3}{\rho_w^2 \nu_w} \right)^{1/4}$$
*Where $V$ is estimated discharge volume, $\Delta = 1 - \rho_o / \rho_w$, $\nu_w$ is kinematic viscosity, and $\sigma_{\text{net}}$ is the net spreading coefficient.*

By taking the observed satellite area $A_{\text{obs}}$ and inverting the spreading equations via bounded binary search, Sea Guardian determines the exact elapsed time since discharge ($T_{\text{spill}}$) with rigorous physical bounds.

### 3. Time-Reversed Lagrangian Particle Tracking (Back-Drift)
To find where the slick originated, the system integrates backward in time ($t = t_0 \to t = t_0 - T_{\text{spill}}$) across dynamic MetOcean wind and current vector grids:
$$\vec{U}_{\text{drift}}(x, y, t) = \vec{u}_{\text{current}}(x, y, t) + \alpha_{\text{wind}} \cdot \mathbf{R}(\theta_{\text{coriolis}}) \vec{u}_{10\text{m}}(x, y, t) + \vec{u}_{\text{Stokes}}(x, y, t)$$
- $\alpha_{\text{wind}} \approx 0.030$ (3% of 10-meter wind speed).
- $\mathbf{R}(\theta_{\text{coriolis}})$ rotates wind drag by the Ekman deflection angle ($15^{\circ}$ clockwise in Northern Hemisphere).
- $\vec{u}_{\text{Stokes}} = \frac{2\pi H_s^2}{16 T_p}$ accounts for non-linear surface wave drift.
- **Monte Carlo Particle Cloud**: 1,000 backward particles subject to turbulent diffusion $\delta x = \sqrt{2 K_h \Delta t} \cdot \mathcal{N}(0, 1)$ define the elliptical **Discharge Origin Envelope ($P_{\text{origin}}$)**.

### 4. Multi-Factor Culprit Attribution Formula
Every candidate vessel in the corridor is evaluated using four independent evidence pillars:
$$S_{\text{culprit}}(i) = \frac{1}{\sqrt{M}} \left[ 0.35 \cdot S_{\text{spatial}} + 0.30 \cdot S_{\text{dark}} + 0.20 \cdot S_{\text{behavior}} + 0.15 \cdot S_{\text{risk}} \right]$$

1. **Spatial Overlap ($S_{\text{spatial}}$)**: Spatio-temporal coincidence between the vessel's reconstructed historical track and the rewind origin envelope $P_{\text{origin}}$ at $t_0 - T_{\text{spill}}$.
2. **AIS Dark Anomaly ($S_{\text{dark}}$)**: Penalty assigned if the vessel experienced an unnotified transponder gap overlapping the computed spill window:
   $$S_{\text{dark}} = 1.0 - \exp\left( -\frac{\Delta t_{\text{blackout}}}{2.5} \right)$$
3. **Behavioral Deviation ($S_{\text{behavior}}$)**: Flagging speed decelerations down to loitering speeds ($< 4$ knots) during transit, characteristic of active bilge pumping.
4. **Target Risk Profiling ($S_{\text{risk}}$)**: Flag state blacklists (Paris MoU / Tokyo MoU), ship class (crude tanker / chemical carrier), and vessel age.

---

## 🗺️ Indian Ocean Maritime Hotspots Benchmark Suite

Sea Guardian comes pre-calibrated with 8 real-world strategic maritime corridors across the Indian Ocean, Arabian Sea, and Bay of Bengal. In **every** scenario, our model demonstrates why standard proximity attribution fails:

| Incident Code | Hotspot Region | Slick ID | Slick Area | Slick Age | Innocent Closest Ship | True Dark Culprit Vessel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `INC-GULF-KUTCH` | Gulf of Kutch Energy Corridor | `SLICK-SAR-2026-KUT-1049` | $0.145\text{ km}^2$ | $1.46\text{ h}$ | MSC GULF TRADER (0.8 km) | MT KUTCH PRIDE (AIS Dark Gap) |
| `INC-MANNAR-PALK` | Gulf of Mannar & Palk Strait | `SLICK-SAR-2026-MAN-2931` | $0.260\text{ km}^2$ | $2.14\text{ h}$ | LANKA CARRIER (1.1 km) | PALK NAVIGATOR (Discharge Loiter) |
| `INC-GOA-MORMUGAO` | Goa Konkan Coast & Mormugao | `SLICK-SAR-2026-GOA-5192` | $0.510\text{ km}^2$ | $3.58\text{ h}$ | ZUARI EXPRESS (1.4 km) | MT KONKAN SHADOW (Dark Bilge) |
| `INC-ARABIAN-MUMBAI`| Mumbai Offshore / Bombay High | `SLICK-SAR-2026-MUM-4802` | $0.875\text{ km}^2$ | $4.85\text{ h}$ | EVER HARMONY (1.2 km) | OCEANIC SENTINEL (Blackout Corridor) |
| `INC-CHENNAI-ENNORE`| Chennai Offshore & Ennore Port| `SLICK-SAR-2026-CHE-8419` | $1.420\text{ km}^2$ | $6.92\text{ h}$ | COROMANDEL TRADER (1.9 km)| CHENNAI BULKER (Speed Drop) |
| `INC-LAKSHADWEEP` | Lakshadweep Sea Deepwater SLOC| `SLICK-SAR-2026-LAK-6512` | $2.150\text{ km}^2$ | $8.74\text{ h}$ | ISLAND RUNNER (2.3 km) | ARABIAN DISPATCHER (Spoofed Track) |
| `INC-BOB-VIZAG` | Visakhapatnam Deepwater SLOC | `SLICK-SAR-2026-VIZ-7914` | $2.950\text{ km}^2$ | $10.80\text{ h}$| VIZAG ENTERPRISE (2.7 km) | ANDHRA VALIANT (Midnight Dump) |
| `INC-MALACCA-NICOBAR`| Great Nicobar / Malacca Gateway| `SLICK-SAR-2026-NIC-3820` | $4.250\text{ km}^2$ | $13.90\text{ h}$| STRAIT TRADER (3.1 km) | MT MALACCA VOYAGER (Major SLOC Dump)|

---

## 💻 Tech Stack & Repository Layout

```
sea_guardian/
├── core/
│   ├── sar_vision/           # SAR radiometry, speckle filters & fractal slick polygons
│   │   ├── preprocessor.py   # Gamma-0 decibel calibration & 5x5 Refined Lee despeckling
│   │   ├── detector.py       # DeepLabV3+ segmentation inference & feature extraction
│   │   ├── slick_geometry.py # Natural dual-layer sheen/core morphology & rewind cone
│   │   └── lookalike_filter.py # False-alarm suppression (low wind / biogenic films)
│   │
│   ├── hydrodynamics/        # Fluid mechanics & backward transport simulation
│   │   ├── fay_spread.py     # 3-phase Fay law & bounded age T_spill inversion
│   │   ├── stokes_drift.py   # Wave-induced Stokes surface drift vector module
│   │   ├── ocean_service.py  # MetOcean weather, Coriolis rotation & CMEMS currents
│   │   └── lagrangian_rewind.py # Monte Carlo backward trajectory tracking
│   │
│   ├── maritime_graph/       # AIS transponder intelligence & graph neural network
│   │   ├── ais_generator.py  # Fleet synthesizer with configurable blackout gaps
│   │   ├── spline_interpolator.py # Hermite cubic spline trajectory reconstructor
│   │   └── st_gnn_model.py   # Spatio-Temporal GNN for transponder gap in-filling
│   │
│   ├── attribution/          # Probabilistic forensics & intelligence agent
│   │   ├── spatial_scorer.py # Hydrodynamic rewind overlap integral (35% weight)
│   │   ├── dark_scorer.py    # Blackout duration & proximity penalty (30% weight)
│   │   ├── behavior_risk.py  # Speed anomalies, loitering & vessel risk (20% + 15%)
│   │   ├── attribution_engine.py # Normalized culprit ranking & confidence scoring
│   │   └── llm_intelligence.py # Bot Guardian Groq Llama-3 Forensic Analyst
│   │
│   └── pipeline.py           # Unified End-to-End Orchestrator
│
├── web/                      # Production Interactive Surveillance Dashboard
│   ├── server.py             # REST API server & HTTP request dispatcher
│   └── templates/
│       └── index.html        # High-contrast dark surveillance UI (Leaflet + Chart.js)
│
├── data/                     # SAR training tiles, labels & hotspot scenarios
├── training/                 # Deep learning training loops (YOLO-seg & DeepLabV3+)
├── weights/                  # Pre-trained deep neural network weights
├── drishti_architecture_blueprint.md # Formal mission architecture specification
├── MODELS_AND_DATASETS.md    # Detailed technical models & dataset documentation
├── requirements.txt          # Python dependencies
└── README.md                 # Project Overview & Architecture Guide
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/ark208/SIH20260-Project-Sea-guardian.git
cd SIH20260-Project-Sea-guardian

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Bot Guardian Chat)
If you wish to interact with the **Bot Guardian** conversational AI analyst in real-time, set your Groq API key in your terminal session:
```bash
# On Linux/macOS:
export GROQ_API_KEY="your_groq_api_key_here"

# On Windows (PowerShell):
$env:GROQ_API_KEY="your_groq_api_key_here"
```
*(The core vision, hydrodynamic rewind, GNN attribution, and web dashboard run fully offline without requiring any external API key).*

### 3. Run Pipeline via Terminal
Execute the end-to-end verification script across maritime incident scenarios:
```bash
python -m core.pipeline
```

### 4. Launch the Surveillance Web Dashboard
Start the local operations server:
```bash
python web/server.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in any modern browser to access:
- Interactive Leaflet ocean tracking map with natural dual-layer slick masks.
- Backward rewind uncertainty cones and particle clouds.
- Benchmark comparison demonstrating why naive proximity models fail.
- Radar charts breakdown across Spatial, Dark, Behavioral, and Risk metrics.
- Bot Guardian AI forensic analyst chat modal.

---

## 📊 Scientific Verification & Documentation

For complete mathematical derivations, neural network hyperparameter settings, dataset benchmarks (Copernicus Sentinel-1 SAR & Spire AIS), and performance metrics:
👉 **Refer to [MODELS_AND_DATASETS.md](MODELS_AND_DATASETS.md)**
