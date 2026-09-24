# Object Dimension Measurement System (Intel RealSense D455f)

> **Real-Time Distance-Invariant 3D Object Metrology & Cuboid Reconstruction System**

---

## 📌 Project Overview

This repository contains an end-to-end computer vision and 3D metrology pipeline using the **Intel RealSense D455f RGB-D camera**. The goal is to accurately calculate the real-world physical dimensions (**Length, Breadth, Height**) of 3D objects (e.g., packages, boxes, parcels) placed in front of the camera in real time — **invariant to the camera-to-object distance**.

By combining metric 3D point cloud deprojection, multi-plane RANSAC surface estimation, 3D orthogonal corner/edge reconstruction, and temporal filtering, the system achieves sub-centimeter accuracy without requiring hardcoded distance priors.

---

## 🚀 Key Capabilities & Implemented Stages

| Stage | Module | Description | Status |
|:---:|:---|:---|:---:|
| **Stage 1** | [`camera_validation.py`](realsense_measurement/camera_validation.py) | Device detection, stream alignment (1280×720 @ 30 FPS RGB+Depth), intrinsics extraction, and live depth probe. | ✅ Complete |
| **Stage 2** | [`depth_to_3d.py`](realsense_measurement/depth_to_3d.py) | Pinhole deprojection model $(u, v, z) \to (X, Y, Z)$ converting 2D depth pixels into calibrated 3D camera coordinates. | ✅ Complete |
| **Stage 3** | [`depth_accuracy_test.py`](realsense_measurement/depth_accuracy_test.py) | Multi-distance depth error characterization, temporal stability measurement, and CSV logging. | ✅ Complete |
| **Stage 3b** | [`depth_repeatability_test.py`](realsense_measurement/depth_repeatability_test.py) | Controlled 3×3 repeatability test suite, linear regression ($R^2 > 0.99$), and error plot generation. | ✅ Complete |
| **Stage 4.1** | [`object_segmentation.py`](realsense_measurement/object_segmentation.py) | Interactive ROI gating, depth histogram clustering, edge-preserving morphology, background suppression, and 3D PLY/CSV export. | ✅ Complete |
| **Stage 5.1–5.5** | [`box_measurement.py`](realsense_measurement/box_measurement.py) | Multi-plane RANSAC segmentation, plane normal orthogonality checks, 3D edge reconstruction ($E_1..E_{12}$), corner calculation ($C_1..C_8$), and live HUD debug mode. | ✅ Complete |
| **Stage 5.6** | [`distance_invariance_validation.py`](realsense_measurement/distance_invariance_validation.py) | Formal multi-distance verification (80 cm, 100 cm, 120 cm), MAE/RMSE calculations, and 4-panel diagnostic plot generator. | ✅ Complete |
| **Stage 6 (Core)** | [`object_detector.py`](realsense_measurement/object_detector.py) | Modular detector interface (`BaseObjectDetector`), temporal bounding box tracker (`DetectionTracker`), and autonomous RGB-D foreground detector. | ✅ Complete |

---

## 🛠️ System Architecture

```
                               ┌──────────────────────────────┐
                               │  Intel RealSense D455f       │
                               │  RGB (1280x720) + Depth Z16  │
                               └──────────────┬───────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────┐
                               │    Stream Alignment (RGB-D)  │
                               │  Camera Intrinsics (fx, fy)  │
                               └──────────────┬───────────────┘
                                              │
                                              ▼
                     ┌───────────────────────────────────────────────────┐
                     │           Object Detection & Localization         │
                     │  • Manual ROI or Autonomous RGB-D Detector        │
                     │  • Temporal Bounding Box Smoother                 │
                     └────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                     ┌───────────────────────────────────────────────────┐
                     │          Foreground Point Cloud Extraction        │
                     │  • Depth Histogram Peak Clustering                │
                     │  • Background & Surface Zeroing                   │
                     │  • Pinhole 3D Deprojection (u, v, z) -> (X, Y, Z) │
                     └────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                     ┌───────────────────────────────────────────────────┐
                     │           3D Geometric Reconstruction             │
                     │  • Multi-Plane RANSAC Normal Estimation           │
                     │  • Orthogonality Check (theta ≈ 90°)              │
                     │  • Edge Support & Vertex Extraction (C1..C8)      │
                     │  • Metric Dimensions (Length, Breadth, Height)    │
                     └────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                     ┌───────────────────────────────────────────────────┐
                     │           Temporal Filtering & HUD Output         │
                     │  • Outlier Rejection & Dimension Smoothing        │
                     │  • Real-time 3D Overlays & Debug Diagnostics      │
                     │  • Automated CSV / PLY / PNG Snapshot Exports     │
                     └───────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
Object--Dimension/
│
├── realsense_measurement/
│   ├── camera_validation.py               # Stage 1: Hardware integrity & alignment check
│   ├── depth_to_3d.py                     # Stage 2: Metric 3D deprojection probe
│   ├── depth_accuracy_test.py             # Stage 3: Depth accuracy characterization
│   ├── depth_repeatability_test.py        # Stage 3b: 3x3 repeatability experiment
│   ├── object_segmentation.py             # Stage 4.1: ROI segmentation & 3D cloud extraction
│   ├── box_measurement.py                 # Stage 5.1-5.5: 3D Cuboid measurement & geometry debug
│   ├── distance_invariance_validation.py  # Stage 5.6: Multi-distance validation suite
│   ├── object_detector.py                 # Stage 6: Modular object detector interface
│   ├── requirements.txt                   # Python library requirements
│   ├── README.md                          # Detailed technical manual
│   └── debug_output/                      # Exported 3D PLY, CSV, and debug snapshots
│
└── README.md                              # This repository documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Intel RealSense D455 / D455f / D435** connected via USB 3.0+ port

### 2. Install Dependencies
```bash
cd realsense_measurement
pip install -r requirements.txt
```

---

## 💻 Quickstart Guide

### 1. Check Camera Health & RGB-D Alignment
```bash
python realsense_measurement/camera_validation.py
```

### 2. Run Interactive 3D Cuboid Dimension Measurement
```bash
python realsense_measurement/box_measurement.py
```
- **Click & drag** an ROI rectangle around the target box.
- Press **`D`** to toggle 3D geometry debug HUD (plane normals, corners $C_1..C_8$, edges $E_1..E_{12}$).
- Press **`G`** to enter ground-truth dimensions for real-time accuracy error tracking.
- Press **`S`** to export a complete debug bundle (PLY point cloud, RGB, depth, and text report).

### 3. Run Distance-Invariance Validation Suite
```bash
python realsense_measurement/distance_invariance_validation.py
```
- Tests object dimensions across multiple distances (e.g., 80 cm, 100 cm, 120 cm).
- Press **`R`** to capture a batch sample at each distance.
- Press **`P`** to generate the 4-panel analysis plot `distance_invariance_plot.png`.

---

## 📊 Summary of Experimental Validation

| Experiment | Target Metric | Achieved Result |
|---|---|---|
| **Depth Repeatability (3×3)** | Frame-to-frame & setup repeatability | $R^2 > 0.999$, temporal $\sigma < 0.25\text{ cm}$ |
| **Object Segmentation** | Background contamination removal | Clean point clouds with zero background bleed |
| **Distance Invariance (80–120 cm)** | Scale invariance across distance | Coefficient of Variation ($CV$) $< 3.0\%$ |
| **Cuboid Orthogonality** | Inter-plane normal angles | Detected faces within $\pm 5^\circ$ of $90^\circ$ |

---

## 🗺️ Roadmap & Upcoming Stages

- [x] **Stage 1**: Camera validation and RGB-D stream alignment
- [x] **Stage 2**: 2D depth pixel to metric 3D coordinate recovery
- [x] **Stage 3 & 3b**: Depth accuracy & controlled repeatability testing
- [x] **Stage 4.1**: Interactive ROI-assisted segmentation & 3D cloud export
- [x] **Stage 5.1–5.5**: Multi-plane RANSAC cuboid reconstruction & edge geometry debug
- [x] **Stage 5.6**: Distance-invariance validation suite & plotting
- [x] **Stage 6 (Core)**: Modular object detector architecture & autonomous baseline
- [ ] **Stage 7**: Multi-object dimensioning & non-cuboid geometry estimation
- [ ] **Stage 8**: Metrology calibration lookup tables & production deployment