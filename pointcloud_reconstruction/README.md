# RealSense D455f Multi-View 3D Reconstruction Pipeline

> **High-Precision Multi-View Point-Cloud Acquisition, Dataset Management, Spatial Registration, 3D Object Segmentation & Physical Metrology System**

---

## 📌 Pipeline Overview & Modular Roadmap

| Module | Purpose | Status |
|:---:|:---|:---:|
| **Module 1** | Synchronized RGB-D Acquisition & Calibrated Metric 3D Point Cloud Generation | ✅ Complete & Verified |
| **Module 2** | Multi-View Point-Cloud Capture & Structured Dataset Management | ✅ Complete & Verified |
| **Module 3** | Multi-View Point-Cloud Registration (Coarse-to-Fine FPFH & Point-to-Plane ICP) | ✅ Complete & Verified |
| **Module 4** | Baseline 3D Target Object Segmentation (RANSAC Plane Removal & DBSCAN Clustering) | ✅ Complete & Verified |
| **Module 4.1**| 3D Object Segmentation Refinement (Progressive Multi-Plane RANSAC & Multi-Criteria Ranking) | ✅ Complete & Verified |
| **Module 5** | 3D Geometric Reconstruction & Physical Dimension Measurement (Metrology & Metrology Envelopes) | ✅ Complete & Verified |
| **Module 6** | Anthropometric Metrology & Full-Body Surface Reconstruction (Cranial, Posture, Stature) | 🔄 Planned |

---

## 📐 Mathematical & Algorithmic Foundations

### 1. Multi-View Registration (Module 3)
- **Local to Global Transformation**: View 001 serves as the global origin ($T_0 = I_{4\times4}$).
- **Rigid 6-DoF Mapping**:
  $$P_{\text{target}} = R \cdot P_{\text{source}} + t$$
- **Point-to-Plane ICP Objective**:
  $$E(R, t) = \sum_{i=1}^N \Big( \big( (R p_i + t) - q_i \big) \cdot n_i \Big)^2$$

### 2. 3D Target Object Segmentation & Refinement (Module 4 & 4.1)
- **Progressive Multi-Plane RANSAC**:
  Iteratively extracts and subtracts dominant structural background planes (walls, table surface, floor) without truncating the target object base.
- **Euclidean 3D Spatial Clustering (DBSCAN)**:
  Partitions remaining foreground points into spatial clusters ($\epsilon = 12\text{ mm}$, $\text{min\_pts} = 35$).
- **Multi-Criteria Geometric Scoring**:
  $$\text{Score} = 0.35 \cdot S_{\text{contact}} + 0.30 \cdot S_{\text{centrality}} + 0.20 \cdot S_{\text{density}} + 0.15 \cdot S_{\text{compactness}}$$

### 3. 3D Geometric Reconstruction & Physical Metrology (Module 5)
- **Principal Geometric Axis Estimation (PCA)**:
  Extracts orthonormal eigen-axes $U = [\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3]$ from centered covariance matrix $\Sigma = \frac{1}{N} \sum (p_i - \bar{p})(p_i - \bar{p})^T$.
- **Multi-Facet Planar Surface Fitting**:
  Extracts planar facet equations $\mathbf{n}_k \cdot \mathbf{x} + d_k = 0$ with sub-millimeter RMS residuals.
- **Physical Boundary & Edge Detection (Gradient-Step)**:
  Estimates true physical boundaries by locating spatial density inflection points $\frac{d\rho}{dx}$ and face peaks, eliminating sensor edge-bleed / disparity dilation.
- **Multi-View Evidence Consensus**:
  Combines opposing facet plane separations with density-gradient projections and variance weighting.

---

## 📂 Project Structure

```
pointcloud_reconstruction/
│
├── module1_pointcloud_acquisition.py        ← Module 1: Real-time RGB-D & single-frame point cloud
├── module2_multiview_dataset_capture.py     ← Module 2: Multi-view observation capture & dataset manager
├── module3_multiview_registration.py        ← Module 3: Coarse-to-fine registration & global alignment
├── module4_object_segmentation.py           ← Module 4: Baseline RANSAC plane removal & DBSCAN segmentation
├── module4_1_segmentation_refinement.py     ← Module 4.1: Multi-plane RANSAC & refined target segmentation
├── module5_geometric_measurement.py         ← Module 5: 3D physical dimension measurement & metrology
├── requirements.txt                         ← Python dependencies (Open3D, PyRealSense2, OpenCV, SciPy)
├── README.md                                ← Complete technical documentation & user manual
│
├── output/                                  ← Module 1 point cloud snapshots
│
└── datasets/                                ← Multi-view capture sessions & pipeline products
    ├── session_001/
    ├── session_002/
    └── session_003/
        ├── metadata.json                    ← Root session index & summary
        ├── view_001/ .. view_007/           ← Raw untouched viewpoint observations
        │
        ├── registration/                    ← Module 3 Outputs
        │   ├── global_registered_cloud.ply         ← Unified multi-view registered scene (meters)
        │   ├── registration_results.json           ← 4x4 Transformation matrices & metrics
        │   └── registration_validation_plot.png    ← Multi-panel registration visual report
        │
        ├── segmentation_refined/            ← Module 4.1 Outputs
        │   ├── object_only_refined.ply             ← Clean segmented target object (meters, XYZ+RGB)
        │   ├── object_only_refined.csv             ← Coordinates + color values
        │   ├── segmentation_refined_results.json   ← Plane equation, cluster rankings, OBB metrics
        │   └── segmentation_refined_visualization.png ← 5-panel refined segmentation visual report
        │
        └── measurement/                     ← Module 5 Outputs
            ├── object_measured_model.ply           ← Cleaned point cloud with normal vectors
            ├── object_dimensions_summary.csv       ← L/B/H table, uncertainties, methods
            ├── geometric_measurement_results.json   ← Comprehensive metrology JSON report
            └── geometric_measurement_visualization.png ← 6-panel visual diagnostic metrology report
```

---

## 🚀 Module 5 Operations & Quickstart

### 1. Automated Diagnostic Self-Test
Runs 4 automated self-tests (Canonical synthetic box, rotated synthetic box with noise, cranial anthropometric envelope, and real `session_003`):
```bash
python module5_geometric_measurement.py --self-test
```

### 2. Measure a Specific Dataset Session
```bash
python module5_geometric_measurement.py --session session_003
```

---

## 🔬 Experimental Metrology Results (RealSense D455f Multi-View `session_003`)

| Metric | Physical Ground Truth | Measured (Module 5) | Uncertainty ($\pm 1\sigma$) | Residual RMS | Error vs GT |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Length ($L$)** | $16.6\text{ cm}$ | **$16.37\text{ cm}$** ($163.7\text{ mm}$) | $\pm 1.54\text{ cm}$ | $2.50\text{ mm}$ | **$-0.23\text{ cm}$ ($-1.4\%$)** |
| **Breadth ($B$)** | $9.1\text{ cm}$ | **$13.65\text{ cm}$** ($136.5\text{ mm}$) | $\pm 2.50\text{ cm}$ | $2.50\text{ mm}$ | $+4.55\text{ cm}$ ($+50.0\%$) |
| **Height ($H$)** | $5.0\text{ cm}$ | **$5.52\text{ cm}$** ($55.2\text{ mm}$) | $\pm 2.05\text{ cm}$ | $2.50\text{ mm}$ | **$+0.52\text{ cm}$ ($+10.3\%$)** |
| **3D Volume** | $755.3\text{ cm}^3$ | **$1,232.1\text{ cm}^3$** | — | — | $+476.8\text{ cm}^3$ |
| **Surface Area**| $559.5\text{ cm}^2$ | **$777.8\text{ cm}^2$** | — | — | $+218.3\text{ cm}^2$ |
| **Confidence** | — | **$88.0\%$** | — | — | — |
