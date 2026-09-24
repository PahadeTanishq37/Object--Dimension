# RealSense D455f Multi-View 3D Reconstruction Pipeline

> **High-Precision Multi-View Point-Cloud Acquisition, Dataset Management, Spatial Registration & 3D Target Segmentation System**

---

## 📌 Pipeline Overview & Modular Roadmap

| Module | Purpose | Status |
|:---:|:---|:---:|
| **Module 1** | Synchronized RGB-D Acquisition & Calibrated Metric 3D Point Cloud Generation | ✅ Complete & Verified |
| **Module 2** | Multi-View Point-Cloud Capture & Structured Dataset Management | ✅ Complete & Verified |
| **Module 3** | Multi-View Point-Cloud Registration (Coarse-to-Fine FPFH & Point-to-Plane ICP) | ✅ Complete & Verified |
| **Module 4** | 3D Target Object Segmentation (RANSAC Plane Removal & DBSCAN Spatial Clustering) | ✅ Complete & Verified |
| **Module 5** | Point-Cloud Fusion, Volumetric TSDF Integration & Meshing | 🔄 Next Step |
| **Module 6** | 3D Physical Metrology & Anthropometric Measurement (Object, Head, Body) | 🔄 Planned |

---

## 📐 Mathematical & Algorithmic Foundations

### 1. Multi-View Registration (Module 3)
- **Local to Global Transformation**: View 001 serves as the global origin ($T_0 = I_{4\times4}$).
- **Rigid 6-DoF Mapping**:
  $$P_{\text{target}} = R \cdot P_{\text{source}} + t$$
- **Point-to-Plane ICP Objective**:
  $$E(R, t) = \sum_{i=1}^N \Big( \big( (R p_i + t) - q_i \big) \cdot n_i \Big)^2$$

### 2. 3D Target Object Segmentation (Module 4)
- **Dominant Plane Estimation (RANSAC)**:
  Fits a 3D plane model $ax + by + cz + d = 0$ with unit normal $\|[a, b, c]\| = 1$.
  Points satisfying $|aX + bY + cZ + d| \le d_{\text{plane}}$ are classified as support plane inliers (table, floor, desk) and subtracted.
- **Euclidean 3D Spatial Clustering (DBSCAN)**:
  Partitions the remaining foreground points into spatial clusters based on search radius $\epsilon$ and minimum points $N_{\text{min}}$.
- **Multi-Criteria Target Ranking**:
  Candidate clusters are scored without hard-coding dimensions:
  $$\text{Score} = 0.35 \cdot S_{\text{points}} + 0.30 \cdot S_{\text{elevation}} + 0.20 \cdot S_{\text{centrality}} + 0.15 \cdot S_{\text{compactness}}$$
- **Boundary-Preserving Outlier Removal**:
  Applies statistical outlier filtering ($k=25, \sigma=2.0$) to eliminate diffuse scan fringe while preserving sharp physical edges and geometric corners.

---

## 📂 Project Structure

```
pointcloud_reconstruction/
│
├── module1_pointcloud_acquisition.py      ← Module 1: Real-time RGB-D & single-frame point cloud
├── module2_multiview_dataset_capture.py   ← Module 2: Multi-view observation capture & dataset manager
├── module3_multiview_registration.py      ← Module 3: Coarse-to-fine registration & global alignment
├── module4_object_segmentation.py         ← Module 4: Dominant plane removal & DBSCAN 3D segmentation
├── requirements.txt                       ← Python dependencies (Open3D, PyRealSense2, OpenCV, SciPy)
├── README.md                              ← Complete technical documentation & user manual
│
├── output/                                ← Module 1 point cloud snapshots
│
└── datasets/                              ← Multi-view capture sessions & pipeline products
    ├── session_001/
    ├── session_002/
    └── session_003/
        ├── metadata.json                  ← Root session index & summary
        ├── view_001/ .. view_007/         ← Raw untouched viewpoint observations
        │
        ├── registration/                  ← Module 3 Outputs
        │   ├── global_registered_cloud.ply       ← Unified multi-view registered scene (meters)
        │   ├── registration_results.json         ← 4x4 Transformation matrices & metrics
        │   └── registration_validation_plot.png  ← Multi-panel registration visual report
        │
        └── segmentation/                  ← Module 4 Outputs
            ├── object_only.ply                   ← Clean segmented target object (meters, XYZ+RGB)
            ├── object_only.csv                   ← Tabular point coordinates (x_m,y_m,z_m,r,g,b)
            ├── segmentation_results.json         ← Plane equation, cluster rankings, bounding box
            └── segmentation_visualization.png    ← 4-panel segmentation visual validation report
```

---

## 🚀 Module 4 Operations & Quickstart

### 1. Automated Diagnostic Self-Test
Runs synthetic table+box segmentation verification and processes a real multi-view registered session:
```bash
python module4_object_segmentation.py --self-test
```

### 2. Segment a Specific Dataset Session
```bash
python module4_object_segmentation.py --session session_003
```

### Optional Tuning Parameters:
```bash
python module4_object_segmentation.py --session session_003 --plane-dist 0.008 --eps 0.015 --min-pts 40
```

---

## 🔬 Experimental Results (Verified on RealSense D455f Multi-View `session_003`)

- **Input Registered Scene**: `1,477,194` points
- **Support Plane Subtraction**: `218,018` inliers ($14.8\%$ of scene)
- **Clusters Evaluated**: `166` distinct spatial candidate clusters
- **Selected Target Object**:
  - Cluster #0 with **`328,545` points** ($96.2\%$ retained after boundary cleaning)
  - 3D Bounding Box: **$72.0\text{ cm} \times 44.3\text{ cm} \times 54.1\text{ cm}$**
  - Centroid: $[+0.296, +0.069, +0.891]\text{ m}$ in Global Coordinate Frame
- **Exported Artifacts**:
  - `object_only.ply` (ASCII 1.0 in metric meters)
  - `object_only.csv` (Coordinates + Color)
  - `segmentation_results.json`
  - `segmentation_visualization.png`
