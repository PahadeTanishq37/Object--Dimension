# RealSense Object Dimension Measurement System

> **Stage 1 — Camera Validation**
> An RGB-D object dimension measurement system using the Intel RealSense D455f.

---

## Project Objective

Build a system that uses an Intel RealSense D455f RGB-D camera to measure the real-world physical dimensions (Length, Breadth, Height) of physical objects placed in front of it — at any distance, without hard-coding the camera-to-object distance.

The system uses metric 3D coordinates derived from depth data and camera intrinsics, making measurements **distance-invariant**: an object measured at 50 cm and at 150 cm will return approximately the same physical dimensions.

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Camera    | Intel RealSense D455f |
| Interface | USB 3.1 Gen 1 (SuperSpeed) |
| Depth technology | Stereo IR + active IR projector |
| Depth range | 0.6 m – 6 m (recommended) |
| Colour resolution | Up to 1920×1080 @ 30 FPS |
| Depth resolution | Up to 1280×720 @ 30 FPS |

---

## Software

| Package | Purpose |
|---------|---------|
| Python 3.10+ | Runtime |
| `pyrealsense2` ≥ 2.54 | Intel RealSense SDK bindings |
| `opencv-python` ≥ 4.8 | Frame display, image processing |
| `numpy` ≥ 1.24 | Numerical array operations |

---

## Project Structure

```
realsense_measurement/
│
├── camera_validation.py   ← Stage 1: Camera health check
├── requirements.txt       ← Python dependencies
└── README.md              ← This file
```

---

## Stage 1 — `camera_validation.py`

### What it does

1. **Device detection** — Scans for any connected RealSense device. Prints
   the device name, serial number, firmware version, and USB descriptor.

2. **Pipeline startup** — Opens colour (BGR8, 1280×720 @ 30 FPS) and
   depth (Z16, 1280×720 @ 30 FPS) streams.

3. **Intrinsics report** — Prints to the terminal:
   - Colour stream resolution
   - Depth stream resolution
   - Depth scale (metres per raw depth unit)
   - Focal lengths `fx`, `fy` (pixels) for both streams
   - Principal point `cx`, `cy` (pixels) for both streams
   - Distortion model and coefficients

4. **Aligned depth** — Uses `rs.align` to reproject the depth frame into
   the colour camera's coordinate system, so that every colour pixel has a
   corresponding depth value.

5. **Live display** — Two windows:
   - `D455f — RGB (aligned)` : live colour feed with a centre crosshair and
     a live depth readout (in cm) at the crosshair position.
   - `D455f — Depth (colourised)` : false-colour (Jet) depth map, same
     crosshair.

6. **Clean shutdown** — Press `Q`, `ESC`, or `Ctrl-C` to stop the pipeline
   and close all windows safely.

---

## How to Install Dependencies

### Option A — pip (recommended)

```bash
cd realsense_measurement
pip install -r requirements.txt
```

### Option B — manual install

```bash
pip install pyrealsense2>=2.54.0 opencv-python>=4.8.0 numpy>=1.24.0
```

> **Note (Windows):** The `pyrealsense2` package on PyPI includes the
> RealSense SDK runtime. You do **not** need a separate SDK installation
> for Python use on Windows, but installing the
> [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases)
> alongside gives you the RealSense Viewer tool for hardware diagnostics.

---

## How to Run

```bash
cd realsense_measurement
python camera_validation.py
```

Press **Q** or **ESC** in any open window, or **Ctrl-C** in the terminal, to exit.

---

## Expected Output

### Terminal output (example)

```
============================================================
  CONNECTED DEVICE
============================================================
  Name                  : Intel RealSense D455
  Serial Number         : 123456789012
  Firmware Version      : 5.15.0.2
  Product Line          : D400
  USB Type              : 3.2
============================================================

[INFO] Starting pipeline ...
[INFO] Pipeline started successfully.

------------------------------------------------------------
  COLOUR STREAM INTRINSICS
------------------------------------------------------------
  Resolution   : 1280 x 720 px
  Focal length : fx = 912.4800 px,  fy = 912.4800 px
  Principal pt : cx = 635.9600 px,  cy = 356.2100 px
  Distortion   : model=distortion.inverse_brown_conrady,  coeffs=[...]

------------------------------------------------------------
  DEPTH STREAM INTRINSICS
------------------------------------------------------------
  Resolution   : 1280 x 720 px
  Focal length : fx = 637.8600 px,  fy = 637.8600 px
  Principal pt : cx = 638.5000 px,  cy = 362.1000 px
  Distortion   : model=distortion.brown_conrady,  coeffs=[0, 0, 0, 0, 0]

------------------------------------------------------------
  DEPTH SCALE
------------------------------------------------------------
  Depth scale  : 0.001 m / depth-unit
  (Raw value 1000 -> 1.0000 m = 100.00 cm)
------------------------------------------------------------

[INFO] Warming up — discarding 10 frames ...
[INFO] Warm-up complete. Displaying live feed.

       Press 'Q' or ESC in either window to quit.
```

### Visual windows

- **RGB window** — Live colour feed. A green crosshair marks the frame centre.
  A text overlay shows the depth at the crosshair (e.g. `Centre depth: 82.3 cm`).
- **Depth window** — Jet colourmap depth image. Near = red/warm, far = blue/cool.

---

## Known Limitations (Stage 1)

- No object detection, segmentation, or dimension measurement is implemented yet.
- Depth accuracy degrades on transparent, specular, or very dark surfaces.
- The D455f has a minimum recommended depth range of ~0.6 m; very close objects
  may return invalid (zero) depth values.
- Colour-depth alignment introduces minor interpolation artefacts at depth
  discontinuities (object edges).
- Depth noise increases with distance; for best accuracy keep objects within
  0.5 m – 2.0 m of the camera.

---

## Stage 2 — `depth_to_3d.py`

### What it does

Validates the full pipeline from a depth-image pixel to a calibrated metric
3D point in the RealSense camera coordinate system.

1. **Live depth window** — false-colour (Jet) depth map.
2. **Interactive mouse probe** — move the mouse over the window; the HUD
   updates in real time showing:
   - Pixel `(u, v)`
   - Raw z16 depth value
   - `Z` in metres and centimetres
   - `X`, `Y`, `Z` in metres and centimetres
3. **Left-click** — prints a full numerical validation report to the terminal.
4. **Auto-probe** — on the first frame the centre pixel is automatically
   probed and reported to the terminal.

### Run

```bash
python depth_to_3d.py
```

### RealSense coordinate system

```
+X →  right    (along sensor width)
+Y ↓  down     (along sensor height)
+Z →  forward  (toward the scene)
```

Origin = depth sensor optical centre. All values in **metres**.

---

## Distance-Invariance Experiment

> **Key concept:** `Z` SHOULD change when you move the camera.
> What must NOT change is the **computed physical size** of an object.

### Setup

Place a flat object (or aim at a flat wall) in front of the D455f.
Run `depth_to_3d.py`, hover the mouse over the object centre, and record
the Z value at each distance.

| Physical distance | Expected Z | Notes |
|-------------------|-----------|-------|
| 50 cm  | ≈ 0.50 m | minimum recommended range |
| 75 cm  | ≈ 0.75 m | |
| 100 cm | ≈ 1.00 m | |
| 125 cm | ≈ 1.25 m | |
| 150 cm | ≈ 1.50 m | |

**What you will observe:**
- `Z` tracks the actual physical distance. ✅ This is correct.
- The pixel size of the object shrinks as distance increases.
- But in Stage 3+ the recovered 3D dimensions will remain constant because
  the smaller pixel count is exactly compensated by the larger Z value in
  the deprojection formula:
  ```
  X_real = (u - cx) × Z / fx
  ```
  Fewer pixels × proportionally larger Z = the same physical size.

### Expected terminal output format

```
==================================================
  PIXEL PROBE
==================================================
  Pixel       : (640, 360)
  Raw depth   : 1003   (z16 uint16)
  Depth scale : 0.00100000 m/unit
  Z (depth)   : 1.0030 m  = 100.30 cm
  --- 3D Point (camera frame) ---
  X           : +0.0000 m  = +0.00 cm  (+right)
  Y           : -0.0000 m  = -0.00 cm  (+down)
  Z           : +1.0030 m  = +100.30 cm  (+forward)
==================================================
```

---

## Stage 3 — `depth_accuracy_test.py`

### What it does

Characterises the D455f's systematic and random depth errors at multiple
distances before using depth data for object measurement.

| Feature | Detail |
|---------|--------|
| ROI sampling | 21×21 px patch, mean/median/std/min/max/valid count |
| Temporal sampling | Collects ~5 s of frames at a locked ROI; reports frame-to-frame stability |
| Depth quality metadata | Reads `depth_fill_rate` and `depth_stdev` from firmware frame metadata |
| CSV logging | Saves every test point to `depth_accuracy_results.csv` |
| Error analysis | Mean error, MAE, RMSE, relative error, linear fit on exit |

### Run

```bash
python depth_accuracy_test.py
```

### Keyboard controls

| Key | Action |
|-----|--------|
| Mouse move | Move ROI (when unlocked) |
| Left-click | Lock / unlock ROI centre |
| `S` | Start a 5-second temporal sample at the current ROI |
| `R` | Record this ROI (prompts for your tape-measure distance) |
| `Q` / `ESC` | Quit and print the final summary |

---

## Camera Reference Point — IMPORTANT

The RealSense **depth value represents distance from the depth sensor's
optical centre**, not from any arbitrary point on the camera housing.

### How to define a repeatable measurement reference

1. Run `depth_to_3d.py` and aim the camera at a flat wall.
2. Place a small piece of tape on the wall at the crosshair centre.
3. Slowly move the camera toward or away from the wall.
4. The reported Z will change. Note the Z value at each position.
5. **The physical reference point** is the depth sensor optical centre.
   To locate it approximately:
   - The D455f has two IR cameras and a projector on its front face.
   - The depth sensor optical centre is approximately behind the
     **left IR camera lens** (when viewed from the front).
   - For practical measurements: hold the camera on a stable tripod,
     measure from the **front face of the camera housing** to the wall,
     then subtract the ~1–2 cm offset to the optical centre.
   - The exact offset can be determined from your Stage 3 linear-fit
     intercept value.
6. **Be consistent**: use the same physical reference point every time.

---

## Controlled Depth Accuracy Experiment (Stage 3)

### Equipment needed

- Intel RealSense D455f on a **stable tripod or flat table**
- A **flat, matte, non-reflective** wall or cardboard panel
- A **tape measure** (metal preferred)
- A computer running `depth_accuracy_test.py`

### Procedure

1. Mount the camera on a tripod. Point it at the flat surface.
   The surface should be **approximately perpendicular** to the camera axis.
2. Run `depth_accuracy_test.py`.
3. Move the mouse ROI to the centre of the flat surface.
   Left-click to **lock** the ROI.
4. For each test distance:

   a. Position the camera at the target distance (tape measure from
      your chosen reference point to the surface).

   b. Wait ~2 seconds for vibrations to settle.

   c. Press **`S`** to collect a 5-second temporal sample.
      **Do not move the camera during collection.**

   d. Press **`R`** to record the measurement.
      When prompted, type the tape-measure distance in cm.

   e. Note the terminal output (error, std).

5. **Recommended distances:**

   | Distance | Why |
   |----------|-----|
   | 75 cm    | Near edge of recommended range |
   | 90 cm    | From your Stage 2 data |
   | 100 cm   | Round number reference |
   | 110 cm   | From your Stage 2 data |
   | 125 cm   | Mid-range |
   | 150 cm   | Far end of close range |

6. **Repeat one distance (e.g. 100 cm) three times** to distinguish
   systematic error from random noise.

7. After all measurements, press `Q` to quit.
   The terminal will print the final error summary.

### Expected CSV output

```csv
timestamp,manual_distance_cm,realsense_mean_cm,realsense_median_cm,std_cm,...
2026-09-21T15:30:00,75.00,68.45,68.50,0.21,...
2026-09-21T15:31:00,90.00,82.70,82.65,0.18,...
2026-09-21T15:32:00,100.00,92.70,92.68,0.15,...
```

### What to send back for analysis

Please share:

1. The **complete terminal output** when you quit (the final summary table).
2. The **`depth_accuracy_results.csv`** file.
3. A note on the **surface type** (wall, cardboard, white paper, etc.)
4. A note on **lighting conditions** (indoor, sunlit window nearby, etc.)
5. Whether the IR projector/emitter appeared to be active (visible in the
   depth colourmap — surfaces should show colour, not black patches).

---

## Stage 3b — `depth_repeatability_test.py`

### What it does

A **controlled 3×3 repeatability experiment** to determine whether depth
error is systematic (fixable) or random (sensor/setup problem).

| Parameter | Value |
|-----------|-------|
| Target distances | 100 cm, 110 cm, 120 cm |
| Repeats per distance | 3 |
| Total measurements | 9 |
| Sample duration | 5 seconds per measurement |
| ROI | 21×21 px — locked once, used for all 9 |
| Primary estimate | Median of per-frame medians |
| Outputs | CSV + PNG plot + terminal summary |

### Run

```bash
python depth_repeatability_test.py
```

### Workflow

1. **ROI selection** — move the mouse over the depth window to position the
   ROI box on the flat surface. Left-click **once** to lock it permanently.
2. **9 measurements** — the program shows you each measurement in sequence
   and asks you to press ENTER when ready.
3. **No camera movement** — only move the flat target surface.
4. **Outputs** on exit:
   - `depth_validation_repeated.csv` — all 9 rows
   - `depth_validation_plot.png` — scatter + ideal line + regression
   - Terminal summary with per-distance and overall statistics

### Output files

| File | Contents |
|------|----------|
| `depth_validation_repeated.csv` | One row per measurement |
| `depth_validation_plot.png` | Scatter plot vs ideal line |

---

## Measurement Terminology

> **Important:** Understanding these terms is critical for correctly
> interpreting depth sensor results.

| Term | Definition | Example |
|------|-----------|---------|
| **Accuracy** | Closeness of a measurement to the true value | RS reads 102.3 cm vs actual 100 cm → 2.3 cm error |
| **Precision** | Tightness of repeated measurements (spread) | 3 repeats: 102.2, 102.3, 102.3 → high precision |
| **Repeatability** | Precision under identical conditions, same operator | Max–min range across repeats at the same distance |
| **Systematic error** | Consistent offset in the same direction | Always reads ~2–7 cm high → likely a bias |
| **Random error** | Unpredictable variation around the mean | std ~0.2 cm → low random error |
| **RMSE** | Root mean square error — combines bias + random error | Lower is better |
| **R²** | How well a linear model fits the data | > 0.999 = very strong linear relationship |

> **Tape-measure uncertainty:** A hand-held tape measure has a typical
> uncertainty of ±2–3 mm due to reading error, tape sag, and inconsistent
> reference point placement. This means a manual reading of "100 cm" may
> actually be 99.8–100.3 cm. Always use the same reference point and
> take multiple manual readings to reduce this uncertainty.

---

## Camera Reference Point — IMPORTANT

The RealSense **depth value represents distance from the depth sensor's
optical centre** — NOT from the front face of the camera housing, NOT from
the USB connector side.

### Consistent reference point procedure

1. Choose a physical mark on the camera (e.g. the centre of the left IR
   camera lens when facing the front).
2. Use the same mark for every tape measurement throughout all experiments.
3. The Stage 3b linear regression intercept will absorb any constant offset
   between your chosen reference mark and the true optical centre.
4. Do **not** attempt to correct this offset manually — let the data show it.

---

## Controlled Repeatability Experiment Procedure

### Equipment

- D455f on a **stable tripod or flat table** — must not move
- A **flat, matte, non-reflective** surface (white cardboard recommended)
- A **metal tape measure**
- A computer running `depth_repeatability_test.py`

### Step-by-step

1. Mount camera on tripod. Point at the flat surface.
2. Run `python depth_repeatability_test.py`
3. Move mouse to the centre of the flat surface → left-click to **lock ROI**.
4. Follow the on-screen instructions for each of the 9 measurements.
5. For **each measurement**:
   - Move the flat surface to the target distance.
   - Verify the surface is visible in the ROI box (live window).
   - Wait ~2 s for vibrations to settle.
   - Press **ENTER** in the terminal.
   - Wait 5 s without touching anything.
6. After all 9, press **Q** to quit.
7. Send back the results (see below).

### What to send back

1. Complete terminal output (final report section)
2. The file `depth_validation_repeated.csv`
3. The file `depth_validation_plot.png`
4. Surface type and lighting conditions
5. Which physical reference point you used for the tape measure

---

---

## Stage 4.1 — `object_segmentation.py` (ROI-Assisted Segmentation)

### What it does

Isolates a foreground target object (rectangular box) using **user-guided ROI selection** combined with **depth histogram clustering**, **depth discontinuity analysis**, and **connected component scoring**. Background surfaces (walls, tables, laptops, floor) are strictly zeroed out and excluded from the resulting 3D point cloud.

| Feature | Detail |
|---------|--------|
| **Manual ROI Gating** | User drags a bounding box ~10–20% larger than the object; strictly zeroes outside |
| **Depth Histogram Mode** | Computes 1 cm resolution histogram inside ROI to find true foreground peak ($Z_{peak}$) |
| **Edge-Preserving Filter** | Conservative morphology preserving physical object perimeter without bloat |
| **Coherent Blob Selection** | Scores connected blobs by area and proximity to ROI center (rejects background corners) |
| **3D Deprojection** | Pinhole model using live color intrinsics: $(u, v, z) \rightarrow (X, Y, Z)$ in metres |
| **Sanity Diagnostics** | Real-time 3D span verification with `"SEGMENTATION INVALID — BACKGROUND CONTAMINATION"` guard |
| **Data Export** | Saves RGB, Binary Mask PNG, Depth PNG, 3D Point Cloud PLY & CSV, and 3D Scatter Plot PNG |

### Interactive Workflow & Controls

```
1. Run: python object_segmentation.py
2. Look at the RGB window: Click and DRAG a rectangle around the target box (~10-20% larger than box).
3. On mouse release, the ROI is LOCKED.
4. The algorithm automatically isolates the box inside the ROI and creates a clean mask.
5. Check HUD: Ensure Diagnostic reads "OBJECT ISOLATED OK" and spans match physical object scale.
6. Press [P] to export a 3D scatter plot of the extracted box point cloud.
7. Press [S] to save the complete snapshot bundle (RGB, Mask, Depth, 3D PLY/CSV).
8. Move box to another distance (e.g. 80 cm, 100 cm, 120 cm) -> Press [N] -> Drag new ROI.
```

| Key / Control | Action |
|---------------|--------|
| **Mouse Drag** | Select rectangular ROI around target object |
| **`N`** | Reset / Draw a new ROI |
| **`S`** | Save snapshot (RGB, binary mask, depth colormap, 3D point cloud `.ply` & `.csv`) |
| **`P`** | Render & save 3D Point Cloud scatter plot (`roi_object_pointcloud_plot.png`) |
| **`Tuning Sliders`** | Fine-tune `Depth Tol (cm)` around foreground peak, `Min Area`, and `Morph Kernel` |
| **`Q` / `ESC`** | Quit cleanly |

---

---

## Stage 5.3 — `box_measurement.py` (3D Geometry Debug & Measurement)

### What it does

Integrates comprehensive **3D Geometric Debug Inspection** into the live measurement pipeline, allowing full visualization and diagnostic inspection of intermediate geometric primitives:
- Detected Plane equations, normals ($N_1, N_2, N_3$), inlier counts, RMS & median residuals, and approximate surface areas.
- Inter-plane angular orthogonality ($\theta_{ij} \approx 90^\circ$).
- Reconstructed 3D Corners ($C_1 \dots C_8$) in camera coordinates.
- Reconstructed 3D Edges ($E_1 \dots E_{12}$) with normalized 3D direction vectors, dimensional classification ($L, B, H$), and point support metrics (number of supporting points and median distance to edge).
- Live on-screen debug HUD toggle (`[D]`) and debug snapshot export (`[S]`) to `debug_output/` containing `.ply`, `.png`, and `.txt` reports.

| Feature | Detail |
|---------|--------|
| **Plane Diagnostics** | RMS & median inlier distance residuals, normal unit vectors, inlier area |
| **Corner Labeling** | Reconstructed $C_1 \dots C_8$ vertices with 3D camera-frame $[X, Y, Z]$ coordinates (cm) |
| **Edge Classification** | $E_1 \dots E_{12}$ categorized as Length ($L$), Breadth ($B$), or Height ($H$) with direction vectors |
| **Edge Point Support** | Number of object points supporting each reconstructed edge and median distance to line |
| **Debug Mode Toggle** | Key `[D]` toggles on-screen edge/corner annotations and prints formatted summary to console |
| **Debug Bundle Export** | Key `[S]` writes RGB, mask, depth, 3D point cloud PLY, 3D plot, and report to `debug_output/` |
| **Ground Truth Analysis** | Key `[G]` inputs actual dimensions to track signed, absolute, and percentage error (%) |

### Interactive Controls

| Key | Action |
|-----|--------|
| **`D`** | **Toggle Debug Mode**: displays edge IDs ($E_1..E_{12}$), corners ($C_1..C_8$), and prints full 3D geometry breakdown to console |
| **`S`** | **Save Debug Snapshot**: saves complete debug bundle (`.ply`, `.png`, `.txt`) to `debug_output/` |
| **`T`** | Record distance-invariance measurement row to `box_measurement_distance_test_v2.csv` |
| **`G`** | Input / update ground-truth physical dimensions (Length, Breadth, Height in cm) |
| **`P`** | Export 3D Cuboid Geometry Debug plot to `box_3d_cuboid_reconstruction_plot.png` |
| **`N`** | Reset / Draw a new ROI |
| **`Q` / `ESC`** | Quit cleanly |

---

## Roadmap

| Stage | Description |
|-------|-------------|
| **1 ✅** | Camera validation and RGB-D streaming foundation |
| **2 ✅** | Depth pixel → metric 3D coordinate recovery |
| **3 ✅** | Depth accuracy characterisation and CSV logging |
| **3b ✅** | Controlled repeatability experiment (3×3 design) |
| **4.1 ✅** | ROI-assisted object segmentation & 3D point cloud extraction |
| **5.3 ✅** | 3D cuboid reconstruction & comprehensive geometry debug visualization |
| 6 | Autonomous 3D object detection & bounding box (replacing manual ROI) |
| 7 | Multi-object dimensioning & complex non-cuboid geometry estimation |
| 8 | Industrial metrology accuracy benchmarking & calibration lookup table |







