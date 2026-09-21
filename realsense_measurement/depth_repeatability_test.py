"""
depth_repeatability_test.py
============================
Stage 3b — RealSense D455f Controlled Repeatability Experiment

Purpose:
    Determine whether depth measurement error is:
      (A) Highly repeatable but systematically biased — fixable with calibration
      (B) Noisy / inconsistent         — sensor or setup problem
      (C) Approximately unbiased       — ready for measurement use

Experiment design:
    3 target distances  ×  3 repeated measurements  =  9 total measurements

    Target distances: 100 cm, 110 cm, 120 cm

    CRITICAL:
      * The D455f MUST remain completely stationary throughout.
      * Only the TARGET SURFACE moves to each new distance.
      * The ROI is locked ONCE at startup and used for all 9 measurements.
      * 5 seconds of depth frames are collected per measurement.
      * The MEDIAN OF FRAME MEDIANS is the primary estimate.

Outputs:
    depth_validation_repeated.csv   — row per measurement
    depth_validation_plot.png       — scatter vs ideal line

Keyboard controls:
    Left-click on depth window  → lock the ROI (only before first measurement)
    Enter                       → confirm prompts in the terminal
    Q / ESC                     → quit early

Author : (your name)
Date   : 2026-09-21
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import sys
import signal
import csv
import math
import time
import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sp_stats   # for linregress

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Experiment parameters
TARGET_DISTANCES_CM = [100, 110, 120]   # cm
N_REPEATS           = 3
SAMPLE_DURATION_S   = 5.0               # seconds per measurement
ROI_HALF            = 10                # → 21×21 px ROI

# Streams
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30
COLOR_WIDTH  = 1280
COLOR_HEIGHT = 720
COLOR_FPS    = 30

# Colouriser
COLORIZER_PRESET = 0   # Jet
DEPTH_VIS_MIN_M  = 0.1
DEPTH_VIS_MAX_M  = 4.0

# Output files (same directory as this script)
_HERE        = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(_HERE, "depth_validation_repeated.csv")
PLOT_PATH    = os.path.join(_HERE, "depth_validation_plot.png")

# Warm-up frames
WARMUP_FRAMES = 20

# HUD / display
HUD_FONT   = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE  = 0.52
HUD_THICK  = 1
HUD_AA     = cv2.LINE_AA
HUD_LINE_H = 21

WIN_DEPTH  = "D455f — Repeatability Test"
WIN_COLOR  = "D455f — RGB"

# ===========================================================================
# Graceful Ctrl-C
# ===========================================================================

_shutdown = False

def _sig_handler(sig, frame):
    global _shutdown
    print("\n[INFO] Ctrl-C — aborting ...")
    _shutdown = True

signal.signal(signal.SIGINT, _sig_handler)


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class DepthCameraInfo:
    """All camera parameters, sourced live from the D455f."""
    width:        int
    height:       int
    fx:           float
    fy:           float
    cx:           float
    cy:           float
    depth_scale:  float
    rs_intr:      rs.intrinsics
    serial:       str
    firmware:     str


@dataclass
class FrameStats:
    """Single-frame ROI statistics."""
    median_m:    float   # NaN if no valid pixels
    mean_m:      float
    std_m:       float
    valid_count: int
    total_count: int


@dataclass
class MeasurementRecord:
    """One complete 5-second measurement."""
    timestamp:          str
    target_distance_cm: float
    repeat_number:      int
    roi_cx:             int
    roi_cy:             int
    roi_width:          int
    roi_height:         int
    # Primary estimate — median of per-frame medians
    realsense_median_cm: float
    # Supporting statistics
    realsense_mean_cm:   float
    temporal_std_cm:     float
    minimum_cm:          float
    maximum_cm:          float
    valid_frames:        int
    total_frames:        int


# ===========================================================================
# CSV schema
# ===========================================================================

_CSV_FIELDS = [
    "timestamp", "target_distance_cm", "repeat_number",
    "roi_cx", "roi_cy", "roi_width", "roi_height",
    "realsense_median_cm", "realsense_mean_cm",
    "temporal_std_cm", "minimum_cm", "maximum_cm",
    "valid_frames", "total_frames",
]


# ===========================================================================
# Camera initialisation
# ===========================================================================

def initialize_camera() -> Tuple[rs.pipeline, rs.pipeline_profile]:
    ctx     = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No Intel RealSense device found.\n"
            "  * Plug in the D455f via USB 3.x.\n"
            "  * Close any app that may have opened it (e.g. RealSense Viewer)."
        )

    device = devices[0]
    _print_device_banner(device)

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT,
                      rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT,
                      rs.format.z16, DEPTH_FPS)

    try:
        profile = pipeline.start(cfg)
    except Exception as e:
        raise RuntimeError(f"Pipeline start failed: {e}")

    return pipeline, profile


def _print_device_banner(device: rs.device) -> None:
    print("\n" + "=" * 62)
    print("  DEVICE")
    print("=" * 62)
    for label, fid in [
        ("Name",         rs.camera_info.name),
        ("Serial",       rs.camera_info.serial_number),
        ("Firmware",     rs.camera_info.firmware_version),
        ("Product line", rs.camera_info.product_line),
        ("USB type",     rs.camera_info.usb_type_descriptor),
    ]:
        try:
            if device.supports(fid):
                print(f"  {label:<16}: {device.get_info(fid)}")
        except Exception:
            pass
    print("=" * 62 + "\n")


def get_camera_info(profile: rs.pipeline_profile) -> DepthCameraInfo:
    """Retrieve ALL parameters dynamically — nothing is hard-coded."""
    vsp   = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr  = vsp.get_intrinsics()
    ds    = profile.get_device().first_depth_sensor().get_depth_scale()
    dev   = profile.get_device()

    serial = fw = "unknown"
    try:
        if dev.supports(rs.camera_info.serial_number):
            serial = dev.get_info(rs.camera_info.serial_number)
        if dev.supports(rs.camera_info.firmware_version):
            fw = dev.get_info(rs.camera_info.firmware_version)
    except Exception:
        pass

    return DepthCameraInfo(
        width=intr.width, height=intr.height,
        fx=intr.fx, fy=intr.fy,
        cx=intr.ppx, cy=intr.ppy,
        depth_scale=ds,
        rs_intr=intr,
        serial=serial, firmware=fw
    )


def print_camera_report(ci: DepthCameraInfo) -> None:
    sep = "-" * 62
    fov = rs.rs2_fov(ci.rs_intr)
    print(sep)
    print("  DEPTH INTRINSICS  (live — NOT hard-coded)")
    print(sep)
    print(f"  Resolution  : {ci.width} x {ci.height} px")
    print(f"  fx / fy     : {ci.fx:.4f}  /  {ci.fy:.4f}  px")
    print(f"  cx / cy     : {ci.cx:.4f}  /  {ci.cy:.4f}  px")
    print(f"  FoV         : H={fov[0]:.2f}°  V={fov[1]:.2f}°")
    print(f"  Depth scale : {ci.depth_scale:.10f}  m/unit")
    print(f"  Serial      : {ci.serial}")
    print(f"  Firmware    : {ci.firmware}")
    print(sep + "\n")


# ===========================================================================
# ROI statistics — single frame
# ===========================================================================

def compute_frame_stats(depth_arr: np.ndarray,
                        u: int, v: int, half: int,
                        depth_scale: float) -> FrameStats:
    """
    Extract depth statistics from a square ROI centred at (u, v).

    Returns FrameStats with median_m = NaN if no valid pixels.
    """
    H, W   = depth_arr.shape
    r0, r1 = max(0, v - half), min(H, v + half + 1)
    c0, c1 = max(0, u - half), min(W, u + half + 1)

    patch  = depth_arr[r0:r1, c0:c1].astype(np.float64)
    mask   = patch > 0
    total  = patch.size
    valid  = int(mask.sum())

    if valid == 0:
        nan = float("nan")
        return FrameStats(nan, nan, nan, 0, total)

    vals_m = patch[mask] * depth_scale
    return FrameStats(
        median_m    = float(np.median(vals_m)),
        mean_m      = float(vals_m.mean()),
        std_m       = float(vals_m.std()),
        valid_count = valid,
        total_count = total,
    )


# ===========================================================================
# Temporal sampling — 5 seconds of frames
# ===========================================================================

def collect_measurement(pipeline:    rs.pipeline,
                        align:       rs.align,
                        ci:          DepthCameraInfo,
                        roi_u:       int,
                        roi_v:       int,
                        target_cm:   float,
                        repeat_num:  int) -> MeasurementRecord:
    """
    Collect SAMPLE_DURATION_S seconds of depth frames at the locked ROI.

    The PRIMARY estimate is the MEDIAN OF FRAME MEDIANS — robust to both
    inter-frame noise and within-frame outliers.

    Returns a MeasurementRecord ready for CSV output.
    """
    frame_medians_m: List[float] = []
    t_start = time.monotonic()
    total_frames = 0

    print(f"\n  [COLLECTING] {SAMPLE_DURATION_S:.0f}s  |  "
          f"distance={target_cm} cm  repeat={repeat_num}"
          f"  ROI=({roi_u},{roi_v})")
    print("  Do NOT move the camera or bump the table.\n")

    while time.monotonic() - t_start < SAMPLE_DURATION_S and not _shutdown:
        frameset = pipeline.wait_for_frames()
        aligned  = align.process(frameset)
        df       = aligned.get_depth_frame()
        if not df:
            continue

        total_frames += 1
        arr   = np.asanyarray(df.get_data())
        fstat = compute_frame_stats(arr, roi_u, roi_v, ROI_HALF, ci.depth_scale)

        if not math.isnan(fstat.median_m):
            frame_medians_m.append(fstat.median_m)

        elapsed = time.monotonic() - t_start
        pct     = min(100, int(elapsed / SAMPLE_DURATION_S * 100))
        if len(frame_medians_m) > 0:
            cur = frame_medians_m[-1] * 100
            print(f"\r  Progress: [{pct:3d}%] frames={len(frame_medians_m):4d}"
                  f"  current={cur:.2f} cm    ", end="", flush=True)

    print()  # newline

    if len(frame_medians_m) == 0:
        # All frames had invalid depth — return NaN record
        nan = float("nan")
        return MeasurementRecord(
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            target_distance_cm=target_cm, repeat_number=repeat_num,
            roi_cx=roi_u, roi_cy=roi_v,
            roi_width=2*ROI_HALF+1, roi_height=2*ROI_HALF+1,
            realsense_median_cm=nan, realsense_mean_cm=nan,
            temporal_std_cm=nan, minimum_cm=nan, maximum_cm=nan,
            valid_frames=0, total_frames=total_frames
        )

    arr_m  = np.array(frame_medians_m)
    med_cm = float(np.median(arr_m)) * 100.0
    avg_cm = float(arr_m.mean())     * 100.0
    std_cm = float(arr_m.std())      * 100.0
    min_cm = float(arr_m.min())      * 100.0
    max_cm = float(arr_m.max())      * 100.0

    return MeasurementRecord(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        target_distance_cm=target_cm, repeat_number=repeat_num,
        roi_cx=roi_u, roi_cy=roi_v,
        roi_width=2*ROI_HALF+1, roi_height=2*ROI_HALF+1,
        realsense_median_cm=med_cm,
        realsense_mean_cm=avg_cm,
        temporal_std_cm=std_cm,
        minimum_cm=min_cm,
        maximum_cm=max_cm,
        valid_frames=len(frame_medians_m),
        total_frames=total_frames,
    )


# ===========================================================================
# CSV helpers
# ===========================================================================

def init_csv(path: str) -> None:
    """Write CSV header (always creates/overwrites for a fresh experiment)."""
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()
    print(f"[CSV] Created: {path}")


def append_csv(path: str, rec: MeasurementRecord) -> None:
    row = {
        "timestamp":             rec.timestamp,
        "target_distance_cm":    f"{rec.target_distance_cm:.2f}",
        "repeat_number":         rec.repeat_number,
        "roi_cx":                rec.roi_cx,
        "roi_cy":                rec.roi_cy,
        "roi_width":             rec.roi_width,
        "roi_height":            rec.roi_height,
        "realsense_median_cm":   f"{rec.realsense_median_cm:.4f}",
        "realsense_mean_cm":     f"{rec.realsense_mean_cm:.4f}",
        "temporal_std_cm":       f"{rec.temporal_std_cm:.4f}",
        "minimum_cm":            f"{rec.minimum_cm:.4f}",
        "maximum_cm":            f"{rec.maximum_cm:.4f}",
        "valid_frames":          rec.valid_frames,
        "total_frames":          rec.total_frames,
    }
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ===========================================================================
# Analysis
# ===========================================================================

def analyse(records: List[MeasurementRecord]) -> dict:
    """
    Compute full error analysis over all records.

    Returns a dict with per-distance stats and overall stats.
    """
    valid = [r for r in records if not math.isnan(r.realsense_median_cm)]
    if not valid:
        return {}

    manual  = np.array([r.target_distance_cm   for r in valid])
    rs_meds = np.array([r.realsense_median_cm  for r in valid])

    errors     = rs_meds - manual
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / manual * 100.0

    # Linear regression  RealSense = slope × Manual + intercept
    reg = sp_stats.linregress(manual, rs_meds)

    # Per-distance breakdown
    per_dist = {}
    for dist in TARGET_DISTANCES_CM:
        grp = [r for r in valid if r.target_distance_cm == dist]
        if not grp:
            continue
        rs_g = np.array([r.realsense_median_cm for r in grp])
        er_g = rs_g - dist
        per_dist[dist] = dict(
            rs_values   = rs_g.tolist(),
            mean_rs     = float(rs_g.mean()),
            std_rs      = float(rs_g.std()),
            range_rs    = float(rs_g.max() - rs_g.min()),
            mean_error  = float(er_g.mean()),
            mae         = float(np.abs(er_g).mean()),
        )

    return dict(
        n              = len(valid),
        manual         = manual.tolist(),
        rs_medians     = rs_meds.tolist(),
        errors         = errors.tolist(),
        abs_errors     = abs_errors.tolist(),
        rel_errors_pct = rel_errors.tolist(),
        mean_error     = float(errors.mean()),
        mae            = float(abs_errors.mean()),
        rmse           = float(np.sqrt((errors**2).mean())),
        mean_rel_pct   = float(rel_errors.mean()),
        slope          = float(reg.slope),
        intercept      = float(reg.intercept),
        r_squared      = float(reg.rvalue ** 2),
        per_dist       = per_dist,
    )


def classify_result(ea: dict) -> str:
    """
    Descriptive classification of the measurement quality.
    Returns one of three categories — no unsupported claims about hardware.
    """
    if not ea:
        return "INSUFFICIENT DATA"

    # Repeatability: max range across all distances
    ranges = [v["range_rs"] for v in ea["per_dist"].values()]
    max_range = max(ranges) if ranges else float("inf")

    mean_err  = abs(ea["mean_error"])
    rmse      = ea["rmse"]
    r_sq      = ea["r_squared"]

    if max_range < 0.5 and rmse > 2.0 and r_sq > 0.99:
        return (
            "(A) HIGHLY REPEATABLE — SYSTEMATIC BIAS PRESENT\n"
            "    Measurements are consistent across repeats but offset from\n"
            "    the manual reference.  A calibration correction may be\n"
            "    appropriate once the reference-point offset is characterised."
        )
    elif max_range > 2.0:
        return (
            "(B) NOISY / INCONSISTENT\n"
            "    Repeatability range exceeds 2 cm.  Check: camera stability,\n"
            "    surface material (avoid specular/transparent), IR projector\n"
            "    activity, and environmental IR interference."
        )
    else:
        return (
            "(C) APPROXIMATELY UNBIASED / MIXED\n"
            "    Errors are within normal sensor tolerance.  Further testing\n"
            "    at additional distances is recommended before applying any\n"
            "    correction."
        )


# ===========================================================================
# Report printer
# ===========================================================================

def print_final_report(records: List[MeasurementRecord],
                       ea: dict) -> None:
    sep = "=" * 54
    thin = "-" * 54

    print("\n" + sep)
    print("  D455f DEPTH REPEATABILITY VALIDATION REPORT")
    print(sep)
    print(f"  Total measurements : {ea.get('n', 0)}")
    print(f"  ROI size           : {2*ROI_HALF+1} × {2*ROI_HALF+1} px")
    print(f"  Sample duration    : {SAMPLE_DURATION_S:.1f} s / measurement")

    if not ea:
        print("  No valid data to report.")
        print(sep)
        return

    # --- Per-distance ---
    for dist in TARGET_DISTANCES_CM:
        pd = ea["per_dist"].get(dist)
        if not pd:
            continue
        print()
        print(f"  Distance : {dist} cm")
        print(thin)
        for i, v in enumerate(pd["rs_values"], 1):
            err = v - dist
            print(f"    Repeat {i}  :  RS={v:.2f} cm   error={err:+.2f} cm")
        print(f"    Mean RS  :  {pd['mean_rs']:.2f} cm")
        print(f"    Std      :  {pd['std_rs']:.3f} cm   (repeatability)")
        print(f"    Range    :  {pd['range_rs']:.3f} cm   (max-min)")
        print(f"    Mean err :  {pd['mean_error']:+.2f} cm")
        print(f"    MAE      :  {pd['mae']:.2f} cm")

    # --- Overall ---
    print()
    print(thin)
    print("  OVERALL STATISTICS")
    print(thin)
    print(f"  Mean signed error : {ea['mean_error']:+.3f} cm"
          f"  {'(RS reads HIGH)' if ea['mean_error']>0 else '(RS reads LOW)'}")
    print(f"  Mean abs error    : {ea['mae']:.3f} cm")
    print(f"  RMSE              : {ea['rmse']:.3f} cm")
    print(f"  Mean rel error    : {ea['mean_rel_pct']:.2f} %")
    print()
    print("  LINEAR REGRESSION  (RealSense = slope × Manual + intercept)")
    print(f"  Slope             : {ea['slope']:.6f}")
    print(f"  Intercept         : {ea['intercept']:.4f} cm")
    print(f"  R²                : {ea['r_squared']:.6f}")
    print()
    print("  DIAGNOSTIC CLASSIFICATION:")
    print(f"  {classify_result(ea)}")
    print()
    print("  NOTE: Slope and intercept are for DIAGNOSIS ONLY.")
    print("        No correction has been applied to raw depth values.")
    print(sep)


# ===========================================================================
# Plot
# ===========================================================================

def save_plot(records: List[MeasurementRecord], ea: dict) -> None:
    """
    Scatter plot: RealSense median vs manual distance.
    Each repeat shown individually.  Ideal (y=x) line overlaid.
    Saved as PNG — no GUI window opened.
    """
    if not ea:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {100: "#e74c3c", 110: "#2980b9", 120: "#27ae60"}
    markers_plotted = set()

    for rec in records:
        if math.isnan(rec.realsense_median_cm):
            continue
        dist = int(rec.target_distance_cm)
        col  = colors.get(dist, "#555555")
        label = f"{dist} cm" if dist not in markers_plotted else None
        ax.scatter(rec.target_distance_cm, rec.realsense_median_cm,
                   color=col, s=80, zorder=5, label=label)
        markers_plotted.add(dist)

    # Per-distance mean markers
    for dist, pd in ea["per_dist"].items():
        col = colors.get(dist, "#555555")
        ax.scatter(dist, pd["mean_rs"], color=col, s=200,
                   marker="D", edgecolors="black", linewidths=1.2,
                   zorder=6)

    # Ideal y = x line
    all_vals = [rec.target_distance_cm for rec in records
                if not math.isnan(rec.realsense_median_cm)]
    x_range = [min(all_vals) - 5, max(all_vals) + 5]
    ax.plot(x_range, x_range, "k--", linewidth=1.2, label="Ideal (RS = Manual)")

    # Regression line
    x_fit = np.array(x_range)
    y_fit = ea["slope"] * x_fit + ea["intercept"]
    ax.plot(x_fit, y_fit, "m-.", linewidth=1.2,
            label=f"Fit: slope={ea['slope']:.4f}  b={ea['intercept']:.2f} cm")

    ax.set_xlabel("Manual distance (cm)", fontsize=12)
    ax.set_ylabel("RealSense median depth (cm)", fontsize=12)
    ax.set_title("D455f Depth Repeatability Validation\n"
                 "Circles = individual repeats  |  Diamonds = group mean",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Annotate errors
    for dist, pd in ea["per_dist"].items():
        ax.annotate(f"err={pd['mean_error']:+.1f} cm",
                    xy=(dist, pd["mean_rs"]),
                    xytext=(dist + 1.5, pd["mean_rs"] + 0.5),
                    fontsize=8, color=colors.get(dist, "#333333"))

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"\n[PLOT] Saved: {PLOT_PATH}")


# ===========================================================================
# Display helpers
# ===========================================================================

def _put(img: np.ndarray, row: int, text: str,
         color=(0, 255, 0)) -> None:
    cv2.putText(img, text, (10, 28 + row * HUD_LINE_H),
                HUD_FONT, HUD_SCALE, color, HUD_THICK, HUD_AA)


def draw_roi(img: np.ndarray, u: int, v: int, half: int,
             locked: bool) -> None:
    H, W = img.shape[:2]
    x0 = max(0, u - half);  x1 = min(W-1, u + half)
    y0 = max(0, v - half);  y1 = min(H-1, v + half)
    color = (0, 255, 255) if locked else (0, 180, 255)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2 if locked else 1,
                  cv2.LINE_AA)
    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)


def draw_roi_hud(img: np.ndarray, fstat: FrameStats,
                 roi_u: int, roi_v: int,
                 locked: bool, n_done: int,
                 n_total: int, sampling: bool) -> None:
    """Render HUD on the depth window."""
    row = 0
    lock_str = "LOCKED" if locked else "click to LOCK"
    _put(img, row, f"ROI: ({roi_u},{roi_v})  21x21  [{lock_str}]",
         (0, 255, 0) if locked else (0, 180, 255)); row += 1

    if math.isnan(fstat.median_m):
        _put(img, row, "Depth UNAVAILABLE in ROI", (0, 60, 255)); row += 1
    else:
        _put(img, row, f"Median Z : {fstat.median_m*100:.2f} cm"); row += 1
        _put(img, row, f"Mean Z   : {fstat.mean_m*100:.2f} cm"); row += 1
        _put(img, row, f"Std      : {fstat.std_m*100:.3f} cm"); row += 1
        _put(img, row, f"Valid px : {fstat.valid_count}/{fstat.total_count}"); row += 1

    _put(img, row, f"Measurements done: {n_done} / {n_total}",
         (200, 200, 0)); row += 1

    if sampling:
        _put(img, row, "** SAMPLING — hold still **", (0, 80, 255)); row += 1

    h = img.shape[0]
    cv2.putText(img,
                "Click=lock ROI  |  Q/ESC=quit",
                (10, h - 10), HUD_FONT, 0.44, (160, 160, 160), 1, HUD_AA)


# ===========================================================================
# ROI selection phase
# ===========================================================================

class _RoiSelect:
    """Mutable state for the ROI selection mouse callback."""
    def __init__(self, u: int, v: int):
        self.u      = u
        self.v      = v
        self.locked = False

    def callback(self, event, x, y, flags, param):
        if self.locked:
            return
        self.u = x
        self.v = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.locked = True
            print(f"\n[ROI] Locked at pixel ({self.u}, {self.v})  "
                  f"size = {2*ROI_HALF+1}x{2*ROI_HALF+1} px")


def wait_for_roi_lock(pipeline: rs.pipeline, align: rs.align,
                      ci: DepthCameraInfo,
                      colorizer: rs.colorizer) -> Tuple[int, int]:
    """
    Show a live depth stream.  User moves the mouse to the desired ROI centre
    and left-clicks ONCE to lock it.  After locking, returns (u, v).

    The ROI cannot be changed without restarting the program.
    """
    print("\n" + "=" * 62)
    print("  ROI SELECTION")
    print("=" * 62)
    print("  Move the mouse over the depth window to position the ROI.")
    print("  Point it at the flat target surface.")
    print("  LEFT-CLICK to lock the ROI for all 9 measurements.")
    print("=" * 62 + "\n")

    sel = _RoiSelect(ci.width // 2, ci.height // 2)

    cv2.namedWindow(WIN_DEPTH, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_COLOR, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_DEPTH, sel.callback)

    while not sel.locked and not _shutdown:
        frameset = pipeline.wait_for_frames()
        aligned  = align.process(frameset)

        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            continue

        arr    = np.asanyarray(df.get_data())
        fstat  = compute_frame_stats(arr, sel.u, sel.v, ROI_HALF, ci.depth_scale)

        depth_disp = np.asanyarray(colorizer.colorize(df).get_data())
        color_disp = np.asanyarray(cf.get_data())

        draw_roi(depth_disp, sel.u, sel.v, ROI_HALF, locked=False)
        draw_roi(color_disp, sel.u, sel.v, ROI_HALF, locked=False)
        draw_roi_hud(depth_disp, fstat, sel.u, sel.v,
                     locked=False, n_done=0, n_total=9, sampling=False)

        cv2.imshow(WIN_DEPTH, depth_disp)
        cv2.imshow(WIN_COLOR, color_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            raise RuntimeError("User quit during ROI selection.")

    if _shutdown:
        raise RuntimeError("Interrupted during ROI selection.")

    return sel.u, sel.v


# ===========================================================================
# Live preview between measurements
# ===========================================================================

def show_live_preview(pipeline: rs.pipeline, align: rs.align,
                      ci: DepthCameraInfo, colorizer: rs.colorizer,
                      roi_u: int, roi_v: int,
                      n_done: int, n_total: int,
                      prompt_msg: str) -> bool:
    """
    Display a live feed with the locked ROI while waiting for the user
    to press ENTER in the terminal.

    Returns False if the user presses Q/ESC (abort).
    """
    import threading

    entered = threading.Event()
    aborted = threading.Event()

    def _terminal_input():
        print(f"\n{prompt_msg}")
        print("  Press ENTER when ready (or type 'q' to quit): ", end="",
              flush=True)
        val = input().strip().lower()
        if val == 'q':
            aborted.set()
        entered.set()

    t = threading.Thread(target=_terminal_input, daemon=True)
    t.start()

    while not entered.is_set() and not _shutdown:
        frameset = pipeline.wait_for_frames()
        aligned  = align.process(frameset)

        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            continue

        arr    = np.asanyarray(df.get_data())
        fstat  = compute_frame_stats(arr, roi_u, roi_v, ROI_HALF, ci.depth_scale)

        depth_disp = np.asanyarray(colorizer.colorize(df).get_data())
        color_disp = np.asanyarray(cf.get_data())

        draw_roi(depth_disp, roi_u, roi_v, ROI_HALF, locked=True)
        draw_roi(color_disp, roi_u, roi_v, ROI_HALF, locked=True)
        draw_roi_hud(depth_disp, fstat, roi_u, roi_v,
                     locked=True, n_done=n_done, n_total=n_total,
                     sampling=False)

        cv2.imshow(WIN_DEPTH, depth_disp)
        cv2.imshow(WIN_COLOR, color_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            aborted.set()
            entered.set()
            break

    return not aborted.is_set() and not _shutdown


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_experiment() -> None:
    global _shutdown

    # --- Camera setup -------------------------------------------------------
    pipeline, profile = initialize_camera()
    ci                = get_camera_info(profile)
    print_camera_report(ci)

    # --- Post-processing objects -------------------------------------------
    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, COLORIZER_PRESET)
    colorizer.set_option(rs.option.min_distance,  DEPTH_VIS_MIN_M)
    colorizer.set_option(rs.option.max_distance,  DEPTH_VIS_MAX_M)

    # --- Warm-up ------------------------------------------------------------
    print(f"[INFO] Warming up ({WARMUP_FRAMES} frames) ...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()
    print("[INFO] Ready.\n")

    # --- CSV initialisation -------------------------------------------------
    init_csv(CSV_PATH)

    records: List[MeasurementRecord] = []

    try:
        # --- ROI selection (ONCE) ------------------------------------------
        roi_u, roi_v = wait_for_roi_lock(pipeline, align, ci, colorizer)

        print(f"\n[ROI] PERMANENTLY LOCKED: centre=({roi_u},{roi_v})  "
              f"size={2*ROI_HALF+1}x{2*ROI_HALF+1} px")
        print("[ROI] The ROI will NOT change for any of the 9 measurements.\n")

        # --- Measurement schedule ------------------------------------------
        # Build ordered list: [(distance, repeat), ...]
        schedule = [
            (dist, rep)
            for dist in TARGET_DISTANCES_CM
            for rep  in range(1, N_REPEATS + 1)
        ]
        n_total = len(schedule)

        print("=" * 54)
        print("  EXPERIMENT SCHEDULE")
        print("=" * 54)
        for i, (d, r) in enumerate(schedule, 1):
            print(f"  {i:2d}. distance={d} cm   repeat={r}")
        print("=" * 54)
        print()
        print("  INSTRUCTIONS:")
        print("  * Keep the D455f completely stationary throughout.")
        print("  * Only move the flat TARGET SURFACE to each distance.")
        print("  * Measure distance from the SAME reference point every time.")
        print("  * When ready for each measurement, press ENTER.")
        print()

        for idx, (target_cm, repeat_num) in enumerate(schedule):
            if _shutdown:
                break

            n_done = idx

            # Prompt user in terminal, live preview in window
            prompt = (
                f"  ── Measurement {idx+1}/{n_total} ──\n"
                f"  Target distance : {target_cm} cm\n"
                f"  Repeat          : {repeat_num} / {N_REPEATS}\n"
                f"  → Move flat surface to {target_cm} cm from camera.\n"
                f"  → Verify the surface is visible in the ROI box."
            )

            ok = show_live_preview(pipeline, align, ci, colorizer,
                                   roi_u, roi_v,
                                   n_done, n_total, prompt)
            if not ok:
                print("[INFO] Experiment aborted by user.")
                break

            # --- 5-second measurement -------------------------------------
            rec = collect_measurement(
                pipeline, align, ci, roi_u, roi_v, target_cm, repeat_num)

            records.append(rec)
            append_csv(CSV_PATH, rec)

            if not math.isnan(rec.realsense_median_cm):
                err = rec.realsense_median_cm - target_cm
                print(f"  ✓  RS median = {rec.realsense_median_cm:.2f} cm   "
                      f"error = {err:+.2f} cm   "
                      f"std = {rec.temporal_std_cm:.3f} cm")
            else:
                print("  ✗  No valid depth in ROI — check surface/IR projector.")

    except RuntimeError as e:
        print(f"\n[ABORT] {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # --- Analysis and report -----------------------------------------------
    if records:
        ea = analyse(records)
        print_final_report(records, ea)
        save_plot(records, ea)
        print(f"\n[CSV]  Results : {CSV_PATH}")
        print(f"[PLOT] Chart   : {PLOT_PATH}")
    else:
        print("[INFO] No measurements recorded — no report generated.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("\n" + "#" * 62)
    print("  RealSense D455f — Depth Repeatability Validation")
    print("  3 distances × 3 repeats = 9 measurements")
    print("#" * 62 + "\n")

    try:
        run_experiment()
    except Exception as err:
        print(f"\n[FATAL] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
