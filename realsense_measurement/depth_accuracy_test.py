"""
depth_accuracy_test.py
======================
Stage 3 — RealSense D455f Depth Accuracy Characterisation

Purpose:
    Diagnose and quantify the systematic and random depth errors of the D455f
    before using depth measurements as the basis for object dimension estimates.

    No object detection, segmentation, or dimension measurement is performed.
    This stage exists purely to characterise depth accuracy across multiple
    real-world distances.

What this script does:
    1.  Connects to the D455f and retrieves intrinsics/depth-scale dynamically.
    2.  Streams aligned RGB + depth frames in a live display window.
    3.  Lets the user drag a configurable ROI (default 21×21 px) over the
        depth image with the mouse.
    4.  Computes real-time depth statistics over the ROI:
            mean, median, std-dev, min, max, valid-pixel count.
    5.  Optionally collects N seconds of temporal samples at the current
        ROI and reports frame-to-frame stability.
    6.  Prompts for the manually measured reference distance and records
        everything to  depth_accuracy_results.csv.
    7.  On exit, prints a final calibration summary:
            mean error, MAE, RMSE, relative error, linear fit.

Keyboard controls (focus the depth window):
    S  →  start a temporal sample collection (≈5 s)
    R  →  record the current ROI statistics (prompts for manual distance)
    Q / ESC  →  quit

Mouse controls:
    Move  →  move ROI centre
    Click →  lock/unlock the ROI centre

Author : (your name)
Date   : 2026-09-21

IMPORTANT — coordinate system note
-------------------------------------
    The D455f reports distance along the Z axis of the DEPTH SENSOR optical
    coordinate system.  The physical origin of this Z axis is the depth sensor
    optical centre — NOT the front face of the camera housing, NOT the USB
    connector side.  See README.md for how to define a repeatable measurement
    reference point.
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
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import pyrealsense2 as rs
import numpy as np
import cv2

# ===========================================================================
# CONFIGURATION  (adjust as needed)
# ===========================================================================

# Depth stream
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30

# RGB stream (used for the colour overlay window)
COLOR_WIDTH  = 1280
COLOR_HEIGHT = 720
COLOR_FPS    = 30

# ROI half-size (pixels).  Full ROI = (2*HALF+1) × (2*HALF+1)
ROI_HALF = 10   # → 21×21 default

# Temporal sample collection duration (seconds)
SAMPLE_DURATION_S = 5.0

# Warm-up frames
WARMUP_FRAMES = 20

# CSV output path (relative to script location)
CSV_FILENAME = "depth_accuracy_results.csv"

# Colouriser settings
COLORIZER_PRESET = 0   # Jet
DEPTH_VIS_MIN_M  = 0.1
DEPTH_VIS_MAX_M  = 4.0

# HUD
HUD_FONT    = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE   = 0.52
HUD_THICK   = 1
HUD_AA      = cv2.LINE_AA
HUD_LINE_H  = 22

WIN_DEPTH = "D455f — Depth Accuracy Test"
WIN_COLOR = "D455f — RGB (aligned)"

# ===========================================================================
# Graceful Ctrl-C
# ===========================================================================

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    print("\n[INFO] Ctrl-C received — shutting down ...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _signal_handler)


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class DepthIntrinsics:
    width:   int
    height:  int
    fx:      float
    fy:      float
    cx:      float
    cy:      float
    rs_intr: rs.intrinsics


@dataclass
class ROIStats:
    """Statistics for one ROI sample (single frame or temporal average)."""
    mean_m:       float
    median_m:     float
    std_m:        float
    min_m:        float
    max_m:        float
    valid_count:  int
    total_count:  int

    @property
    def mean_cm(self)   -> float: return self.mean_m   * 100.0
    @property
    def median_cm(self) -> float: return self.median_m * 100.0
    @property
    def std_cm(self)    -> float: return self.std_m    * 100.0
    @property
    def min_cm(self)    -> float: return self.min_m    * 100.0
    @property
    def max_cm(self)    -> float: return self.max_m    * 100.0
    @property
    def valid_ratio(self) -> float:
        return self.valid_count / self.total_count if self.total_count > 0 else 0.0


@dataclass
class TemporalStats:
    """Statistics collected across multiple frames at a fixed ROI."""
    frame_medians_m: List[float]       # one median per frame
    frame_count:     int

    @property
    def mean_m(self)  -> float: return float(np.mean(self.frame_medians_m))
    @property
    def std_m(self)   -> float: return float(np.std(self.frame_medians_m))
    @property
    def min_m(self)   -> float: return float(np.min(self.frame_medians_m))
    @property
    def max_m(self)   -> float: return float(np.max(self.frame_medians_m))
    @property
    def mean_cm(self) -> float: return self.mean_m * 100.0
    @property
    def std_cm(self)  -> float: return self.std_m  * 100.0
    @property
    def min_cm(self)  -> float: return self.min_m  * 100.0
    @property
    def max_cm(self)  -> float: return self.max_m  * 100.0


@dataclass
class TestRecord:
    """One row in the CSV output."""
    timestamp:             str
    manual_distance_cm:    float
    realsense_mean_cm:     float
    realsense_median_cm:   float
    std_cm:                float
    min_cm:                float
    max_cm:                float
    valid_pixel_count:     int
    total_pixel_count:     int
    temporal_frames:       int
    temporal_std_cm:       float   # std of per-frame medians
    depth_fill_rate:       float   # from frame metadata (0–1) or -1 if unavailable
    depth_stdev_meta:      float   # from frame metadata or -1 if unavailable
    roi_half_px:           int
    depth_scale:           float
    notes:                 str = ""


# ===========================================================================
# Camera initialisation
# ===========================================================================

def initialize_camera() -> Tuple[rs.pipeline, rs.pipeline_profile]:
    """Detect D455f, configure and start depth + colour pipeline."""
    ctx     = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No Intel RealSense device detected.\n"
            "  * Ensure D455f is connected via USB 3.x.\n"
            "  * Close other apps that may hold the camera (RealSense Viewer, etc.)."
        )

    device = devices[0]
    _print_device_info(device)

    pipeline = rs.pipeline()
    cfg      = rs.config()

    cfg.enable_stream(rs.stream.color, COLOR_WIDTH,  COLOR_HEIGHT,
                      rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH,  DEPTH_HEIGHT,
                      rs.format.z16,  DEPTH_FPS)

    try:
        profile = pipeline.start(cfg)
    except Exception as exc:
        raise RuntimeError(f"Pipeline start failed:\n  {exc}")

    return pipeline, profile


def _print_device_info(device: rs.device) -> None:
    print("\n" + "=" * 60)
    print("  CONNECTED DEVICE")
    print("=" * 60)
    for label, field_id in [
        ("Name",         rs.camera_info.name),
        ("Serial",       rs.camera_info.serial_number),
        ("Firmware",     rs.camera_info.firmware_version),
        ("Product line", rs.camera_info.product_line),
        ("USB type",     rs.camera_info.usb_type_descriptor),
    ]:
        try:
            if device.supports(field_id):
                print(f"  {label:<16}: {device.get_info(field_id)}")
        except Exception:
            pass
    print("=" * 60 + "\n")


# ===========================================================================
# Intrinsics and depth scale
# ===========================================================================

def get_depth_intrinsics(profile: rs.pipeline_profile) -> DepthIntrinsics:
    """Retrieve depth intrinsics DYNAMICALLY from the active profile."""
    vsp  = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = vsp.get_intrinsics()
    return DepthIntrinsics(
        width=intr.width, height=intr.height,
        fx=intr.fx, fy=intr.fy,
        cx=intr.ppx, cy=intr.ppy,
        rs_intr=intr
    )


def get_depth_scale(profile: rs.pipeline_profile) -> float:
    """Return metres per raw depth unit."""
    return profile.get_device().first_depth_sensor().get_depth_scale()


def print_setup_report(di: DepthIntrinsics, depth_scale: float) -> None:
    sep = "-" * 62
    fov = rs.rs2_fov(di.rs_intr)
    print(sep)
    print("  DEPTH INTRINSICS  (live from camera — never hard-coded)")
    print(sep)
    print(f"  Resolution   : {di.width} x {di.height} px")
    print(f"  Focal length : fx={di.fx:.4f}  fy={di.fy:.4f}  [px]")
    print(f"  Principal pt : cx={di.cx:.4f}  cy={di.cy:.4f}  [px]")
    print(f"  FoV          : H={fov[0]:.2f}°  V={fov[1]:.2f}°")
    print(f"  Depth scale  : {depth_scale:.10f} m/unit")
    print(f"  Stereo base  : see sensor options")
    print(sep + "\n")

    print(sep)
    print("  DEPTH QUALITY NOTE")
    print(sep)
    print("  Depth fill rate and per-frame std-dev will be read from")
    print("  frame metadata (frame_metadata_value.depth_fill_rate and")
    print("  frame_metadata_value.depth_stdev) if supported by this")
    print("  firmware.  A value of -1 in the CSV means unavailable.")
    print(sep + "\n")


# ===========================================================================
# ROI sampling
# ===========================================================================

def compute_roi_stats(depth_arr: np.ndarray,
                      u: int, v: int,
                      half: int,
                      depth_scale: float) -> ROIStats:
    """
    Extract depth statistics from a square ROI centred at (u, v).

    Parameters
    ----------
    depth_arr   : uint16 numpy array, shape (H, W)
    u, v        : ROI centre (column, row)
    half        : half-side length in pixels
    depth_scale : metres per raw depth unit

    Returns
    -------
    ROIStats with all values in metres.

    Why ROI instead of a single pixel?
    ------------------------------------
    A single depth pixel has high variance due to stereo matching noise,
    IR speckle, and quantisation.  Averaging/median over a small patch
    dramatically reduces this noise without introducing systematic bias
    (assuming the surface is flat and approximately parallel to the sensor).
    """
    H, W = depth_arr.shape
    r0 = max(0, v - half);  r1 = min(H, v + half + 1)
    c0 = max(0, u - half);  c1 = min(W, u + half + 1)

    patch      = depth_arr[r0:r1, c0:c1].astype(np.float64)
    valid_mask = patch > 0
    total      = patch.size
    valid      = int(valid_mask.sum())

    if valid == 0:
        nan = float("nan")
        return ROIStats(nan, nan, nan, nan, nan, 0, total)

    vals_m = patch[valid_mask] * depth_scale

    return ROIStats(
        mean_m      = float(vals_m.mean()),
        median_m    = float(np.median(vals_m)),
        std_m       = float(vals_m.std()),
        min_m       = float(vals_m.min()),
        max_m       = float(vals_m.max()),
        valid_count = valid,
        total_count = total
    )


# ===========================================================================
# Frame metadata helpers (depth quality)
# ===========================================================================

def _try_get_metadata(frame: rs.frame,
                      key: rs.frame_metadata_value) -> float:
    """
    Safely read a frame metadata field.

    Returns the value as float if supported, otherwise -1.0.
    This is the correct way to query metadata — the API exists in pyrealsense2
    v2.54+ via frame.supports_frame_metadata() and frame.get_frame_metadata().
    """
    try:
        if frame.supports_frame_metadata(key):
            return float(frame.get_frame_metadata(key))
    except Exception:
        pass
    return -1.0


def get_depth_quality_metadata(depth_frame: rs.depth_frame) -> Tuple[float, float]:
    """
    Read depth_fill_rate and depth_stdev from frame metadata.

    depth_fill_rate : fraction of valid pixels in the full depth frame (0–1).
                      Values close to 1.0 indicate clean depth data.
    depth_stdev     : per-frame depth standard deviation reported by firmware.

    Both return -1.0 if the current firmware does not expose them.
    """
    fill_rate = _try_get_metadata(depth_frame,
                                  rs.frame_metadata_value.depth_fill_rate)
    stdev     = _try_get_metadata(depth_frame,
                                  rs.frame_metadata_value.depth_stdev)
    return fill_rate, stdev


# ===========================================================================
# Temporal sampling (multi-frame collection)
# ===========================================================================

def collect_temporal_samples(pipeline:    rs.pipeline,
                             align:       rs.align,
                             di:          DepthIntrinsics,
                             depth_scale: float,
                             roi_u:       int,
                             roi_v:       int,
                             roi_half:    int,
                             duration_s:  float) -> TemporalStats:
    """
    Collect depth ROI medians over `duration_s` seconds at a fixed position.

    This answers: "How stable is the D455f depth value on a stationary surface?"

    Parameters
    ----------
    pipeline, align : active pipeline + aligner
    di              : DepthIntrinsics
    depth_scale     : float
    roi_u, roi_v    : locked ROI centre (pixels)
    roi_half        : ROI half-size
    duration_s      : collection window (seconds)

    Returns
    -------
    TemporalStats containing one median per acquired frame.
    """
    medians: List[float] = []
    t_start = time.monotonic()

    print(f"\n[SAMPLE] Collecting {duration_s:.1f}s of temporal samples ...")
    print(f"         ROI centre: ({roi_u}, {roi_v})  half={roi_half} px")
    print("         Do NOT move the camera.")

    while time.monotonic() - t_start < duration_s:
        frameset = pipeline.wait_for_frames()
        aligned  = align.process(frameset)
        df       = aligned.get_depth_frame()
        if not df:
            continue

        arr   = np.asanyarray(df.get_data())
        stats = compute_roi_stats(arr, roi_u, roi_v, roi_half, depth_scale)

        if not math.isnan(stats.median_m):
            medians.append(stats.median_m)

        elapsed = time.monotonic() - t_start
        pct     = min(100, int(elapsed / duration_s * 100))
        print(f"\r         Progress: {pct:3d}%  frames={len(medians):4d}"
              f"  current median={stats.median_cm:.2f} cm      ",
              end="", flush=True)

    print()  # newline after progress
    return TemporalStats(frame_medians_m=medians, frame_count=len(medians))


# ===========================================================================
# CSV logging
# ===========================================================================

_CSV_FIELDS = [
    "timestamp", "manual_distance_cm",
    "realsense_mean_cm", "realsense_median_cm",
    "std_cm", "min_cm", "max_cm",
    "valid_pixel_count", "total_pixel_count",
    "temporal_frames", "temporal_std_cm",
    "depth_fill_rate", "depth_stdev_meta",
    "roi_half_px", "depth_scale", "notes"
]


def init_csv(path: str) -> bool:
    """Create CSV with header row if it doesn't already exist."""
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not exists:
            writer.writeheader()
    return exists


def append_csv(path: str, record: TestRecord) -> None:
    """Append one TestRecord to the CSV file."""
    row = {
        "timestamp":             record.timestamp,
        "manual_distance_cm":    f"{record.manual_distance_cm:.2f}",
        "realsense_mean_cm":     f"{record.realsense_mean_cm:.4f}",
        "realsense_median_cm":   f"{record.realsense_median_cm:.4f}",
        "std_cm":                f"{record.std_cm:.4f}",
        "min_cm":                f"{record.min_cm:.4f}",
        "max_cm":                f"{record.max_cm:.4f}",
        "valid_pixel_count":     record.valid_pixel_count,
        "total_pixel_count":     record.total_pixel_count,
        "temporal_frames":       record.temporal_frames,
        "temporal_std_cm":       f"{record.temporal_std_cm:.4f}",
        "depth_fill_rate":       f"{record.depth_fill_rate:.4f}",
        "depth_stdev_meta":      f"{record.depth_stdev_meta:.4f}",
        "roi_half_px":           record.roi_half_px,
        "depth_scale":           f"{record.depth_scale:.10f}",
        "notes":                 record.notes,
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writerow(row)


# ===========================================================================
# Error analysis
# ===========================================================================

def compute_error_analysis(records: List[TestRecord]) -> dict:
    """
    Compute calibration error metrics from all recorded test points.

    Returns a dict with:
        n               : number of test points
        errors_cm       : signed errors (RealSense - manual)
        abs_errors_cm   : absolute errors
        rel_errors_pct  : relative errors
        mean_error_cm   : mean signed error  (bias)
        mae_cm          : mean absolute error
        rmse_cm         : root mean square error
        mean_rel_pct    : mean relative error
        slope, intercept: linear fit  manual = slope × RealSense + intercept
                          (only if n >= 2)
    """
    if len(records) == 0:
        return {}

    manual   = np.array([r.manual_distance_cm   for r in records])
    rs_vals  = np.array([r.realsense_median_cm  for r in records])

    errors      = rs_vals - manual
    abs_errors  = np.abs(errors)
    rel_errors  = abs_errors / manual * 100.0

    result = dict(
        n             = len(records),
        errors_cm     = errors.tolist(),
        abs_errors_cm = abs_errors.tolist(),
        rel_errors_pct= rel_errors.tolist(),
        mean_error_cm = float(errors.mean()),
        mae_cm        = float(abs_errors.mean()),
        rmse_cm       = float(np.sqrt((errors ** 2).mean())),
        mean_rel_pct  = float(rel_errors.mean()),
        slope         = None,
        intercept     = None,
    )

    # Linear fit: manual = slope × realsense + intercept
    # This models: true_distance = a × sensor_reading + b
    if len(records) >= 2:
        coeffs             = np.polyfit(rs_vals, manual, deg=1)
        result["slope"]     = float(coeffs[0])
        result["intercept"] = float(coeffs[1])

    return result


def print_error_table(records: List[TestRecord]) -> None:
    """Print per-point error table to terminal."""
    if not records:
        return
    print("\n" + "=" * 78)
    print("  PER-POINT ERROR TABLE")
    print("=" * 78)
    hdr = (f"{'Manual':>10} {'RS Median':>12} {'Error':>10} "
           f"{'Abs Err':>10} {'Rel %':>8} {'Std':>8} {'Valid%':>8}")
    print(hdr)
    print("-" * 78)
    for r in records:
        err = r.realsense_median_cm - r.manual_distance_cm
        rel = abs(err) / r.manual_distance_cm * 100.0
        pct = r.valid_pixel_count / r.total_pixel_count * 100.0
        print(f"  {r.manual_distance_cm:>8.2f} cm"
              f"  {r.realsense_median_cm:>10.2f} cm"
              f"  {err:>+9.2f} cm"
              f"  {abs(err):>8.2f} cm"
              f"  {rel:>7.2f}%"
              f"  {r.std_cm:>6.3f} cm"
              f"  {pct:>6.1f}%")
    print("=" * 78)


def print_final_summary(records: List[TestRecord]) -> None:
    """Print overall calibration summary to terminal."""
    ea = compute_error_analysis(records)
    if not ea:
        print("\n[INFO] No test records collected — no summary to display.")
        return

    print("\n" + "#" * 62)
    print("  FINAL DEPTH ACCURACY SUMMARY")
    print("#" * 62)
    print(f"  Test points collected : {ea['n']}")
    print(f"  Mean error (bias)     : {ea['mean_error_cm']:+.3f} cm")
    print(f"    (negative = RealSense reads LESS than actual)")
    print(f"  Mean absolute error   : {ea['mae_cm']:.3f} cm")
    print(f"  RMSE                  : {ea['rmse_cm']:.3f} cm")
    print(f"  Mean relative error   : {ea['mean_rel_pct']:.2f}%")
    if ea["slope"] is not None:
        print(f"  Linear fit (manual = slope × RS + intercept):")
        print(f"    slope     = {ea['slope']:.6f}")
        print(f"    intercept = {ea['intercept']:.4f} cm")
        print(f"  NOTE: These coefficients are for DIAGNOSTIC purposes only.")
        print(f"        Do NOT apply them as a correction without further analysis.")
    print("#" * 62)
    print()

    print_error_table(records)


# ===========================================================================
# Mouse state
# ===========================================================================

class RoiState:
    """Mutable ROI centre updated by the mouse callback."""
    def __init__(self, u: int, v: int):
        self.u      = u
        self.v      = v
        self.locked = False   # True → mouse clicks move the ROI, not hover


def make_mouse_callback(state: RoiState):
    def callback(event, x, y, flags, param):
        if not state.locked:
            state.u = x
            state.v = y
        if event == cv2.EVENT_LBUTTONDOWN:
            state.locked = not state.locked
    return callback


# ===========================================================================
# HUD / overlay rendering
# ===========================================================================

def _put(img: np.ndarray, row: int, text: str, color=(0, 255, 0)) -> None:
    cv2.putText(img, text,
                (10, 28 + row * HUD_LINE_H),
                HUD_FONT, HUD_SCALE, color, HUD_THICK, HUD_AA)


def draw_roi_rect(img: np.ndarray, u: int, v: int, half: int,
                  color=(0, 255, 255)) -> None:
    """Draw the ROI bounding rectangle."""
    H, W = img.shape[:2]
    x0 = max(0, u - half);  x1 = min(W - 1, u + half)
    y0 = max(0, v - half);  y1 = min(H - 1, v + half)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, stats: ROIStats,
             roi: RoiState, fill_rate: float, locked: bool,
             sampling: bool, n_records: int) -> None:
    """Render real-time statistics HUD."""
    roi_size = 2 * ROI_HALF + 1
    lock_str = "LOCKED" if locked else "floating"

    row = 0
    _put(img, row, f"ROI centre: ({roi.u}, {roi.v})  [{lock_str}]  "
                   f"size={roi_size}x{roi_size}"); row += 1

    if math.isnan(stats.mean_m):
        _put(img, row, "Depth UNAVAILABLE in ROI", (0, 60, 255)); row += 1
    else:
        _put(img, row, f"Mean Z   : {stats.mean_cm:.2f} cm"); row += 1
        _put(img, row, f"Median Z : {stats.median_cm:.2f} cm"); row += 1
        _put(img, row, f"Std Dev  : {stats.std_cm:.3f} cm"); row += 1
        _put(img, row, f"Min/Max  : {stats.min_cm:.2f} / {stats.max_cm:.2f} cm"); row += 1
        _put(img, row, f"Valid px : {stats.valid_count} / {stats.total_count}"
                       f"  ({stats.valid_ratio*100:.1f}%)"); row += 1

    if fill_rate >= 0:
        _put(img, row, f"Frame fill rate : {fill_rate*100:.1f}%",
             (180, 180, 0)); row += 1

    if sampling:
        _put(img, row, "** SAMPLING — do not move camera **",
             (0, 100, 255)); row += 1

    # Bottom line
    h = img.shape[0]
    tips = ("S=sample  R=record  Click=lock/unlock ROI  Q/ESC=quit"
            f"  Records: {n_records}")
    cv2.putText(img, tips, (10, h - 10),
                HUD_FONT, 0.44, (170, 170, 170), 1, HUD_AA)


# ===========================================================================
# Main loop
# ===========================================================================

def run_accuracy_test() -> None:
    global _shutdown_requested

    # --- Init camera -------------------------------------------------------
    pipeline, profile = initialize_camera()
    di          = get_depth_intrinsics(profile)
    depth_scale = get_depth_scale(profile)
    print_setup_report(di, depth_scale)

    # --- CSV ---------------------------------------------------------------
    csv_path    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               CSV_FILENAME)
    csv_existed = init_csv(csv_path)
    print(f"[CSV] {'Appending to' if csv_existed else 'Created'}: {csv_path}\n")

    # --- Processing helpers ------------------------------------------------
    align      = rs.align(rs.stream.color)
    colorizer  = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, COLORIZER_PRESET)
    colorizer.set_option(rs.option.min_distance,  DEPTH_VIS_MIN_M)
    colorizer.set_option(rs.option.max_distance,  DEPTH_VIS_MAX_M)

    # --- Warm-up -----------------------------------------------------------
    print(f"[INFO] Warming up ({WARMUP_FRAMES} frames) ...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()
    print("[INFO] Ready.\n")
    print("  Controls:")
    print("    Move mouse  → move ROI  |  Left-click  → lock/unlock ROI")
    print("    S           → run temporal sample collection (~5 s)")
    print("    R           → record current ROI stats (prompts for distance)")
    print("    Q / ESC     → quit and print summary\n")

    # --- State -------------------------------------------------------------
    roi    = RoiState(di.width // 2, di.height // 2)
    records: List[TestRecord] = []

    # Flags driven by key presses
    do_sample  = False
    do_record  = False
    is_sampling = False

    # Last depth frame reference for temporal collection
    last_depth_arr: Optional[np.ndarray] = None
    last_fill_rate = -1.0
    last_stdev_meta = -1.0

    # Windows and mouse callback
    cv2.namedWindow(WIN_DEPTH, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_COLOR, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_DEPTH, make_mouse_callback(roi))

    try:
        while not _shutdown_requested:
            # --- Acquire frames -------------------------------------------
            frameset = pipeline.wait_for_frames()
            aligned  = align.process(frameset)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_arr = np.asanyarray(depth_frame.get_data())
            last_depth_arr = depth_arr

            last_fill_rate,  last_stdev_meta = get_depth_quality_metadata(
                depth_frame)

            # --- ROI stats (current frame) --------------------------------
            u = np.clip(roi.u, ROI_HALF, di.width  - ROI_HALF - 1)
            v = np.clip(roi.v, ROI_HALF, di.height - ROI_HALF - 1)
            stats = compute_roi_stats(depth_arr, u, v, ROI_HALF, depth_scale)

            # --- Build display frames ------------------------------------
            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())
            color_image    = np.asanyarray(color_frame.get_data())

            depth_disp = depth_colormap.copy()
            color_disp = color_image.copy()

            draw_roi_rect(depth_disp, u, v, ROI_HALF)
            draw_roi_rect(color_disp, u, v, ROI_HALF)
            draw_hud(depth_disp, stats, roi, last_fill_rate,
                     roi.locked, is_sampling, len(records))

            cv2.imshow(WIN_DEPTH, depth_disp)
            cv2.imshow(WIN_COLOR, color_disp)

            # --- Key handling --------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Quit key pressed.")
                break

            elif key in (ord('s'), ord('S')):
                do_sample = True

            elif key in (ord('r'), ord('R')):
                do_record = True

            # --- Temporal sampling (blocking within the loop) -----------
            if do_sample and not is_sampling:
                do_sample   = False
                is_sampling = True

                t_stats = collect_temporal_samples(
                    pipeline, align, di, depth_scale,
                    u, v, ROI_HALF, SAMPLE_DURATION_S)

                is_sampling = False

                print(f"\n[SAMPLE] Collected {t_stats.frame_count} frames")
                print(f"         Mean of medians : {t_stats.mean_cm:.2f} cm")
                print(f"         Std  of medians : {t_stats.std_cm:.3f} cm")
                print(f"         Min/Max         : "
                      f"{t_stats.min_cm:.2f} / {t_stats.max_cm:.2f} cm\n")

            # --- Record a test point ------------------------------------
            if do_record:
                do_record = False

                if math.isnan(stats.median_m):
                    print("[RECORD] No valid depth in ROI — cannot record.")
                else:
                    manual_str = input(
                        "\n  Enter manually measured distance (cm): ").strip()
                    try:
                        manual_cm = float(manual_str)
                    except ValueError:
                        print("[RECORD] Invalid number — skipped.")
                        continue

                    # Run a quick temporal sample for the recorded point
                    print("[RECORD] Running quick temporal sample ...")
                    t_stats = collect_temporal_samples(
                        pipeline, align, di, depth_scale,
                        u, v, ROI_HALF, SAMPLE_DURATION_S)

                    # Recompute fresh single-frame stats for the CSV
                    frameset2 = pipeline.wait_for_frames()
                    aligned2  = align.process(frameset2)
                    df2       = aligned2.get_depth_frame()
                    arr2      = np.asanyarray(df2.get_data()) if df2 else depth_arr
                    stats2    = compute_roi_stats(arr2, u, v, ROI_HALF, depth_scale)
                    fill2, stdev2 = get_depth_quality_metadata(
                        df2 if df2 else depth_frame)

                    rec = TestRecord(
                        timestamp           = datetime.datetime.now().isoformat(
                                             timespec="seconds"),
                        manual_distance_cm  = manual_cm,
                        realsense_mean_cm   = t_stats.mean_cm,
                        realsense_median_cm = t_stats.mean_cm,  # mean of medians
                        std_cm              = t_stats.std_cm,
                        min_cm              = t_stats.min_cm,
                        max_cm              = t_stats.max_cm,
                        valid_pixel_count   = stats2.valid_count,
                        total_pixel_count   = stats2.total_count,
                        temporal_frames     = t_stats.frame_count,
                        temporal_std_cm     = t_stats.std_cm,
                        depth_fill_rate     = fill2,
                        depth_stdev_meta    = stdev2,
                        roi_half_px         = ROI_HALF,
                        depth_scale         = depth_scale,
                    )
                    records.append(rec)
                    append_csv(csv_path, rec)

                    err = rec.realsense_median_cm - manual_cm
                    print(f"\n[RECORD] Saved:")
                    print(f"         Manual      : {manual_cm:.2f} cm")
                    print(f"         RS median   : {rec.realsense_median_cm:.2f} cm")
                    print(f"         Error       : {err:+.2f} cm")
                    print(f"         Temporal std: {rec.temporal_std_cm:.3f} cm")
                    print(f"         CSV row     : {len(records)}\n")

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # --- Final summary ----------------------------------------------------
    print_final_summary(records)
    if records:
        print(f"[CSV] Results saved to: {csv_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    print("\n" + "#" * 62)
    print("  RealSense D455f — Stage 3: Depth Accuracy Characterisation")
    print("#" * 62 + "\n")

    try:
        run_accuracy_test()
    except RuntimeError as err:
        print(f"\n[FATAL] {err}")
        sys.exit(1)
    except Exception as err:
        print(f"\n[FATAL] Unexpected error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
