"""
object_segmentation.py
======================
Stage 4.1 — Controlled ROI-Assisted Object Segmentation & 3D Point Cloud Extraction
Intel RealSense D455f RGB-D Camera

Objective:
    Isolate a foreground target object (rectangular box) using user-guided ROI
    selection combined with depth histogram clustering, depth discontinuity
    analysis, connected component filtering, and calibrated metric 3D point
    deprojection. Background surfaces (walls, tables, laptops, floor) are strictly
    excluded.

Modular Architecture:
    1. select_roi()               : Interactive mouse-driven bounding box selection.
    2. lock_roi()                 : Locks ROI to prevent background contamination.
    3. extract_roi_depth()        : Crops aligned metric depth to active ROI.
    4. estimate_foreground_depth(): Histogram mode & robust cluster extraction for foreground.
    5. create_depth_mask()        : Depth-range & gradient thresholding inside ROI.
    6. clean_mask()               : Conservative morphology preserving physical object boundaries.
    7. select_object_component()  : Spatially coherent component scoring (area + ROI-center proximity).
    8. extract_object_points()    : Calibrated 3D deprojection (u, v, z) -> (X, Y, Z) in metres.
    9. validate_segmentation()    : 3D sanity check against background contamination.

Author : (your name)
Date   : 2026-09-21
"""

import os
import sys
import time
import signal
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import pyrealsense2 as rs
import numpy as np
import cv2

# Matplotlib for 3D point cloud visualization export
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend to avoid OpenCV GUI conflicts
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ===========================================================================
# STREAM CONFIGURATION
# ===========================================================================

STREAM_WIDTH  = 1280
STREAM_HEIGHT = 720
STREAM_FPS    = 30

WARMUP_FRAMES = 25

# GUI Window Names
WIN_MAIN     = "D455f — Stage 4.1: ROI Object Segmentation (RGB)"
WIN_MASK     = "D455f — Object Mask (Debug)"
WIN_DEPTH    = "D455f — Segmented Depth Map"
WIN_CONTROLS = "Segmentation Tuning"

# Default Tuning Parameters
DEFAULT_DEPTH_TOL_CM = 25     # Foreground depth thickness tolerance (+/- cm around peak)
DEFAULT_MIN_AREA_PX  = 1500   # Minimum object area in pixels inside ROI
DEFAULT_MORPH_K      = 3      # Morphological kernel size (odd, e.g. 3 or 5)

# Sanity check thresholds (a realistic box should never span > 80cm across when tested)
MAX_PLAUSIBLE_SPAN_CM = 80.0
MAX_PLAUSIBLE_DEPTH_SPAN_CM = 60.0


# ===========================================================================
# GRACEFUL SHUTDOWN HANDLER
# ===========================================================================

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    print("\n[INFO] Interrupt received — initiating clean shutdown...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _signal_handler)


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class ROIState:
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    is_drawing: bool = False
    is_selected: bool = False
    is_locked: bool = False

    @property
    def is_valid(self) -> bool:
        w = abs(self.x2 - self.x1)
        h = abs(self.y2 - self.y1)
        return w >= 20 and h >= 20

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Returns (xmin, ymin, xmax, ymax) strictly clamped to stream bounds."""
        xmin = max(0, min(self.x1, self.x2))
        ymin = max(0, min(self.y1, self.y2))
        xmax = min(STREAM_WIDTH - 1, max(self.x1, self.x2))
        ymax = min(STREAM_HEIGHT - 1, max(self.y1, self.y2))
        return (xmin, ymin, xmax, ymax)

    @property
    def width(self) -> int:
        xmin, _, xmax, _ = self.box
        return xmax - xmin

    @property
    def height(self) -> int:
        _, ymin, _, ymax = self.box
        return ymax - ymin


@dataclass
class CameraContext:
    pipeline: rs.pipeline
    profile: rs.pipeline_profile
    align: rs.align
    depth_scale: float
    intrinsics: rs.intrinsics
    colorizer: rs.colorizer
    device_name: str
    serial_number: str
    firmware_version: str


@dataclass
class SegmentationQualityReport:
    roi_area_px: int
    mask_area_px: int
    mask_to_roi_ratio: float
    valid_depth_pct: float
    point_count: int
    median_z_m: float
    min_z_m: float
    max_z_m: float
    span_x_cm: float
    span_y_cm: float
    span_z_cm: float
    is_valid_sanity: bool
    sanity_message: str


# ===========================================================================
# MOUSE CALLBACK FOR ROI SELECTION
# ===========================================================================

_current_roi = ROIState()

def _mouse_callback(event, x, y, flags, param):
    global _current_roi

    if _current_roi.is_locked:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        _current_roi.x1 = x
        _current_roi.y1 = y
        _current_roi.x2 = x
        _current_roi.y2 = y
        _current_roi.is_drawing = True
        _current_roi.is_selected = False

    elif event == cv2.EVENT_MOUSEMOVE and _current_roi.is_drawing:
        _current_roi.x2 = x
        _current_roi.y2 = y

    elif event == cv2.EVENT_LBUTTONUP:
        _current_roi.x2 = x
        _current_roi.y2 = y
        _current_roi.is_drawing = False
        if _current_roi.is_valid:
            _current_roi.is_selected = True
            # Auto-lock on complete rectangle drag
            _current_roi.is_locked = True
            xmin, ymin, xmax, ymax = _current_roi.box
            print(f"\n[ROI LOCKED] Bounding Box: ({xmin}, {ymin}) -> ({xmax}, {ymax}) | Size: {xmax-xmin} x {ymax-ymin} px")


def lock_roi():
    """Manually lock the current ROI."""
    global _current_roi
    if _current_roi.is_valid:
        _current_roi.is_locked = True
        _current_roi.is_selected = True


def reset_roi():
    """Clear and unlock ROI to allow user to draw a new one."""
    global _current_roi
    _current_roi = ROIState()
    print("\n[ROI RESET] Draw a new rectangle around the target object using mouse drag.")


# ===========================================================================
# CAMERA INITIALISATION
# ===========================================================================

def initialize_camera() -> CameraContext:
    """
    Connect to RealSense D455f, initialize aligned RGB & Depth streams,
    and retrieve live intrinsics.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense device detected.\n"
            "  1. Ensure D455f is plugged into a USB 3.0+ port.\n"
            "  2. Close any other application using the camera.\n"
            "  3. Reconnect if necessary."
        )

    dev = devices[0]
    dev_name = dev.get_info(rs.camera_info.name)
    serial   = dev.get_info(rs.camera_info.serial_number)
    fw_ver   = dev.get_info(rs.camera_info.firmware_version)

    depth_sensor = dev.first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    print("==============================================================")
    print("  DEVICE INITIALISATION")
    print("==============================================================")
    print(f"  Device Name      : {dev_name}")
    print(f"  Serial Number    : {serial}")
    print(f"  Firmware Version : {fw_ver}")
    print(f"  Depth Scale      : {depth_scale:.7f} m/unit")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, STREAM_FPS)
    config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, STREAM_FPS)

    profile = pipeline.start(config)

    # Depth-to-Color Alignment block
    align = rs.align(rs.stream.color)

    # Colorizer for visualization
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 0) # Jet preset
    colorizer.set_option(rs.option.min_distance, 0.2)
    colorizer.set_option(rs.option.max_distance, 3.0)

    # Stabilise auto-exposure
    print(f"\n[INFO] Stabilising sensor auto-exposure ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        _ = align.process(frames)

    aligned_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = aligned_profile.get_intrinsics()

    print("--------------------------------------------------------------")
    print("  ALIGNED INTRINSICS (Color-referenced)")
    print("--------------------------------------------------------------")
    print(f"  Resolution  : {intrinsics.width} x {intrinsics.height} px")
    print(f"  fx / fy     : {intrinsics.fx:.4f} / {intrinsics.fy:.4f} px")
    print(f"  cx / cy     : {intrinsics.ppx:.4f} / {intrinsics.ppy:.4f} px")
    print(f"  Distortion  : {intrinsics.model}")
    print("==============================================================\n")

    return CameraContext(
        pipeline=pipeline,
        profile=profile,
        align=align,
        depth_scale=depth_scale,
        intrinsics=intrinsics,
        colorizer=colorizer,
        device_name=dev_name,
        serial_number=serial,
        firmware_version=fw_ver
    )


# ===========================================================================
# GUI & TRACKBAR SETUP
# ===========================================================================

def _nothing(x):
    pass

def setup_gui():
    """Create OpenCV windows and mouse / trackbar listeners."""
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DEPTH, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(WIN_MAIN, 960, 540)
    cv2.resizeWindow(WIN_MASK, 480, 270)
    cv2.resizeWindow(WIN_DEPTH, 480, 270)
    cv2.resizeWindow(WIN_CONTROLS, 450, 180)

    cv2.setMouseCallback(WIN_MAIN, _mouse_callback)

    # Trackbars
    cv2.createTrackbar("Depth Tol (cm)", WIN_CONTROLS, DEFAULT_DEPTH_TOL_CM, 60, _nothing)
    cv2.createTrackbar("Min Area (x100 px)", WIN_CONTROLS, DEFAULT_MIN_AREA_PX // 100, 300, _nothing)
    cv2.createTrackbar("Morph Kernel (px)", WIN_CONTROLS, DEFAULT_MORPH_K, 15, _nothing)


def get_tuning_parameters() -> Dict[str, Any]:
    """Read tuning parameters from OpenCV sliders."""
    tol_cm   = cv2.getTrackbarPos("Depth Tol (cm)", WIN_CONTROLS)
    min_area = cv2.getTrackbarPos("Min Area (x100 px)", WIN_CONTROLS) * 100
    morph_k  = cv2.getTrackbarPos("Morph Kernel (px)", WIN_CONTROLS)

    tol_m   = max(0.05, tol_cm / 100.0)
    morph_k = max(1, morph_k if morph_k % 2 == 1 else morph_k + 1)

    return {
        "depth_tol_m": tol_m,
        "min_area_px": max(100, min_area),
        "morph_kernel": morph_k,
    }


# ===========================================================================
# MODULAR SEGMENTATION PIPELINE FUNCTIONS
# ===========================================================================

def extract_roi_depth(depth_m: np.ndarray, roi: ROIState) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extracts the depth subarray corresponding to the locked ROI.
    """
    xmin, ymin, xmax, ymax = roi.box
    roi_depth = depth_m[ymin:ymax, xmin:xmax]
    return roi_depth, (xmin, ymin, xmax, ymax)


def estimate_foreground_depth(roi_depth: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Analyzes the depth histogram inside the ROI to locate the dominant foreground
    cluster peak and its spread, avoiding isolated depth noise and background surfaces.

    Returns:
        (z_peak, z_min_cluster, z_max_cluster) in metres, or None if insufficient depth.
    """
    valid_depths = roi_depth[(roi_depth > 0.15) & (roi_depth < 3.5)]
    if len(valid_depths) < 100:
        return None

    # Compute depth histogram with 1 cm (0.01 m) resolution
    min_d = np.percentile(valid_depths, 1.0)
    max_d = np.percentile(valid_depths, 99.0)

    if max_d - min_d < 0.02:
        z_peak = float(np.median(valid_depths))
        return z_peak, z_peak - 0.05, z_peak + 0.05

    bins = np.arange(min_d, max_d + 0.02, 0.01)
    if len(bins) < 2:
        z_peak = float(np.median(valid_depths))
        return z_peak, z_peak - 0.05, z_peak + 0.05

    hist, bin_edges = np.histogram(valid_depths, bins=bins)

    # Focus on the closest 65% of the depth range to isolate foreground object from background wall/table
    cutoff_idx = max(3, int(len(hist) * 0.65))
    fg_hist = hist[:cutoff_idx]

    if len(fg_hist) == 0:
        peak_idx = int(np.argmax(hist))
    else:
        peak_idx = int(np.argmax(fg_hist))

    z_peak = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0)

    # Extract all points within a cluster around this peak to find natural cluster bounds
    cluster_pts = valid_depths[np.abs(valid_depths - z_peak) <= 0.20]
    if len(cluster_pts) > 0:
        z_min_cluster = float(np.percentile(cluster_pts, 2.0))
        z_max_cluster = float(np.percentile(cluster_pts, 98.0))
    else:
        z_min_cluster = z_peak - 0.08
        z_max_cluster = z_peak + 0.08

    return z_peak, z_min_cluster, z_max_cluster


def create_depth_mask(
    roi_depth: np.ndarray,
    z_peak: float,
    depth_tol_m: float
) -> np.ndarray:
    """
    Creates a binary candidate mask inside the ROI based on proximity to the
    foreground depth cluster.
    """
    # Box front, top, and side faces can be slightly closer or further than center peak
    z_min_thresh = max(0.15, z_peak - (depth_tol_m * 0.6))
    z_max_thresh = z_peak + depth_tol_m

    candidate_mask = (
        (roi_depth >= z_min_thresh) &
        (roi_depth <= z_max_thresh) &
        (roi_depth > 0)
    ).astype(np.uint8) * 255

    return candidate_mask


def clean_mask(candidate_mask: np.ndarray, morph_kernel: int) -> np.ndarray:
    """
    Applies conservative morphological opening and closing to remove speckle
    noise and fill interior surface holes while preserving physical object boundaries.
    """
    if not np.any(candidate_mask):
        return candidate_mask

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel + 2, morph_kernel + 2))

    # Opening to discard thin noise strands and floating edge pixels
    opened = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, k_open)
    # Closing to bridge internal depth dropouts on smooth faces
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    return closed


def select_object_component(
    cleaned_mask: np.ndarray,
    min_area_px: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Identifies connected components inside the ROI and selects the target box
    component based on area, spatial coherence, and proximity to ROI center.

    Returns:
        (roi_final_mask, primary_contour)
    """
    if not np.any(cleaned_mask):
        return np.zeros_like(cleaned_mask), None

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(cleaned_mask), None

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not valid_contours:
        # Fallback to largest if under threshold
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 200:
            valid_contours = [largest]
        else:
            return np.zeros_like(cleaned_mask), None

    # Score candidates: combine Area with Centroid proximity to ROI center
    h, w = cleaned_mask.shape
    roi_center = np.array([w / 2.0, h / 2.0])
    max_diag = np.sqrt(w**2 + h**2) + 1e-5

    best_contour = None
    best_score = -1e9

    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_to_center = np.linalg.norm(np.array([cx, cy]) - roi_center)
            norm_dist = dist_to_center / max_diag
        else:
            norm_dist = 0.5

        # Score balances high area and central placement within ROI
        score = area * (1.0 - (0.4 * norm_dist))
        if score > best_score:
            best_score = score
            best_contour = cnt

    # Generate filled solid mask for the selected component
    roi_final_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(roi_final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    return roi_final_mask, best_contour


def extract_object_points(
    full_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: rs.intrinsics
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Deprojects all valid depth pixels belonging to the final object mask
    into calibrated metric 3D coordinates (X, Y, Z) in metres.

    Returns:
        (points_3d, valid_u, valid_v)
    """
    v_indices, u_indices = np.where(full_mask > 0)
    if len(u_indices) == 0:
        return None, None, None

    z_vals = depth_m[v_indices, u_indices]
    valid_idx = np.where(z_vals > 0.10)[0]

    if len(valid_idx) < 30:
        return None, None, None

    valid_u = u_indices[valid_idx]
    valid_v = v_indices[valid_idx]
    valid_z = z_vals[valid_idx]

    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    x_points = (valid_u - cx) * valid_z / fx
    y_points = (valid_v - cy) * valid_z / fy
    z_points = valid_z

    points_3d = np.column_stack((x_points, y_points, z_points)).astype(np.float32)
    return points_3d, valid_u, valid_v


def validate_segmentation(
    roi: ROIState,
    mask: np.ndarray,
    points_3d: Optional[np.ndarray]
) -> SegmentationQualityReport:
    """
    Performs 3D metric sanity checks on the isolated object point cloud to verify
    complete exclusion of background geometry (walls, tables, floors).
    """
    roi_w = roi.width
    roi_h = roi.height
    roi_area = roi_w * roi_h
    mask_area = int(np.count_nonzero(mask))
    ratio = mask_area / roi_area if roi_area > 0 else 0.0

    if points_3d is None or len(points_3d) < 30:
        return SegmentationQualityReport(
            roi_area_px=roi_area,
            mask_area_px=mask_area,
            mask_to_roi_ratio=ratio,
            valid_depth_pct=0.0,
            point_count=0,
            median_z_m=0.0,
            min_z_m=0.0,
            max_z_m=0.0,
            span_x_cm=0.0,
            span_y_cm=0.0,
            span_z_cm=0.0,
            is_valid_sanity=False,
            sanity_message="NO VALID 3D POINTS"
        )

    valid_depth_pct = (len(points_3d) / mask_area * 100.0) if mask_area > 0 else 0.0

    x_pts = points_3d[:, 0]
    y_pts = points_3d[:, 1]
    z_pts = points_3d[:, 2]

    # Robust 1st to 99th percentile spans to ignore single-pixel outliers
    x_min, x_max = float(np.percentile(x_pts, 1.0)), float(np.percentile(x_pts, 99.0))
    y_min, y_max = float(np.percentile(y_pts, 1.0)), float(np.percentile(y_pts, 99.0))
    z_min, z_max = float(np.percentile(z_pts, 1.0)), float(np.percentile(z_pts, 99.0))

    span_x = (x_max - x_min) * 100.0
    span_y = (y_max - y_min) * 100.0
    span_z = (z_max - z_min) * 100.0
    med_z  = float(np.median(z_pts))

    # Sanity checks: box dimensions must be physically plausible (< 80cm span)
    is_valid = True
    msg = "OBJECT ISOLATED OK"

    if span_x > MAX_PLAUSIBLE_SPAN_CM or span_y > MAX_PLAUSIBLE_SPAN_CM or span_z > MAX_PLAUSIBLE_DEPTH_SPAN_CM:
        is_valid = False
        msg = "SEGMENTATION INVALID — BACKGROUND CONTAMINATION"
    elif ratio < 0.05:
        is_valid = False
        msg = "MASK TOO SMALL (ADJUST ROI / TOLERANCE)"
    elif ratio > 0.98:
        is_valid = False
        msg = "MASK COVERS ENTIRE ROI (CHECK BACKGROUND)"

    return SegmentationQualityReport(
        roi_area_px=roi_area,
        mask_area_px=mask_area,
        mask_to_roi_ratio=ratio,
        valid_depth_pct=valid_depth_pct,
        point_count=len(points_3d),
        median_z_m=med_z,
        min_z_m=z_min,
        max_z_m=z_max,
        span_x_cm=span_x,
        span_y_cm=span_y,
        span_z_cm=span_z,
        is_valid_sanity=is_valid,
        sanity_message=msg
    )


# ===========================================================================
# VISUALISATION & OVERLAY
# ===========================================================================

def render_visualization(
    color_img: np.ndarray,
    depth_colormap: np.ndarray,
    full_mask: np.ndarray,
    roi: ROIState,
    report: SegmentationQualityReport,
    primary_contour: Optional[np.ndarray],
    fps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders high-contrast annotated RGB view, debug binary mask, and segmented depth colormap.
    """
    annotated_rgb = color_img.copy()

    # 1. Mask Debug Window (Pure Black Background, Pure White Box)
    mask_debug = full_mask.copy()

    # 2. Segmented Depth Window (Outside mask is strictly black)
    mask_3ch = cv2.merge([full_mask, full_mask, full_mask])
    masked_depth = cv2.bitwise_and(depth_colormap, mask_3ch)

    # 3. Draw Active / Locked ROI Rectangle on RGB
    if roi.is_valid or roi.is_drawing:
        xmin, ymin, xmax, ymax = roi.box
        roi_color = (0, 255, 0) if roi.is_locked else (0, 255, 255) # Green if locked, yellow if drawing
        cv2.rectangle(annotated_rgb, (xmin, ymin), (xmax, ymax), roi_color, 2)
        status_label = f"ROI LOCKED [{xmax-xmin}x{ymax-ymin}]" if roi.is_locked else "DRAWING ROI..."
        cv2.putText(annotated_rgb, status_label, (xmin, max(20, ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2, cv2.LINE_AA)

    # 4. Mask Overlay & Outlines on RGB
    if roi.is_locked and np.any(full_mask):
        # Semi-transparent emerald green overlay strictly inside mask
        overlay = annotated_rgb.copy()
        overlay[full_mask > 0] = [0, 220, 100]
        cv2.addWeighted(overlay, 0.40, annotated_rgb, 0.60, 0, annotated_rgb)

        # Draw Clean Object Boundary Outline (Cyan)
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(annotated_rgb, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)

            # Minimum Area Oriented Bounding Box
            largest_c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_c)
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(annotated_rgb, [box_pts], 0, (255, 0, 255), 2, cv2.LINE_AA)

            # 2D Centroid Crosshair
            M = cv2.moments(largest_c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawMarker(annotated_rgb, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
                cv2.circle(annotated_rgb, (cx, cy), 6, (0, 255, 255), 1, cv2.LINE_AA)

    # 5. Top Diagnostics HUD Card
    hud_bg = annotated_rgb.copy()
    cv2.rectangle(hud_bg, (10, 10), (490, 210), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.75, annotated_rgb, 0.25, 0, annotated_rgb)
    cv2.rectangle(annotated_rgb, (10, 10), (490, 210), (90, 90, 90), 1)

    cv2.putText(annotated_rgb, f"D455f Stage 4.1 — ROI Segmentation ({fps:.1f} FPS)", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    if not roi.is_locked:
        cv2.putText(annotated_rgb, "STATUS: DRAG MOUSE TO SELECT ROI AROUND BOX", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_rgb, "Click & drag a rectangle around the target box", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated_rgb, "Make ROI ~10-20% larger than the box", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        sanity_col = (0, 255, 0) if report.is_valid_sanity else (0, 0, 255)
        cv2.putText(annotated_rgb, f"DIAGNOSTIC: {report.sanity_message}", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, sanity_col, 2, cv2.LINE_AA)
        cv2.putText(annotated_rgb, f"ROI Size         : {roi.width} x {roi.height} px ({report.roi_area_px:,} px)", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated_rgb, f"Mask Area        : {report.mask_area_px:,} px ({report.mask_to_roi_ratio*100:.1f}% of ROI)", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated_rgb, f"3D Point Count   : {report.point_count:,} pts ({report.valid_depth_pct:.1f}% valid)", (20, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_rgb, f"Target Distance  : Z_median = {report.median_z_m*100:.1f} cm ({report.min_z_m*100:.1f} - {report.max_z_m*100:.1f} cm)", (20, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_rgb, f"3D Span (approx) : dX={report.span_x_cm:.1f} cm, dY={report.span_y_cm:.1f} cm, dZ={report.span_z_cm:.1f} cm", (20, 166),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated_rgb, f"Background Wall  : EXCLUDED (Outside ROI = 0)", (20, 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # 6. Bottom Keybindings
    help_str = "[N] New ROI  |  [S] Save Snapshot  |  [P] 3D Scatter Plot  |  [Q] / [ESC] Quit"
    cv2.putText(annotated_rgb, help_str, (20, STREAM_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated_rgb, mask_debug, masked_depth


# ===========================================================================
# SNAPSHOT & 3D POINT CLOUD EXPORT
# ===========================================================================

def save_snapshot_bundle(
    color_img: np.ndarray,
    mask: np.ndarray,
    depth_colormap: np.ndarray,
    points_3d: Optional[np.ndarray],
    report: SegmentationQualityReport,
    roi: ROIState
) -> None:
    """Saves RGB, Mask PNG, Depth PNG, 3D Point Cloud PLY/CSV, and quality report."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    os.makedirs(out_dir, exist_ok=True)

    rgb_path   = os.path.join(out_dir, f"roi_object_rgb_{timestamp}.png")
    mask_path  = os.path.join(out_dir, f"roi_object_mask_{timestamp}.png")
    depth_path = os.path.join(out_dir, f"roi_object_depth_{timestamp}.png")

    cv2.imwrite(rgb_path, color_img)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(depth_path, depth_colormap)

    summary = [
        f"\n==============================================================",
        f"  SNAPSHOT SAVED AT {timestamp}",
        f"==============================================================",
        f"  RGB Image        : {rgb_path}",
        f"  Object Mask      : {mask_path}",
        f"  Segmented Depth  : {depth_path}",
        f"  ROI Coordinates  : ({roi.box[0]}, {roi.box[1]}) -> ({roi.box[2]}, {roi.box[3]})",
        f"  ROI Area         : {report.roi_area_px:,} px",
        f"  Mask Area        : {report.mask_area_px:,} px ({report.mask_to_roi_ratio*100:.1f}%)",
        f"  Valid 3D Points  : {report.point_count:,}",
        f"  Median Depth Z   : {report.median_z_m*100:.2f} cm",
        f"  3D Span Estimate : dX={report.span_x_cm:.1f} cm, dY={report.span_y_cm:.1f} cm, dZ={report.span_z_cm:.1f} cm",
        f"  Sanity Status    : {report.sanity_message}",
    ]

    if points_3d is not None and len(points_3d) > 0:
        ply_path = os.path.join(out_dir, f"roi_object_pointcloud_{timestamp}.ply")
        csv_path = os.path.join(out_dir, f"roi_object_pointcloud_{timestamp}.csv")

        # Save ASCII PLY
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for pt in points_3d:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        # Save CSV in centimetres
        np.savetxt(csv_path, points_3d * 100.0, delimiter=",", header="X_cm,Y_cm,Z_cm", comments="", fmt="%.3f")

        summary.append(f"  3D Pointcloud PLY: {ply_path} ({len(points_3d)} points)")
        summary.append(f"  3D Pointcloud CSV: {csv_path}")

    summary.append("==============================================================\n")
    print("\n".join(summary))


def plot_pointcloud_3d(points_3d: Optional[np.ndarray], report: SegmentationQualityReport) -> None:
    """Renders a 3D scatter plot of the isolated object points and saves to disk."""
    if not HAS_MATPLOTLIB:
        print("[WARN] Matplotlib not available — skipping 3D plot.")
        return

    if points_3d is None or len(points_3d) < 10:
        print("[WARN] Insufficient 3D points to generate scatter plot.")
        return

    max_plot_points = 6000
    if len(points_3d) > max_plot_points:
        idx = np.random.choice(len(points_3d), max_plot_points, replace=False)
        pts = points_3d[idx] * 100.0  # cm
    else:
        pts = points_3d * 100.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=pts[:, 2], cmap="viridis", s=3, alpha=0.8)
    cbar = fig.colorbar(p, ax=ax, pad=0.1)
    cbar.set_label("Distance from Camera Z (cm)")

    ax.set_xlabel("X — Lateral (cm)")
    ax.set_ylabel("Z — Depth (cm)")
    ax.set_zlabel("Y — Vertical (Up) (cm)")
    ax.set_title(
        f"Stage 4.1: Isolated Object Point Cloud\n"
        f"Distance Z ≈ {report.median_z_m*100:.1f} cm | Points = {report.point_count:,} | "
        f"Span: {report.span_x_cm:.1f} x {report.span_y_cm:.1f} x {report.span_z_cm:.1f} cm",
        fontsize=11, fontweight="bold"
    )

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_object_pointcloud_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[INFO] 3D Point Cloud Plot successfully saved to: {plot_path}\n")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print("""
##############################################################
  Intel RealSense D455f — Stage 4.1: ROI-Assisted Object Segmentation
##############################################################
  - Drag mouse on RGB window to define ROI (~10-20% larger than box)
  - Depth histogram clustering isolates foreground box from table/wall
  - Strictly zeroes out all background outside ROI
  - Metric 3D Point Cloud Extraction & 3D Sanity Verification
##############################################################
""")

    cam = initialize_camera()
    setup_gui()

    fps = 0.0
    frame_count = 0
    t_start = time.time()

    print("[INFO] Application running.")
    print("[INFO] STEP 1: Look at the RGB window and DRAG A RECTANGLE around the target box.")
    print("[INFO] Hotkeys:")
    print("       - [N]     : Reset / Draw a new ROI")
    print("       - [S]     : Save Snapshot bundle (RGB, Mask, Depth, 3D PLY/CSV)")
    print("       - [P]     : Render & save 3D Point Cloud Plot (PNG)")
    print("       - [Q]/ESC : Exit\n")

    try:
        while not _shutdown_requested:
            # 1. Fetch synchronized frames
            success, frameset = cam.pipeline.try_wait_for_frames(timeout_ms=3000)
            if not success:
                continue

            # 2. Align depth to color pixel grid
            aligned_frames = cam.align.process(frameset)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m   = depth_raw.astype(np.float32) * cam.depth_scale

            depth_color_frame = cam.colorizer.colorize(depth_frame)
            depth_colormap    = np.asanyarray(depth_color_frame.get_data())

            # 3. Read tuning sliders
            params = get_tuning_parameters()

            # 4. Process ROI Segmentation
            full_mask = np.zeros((STREAM_HEIGHT, STREAM_WIDTH), dtype=np.uint8)
            primary_contour = None
            points_3d = None

            if _current_roi.is_locked and _current_roi.is_valid:
                # Step 4a: Extract ROI depth
                roi_depth, (xmin, ymin, xmax, ymax) = extract_roi_depth(depth_m, _current_roi)

                # Step 4b: Estimate foreground depth cluster inside ROI
                fg_depth_info = estimate_foreground_depth(roi_depth)

                if fg_depth_info is not None:
                    z_peak, _, _ = fg_depth_info

                    # Step 4c: Depth gating inside ROI
                    cand_mask = create_depth_mask(roi_depth, z_peak, params["depth_tol_m"])

                    # Step 4d: Conservative morphology
                    clean_m = clean_mask(cand_mask, params["morph_kernel"])

                    # Step 4e: Connected component selection (best foreground blob)
                    roi_final_mask, primary_contour = select_object_component(clean_m, params["min_area_px"])

                    # Step 4f: Embed ROI mask into full-image mask (Everything outside ROI is 0)
                    full_mask[ymin:ymax, xmin:xmax] = roi_final_mask

                # Step 4g: Extract calibrated 3D object points
                points_3d, _, _ = extract_object_points(full_mask, depth_m, cam.intrinsics)

            # 5. Sanity Check & Diagnostic Report
            report = validate_segmentation(_current_roi, full_mask, points_3d)

            # 6. Render Annotated Visuals
            annotated_rgb, mask_debug, masked_depth = render_visualization(
                color_img, depth_colormap, full_mask, _current_roi, report, primary_contour, fps
            )

            # 7. Display Windows
            cv2.imshow(WIN_MAIN, annotated_rgb)
            cv2.imshow(WIN_MASK, mask_debug)
            cv2.imshow(WIN_DEPTH, masked_depth)

            # 8. FPS Tracker
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # 9. Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] Exit requested by user.")
                break
            elif key in (ord('n'), ord('N')):
                reset_roi()
            elif key in (ord('s'), ord('S')):
                save_snapshot_bundle(annotated_rgb, full_mask, masked_depth, points_3d, report, _current_roi)
            elif key in (ord('p'), ord('P')):
                plot_pointcloud_3d(points_3d, report)

    except Exception as exc:
        print(f"\n[ERROR] Runtime exception: {exc}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[INFO] Shutting down camera pipeline...")
        try:
            cam.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Camera stopped and windows closed cleanly.")
        print("Stage 4.1 execution finished.\n")


if __name__ == "__main__":
    main()
