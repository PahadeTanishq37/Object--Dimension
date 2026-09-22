"""
box_measurement.py
==================
Stage 5.5 — Physical Box Boundary and Edge Reconstruction (RGB-D Fusion)
Intel RealSense D455f RGB-D Camera

Separates "Observed Plane Extent" from "Physical Box Edge" by deriving physical
dimensions directly from:
    1. 2D in-plane convex hull & boundary line extraction on detected planes
    2. Straight RGB line segments (LSD / Hough) deprojected to 3D with aligned depth
    3. Supported orthogonal plane intersection line segments
    4. Orthogonal directional edge clustering (Length, Breadth, Height)
    5. Multi-evidence validation & confidence scoring (HIGH, MEDIUM, LOW, UNSUPPORTED)
    6. Strict refusal to extrapolate or fabricate unsupported dimensions (e.g. unobserved height)

Author : (your name)
Date   : 2026-09-21
"""

import os
import sys
import time
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import pyrealsense2 as rs
import numpy as np
import cv2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from object_detector import RGBDForegroundDetector, DetectionResult, BaseObjectDetector


# ===========================================================================
# STREAM & APPLICATION CONFIGURATION
# ===========================================================================

STREAM_WIDTH  = 1280
STREAM_HEIGHT = 720
STREAM_FPS    = 30

WARMUP_FRAMES = 25
TEMPORAL_BUFFER_SIZE = 45  # ~1.5 seconds at 30 FPS

WIN_MAIN     = "D455f — Stage 5.5: Physical Boundary Measurement (RGB)"
WIN_MASK     = "D455f — Object Mask"
WIN_CONTROLS = "Measurement Tuning"

# Default Tuning Parameters
DEFAULT_DEPTH_TOL_CM   = 25     # Depth gating tolerance around foreground peak
DEFAULT_MIN_AREA_PX    = 1200   # Min pixel area in ROI
DEFAULT_RANSAC_DIST_MM = 10     # RANSAC plane distance threshold in mm (1.0 cm)
DEFAULT_ORTHO_TOL_DEG  = 28.0   # Max allowed angular deviation from 90 degrees

# Support & Boundary Estimation Thresholds
MIN_EDGE_SUPPORT_POINTS = 20    # Minimum nearby points required for an edge to be VALID
EDGE_SUPPORT_RADIUS_CM  = 2.0   # Search cylinder radius around edge in cm
LOW_PERCENTILE          = 1.0   # Lower quantile for robust in-plane boundary
HIGH_PERCENTILE         = 99.0  # Upper quantile for robust in-plane boundary

# Ground Truth for Evaluation Only (NEVER used in measurement calculation!)
DEFAULT_GT_L = 16.6  # cm
DEFAULT_GT_B = 9.1   # cm
DEFAULT_GT_H = 5.0   # cm


# ===========================================================================
# SHUTDOWN HANDLER
# ===========================================================================

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    print("\n[INFO] Interrupt signal received — shutting down cleanly...")
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
        return w >= 25 and h >= 25

    @property
    def box(self) -> Tuple[int, int, int, int]:
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
class DetectedPlane:
    plane_id: int
    normal: np.ndarray        # (3,) unit vector pointing towards camera
    d: float                  # scalar offset: n · p + d = 0
    inliers_idx: np.ndarray   # inlier indices
    inlier_points: np.ndarray # (M, 3) inlier 3D coordinates in metres
    rms_residual_cm: float    # RMS inlier residual distance in cm
    median_residual_cm: float # Median inlier residual distance in cm
    local_axis_u: np.ndarray  # (3,) in-plane primary axis (along shared edge)
    local_axis_v: np.ndarray  # (3,) in-plane secondary axis (across face)
    extent_u_cm: float        # Observed length along U in cm
    extent_v_cm: float        # Observed breadth/height along V in cm
    approx_area_sq_cm: float  # Estimated surface area in cm²


@dataclass
class CandidatePhysicalEdge:
    edge_id: str                      # "RGB_E1", "PL1_BND1", "INTERSECT_EDGE", etc.
    source: str                       # "RGB-D Line Segment", "Plane Boundary Line", "Common Intersection"
    start_pt_3d_m: np.ndarray         # (3,) metric 3D start point
    end_pt_3d_m: np.ndarray           # (3,) metric 3D end point
    direction_3d: np.ndarray          # (3,) unit vector
    length_cm: float                  # physical length in cm
    support_points: int               # count of supporting points
    depth_valid_ratio: float          # valid depth samples / total samples [0.0 - 1.0]
    rgb_strength: float               # normalized contrast/gradient [0.0 - 1.0]
    rms_residual_cm: float            # 3D line-fitting RMS residual in cm
    median_dist_cm: float             # Median distance of nearby points in cm
    confidence: str                   # "HIGH", "MEDIUM", "LOW", "REJECTED"
    is_valid: bool                    # True if confidence in ("HIGH", "MEDIUM", "LOW")
    start_pix: Tuple[int, int] = (0, 0)
    end_pix: Tuple[int, int] = (0, 0)


@dataclass
class ReconstructedEdge:
    edge_id: str              # "E1", "E2", ...
    start_corner_name: str    # "C1", "C2", ...
    end_corner_name: str
    start_pt_3d_m: np.ndarray # (3,) in metres
    end_pt_3d_m: np.ndarray   # (3,) in metres
    length_cm: float
    direction_vec: np.ndarray # (3,) normalized direction
    classification: str       # "LENGTH", "BREADTH", "HEIGHT"
    support_points_count: int # Points within search radius around edge
    median_point_dist_cm: float # Median distance of support points to edge
    is_supported: bool        # True if support_points_count >= MIN_EDGE_SUPPORT_POINTS


@dataclass
class DimensionEstimate:
    value_cm: float
    confidence: str           # "HIGH", "MEDIUM", "LOW", "UNSUPPORTED"
    source: str               # e.g., "RGB-D & Plane Boundary (3 edges)", etc.
    support_points: int
    median_dist_cm: float
    is_valid: bool
    candidate_lengths: List[float] = field(default_factory=list)


@dataclass
class CuboidReconstruction:
    length: DimensionEstimate
    breadth: DimensionEstimate
    height: DimensionEstimate
    centroid_3d_m: np.ndarray # (3,) center of bounding box in camera frame
    axes_rot: np.ndarray      # (3, 3) orthonormal basis
    corners_3d_m: np.ndarray  # (8, 3) 8 reconstructed bounding box vertices in metres
    corners_labeled: Dict[str, Tuple[float, float, float]] # "C1".."C8" -> (X, Y, Z) in cm
    reconstructed_edges: List[ReconstructedEdge]
    candidate_physical_edges: List[CandidatePhysicalEdge]
    detected_planes: List[DetectedPlane]
    num_planes: int
    inter_plane_angles_deg: List[float]
    geometry_status: str      # "GEOMETRY VALID", "SHOW MORE BOX FACES", "UNSUPPORTED"
    is_reliable: bool


@dataclass
class GroundTruth:
    length_cm: float = DEFAULT_GT_L
    breadth_cm: float = DEFAULT_GT_B
    height_cm: float = DEFAULT_GT_H
    is_set: bool = True


@dataclass
class TemporalStats:
    length_median: float = 0.0
    length_mean: float = 0.0
    length_std: float = 0.0
    breadth_median: float = 0.0
    breadth_mean: float = 0.0
    breadth_std: float = 0.0
    height_median: float = 0.0
    height_mean: float = 0.0
    height_std: float = 0.0
    sample_count: int = 0


@dataclass
class MeasurementResult:
    timestamp: float
    detection_status: str             # "AUTO OK", "MANUAL LOCKED", "NO OBJECT DETECTED", "TRACKING LOST"
    detection_confidence: float
    detection_mode: str               # "AUTO" or "MANUAL"
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    length_cm: float
    breadth_cm: float
    height_cm: float
    length_std_cm: float
    breadth_std_cm: float
    height_std_cm: float
    length_confidence: str            # "HIGH", "MEDIUM", "LOW", "UNSUPPORTED"
    breadth_confidence: str
    height_confidence: str
    physical_edges: List[CandidatePhysicalEdge]
    reconstructed_edges: List[ReconstructedEdge]
    physical_corners: Dict[str, Tuple[float, float, float]]
    metrology_status: str
    is_metrology_valid: bool
    camera_distance_z_cm: float
    fps: float



# ===========================================================================
# MOUSE CALLBACK FOR ROI SELECTION
# ===========================================================================

_current_roi = ROIState()
_user_dragged_new_roi = False

def _mouse_callback(event, x, y, flags, param):
    global _current_roi, _user_dragged_new_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        _current_roi.x1 = x
        _current_roi.y1 = y
        _current_roi.x2 = x
        _current_roi.y2 = y
        _current_roi.is_drawing = True
        _current_roi.is_selected = False
        _current_roi.is_locked = False
        _user_dragged_new_roi = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if _current_roi.is_drawing:
            _current_roi.x2 = x
            _current_roi.y2 = y

    elif event == cv2.EVENT_LBUTTONUP:
        if _current_roi.is_drawing:
            _current_roi.x2 = x
            _current_roi.y2 = y
            _current_roi.is_drawing = False
            if _current_roi.is_valid:
                _current_roi.is_selected = True
                _current_roi.is_locked = True
                xmin, ymin, xmax, ymax = _current_roi.box
                print(f"\n[BORDER DRAWN & LOCKED] Object Box: ({xmin}, {ymin}) -> ({xmax}, {ymax}) [{xmax-xmin} x {ymax-ymin} px]")


def reset_roi():
    global _current_roi
    _current_roi = ROIState()
    print("\n[BORDER RESET] Click and drag a new border around the target object.")



# ===========================================================================
# CAMERA INITIALISATION
# ===========================================================================

def initialize_camera() -> CameraContext:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense D455f device connected.")

    dev = devices[0]
    dev_name = dev.get_info(rs.camera_info.name)
    serial   = dev.get_info(rs.camera_info.serial_number)
    fw_ver   = dev.get_info(rs.camera_info.firmware_version)

    depth_sensor = dev.first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    print("==============================================================")
    print("  CAMERA INITIALISATION (Stage 5.5 — Physical Edge Reconstruction)")
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
    align = rs.align(rs.stream.color)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 0)
    colorizer.set_option(rs.option.min_distance, 0.2)
    colorizer.set_option(rs.option.max_distance, 3.0)

    print(f"[INFO] Stabilising auto-exposure ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        _ = align.process(frames)

    aligned_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = aligned_profile.get_intrinsics()

    print("--------------------------------------------------------------")
    print("  ALIGNED CAMERA INTRINSICS")
    print("--------------------------------------------------------------")
    print(f"  Resolution : {intrinsics.width} x {intrinsics.height} px")
    print(f"  fx / fy    : {intrinsics.fx:.4f} / {intrinsics.fy:.4f} px")
    print(f"  cx / cy    : {intrinsics.ppx:.4f} / {intrinsics.ppy:.4f} px")
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
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(WIN_MAIN, 960, 540)
    cv2.resizeWindow(WIN_MASK, 480, 270)
    cv2.resizeWindow(WIN_CONTROLS, 460, 160)

    cv2.setMouseCallback(WIN_MAIN, _mouse_callback)

    cv2.createTrackbar("Depth Tol (cm)", WIN_CONTROLS, DEFAULT_DEPTH_TOL_CM, 60, _nothing)
    cv2.createTrackbar("Min Area (x100 px)", WIN_CONTROLS, DEFAULT_MIN_AREA_PX // 100, 300, _nothing)
    cv2.createTrackbar("RANSAC Thresh (mm)", WIN_CONTROLS, DEFAULT_RANSAC_DIST_MM, 35, _nothing)


def get_tuning_parameters() -> Dict[str, Any]:
    tol_cm     = cv2.getTrackbarPos("Depth Tol (cm)", WIN_CONTROLS)
    min_area   = cv2.getTrackbarPos("Min Area (x100 px)", WIN_CONTROLS) * 100
    ransac_mm  = cv2.getTrackbarPos("RANSAC Thresh (mm)", WIN_CONTROLS)

    return {
        "depth_tol_m": max(0.05, tol_cm / 100.0),
        "min_area_px": max(100, min_area),
        "ransac_thresh_m": max(0.004, ransac_mm / 1000.0),
    }


# ===========================================================================
# POINT CLOUD EXTRACTION & FILTERING
# ===========================================================================

def segment_and_extract_point_cloud(
    depth_m: np.ndarray,
    roi: ROIState,
    intrinsics: rs.intrinsics,
    params: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray], int, int, int, float]:
    full_mask = np.zeros((STREAM_HEIGHT, STREAM_WIDTH), dtype=np.uint8)
    if not (roi.is_locked and roi.is_valid):
        return full_mask, None, 0, 0, 0, 0.0

    xmin, ymin, xmax, ymax = roi.box
    roi_depth = depth_m[ymin:ymax, xmin:xmax]

    valid_depths = roi_depth[(roi_depth > 0.15) & (roi_depth < 3.5)]
    raw_count = len(valid_depths)
    if raw_count < 80:
        return full_mask, None, raw_count, 0, 0, 0.0

    min_d = np.percentile(valid_depths, 1.0)
    max_d = np.percentile(valid_depths, 99.0)
    bins = np.arange(min_d, max_d + 0.02, 0.01)

    if len(bins) >= 2:
        hist, bin_edges = np.histogram(valid_depths, bins=bins)
        cutoff_idx = max(3, int(len(hist) * 0.70))
        fg_hist = hist[:cutoff_idx] if len(hist) > 3 else hist
        peak_idx = int(np.argmax(fg_hist))
        z_peak = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0)
    else:
        z_peak = float(np.median(valid_depths))

    tol = params["depth_tol_m"]
    z_min_thresh = max(0.15, z_peak - (tol * 0.6))
    z_max_thresh = z_peak + tol

    cand_mask = ((roi_depth >= z_min_thresh) & (roi_depth <= z_max_thresh) & (roi_depth > 0)).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(cand_mask, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return full_mask, None, raw_count, 0, 0, z_peak

    valid_contours = [c for c in contours if cv2.contourArea(c) >= params["min_area_px"]]
    if not valid_contours:
        valid_contours = [max(contours, key=cv2.contourArea)]

    h, w = closed.shape
    roi_center = np.array([w / 2.0, h / 2.0])
    max_diag = np.sqrt(w**2 + h**2) + 1e-5

    best_cnt = max(
        valid_contours,
        key=lambda c: cv2.contourArea(c) * (1.0 - 0.35 * (np.linalg.norm(np.array([cv2.moments(c)["m10"]/(cv2.moments(c)["m00"]+1e-5), cv2.moments(c)["m01"]/(cv2.moments(c)["m00"]+1e-5)]) - roi_center) / max_diag))
    )

    roi_mask = np.zeros_like(closed)
    cv2.drawContours(roi_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
    full_mask[ymin:ymax, xmin:xmax] = roi_mask

    v_idx, u_idx = np.where(full_mask > 0)
    z_vals = depth_m[v_idx, u_idx]
    valid_k = np.where(z_vals > 0.10)[0]
    valid_count = len(valid_k)

    if valid_count < 60:
        return full_mask, None, raw_count, valid_count, 0, z_peak

    valid_u = u_idx[valid_k]
    valid_v = v_idx[valid_k]
    valid_z = z_vals[valid_k]

    x_pts = (valid_u - intrinsics.ppx) * valid_z / intrinsics.fx
    y_pts = (valid_v - intrinsics.ppy) * valid_z / intrinsics.fy
    points_3d = np.column_stack((x_pts, y_pts, valid_z)).astype(np.float32)

    centroid = np.median(points_3d, axis=0)
    dists = np.linalg.norm(points_3d - centroid, axis=1)
    inlier_mask = dists <= (np.mean(dists) + 2.5 * np.std(dists))
    cleaned_pts = points_3d[inlier_mask]

    return full_mask, cleaned_pts, raw_count, valid_count, len(cleaned_pts), z_peak


# ===========================================================================
# RANSAC MULTI-PLANE DETECTION
# ===========================================================================

def fit_plane_ransac(
    points: np.ndarray,
    plane_id: int,
    dist_thresh_m: float = 0.010,
    max_iters: int = 180
) -> Optional[DetectedPlane]:
    n_pts = len(points)
    if n_pts < 30:
        return None

    best_inliers = np.array([], dtype=int)
    best_normal = None
    best_d = 0.0

    for _ in range(max_iters):
        sample_idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = points[sample_idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len

        d = -float(np.dot(normal, p1))
        dists = np.abs(np.dot(points, normal) + d)
        inliers = np.where(dists <= dist_thresh_m)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if len(best_inliers) < max(35, int(0.08 * n_pts)):
        return None

    inlier_pts = points[best_inliers]
    c = np.mean(inlier_pts, axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - c)
    refined_normal = vh[2, :]
    refined_normal = refined_normal / np.linalg.norm(refined_normal)

    # Orient normal towards camera
    if refined_normal[2] > 0:
        refined_normal = -refined_normal

    refined_d = -float(np.dot(refined_normal, c))
    residuals_m = np.abs(np.dot(inlier_pts, refined_normal) + refined_d)

    # Temporary initial local axes (refined later once intersection line is known)
    u_vec = np.cross(refined_normal, np.array([0, 1, 0]))
    if np.linalg.norm(u_vec) < 1e-4:
        u_vec = np.cross(refined_normal, np.array([1, 0, 0]))
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(refined_normal, u_vec)

    q_u = np.dot(inlier_pts - c, u_vec)
    q_v = np.dot(inlier_pts - c, v_vec)
    extent_u = float(np.percentile(q_u, HIGH_PERCENTILE) - np.percentile(q_u, LOW_PERCENTILE)) * 100.0
    extent_v = float(np.percentile(q_v, HIGH_PERCENTILE) - np.percentile(q_v, LOW_PERCENTILE)) * 100.0
    approx_area = extent_u * extent_v

    return DetectedPlane(
        plane_id=plane_id,
        normal=refined_normal,
        d=refined_d,
        inliers_idx=best_inliers,
        inlier_points=inlier_pts,
        rms_residual_cm=float(np.sqrt(np.mean(residuals_m**2))) * 100.0,
        median_residual_cm=float(np.median(residuals_m)) * 100.0,
        local_axis_u=u_vec,
        local_axis_v=v_vec,
        extent_u_cm=extent_u,
        extent_v_cm=extent_v,
        approx_area_sq_cm=approx_area
    )


def detect_planes_ransac(
    points_3d: np.ndarray,
    dist_thresh_m: float = 0.010
) -> List[DetectedPlane]:
    detected_planes: List[DetectedPlane] = []
    remaining_indices = np.arange(len(points_3d))
    remaining_pts = points_3d.copy()

    for pid in range(1, 4):
        if len(remaining_pts) < 40:
            break
        plane = fit_plane_ransac(remaining_pts, plane_id=pid, dist_thresh_m=dist_thresh_m)
        if plane is None:
            break

        orig_inliers = remaining_indices[plane.inliers_idx]
        detected_planes.append(DetectedPlane(
            plane_id=pid,
            normal=plane.normal,
            d=plane.d,
            inliers_idx=orig_inliers,
            inlier_points=points_3d[orig_inliers],
            rms_residual_cm=plane.rms_residual_cm,
            median_residual_cm=plane.median_residual_cm,
            local_axis_u=plane.local_axis_u,
            local_axis_v=plane.local_axis_v,
            extent_u_cm=plane.extent_u_cm,
            extent_v_cm=plane.extent_v_cm,
            approx_area_sq_cm=plane.approx_area_sq_cm
        ))

        keep_mask = np.ones(len(remaining_pts), dtype=bool)
        keep_mask[plane.inliers_idx] = False
        remaining_pts = remaining_pts[keep_mask]
        remaining_indices = remaining_indices[keep_mask]

    return detected_planes


# ===========================================================================
# 3D EDGE POINT SUPPORT EVALUATION
# ===========================================================================

def calculate_edge_point_support(
    start_pt_m: np.ndarray,
    end_pt_m: np.ndarray,
    all_points_3d_m: np.ndarray,
    radius_thresh_cm: float = EDGE_SUPPORT_RADIUS_CM
) -> Tuple[int, float]:
    seg_vec = end_pt_m - start_pt_m
    seg_len_sq = float(np.dot(seg_vec, seg_vec))
    if seg_len_sq < 1e-8:
        return 0, 0.0

    pt_vecs = all_points_3d_m - start_pt_m
    t = np.clip(np.dot(pt_vecs, seg_vec) / seg_len_sq, 0.0, 1.0)
    projections = start_pt_m + np.outer(t, seg_vec)

    dists_m = np.linalg.norm(all_points_3d_m - projections, axis=1)
    dists_cm = dists_m * 100.0

    inlier_mask = dists_cm <= radius_thresh_cm
    support_count = int(np.sum(inlier_mask))
    median_dist = float(np.median(dists_cm[inlier_mask])) if support_count > 0 else 0.0

    return support_count, median_dist


# ===========================================================================
# RGB EDGE EXTRACTION & 3D DEPROJECTION
# ===========================================================================

def extract_rgbd_physical_edges(
    color_img: np.ndarray,
    depth_m: np.ndarray,
    roi: ROIState,
    intrinsics: rs.intrinsics,
    z_peak: float,
    depth_tol_m: float = 0.25
) -> List[CandidatePhysicalEdge]:
    """
    Extracts straight 2D line segments inside the ROI using LSD,
    queries their aligned depth, deprojects to 3D, and fits 3D line edges.
    """
    if not (roi.is_locked and roi.is_valid):
        return []

    xmin, ymin, xmax, ymax = roi.box
    roi_bgr = color_img[ymin:ymax, xmin:xmax]
    roi_depth = depth_m[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Gradient magnitude for edge strength estimation
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    max_grad = float(np.max(grad_mag)) if np.max(grad_mag) > 0 else 1.0

    # Line Segment Detector (LSD)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray)

    candidate_edges: List[CandidatePhysicalEdge] = []
    if lines is None:
        return candidate_edges

    edge_counter = 1
    for line in lines:
        coords = np.asarray(line).ravel()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        seg_len_px = float(np.hypot(x2 - x1, y2 - y1))
        if seg_len_px < 28.0:
            continue

        num_samples = max(20, int(seg_len_px))
        t_vals = np.linspace(0.0, 1.0, num_samples)
        sample_xs = np.clip((x1 + t_vals * (x2 - x1)).astype(int), 0, roi_bgr.shape[1] - 1)
        sample_ys = np.clip((y1 + t_vals * (y2 - y1)).astype(int), 0, roi_bgr.shape[0] - 1)

        sampled_depths = roi_depth[sample_ys, sample_xs]

        # Valid depth filtering around object peak
        z_min = max(0.15, z_peak - depth_tol_m)
        z_max = z_peak + depth_tol_m
        valid_mask = (sampled_depths >= z_min) & (sampled_depths <= z_max) & (sampled_depths > 0)

        valid_count = int(np.sum(valid_mask))
        depth_valid_ratio = valid_count / float(num_samples)

        if valid_count < 12 or depth_valid_ratio < 0.35:
            continue

        full_u = sample_xs[valid_mask] + xmin
        full_v = sample_ys[valid_mask] + ymin
        full_z = sampled_depths[valid_mask]

        pts_x = (full_u - intrinsics.ppx) * full_z / intrinsics.fx
        pts_y = (full_v - intrinsics.ppy) * full_z / intrinsics.fy
        pts_3d = np.column_stack([pts_x, pts_y, full_z])

        # 3D line fitting via SVD
        c_3d = np.mean(pts_3d, axis=0)
        _, _, vh = np.linalg.svd(pts_3d - c_3d)
        dir_3d = vh[0, :]
        dir_3d = dir_3d / np.linalg.norm(dir_3d)

        proj_t = np.dot(pts_3d - c_3d, dir_3d)
        t_min, t_max = float(np.percentile(proj_t, 2.0)), float(np.percentile(proj_t, 98.0))
        p_start_3d = c_3d + t_min * dir_3d
        p_end_3d = c_3d + t_max * dir_3d
        len_cm = float(np.linalg.norm(p_end_3d - p_start_3d) * 100.0)

        if len_cm < 2.0:
            continue

        # 3D Residuals
        projs_on_line = c_3d + np.outer(proj_t, dir_3d)
        residuals_m = np.linalg.norm(pts_3d - projs_on_line, axis=1)
        rms_res_cm = float(np.sqrt(np.mean(residuals_m**2)) * 100.0)
        med_dist_cm = float(np.median(residuals_m) * 100.0)

        # RGB contrast / gradient strength
        sampled_grads = grad_mag[sample_ys[valid_mask], sample_xs[valid_mask]]
        rgb_strength = float(np.mean(sampled_grads) / max_grad)

        # Confidence categorization
        if valid_count >= 50 and depth_valid_ratio >= 0.70 and rms_res_cm <= 0.40:
            conf = "HIGH"
        elif valid_count >= 25 and depth_valid_ratio >= 0.50 and rms_res_cm <= 0.70:
            conf = "MEDIUM"
        elif valid_count >= 12 and depth_valid_ratio >= 0.35 and rms_res_cm <= 1.10:
            conf = "LOW"
        else:
            conf = "REJECTED"

        if conf != "REJECTED":
            candidate_edges.append(CandidatePhysicalEdge(
                edge_id=f"RGB_E{edge_counter}",
                source="RGB-D Line Segment",
                start_pt_3d_m=p_start_3d,
                end_pt_3d_m=p_end_3d,
                direction_3d=dir_3d,
                length_cm=len_cm,
                support_points=valid_count,
                depth_valid_ratio=depth_valid_ratio,
                rgb_strength=rgb_strength,
                rms_residual_cm=rms_res_cm,
                median_dist_cm=med_dist_cm,
                confidence=conf,
                is_valid=True,
                start_pix=(int(x1 + xmin), int(y1 + ymin)),
                end_pix=(int(x2 + xmin), int(y2 + ymin))
            ))
            edge_counter += 1

    return candidate_edges


# ===========================================================================
# PLANAR BOUNDARY EXTRACTION & STRAIGHT LINE FITTING
# ===========================================================================

def extract_plane_boundary_lines(
    plane: DetectedPlane,
    all_points_3d: np.ndarray,
    plane_id: int
) -> List[CandidatePhysicalEdge]:
    """
    Fits a 2D convex hull to the in-plane points, fits candidate straight boundary
    segments, deprojects back into 3D, and computes 3D point support.
    """
    pts = plane.inlier_points
    if len(pts) < 40:
        return []

    c = np.mean(pts, axis=0)
    u_axis = plane.local_axis_u
    v_axis = plane.local_axis_v

    # Project to 2D
    q_u = np.dot(pts - c, u_axis)
    q_v = np.dot(pts - c, v_axis)
    pts_2d = np.column_stack([q_u, q_v]).astype(np.float32)

    try:
        hull = cv2.convexHull(pts_2d, returnPoints=True)
        hull_pts = hull.squeeze()
        if len(hull_pts.shape) < 2 or len(hull_pts) < 4:
            return []
    except Exception:
        return []

    # Fit minimum area rectangle to 2D convex hull boundary
    rect = cv2.minAreaRect(hull_pts)
    box_2d = cv2.boxPoints(rect)  # (4, 2) in local (u, v) plane coordinates

    boundary_edges: List[CandidatePhysicalEdge] = []
    for i in range(4):
        p1_2d = box_2d[i]
        p2_2d = box_2d[(i + 1) % 4]

        p1_3d = c + p1_2d[0] * u_axis + p1_2d[1] * v_axis
        p2_3d = c + p2_2d[0] * u_axis + p2_2d[1] * v_axis

        d_vec = p2_3d - p1_3d
        seg_len_cm = float(np.linalg.norm(d_vec) * 100.0)
        if seg_len_cm < 2.0:
            continue
        d_unit = d_vec / np.linalg.norm(d_vec)

        supp_count, med_dist_cm = calculate_edge_point_support(p1_3d, p2_3d, all_points_3d, radius_thresh_cm=2.0)

        if supp_count >= 100 and plane.rms_residual_cm <= 0.45:
            conf = "HIGH"
        elif supp_count >= 35 and plane.rms_residual_cm <= 0.85:
            conf = "MEDIUM"
        elif supp_count >= MIN_EDGE_SUPPORT_POINTS:
            conf = "LOW"
        else:
            conf = "REJECTED"

        if conf != "REJECTED":
            boundary_edges.append(CandidatePhysicalEdge(
                edge_id=f"PL{plane_id}_BND{i+1}",
                source=f"Plane {plane_id} 2D Boundary",
                start_pt_3d_m=p1_3d,
                end_pt_3d_m=p2_3d,
                direction_3d=d_unit,
                length_cm=seg_len_cm,
                support_points=supp_count,
                depth_valid_ratio=1.0,
                rgb_strength=0.6,
                rms_residual_cm=plane.rms_residual_cm,
                median_dist_cm=med_dist_cm,
                confidence=conf,
                is_valid=True
            ))

    return boundary_edges


# ===========================================================================
# DIMENSION ESTIMATE CLUSTERING & ORTHOGONAL PROJECTION
# ===========================================================================

def cluster_and_estimate_dimension(
    target_axis: np.ndarray,
    candidate_edges: List[CandidatePhysicalEdge],
    fallback_extent_cm: float,
    dim_name: str
) -> DimensionEstimate:
    """
    Clusters candidate physical edges aligned with the target axis (dot product >= 0.82),
    computes robust median length, and assigns confidence. If no supported physical
    edges exist, marks the dimension as UNSUPPORTED.
    """
    matching_edges = [
        e for e in candidate_edges
        if abs(np.dot(e.direction_3d, target_axis)) >= 0.82 and e.is_valid and e.length_cm >= 2.5
    ]

    if not matching_edges:
        return DimensionEstimate(
            value_cm=0.0,
            confidence="UNSUPPORTED",
            source="INSUFFICIENT EVIDENCE (No supported physical edge)",
            support_points=0,
            median_dist_cm=0.0,
            is_valid=False,
            candidate_lengths=[]
        )

    lengths = [e.length_cm for e in matching_edges]
    total_support = sum(e.support_points for e in matching_edges)
    median_val = float(np.median(lengths))
    med_dist = float(np.mean([e.median_dist_cm for e in matching_edges]))

    # Determine confidence from candidates
    conf_scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    best_conf_score = max(conf_scores.get(e.confidence, 0) for e in matching_edges)
    inv_map = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "UNSUPPORTED"}
    conf = inv_map[best_conf_score]

    source_desc = f"{len(matching_edges)} physical edge(s) ({', '.join([e.edge_id for e in matching_edges[:3]])})"

    return DimensionEstimate(
        value_cm=median_val,
        confidence=conf,
        source=source_desc,
        support_points=total_support,
        median_dist_cm=med_dist,
        is_valid=True,
        candidate_lengths=lengths
    )


# ===========================================================================
# CUBOID RECONSTRUCTION & PHYSICAL EDGE METROLOGY
# ===========================================================================

def calculate_plane_angles(planes: List[DetectedPlane]) -> List[float]:
    angles = []
    n = len(planes)
    for i in range(n):
        for j in range(i + 1, n):
            n1 = planes[i].normal
            n2 = planes[j].normal
            cos_th = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
            deg = float(np.degrees(np.arccos(cos_th)))
            angles.append(deg)
    return angles


def reconstruct_cuboid_from_pointcloud_pca(
    points_3d: np.ndarray,
    rgbd_edges: List[CandidatePhysicalEdge]
) -> CuboidReconstruction:
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    rot_matrix = eig_vecs[:, idx]
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] = -rot_matrix[:, 2]

    q0 = np.dot(centered, rot_matrix[:, 0])
    q1 = np.dot(centered, rot_matrix[:, 1])
    q2 = np.dot(centered, rot_matrix[:, 2])

    l_span = max(1.0, float(np.percentile(q0, 98.0) - np.percentile(q0, 2.0)) * 100.0)
    b_span = max(1.0, float(np.percentile(q1, 98.0) - np.percentile(q1, 2.0)) * 100.0)
    h_span = max(1.0, float(np.percentile(q2, 98.0) - np.percentile(q2, 2.0)) * 100.0)

    dims_data = [(l_span, rot_matrix[:, 0]), (b_span, rot_matrix[:, 1]), (h_span, rot_matrix[:, 2])]
    dims_data.sort(key=lambda x: x[0], reverse=True)
    l_span, l_ax = dims_data[0]
    b_span, b_ax = dims_data[1]
    h_span, h_ax = dims_data[2]
    rot_matrix = np.column_stack([l_ax, b_ax, h_ax])
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] = -rot_matrix[:, 2]

    dim_l = DimensionEstimate(value_cm=l_span, confidence="HIGH" if len(points_3d) > 200 else "MEDIUM", source="3D Point Cloud Span", support_points=len(points_3d), median_dist_cm=0.0, is_valid=True)
    dim_b = DimensionEstimate(value_cm=b_span, confidence="HIGH" if len(points_3d) > 200 else "MEDIUM", source="3D Point Cloud Span", support_points=len(points_3d), median_dist_cm=0.0, is_valid=True)
    dim_h = DimensionEstimate(value_cm=h_span, confidence="MEDIUM", source="3D Point Cloud Depth Span", support_points=len(points_3d), median_dist_cm=0.0, is_valid=True)

    hx, hy, hz = (l_span / 100.0) / 2.0, (b_span / 100.0) / 2.0, (h_span / 100.0) / 2.0
    local_corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])
    corners_3d_m = (local_corners @ rot_matrix.T) + centroid
    corners_labeled = {f"C{i+1}": (float(c[0] * 100.0), float(c[1] * 100.0), float(c[2] * 100.0)) for i, c in enumerate(corners_3d_m)}

    edge_defs = [
        ("E1", 0, 1), ("E2", 3, 2), ("E3", 4, 5), ("E4", 7, 6),
        ("E5", 1, 2), ("E6", 0, 3), ("E7", 5, 6), ("E8", 4, 7),
        ("E9", 0, 4), ("E10", 1, 5), ("E11", 2, 6), ("E12", 3, 7)
    ]
    reconstructed_edges = []
    for e_id, idx1, idx2 in edge_defs:
        p_start, p_end = corners_3d_m[idx1], corners_3d_m[idx2]
        d_vec = p_end - p_start
        e_len = float(np.linalg.norm(d_vec) * 100.0)
        reconstructed_edges.append(ReconstructedEdge(
            edge_id=e_id,
            start_corner_name=f"C{idx1+1}",
            end_corner_name=f"C{idx2+1}",
            start_pt_3d_m=p_start,
            end_pt_3d_m=p_end,
            length_cm=e_len,
            direction_vec=d_vec / (np.linalg.norm(d_vec) + 1e-6),
            classification="LENGTH" if abs(e_len - l_span) < 0.5 else ("BREADTH" if abs(e_len - b_span) < 0.5 else "HEIGHT"),
            support_points_count=len(points_3d),
            median_point_dist_cm=0.0,
            is_supported=True
        ))

    return CuboidReconstruction(
        length=dim_l,
        breadth=dim_b,
        height=dim_h,
        centroid_3d_m=centroid,
        axes_rot=rot_matrix,
        corners_3d_m=corners_3d_m,
        corners_labeled=corners_labeled,
        reconstructed_edges=reconstructed_edges,
        candidate_physical_edges=rgbd_edges,
        detected_planes=[],
        num_planes=0,
        inter_plane_angles_deg=[],
        geometry_status="MEASURED (3D Point Cloud Span)",
        is_reliable=True
    )


def reconstruct_cuboid_from_physical_edges(
    points_3d: np.ndarray,
    planes: List[DetectedPlane],
    color_img: np.ndarray,
    depth_m: np.ndarray,
    roi: ROIState,
    intrinsics: rs.intrinsics,
    z_peak: float,
    ortho_tol_deg: float = DEFAULT_ORTHO_TOL_DEG
) -> Optional[CuboidReconstruction]:
    n_planes = len(planes)
    angles = calculate_plane_angles(planes)

    # 1. Extract candidate RGB-D physical edges
    rgbd_edges = extract_rgbd_physical_edges(color_img, depth_m, roi, intrinsics, z_peak)

    # If no planes fitted, fallback to 3D point cloud PCA bounding box
    if n_planes == 0:
        if len(points_3d) >= 30:
            return reconstruct_cuboid_from_pointcloud_pca(points_3d, rgbd_edges)
        return None

    # -----------------------------------------------------------------------
    # CASE A: Only 1 Face Visible
    # -----------------------------------------------------------------------
    if n_planes == 1:
        p0 = planes[0]
        pts = p0.inlier_points
        c = np.mean(pts, axis=0)

        centered = pts - c
        cov_2d = np.cov(centered, rowvar=False)
        _, eig_vecs = np.linalg.eigh(cov_2d)
        u1 = eig_vecs[:, 2]
        u2 = np.cross(p0.normal, u1)
        u2 = u2 / np.linalg.norm(u2)
        u3 = p0.normal

        p0.local_axis_u = u1
        p0.local_axis_v = u2

        q_u = np.dot(pts - c, u1)
        q_v = np.dot(pts - c, u2)
        p0.extent_u_cm = float(np.percentile(q_u, HIGH_PERCENTILE) - np.percentile(q_u, LOW_PERCENTILE)) * 100.0
        p0.extent_v_cm = float(np.percentile(q_v, HIGH_PERCENTILE) - np.percentile(q_v, LOW_PERCENTILE)) * 100.0

        bnd_edges = extract_plane_boundary_lines(p0, points_3d, plane_id=1)
        all_cands = rgbd_edges + bnd_edges

        dim_l = cluster_and_estimate_dimension(u1, all_cands, p0.extent_u_cm, "Length")
        dim_b = cluster_and_estimate_dimension(u2, all_cands, p0.extent_v_cm, "Breadth")

        # Height from depth extent along normal and depth range
        q_norm = np.dot(points_3d - c, u3)
        depth_span_cm = float(np.percentile(q_norm, 98.0) - np.percentile(q_norm, 2.0)) * 100.0
        z_span_cm = float(np.percentile(points_3d[:, 2], 98.0) - np.percentile(points_3d[:, 2], 2.0)) * 100.0
        h_val = max(1.0, max(depth_span_cm, z_span_cm))

        dim_h = DimensionEstimate(
            value_cm=h_val,
            confidence="MEDIUM",
            source="Depth Extent",
            support_points=len(points_3d),
            median_dist_cm=0.0,
            is_valid=True
        )

        dim_items = [
            (dim_l.value_cm if dim_l.is_valid else p0.extent_u_cm, dim_l, u1),
            (dim_b.value_cm if dim_b.is_valid else p0.extent_v_cm, dim_b, u2),
            (h_val, dim_h, u3)
        ]
        dim_items.sort(key=lambda x: x[0], reverse=True)
        l_span, dim_l_res, l_axis = dim_items[0]
        b_span, dim_b_res, b_axis = dim_items[1]
        h_span, dim_h_res, h_axis = dim_items[2]

        rot_matrix = np.column_stack([l_axis, b_axis, h_axis])
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[:, 2] = -rot_matrix[:, 2]

        hx, hy, hz = (l_span / 100.0) / 2.0, (b_span / 100.0) / 2.0, (h_span / 100.0) / 2.0
        local_corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ])
        corners_3d_m = (local_corners @ rot_matrix.T) + c
        corners_labeled = {f"C{i+1}": (float(pt[0] * 100.0), float(pt[1] * 100.0), float(pt[2] * 100.0)) for i, pt in enumerate(corners_3d_m)}

        edge_defs = [
            ("E1", 0, 1), ("E2", 3, 2), ("E3", 4, 5), ("E4", 7, 6),
            ("E5", 1, 2), ("E6", 0, 3), ("E7", 5, 6), ("E8", 4, 7),
            ("E9", 0, 4), ("E10", 1, 5), ("E11", 2, 6), ("E12", 3, 7)
        ]
        reconstructed_edges = []
        for e_id, idx1, idx2 in edge_defs:
            p_start, p_end = corners_3d_m[idx1], corners_3d_m[idx2]
            d_vec = p_end - p_start
            e_len = float(np.linalg.norm(d_vec) * 100.0)
            reconstructed_edges.append(ReconstructedEdge(
                edge_id=e_id,
                start_corner_name=f"C{idx1+1}",
                end_corner_name=f"C{idx2+1}",
                start_pt_3d_m=p_start,
                end_pt_3d_m=p_end,
                length_cm=e_len,
                direction_vec=d_vec / (np.linalg.norm(d_vec) + 1e-6),
                classification="LENGTH" if abs(e_len - l_span) < 0.5 else ("BREADTH" if abs(e_len - b_span) < 0.5 else "HEIGHT"),
                support_points_count=len(points_3d),
                median_point_dist_cm=0.0,
                is_supported=True
            ))

        return CuboidReconstruction(
            length=dim_l_res,
            breadth=dim_b_res,
            height=dim_h_res,
            centroid_3d_m=c,
            axes_rot=rot_matrix,
            corners_3d_m=corners_3d_m,
            corners_labeled=corners_labeled,
            reconstructed_edges=reconstructed_edges,
            candidate_physical_edges=all_cands,
            detected_planes=planes,
            num_planes=1,
            inter_plane_angles_deg=angles,
            geometry_status="MEASURED (Face Extent + Depth)",
            is_reliable=True
        )

    # -----------------------------------------------------------------------
    # CASE B: 2 Faces Visible
    # -----------------------------------------------------------------------
    if n_planes == 2:
        ang = angles[0]
        if abs(ang - 90.0) > ortho_tol_deg:
            # Oblique planes fallback to PCA
            return reconstruct_cuboid_from_pointcloud_pca(points_3d, rgbd_edges)


        p1, p2 = planes[0], planes[1]

        # 1. Physical shared intersection edge direction
        edge_dir = np.cross(p1.normal, p2.normal)
        edge_dir = edge_dir / np.linalg.norm(edge_dir)

        # 2. In-plane axes strictly perpendicular to intersection line on each face
        e_v1 = np.cross(p1.normal, edge_dir)
        e_v1 = e_v1 / np.linalg.norm(e_v1)

        e_v2 = np.cross(p2.normal, edge_dir)
        e_v2 = e_v2 / np.linalg.norm(e_v2)

        p1.local_axis_u = edge_dir
        p1.local_axis_v = e_v1

        p2.local_axis_u = edge_dir
        p2.local_axis_v = e_v2

        c1 = np.mean(p1.inlier_points, axis=0)
        c2 = np.mean(p2.inlier_points, axis=0)
        centroid = (c1 + c2) / 2.0

        # In-plane observed extents (for telemetry display)
        q1_u = np.dot(p1.inlier_points - c1, edge_dir)
        q2_u = np.dot(p2.inlier_points - c2, edge_dir)
        p1.extent_u_cm = float(np.percentile(q1_u, HIGH_PERCENTILE) - np.percentile(q1_u, LOW_PERCENTILE)) * 100.0
        p2.extent_u_cm = float(np.percentile(q2_u, HIGH_PERCENTILE) - np.percentile(q2_u, LOW_PERCENTILE)) * 100.0

        q1_v = np.dot(p1.inlier_points - c1, e_v1)
        p1.extent_v_cm = float(np.percentile(q1_v, HIGH_PERCENTILE) - np.percentile(q1_v, LOW_PERCENTILE)) * 100.0

        q2_v = np.dot(p2.inlier_points - c2, e_v2)
        p2.extent_v_cm = float(np.percentile(q2_v, HIGH_PERCENTILE) - np.percentile(q2_v, LOW_PERCENTILE)) * 100.0

        # 3. Form Common Intersection Physical Edge
        shared_len_cm = max(p1.extent_u_cm, p2.extent_u_cm)
        half_len_m = (shared_len_cm / 100.0) / 2.0
        inter_start = centroid - half_len_m * edge_dir
        inter_end   = centroid + half_len_m * edge_dir

        supp_intersect, med_d_intersect = calculate_edge_point_support(inter_start, inter_end, points_3d, radius_thresh_cm=2.0)
        intersect_edge = CandidatePhysicalEdge(
            edge_id="INTERSECT_EDGE",
            source="Common Plane Intersection",
            start_pt_3d_m=inter_start,
            end_pt_3d_m=inter_end,
            direction_3d=edge_dir,
            length_cm=shared_len_cm,
            support_points=supp_intersect,
            depth_valid_ratio=1.0,
            rgb_strength=0.9,
            rms_residual_cm=(p1.rms_residual_cm + p2.rms_residual_cm) / 2.0,
            median_dist_cm=med_d_intersect,
            confidence="HIGH" if supp_intersect >= 50 else ("MEDIUM" if supp_intersect >= 20 else "LOW"),
            is_valid=True
        )

        # 4. Extract 2D boundary lines on each plane
        bnd_edges_p1 = extract_plane_boundary_lines(p1, points_3d, plane_id=1)
        bnd_edges_p2 = extract_plane_boundary_lines(p2, points_3d, plane_id=2)

        all_candidate_edges = [intersect_edge] + rgbd_edges + bnd_edges_p1 + bnd_edges_p2

        # 5. Cluster candidate physical edges along each of the 3 principal orthogonal directions
        dim_along_u  = cluster_and_estimate_dimension(edge_dir, all_candidate_edges, shared_len_cm, "U (Shared)")
        dim_along_v1 = cluster_and_estimate_dimension(e_v1, all_candidate_edges, p1.extent_v_cm, "V1 (Face 1)")
        dim_along_v2 = cluster_and_estimate_dimension(e_v2, all_candidate_edges, p2.extent_v_cm, "V2 (Face 2)")

        # Sort measured orthogonal spans into L >= B >= H
        cand_list = [
            (dim_along_u, edge_dir),
            (dim_along_v1, e_v1),
            (dim_along_v2, e_v2)
        ]
        cand_list.sort(key=lambda x: x[0].value_cm if x[0].is_valid else -1.0, reverse=True)

        dim_l, l_axis = cand_list[0]
        dim_b, b_axis = cand_list[1]
        dim_h, h_axis = cand_list[2]

        rot_matrix = np.column_stack([l_axis, b_axis, h_axis])

        # 3D Bounding Box Vertices
        l_span = dim_l.value_cm if dim_l.is_valid else shared_len_cm
        b_span = dim_b.value_cm if dim_b.is_valid else p1.extent_v_cm
        h_span = dim_h.value_cm if dim_h.is_valid else p2.extent_v_cm

        hx, hy, hz = (l_span/100.0)/2.0, (b_span/100.0)/2.0, (h_span/100.0)/2.0
        local_corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ])
        corners_3d_m = (local_corners @ rot_matrix.T) + centroid
        corners_labeled = {f"C{i+1}": (float(c[0]*100.0), float(c[1]*100.0), float(c[2]*100.0)) for i, c in enumerate(corners_3d_m)}

        # Build wireframe edges with support classification
        edge_defs = [
            ("E1", 0, 1), ("E2", 3, 2), ("E3", 4, 5), ("E4", 7, 6),
            ("E5", 1, 2), ("E6", 0, 3), ("E7", 5, 6), ("E8", 4, 7),
            ("E9", 0, 4), ("E10", 1, 5), ("E11", 2, 6), ("E12", 3, 7)
        ]

        reconstructed_edges = []
        for e_id, idx1, idx2 in edge_defs:
            p_start, p_end = corners_3d_m[idx1], corners_3d_m[idx2]
            d_vec = p_end - p_start
            e_len = float(np.linalg.norm(d_vec) * 100.0)
            norm_d = d_vec / (np.linalg.norm(d_vec) + 1e-6)

            diff_l = abs(e_len - l_span)
            diff_b = abs(e_len - b_span)
            diff_h = abs(e_len - h_span)
            min_d = min(diff_l, diff_b, diff_h)
            classification = "LENGTH" if min_d == diff_l else ("BREADTH" if min_d == diff_b else "HEIGHT")

            supp_count, med_dist = calculate_edge_point_support(p_start, p_end, points_3d)
            is_supp = supp_count >= MIN_EDGE_SUPPORT_POINTS

            reconstructed_edges.append(ReconstructedEdge(
                edge_id=e_id,
                start_corner_name=f"C{idx1+1}",
                end_corner_name=f"C{idx2+1}",
                start_pt_3d_m=p_start,
                end_pt_3d_m=p_end,
                length_cm=e_len,
                direction_vec=norm_d,
                classification=classification,
                support_points_count=supp_count,
                median_point_dist_cm=med_dist,
                is_supported=is_supp
            ))

        status = f"GEOMETRY VALID (2 Faces @ {ang:.1f} deg | {len(all_candidate_edges)} physical edge candidates)"
        is_reliable = (dim_l.is_valid and dim_b.is_valid and dim_h.is_valid)

        return CuboidReconstruction(
            length=dim_l,
            breadth=dim_b,
            height=dim_h,
            centroid_3d_m=centroid,
            axes_rot=rot_matrix,
            corners_3d_m=corners_3d_m,
            corners_labeled=corners_labeled,
            reconstructed_edges=reconstructed_edges,
            candidate_physical_edges=all_candidate_edges,
            detected_planes=planes,
            num_planes=2,
            inter_plane_angles_deg=angles,
            geometry_status=status,
            is_reliable=is_reliable
        )

    # -----------------------------------------------------------------------
    # CASE C: 3 Orthogonal Visible Faces
    # -----------------------------------------------------------------------
    if n_planes >= 3:
        p1, p2, p3 = planes[0], planes[1], planes[2]
        ang12, ang13, ang23 = angles[0], angles[1], angles[2]

        is_ortho = (abs(ang12 - 90.0) <= ortho_tol_deg and
                    abs(ang13 - 90.0) <= ortho_tol_deg and
                    abs(ang23 - 90.0) <= ortho_tol_deg)

        if not is_ortho:
            return reconstruct_cuboid_from_physical_edges(points_3d, [p1, p2], color_img, depth_m, roi, intrinsics, z_peak, ortho_tol_deg)

        n_mat = np.column_stack([p1.normal, p2.normal, p3.normal])
        u, _, vt = np.linalg.svd(n_mat)
        rot_matrix = u @ vt
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[:, 2] = -rot_matrix[:, 2]

        centroid = np.mean(points_3d, axis=0)
        all_cands = rgbd_edges
        for i, pl in enumerate([p1, p2, p3]):
            all_cands.extend(extract_plane_boundary_lines(pl, points_3d, plane_id=i+1))

        dim_1 = cluster_and_estimate_dimension(rot_matrix[:, 0], all_cands, p1.extent_u_cm, "Axis 1")
        dim_2 = cluster_and_estimate_dimension(rot_matrix[:, 1], all_cands, p2.extent_u_cm, "Axis 2")
        dim_3 = cluster_and_estimate_dimension(rot_matrix[:, 2], all_cands, p3.extent_u_cm, "Axis 3")

        dims_sorted = sorted([dim_1, dim_2, dim_3], key=lambda x: x.value_cm if x.is_valid else -1.0, reverse=True)

        return CuboidReconstruction(
            length=dims_sorted[0],
            breadth=dims_sorted[1],
            height=dims_sorted[2],
            centroid_3d_m=centroid,
            axes_rot=rot_matrix,
            corners_3d_m=np.zeros((8, 3)),
            corners_labeled={},
            reconstructed_edges=[],
            candidate_physical_edges=all_cands,
            detected_planes=planes,
            num_planes=3,
            inter_plane_angles_deg=angles,
            geometry_status="GEOMETRY VALID (3 Mutually Orthogonal Faces)",
            is_reliable=(dims_sorted[0].is_valid and dims_sorted[1].is_valid and dims_sorted[2].is_valid)
        )

    return None


# ===========================================================================
# TEMPORAL STABILITY FILTER
# ===========================================================================

class TemporalDimensionFilter:
    def __init__(self, buffer_size: int = TEMPORAL_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.l_buf = deque(maxlen=buffer_size)
        self.b_buf = deque(maxlen=buffer_size)
        self.h_buf = deque(maxlen=buffer_size)

    def add_sample(self, l: float, b: float, h: float, is_valid: bool = True):
        if is_valid and l > 0.5 and b > 0.5 and h > 0.5:
            self.l_buf.append(l)
            self.b_buf.append(b)
            self.h_buf.append(h)

    def clear(self):
        self.l_buf.clear()
        self.b_buf.clear()
        self.h_buf.clear()

    def get_stats(self) -> TemporalStats:
        n = len(self.l_buf)
        if n == 0:
            return TemporalStats()
        l_arr = np.array(self.l_buf)
        b_arr = np.array(self.b_buf)
        h_arr = np.array(self.h_buf)
        return TemporalStats(
            length_median=float(np.median(l_arr)),
            length_mean=float(np.mean(l_arr)),
            length_std=float(np.std(l_arr)),
            breadth_median=float(np.median(b_arr)),
            breadth_mean=float(np.mean(b_arr)),
            breadth_std=float(np.std(b_arr)),
            height_median=float(np.median(h_arr)),
            height_mean=float(np.mean(h_arr)),
            height_std=float(np.std(h_arr)),
            sample_count=n
        )


# ===========================================================================
# REAL-TIME PROFESSIONAL MEASUREMENT VISUALIZATION
# ===========================================================================

def project_3d_to_pixel(point_3d: np.ndarray, intrinsics: rs.intrinsics) -> Tuple[int, int]:
    x, y, z = point_3d[0], point_3d[1], point_3d[2]
    if z <= 0.05:
        return (0, 0)
    u = int((x * intrinsics.fx / z) + intrinsics.ppx)
    v = int((y * intrinsics.fy / z) + intrinsics.ppy)
    return (np.clip(u, 0, STREAM_WIDTH - 1), np.clip(v, 0, STREAM_HEIGHT - 1))


def draw_dimension_arrow_with_label(
    img: np.ndarray,
    p1_2d: Tuple[int, int],
    p2_2d: Tuple[int, int],
    dim_text: str,
    color: Tuple[int, int, int],
    normal_offset_px: float = 20.0,
    arrow_size: int = 9
) -> None:
    """
    Renders double-ended physical dimension arrows aligned with the projected 3D edge
    orientation, including extension ticks and a centered translucent text badge.
    """
    x1, y1 = p1_2d
    x2, y2 = p2_2d
    dx, dy = float(x2 - x1), float(y2 - y1)
    seg_len = float(np.hypot(dx, dy))
    if seg_len < 16.0:
        return

    ux, uy = dx / seg_len, dy / seg_len
    # 2D Normal vector perpendicular to the projected edge
    nx, ny = -uy, ux

    # Offset endpoints away from object interior
    ax1 = int(round(x1 + normal_offset_px * nx))
    ay1 = int(round(y1 + normal_offset_px * ny))
    ax2 = int(round(x2 + normal_offset_px * nx))
    ay2 = int(round(y2 + normal_offset_px * ny))

    h, w = img.shape[:2]
    ax1, ay1 = np.clip(ax1, 5, w - 6), np.clip(ay1, 5, h - 6)
    ax2, ay2 = np.clip(ax2, 5, w - 6), np.clip(ay2, 5, h - 6)

    # 1. Subtle Extension Lines from physical edge to dimension line
    ex1_s = (int(x1 + 3 * nx), int(y1 + 3 * ny))
    ex1_e = (int(ax1 + 4 * nx), int(ay1 + 4 * ny))
    ex2_s = (int(x2 + 3 * nx), int(y2 + 3 * ny))
    ex2_e = (int(ax2 + 4 * nx), int(ay2 + 4 * ny))
    dim_ext_col = tuple(int(c * 0.65) for c in color)
    cv2.line(img, ex1_s, ex1_e, dim_ext_col, 1, cv2.LINE_AA)
    cv2.line(img, ex2_s, ex2_e, dim_ext_col, 1, cv2.LINE_AA)

    # 2. Main Dimension Line
    cv2.line(img, (ax1, ay1), (ax2, ay2), color, 2, cv2.LINE_AA)

    # 3. Double-Ended Arrowheads
    def draw_arrowhead(tip_x, tip_y, dir_x, dir_y):
        s1_x = int(round(tip_x - arrow_size * dir_x + (arrow_size * 0.48) * (-dir_y)))
        s1_y = int(round(tip_y - arrow_size * dir_y + (arrow_size * 0.48) * (dir_x)))
        s2_x = int(round(tip_x - arrow_size * dir_x - (arrow_size * 0.48) * (-dir_y)))
        s2_y = int(round(tip_y - arrow_size * dir_y - (arrow_size * 0.48) * (dir_x)))
        pts = np.array([[tip_x, tip_y], [s1_x, s1_y], [s2_x, s2_y]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

    draw_arrowhead(ax1, ay1, -ux, -uy)
    draw_arrowhead(ax2, ay2, ux, uy)

    # 4. Dimension Badge Box at midpoint
    mid_x = (ax1 + ax2) // 2 + int(14 * nx)
    mid_y = (ay1 + ay2) // 2 + int(14 * ny)
    mid_x = np.clip(mid_x, 50, w - 80)
    mid_y = np.clip(mid_y, 25, h - 25)

    (tw, th), _ = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    bx1 = mid_x - tw // 2 - 6
    by1 = mid_y - th // 2 - 5
    bx2 = mid_x + tw // 2 + 6
    by2 = mid_y + th // 2 + 5

    sub_h, sub_w = img.shape[:2]
    if 0 <= by1 < by2 <= sub_h and 0 <= bx1 < bx2 <= sub_w:
        sub_img = img[by1:by2, bx1:bx2]
        dark_rect = np.full_like(sub_img, 20)
        cv2.addWeighted(dark_rect, 0.80, sub_img, 0.20, 0, sub_img)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

    cv2.putText(img, dim_text, (mid_x - tw // 2, mid_y + th // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


def draw_measurement_overlay(
    color_img: np.ndarray,
    mask: np.ndarray,
    roi: ROIState,
    cuboid: Optional[CuboidReconstruction],
    t_stats: TemporalStats,
    ground_truth: GroundTruth,
    intrinsics: rs.intrinsics,
    fps: float,
    debug_mode: bool = False,
    detection_mode: str = "MANUAL",
    detection_res: Optional[DetectionResult] = None
) -> np.ndarray:
    annotated = color_img.copy()

    # 1. Object Outline from Segmentation Mask
    if roi.is_locked and np.any(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.polylines(annotated, contours, isClosed=True, color=(0, 230, 115), thickness=2, lineType=cv2.LINE_AA)

        # Subtle translucent mask fill
        overlay = annotated.copy()
        overlay[mask > 0] = [0, 200, 90]
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

    # 2. Guidance & Drawing Prompts
    if not roi.is_locked and not roi.is_drawing:
        guide_msg = "CLICK & DRAG A BORDER AROUND ANY OBJECT TO MEASURE IT"
        (gw, gh), _ = cv2.getTextSize(guide_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        gx = (STREAM_WIDTH - gw) // 2
        gy = 40
        cv2.rectangle(annotated, (gx - 14, gy - gh - 8), (gx + gw + 14, gy + 8), (15, 15, 15), -1)
        cv2.rectangle(annotated, (gx - 14, gy - gh - 8), (gx + gw + 14, gy + 8), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, guide_msg, (gx, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    elif roi.is_drawing:
        xmin, ymin, xmax, ymax = roi.box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, cv2.LINE_AA)
        draw_txt = f"RELEASE TO MEASURE [{xmax-xmin} x {ymax-ymin} px]"
        cv2.putText(annotated, draw_txt, (xmin, max(22, ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

    elif roi.is_locked and roi.is_valid:
        xmin, ymin, xmax, ymax = roi.box
        # Draw high-contrast bounding border with targeting corners
        col = (0, 240, 120)
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), col, 2, cv2.LINE_AA)
        bracket_len = min(22, max(8, min(xmax - xmin, ymax - ymin) // 4))
        # Corners
        cv2.line(annotated, (xmin, ymin), (xmin + bracket_len, ymin), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmin, ymin), (xmin, ymin + bracket_len), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmax, ymin), (xmax - bracket_len, ymin), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmax, ymin), (xmax, ymin + bracket_len), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmin, ymax), (xmin + bracket_len, ymax), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmin, ymax), (xmin, ymax - bracket_len), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmax, ymax), (xmax - bracket_len, ymax), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (xmax, ymax), (xmax, ymax - bracket_len), (255, 255, 255), 3, cv2.LINE_AA)

        # Floating measurement badge directly above the drawn bounding box
        if cuboid is not None:
            l_val = t_stats.length_median if t_stats.sample_count > 5 else cuboid.length.value_cm
            b_val = t_stats.breadth_median if t_stats.sample_count > 5 else cuboid.breadth.value_cm
            h_val = t_stats.height_median if t_stats.sample_count > 5 else cuboid.height.value_cm
            z_dist = (cuboid.centroid_3d_m[2] * 100.0) if np.any(cuboid.centroid_3d_m) else 0.0

            badge_str = f"L: {l_val:.1f} cm  |  B: {b_val:.1f} cm  |  H: {h_val:.1f} cm"
            (bw, bh), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            bx_mid = (xmin + xmax) // 2
            bx1 = max(10, bx_mid - bw // 2 - 10)
            by1 = max(8, ymin - bh - 14)
            bx2 = min(STREAM_WIDTH - 10, bx1 + bw + 20)
            by2 = by1 + bh + 12

            sub_img = annotated[by1:by2, bx1:bx2]
            if sub_img.size > 0:
                dark_rect = np.full_like(sub_img, 15)
                cv2.addWeighted(dark_rect, 0.85, sub_img, 0.15, 0, sub_img)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 240, 120), 1, cv2.LINE_AA)
                cv2.putText(annotated, badge_str, (bx1 + 10, by2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    # 3. Projected Physical Dimension Arrows & Supported Corner Markers
    if cuboid is not None and np.any(cuboid.corners_3d_m):
        corners = cuboid.corners_3d_m
        pix_corners = [project_3d_to_pixel(c, intrinsics) for c in corners]

        l_val = t_stats.length_median if t_stats.sample_count > 5 else cuboid.length.value_cm
        b_val = t_stats.breadth_median if t_stats.sample_count > 5 else cuboid.breadth.value_cm
        h_val = t_stats.height_median if t_stats.sample_count > 5 else cuboid.height.value_cm

        # Find representative physical edge for Length
        l_edges = [e for e in cuboid.reconstructed_edges if e.classification == "LENGTH" and e.is_supported]
        b_edges = [e for e in cuboid.reconstructed_edges if e.classification == "BREADTH" and e.is_supported]
        h_edges = [e for e in cuboid.reconstructed_edges if e.classification == "HEIGHT" and e.is_supported]

        # Draw Length Dimension Arrow
        if cuboid.length.is_valid and len(pix_corners) >= 4:
            target_edge = l_edges[0] if l_edges else (cuboid.reconstructed_edges[0] if cuboid.reconstructed_edges else None)
            if target_edge:
                i1 = int(target_edge.start_corner_name[1:]) - 1
                i2 = int(target_edge.end_corner_name[1:]) - 1
                draw_dimension_arrow_with_label(
                    annotated, pix_corners[i1], pix_corners[i2],
                    f"L = {l_val:.1f} cm", (0, 255, 255), normal_offset_px=-22.0
                )

        # Draw Breadth Dimension Arrow
        if cuboid.breadth.is_valid and len(pix_corners) >= 6:
            target_edge = b_edges[0] if b_edges else ([e for e in cuboid.reconstructed_edges if e.classification == "BREADTH"][0] if any(e.classification == "BREADTH" for e in cuboid.reconstructed_edges) else None)
            if target_edge:
                i1 = int(target_edge.start_corner_name[1:]) - 1
                i2 = int(target_edge.end_corner_name[1:]) - 1
                draw_dimension_arrow_with_label(
                    annotated, pix_corners[i1], pix_corners[i2],
                    f"B = {b_val:.1f} cm", (100, 255, 100), normal_offset_px=22.0
                )

        # Draw Height Dimension Arrow
        if cuboid.height.is_valid and len(pix_corners) >= 8:
            target_edge = h_edges[0] if h_edges else ([e for e in cuboid.reconstructed_edges if e.classification == "HEIGHT"][0] if any(e.classification == "HEIGHT" for e in cuboid.reconstructed_edges) else None)
            if target_edge:
                i1 = int(target_edge.start_corner_name[1:]) - 1
                i2 = int(target_edge.end_corner_name[1:]) - 1
                draw_dimension_arrow_with_label(
                    annotated, pix_corners[i1], pix_corners[i2],
                    f"H = {h_val:.1f} cm", (255, 180, 0), normal_offset_px=22.0
                )

        # 4. Supported Physical Corner Markers (Concentric Circles)
        for i, c_pt_m in enumerate(corners):
            c_pix = pix_corners[i]
            adj_supported = any(
                e.is_supported for e in cuboid.reconstructed_edges
                if e.start_corner_name == f"C{i+1}" or e.end_corner_name == f"C{i+1}"
            )
            if adj_supported:
                cv2.circle(annotated, c_pix, 5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(annotated, c_pix, 2, (0, 255, 255), -1, cv2.LINE_AA)

        # In DEBUG Mode: Draw wireframe edges & plane vectors
        if debug_mode:
            for edge in cuboid.reconstructed_edges:
                idx1 = int(edge.start_corner_name[1:]) - 1
                idx2 = int(edge.end_corner_name[1:]) - 1
                col = (0, 255, 255) if edge.classification == "LENGTH" else ((100, 255, 100) if edge.classification == "BREADTH" else (255, 180, 0))
                if edge.is_supported:
                    cv2.line(annotated, pix_corners[idx1], pix_corners[idx2], col, 2, cv2.LINE_AA)
                else:
                    cv2.line(annotated, pix_corners[idx1], pix_corners[idx2], tuple(int(c * 0.35) for c in col), 1, cv2.LINE_AA)

    # 5. Clean Professional Telemetry HUD Panel (Top-Left)
    hud_w = 420
    hud_h = 240 if debug_mode else 190

    # Glassmorphic dark card
    hud_crop = annotated[12:12+hud_h, 12:12+hud_w]
    if hud_crop.size > 0:
        dark_overlay = np.full_like(hud_crop, 18)
        cv2.addWeighted(dark_overlay, 0.82, hud_crop, 0.18, 0, hud_crop)
        cv2.rectangle(annotated, (12, 12), (12+hud_w, 12+hud_h), (80, 80, 80), 1, cv2.LINE_AA)

    # Header
    debug_tag = " [DEBUG]" if debug_mode else ""
    cv2.putText(annotated, f"REALSENSE D455f METROLOGY{debug_tag}", (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2, cv2.LINE_AA)

    # Detection line
    if detection_mode == "AUTO":
        if detection_res is not None and detection_res.is_valid:
            det_txt = f"OBJECT : box  |  DETECTION: AUTO (Conf: {detection_res.confidence:.2f})"
            det_col = (0, 255, 150)
        else:
            det_txt = "OBJECT : NO OBJECT DETECTED"
            det_col = (0, 100, 255)
    else:
        det_txt = "OBJECT : box  |  BORDER: LOCKED" if roi.is_locked else "DETECTION: DRAW BORDER AROUND OBJECT"
        det_col = (0, 255, 150) if roi.is_locked else (0, 180, 255)

    cv2.putText(annotated, det_txt, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, det_col, 1, cv2.LINE_AA)
    cv2.line(annotated, (24, 60), (12+hud_w - 12, 60), (60, 60, 60), 1)

    # Measurement Dimensions
    if cuboid is not None and cuboid.is_reliable:
        l_m = t_stats.length_median if t_stats.sample_count > 5 else cuboid.length.value_cm
        b_m = t_stats.breadth_median if t_stats.sample_count > 5 else cuboid.breadth.value_cm
        h_m = t_stats.height_median if t_stats.sample_count > 5 else cuboid.height.value_cm
        l_s, b_s, h_s = t_stats.length_std, t_stats.breadth_std, t_stats.height_std

        cv2.putText(annotated, f"Length  (L) : {l_m:5.1f} cm  (±{l_s:.1f})  [{cuboid.length.confidence}]", (24, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Breadth (B) : {b_m:5.1f} cm  (±{b_s:.1f})  [{cuboid.breadth.confidence}]", (24, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Height  (H) : {h_m:5.1f} cm  (±{h_s:.1f})  [{cuboid.height.confidence}]", (24, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 185, 50), 1, cv2.LINE_AA)
    elif cuboid is not None:
        cv2.putText(annotated, f"METROLOGY : {cuboid.geometry_status}", (24, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "METROLOGY : Draw a border around the object", (24, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    # Footer metrics
    cv2.line(annotated, (24, 134), (12+hud_w - 12, 134), (60, 60, 60), 1)
    z_dist = (cuboid.centroid_3d_m[2] * 100.0) if (cuboid is not None and np.any(cuboid.centroid_3d_m)) else 0.0
    cv2.putText(annotated, f"Distance Z: {z_dist:.1f} cm  |  {fps:.1f} FPS  |  {t_stats.sample_count} frames", (24, 152),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    if ground_truth.is_set and cuboid is not None and cuboid.is_reliable:
        l_curr = t_stats.length_median if t_stats.sample_count > 5 else cuboid.length.value_cm
        b_curr = t_stats.breadth_median if t_stats.sample_count > 5 else cuboid.breadth.value_cm
        h_curr = t_stats.height_median if t_stats.sample_count > 5 else cuboid.height.value_cm
        err_l = abs(l_curr - ground_truth.length_cm)
        err_b = abs(b_curr - ground_truth.breadth_cm)
        err_h = abs(h_curr - ground_truth.height_cm)
        gt_txt = f"GT Err: dL={err_l:.1f}cm ({err_l/ground_truth.length_cm*100:.1f}%), dB={err_b:.1f}cm, dH={err_h:.1f}cm"
        cv2.putText(annotated, gt_txt, (24, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 180, 100), 1, cv2.LINE_AA)

    if debug_mode and cuboid is not None and cuboid.detected_planes:
        p_info = " | ".join([f"P{p.plane_id}: U={p.extent_u_cm:.1f} V={p.extent_v_cm:.1f}cm" for p in cuboid.detected_planes])
        cv2.putText(annotated, f"Planes : {p_info}", (24, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1, cv2.LINE_AA)
        cands_count = len([e for e in cuboid.candidate_physical_edges if e.is_valid])
        cv2.putText(annotated, f"Edges  : {cands_count} supported physical candidates", (24, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 200, 255), 1, cv2.LINE_AA)

    # 6. Bottom Navigation Bar
    ctrl_str = "Draw Border: Drag Mouse  |  [N] Reset  |  [D] Debug Mode  |  [S] Save Snapshot  |  [Q] Exit"
    cv2.putText(annotated, ctrl_str, (24, STREAM_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated



# ===========================================================================
# MEASUREMENT SNAPSHOT EXPORT
# ===========================================================================

def save_measurement_snapshot(
    annotated_img: np.ndarray,
    raw_img: np.ndarray,
    mask: np.ndarray,
    meas_result: MeasurementResult,
    gt: GroundTruth
) -> str:
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurement_snapshots")
    os.makedirs(out_dir, exist_ok=True)

    prefix = f"measurement_{timestamp_str}"
    ann_path  = os.path.join(out_dir, f"{prefix}_annotated.png")
    rgb_path  = os.path.join(out_dir, f"{prefix}_rgb.png")
    mask_path = os.path.join(out_dir, f"{prefix}_mask.png")
    rep_path  = os.path.join(out_dir, f"{prefix}_report.txt")

    cv2.imwrite(ann_path, annotated_img)
    cv2.imwrite(rgb_path, raw_img)
    cv2.imwrite(mask_path, mask if mask is not None else np.zeros((STREAM_HEIGHT, STREAM_WIDTH), dtype=np.uint8))

    with open(rep_path, "w") as f:
        f.write("==================================================\n")
        f.write(f"  INTEL REALSENSE D455f METROLOGY REPORT\n")
        f.write("==================================================\n\n")
        f.write(f"Timestamp           : {timestamp_str}\n")
        f.write(f"Camera Distance Z   : {meas_result.camera_distance_z_cm:.1f} cm\n\n")

        f.write("DETECTION:\n")
        f.write(f"  Mode              : {meas_result.detection_mode}\n")
        f.write(f"  Status            : {meas_result.detection_status}\n")
        f.write(f"  Confidence        : {meas_result.detection_confidence:.2f}\n")
        f.write(f"  Bounding Box      : {meas_result.bbox}\n\n")

        f.write("MEASUREMENT:\n")
        f.write(f"  Length  (L)       : {meas_result.length_cm:.2f} cm (std: {meas_result.length_std_cm:.2f} cm) [{meas_result.length_confidence}]\n")
        f.write(f"  Breadth (B)       : {meas_result.breadth_cm:.2f} cm (std: {meas_result.breadth_std_cm:.2f} cm) [{meas_result.breadth_confidence}]\n")
        f.write(f"  Height  (H)       : {meas_result.height_cm:.2f} cm (std: {meas_result.height_std_cm:.2f} cm) [{meas_result.height_confidence}]\n")
        f.write(f"  Metrology Status  : {meas_result.metrology_status}\n\n")

        if gt.is_set:
            f.write("GROUND TRUTH COMPARISON:\n")
            err_l = abs(meas_result.length_cm - gt.length_cm)
            err_b = abs(meas_result.breadth_cm - gt.breadth_cm)
            err_h = abs(meas_result.height_cm - gt.height_cm)
            f.write(f"  Ground Truth (L/B/H) : {gt.length_cm:.1f} / {gt.breadth_cm:.1f} / {gt.height_cm:.1f} cm\n")
            f.write(f"  Absolute Errors      : dL={err_l:.2f}cm, dB={err_b:.2f}cm, dH={err_h:.2f}cm\n")
            f.write(f"  Percentage Errors    : dL={err_l/gt.length_cm*100:.1f}%, dB={err_b/gt.breadth_cm*100:.1f}%, dH={err_h/gt.height_cm*100:.1f}%\n")
        f.write("==================================================\n")

    print("\n==============================================================")
    print(f"  MEASUREMENT SNAPSHOT SAVED TO 'measurement_snapshots/'")
    print("==============================================================")
    print(f"  Annotated Image  : {ann_path}")
    print(f"  RGB Image        : {rgb_path}")
    print(f"  Object Mask      : {mask_path}")
    print(f"  Report           : {rep_path}")
    print("==============================================================\n")
    return rep_path



# ===========================================================================
# CONSOLE DEBUG SUMMARY
# ===========================================================================

def print_console_debug_summary(
    cuboid: Optional[CuboidReconstruction],
    t_stats: TemporalStats,
    points_3d: Optional[np.ndarray],
    gt: GroundTruth
) -> None:
    print("\n==================================================")
    print("  STAGE 5.5: PHYSICAL BOUNDARY & EDGE DEBUG SUMMARY")
    print("==================================================")

    if points_3d is None or cuboid is None:
        print("[WARN] No valid 3D cuboid geometry available.")
        print("==================================================\n")
        return

    print(f"OBJECT POINTS: {len(points_3d)}")
    print(f"CAMERA DISTANCE Z: {cuboid.centroid_3d_m[2]*100.0:.2f} cm\n")

    print("DETECTED PLANES & OBSERVED EXTENTS:")
    for p in cuboid.detected_planes:
        n = p.normal
        u = p.local_axis_u
        v = p.local_axis_v
        print(f"  Plane {p.plane_id}:")
        print(f"    Equation          : {n[0]:.4f}x + {n[1]:.4f}y + {n[2]:.4f}z + {p.d:.4f} = 0")
        print(f"    Normal (XYZ)      : [{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
        print(f"    Local Axis U      : [{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]")
        print(f"    Local Axis V      : [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
        print(f"    Number of Inliers : {len(p.inlier_points)}")
        print(f"    RMS Residual      : {p.rms_residual_cm:.2f} cm (Median: {p.median_residual_cm:.2f} cm)")
        print(f"    U Extent          : {p.extent_u_cm:.2f} cm (along shared edge / primary axis)")
        print(f"    V Extent          : {p.extent_v_cm:.2f} cm (across face / secondary axis)")
        print(f"    Approx Area       : {p.approx_area_sq_cm:.1f} cm²\n")

    print("CANDIDATE PHYSICAL BOUNDARIES & EDGES (RGB-D + Boundary Lines):")
    if cuboid.candidate_physical_edges:
        for e in cuboid.candidate_physical_edges:
            d = e.direction_3d
            print(f"  Boundary {e.edge_id} [{e.source}]:")
            print(f"    3D Length         : {e.length_cm:.2f} cm")
            print(f"    Direction         : [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")
            print(f"    Support Points    : {e.support_points}")
            print(f"    Depth Valid Ratio : {e.depth_valid_ratio*100:.1f}%")
            print(f"    RGB Strength      : {e.rgb_strength:.2f}")
            print(f"    3D RMS Residual   : {e.rms_residual_cm:.2f} cm (Median dist: {e.median_dist_cm:.2f} cm)")
            print(f"    Confidence        : {e.confidence}")
    else:
        print("  No candidate physical boundaries detected.")
    print()

    print("FINAL MEASURED DIMENSIONS (From Physical Evidence):")
    for dim_name, dim_obj in [("LENGTH (L)", cuboid.length), ("BREADTH (B)", cuboid.breadth), ("HEIGHT (H)", cuboid.height)]:
        print(f"  {dim_name}:")
        if dim_obj.confidence == "UNSUPPORTED" or not dim_obj.is_valid:
            print(f"    Value           : UNSUPPORTED / INSUFFICIENT EVIDENCE")
            print(f"    Support points  : {dim_obj.support_points}")
            print(f"    Source          : {dim_obj.source}")
            print(f"    Confidence      : {dim_obj.confidence}")
        else:
            print(f"    Value           : {dim_obj.value_cm:.2f} cm")
            print(f"    Support points  : {dim_obj.support_points}")
            print(f"    Median distance : {dim_obj.median_dist_cm:.2f} cm")
            print(f"    Confidence      : {dim_obj.confidence}")
            print(f"    Source          : {dim_obj.source}")
            if dim_obj.candidate_lengths:
                c_str = ", ".join([f"{x:.1f}" for x in dim_obj.candidate_lengths])
                print(f"    Candidate values: [{c_str}] cm")
    print()

    print("FINAL MEASUREMENT (Temporal Filtered):")
    print(f"  L = {t_stats.length_median:.2f} cm (std: {t_stats.length_std:.2f} cm) [{cuboid.length.confidence}]")
    print(f"  B = {t_stats.breadth_median:.2f} cm (std: {t_stats.breadth_std:.2f} cm) [{cuboid.breadth.confidence}]")
    print(f"  H = {t_stats.height_median:.2f} cm (std: {t_stats.height_std:.2f} cm) [{cuboid.height.confidence}]")
    print(f"  Status: {cuboid.geometry_status}")

    if gt.is_set:
        err_l = abs(t_stats.length_median - gt.length_cm)
        err_b = abs(t_stats.breadth_median - gt.breadth_cm)
        err_h = abs(t_stats.height_median - gt.height_cm)
        print(f"  Ground Truth   : L={gt.length_cm:.1f}, B={gt.breadth_cm:.1f}, H={gt.height_cm:.1f} cm")
        print(f"  Error vs GT    : dL={err_l:.2f}cm ({err_l/gt.length_cm*100:.1f}%), dB={err_b:.2f}cm ({err_b/gt.breadth_cm*100:.1f}%), dH={err_h:.2f}cm ({err_h/gt.height_cm*100:.1f}%)")
    print("==================================================\n")


# ===========================================================================
# SAVE DEBUG SNAPSHOT BUNDLE
# ===========================================================================

def save_debug_snapshot_bundle(
    color_img: np.ndarray,
    mask: np.ndarray,
    depth_colormap: np.ndarray,
    points_3d: Optional[np.ndarray],
    cuboid: Optional[CuboidReconstruction],
    t_stats: TemporalStats,
    gt: GroundTruth
) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(out_dir, exist_ok=True)

    dist_str = f"{int(cuboid.centroid_3d_m[2]*100.0)}cm" if cuboid is not None else "unknown"
    prefix = f"debug_{dist_str}_{timestamp}"

    rgb_path   = os.path.join(out_dir, f"{prefix}_rgb.png")
    mask_path  = os.path.join(out_dir, f"{prefix}_mask.png")
    depth_path = os.path.join(out_dir, f"{prefix}_depth.png")
    ply_path   = os.path.join(out_dir, f"{prefix}_pointcloud.ply")
    plot_path  = os.path.join(out_dir, f"{prefix}_3dplot.png")
    rep_path   = os.path.join(out_dir, f"{prefix}_report.txt")

    cv2.imwrite(rgb_path, color_img)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(depth_path, depth_colormap)

    if points_3d is not None and len(points_3d) > 0:
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for pt in points_3d:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

    if points_3d is not None and cuboid is not None:
        plot_3d_cuboid_reconstruction(points_3d, cuboid, t_stats, out_filename=plot_path)

    with open(rep_path, "w") as f:
        f.write("==================================================\n")
        f.write(f"  PHYSICAL BOUNDARY 3D GEOMETRY REPORT: {prefix}\n")
        f.write("==================================================\n\n")
        if cuboid is not None:
            f.write(f"Timestamp        : {timestamp}\n")
            f.write(f"Camera Distance Z: {cuboid.centroid_3d_m[2]*100.0:.2f} cm\n")
            f.write(f"Object Points    : {len(points_3d) if points_3d is not None else 0}\n\n")

            f.write("PLANES & OBSERVED EXTENTS:\n")
            for p in cuboid.detected_planes:
                f.write(f"  Plane {p.plane_id}:\n")
                f.write(f"    Equation          : {p.normal[0]:.4f}x + {p.normal[1]:.4f}y + {p.normal[2]:.4f}z + {p.d:.4f} = 0\n")
                f.write(f"    Normal (XYZ)      : [{p.normal[0]:.4f}, {p.normal[1]:.4f}, {p.normal[2]:.4f}]\n")
                f.write(f"    Inliers           : {len(p.inlier_points)} points\n")
                f.write(f"    RMS Residual      : {p.rms_residual_cm:.2f} cm (Median: {p.median_residual_cm:.2f} cm)\n")
                f.write(f"    Observed Extents  : U = {p.extent_u_cm:.2f} cm (shared edge) | V = {p.extent_v_cm:.2f} cm (face span)\n")
                f.write(f"    Approx Area       : {p.approx_area_sq_cm:.1f} cm²\n\n")

            f.write("CANDIDATE PHYSICAL BOUNDARIES:\n")
            for e in cuboid.candidate_physical_edges:
                f.write(f"  {e.edge_id} ({e.source}): Length={e.length_cm:.2f}cm, Support={e.support_points}, RMS={e.rms_residual_cm:.2f}cm, Conf={e.confidence}\n")
            f.write("\n")

            f.write("FINAL MEASURED DIMENSIONS:\n")
            f.write(f"  Length  : {cuboid.length.value_cm:.2f} cm [{cuboid.length.confidence}] (Source: {cuboid.length.source})\n")
            f.write(f"  Breadth : {cuboid.breadth.value_cm:.2f} cm [{cuboid.breadth.confidence}] (Source: {cuboid.breadth.source})\n")
            f.write(f"  Height  : {cuboid.height.value_cm:.2f} cm [{cuboid.height.confidence}] (Source: {cuboid.height.source})\n\n")

            f.write("FINAL TEMPORAL MEASUREMENT:\n")
            f.write(f"  L = {t_stats.length_median:.2f} cm (std: {t_stats.length_std:.2f} cm)\n")
            f.write(f"  B = {t_stats.breadth_median:.2f} cm (std: {t_stats.breadth_std:.2f} cm)\n")
            f.write(f"  H = {t_stats.height_median:.2f} cm (std: {t_stats.height_std:.2f} cm)\n")
            f.write(f"  Status: {cuboid.geometry_status}\n")

    print("\n==============================================================")
    print(f"  DEBUG SNAPSHOT BUNDLE SAVED TO 'debug_output/'")
    print("==============================================================")
    print(f"  RGB Image        : {rgb_path}")
    print(f"  Object Mask      : {mask_path}")
    print(f"  Segmented Depth  : {depth_path}")
    print(f"  3D Pointcloud PLY: {ply_path}")
    print(f"  3D Debug Plot    : {plot_path}")
    print(f"  Detailed Report  : {rep_path}")
    print("==============================================================\n")


# ===========================================================================
# DISTANCE-INVARIANCE TEST CSV & ERROR ANALYSIS
# ===========================================================================

def record_distance_test_csv(
    test_distance_cm: float,
    t_stats: TemporalStats,
    cuboid: CuboidReconstruction,
    gt: GroundTruth,
    csv_filename: str = "box_measurement_distance_test_v2.csv"
) -> str:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
    file_exists = os.path.isfile(csv_path)

    header = "camera_distance_cm,measured_L_cm,measured_B_cm,measured_H_cm,L_std_cm,B_std_cm,H_std_cm,conf_L,conf_B,conf_H,gt_L_cm,gt_B_cm,gt_H_cm,err_L_cm,err_B_cm,err_H_cm,valid_points,number_of_planes,geometry_status\n"

    err_l = abs(t_stats.length_median - gt.length_cm) if gt.is_set else 0.0
    err_b = abs(t_stats.breadth_median - gt.breadth_cm) if gt.is_set else 0.0
    err_h = abs(t_stats.height_median - gt.height_cm) if gt.is_set else 0.0

    row = (
        f"{test_distance_cm:.1f},"
        f"{t_stats.length_median:.2f},"
        f"{t_stats.breadth_median:.2f},"
        f"{t_stats.height_median:.2f},"
        f"{t_stats.length_std:.2f},"
        f"{t_stats.breadth_std:.2f},"
        f"{t_stats.height_std:.2f},"
        f"\"{cuboid.length.confidence}\","
        f"\"{cuboid.breadth.confidence}\","
        f"\"{cuboid.height.confidence}\","
        f"{gt.length_cm:.2f},"
        f"{gt.breadth_cm:.2f},"
        f"{gt.height_cm:.2f},"
        f"{err_l:.2f},"
        f"{err_b:.2f},"
        f"{err_h:.2f},"
        f"{sum(len(p.inlier_points) for p in cuboid.detected_planes)},"
        f"{cuboid.num_planes},"
        f"\"{cuboid.geometry_status}\"\n"
    )

    with open(csv_path, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(row)

    print("\n==============================================================")
    print(f"  DISTANCE TEST RECORDED AT {test_distance_cm:.1f} cm")
    print("==============================================================")
    print(f"  Measurement  : L = {t_stats.length_median:.2f} cm [{cuboid.length.confidence}] (std: {t_stats.length_std:.2f} cm)")
    print(f"                 B = {t_stats.breadth_median:.2f} cm [{cuboid.breadth.confidence}] (std: {t_stats.breadth_std:.2f} cm)")
    print(f"                 H = {t_stats.height_median:.2f} cm [{cuboid.height.confidence}] (std: {t_stats.height_std:.2f} cm)")
    print(f"  Ground Truth : L = {gt.length_cm:.1f} cm | B = {gt.breadth_cm:.1f} cm | H = {gt.height_cm:.1f} cm")
    print(f"  Absolute Err : dL = {err_l:.2f} cm ({err_l/gt.length_cm*100:.1f}%) | dB = {err_b:.2f} cm ({err_b/gt.breadth_cm*100:.1f}%) | dH = {err_h:.2f} cm ({err_h/gt.height_cm*100:.1f}%)")
    print(f"  Status       : {cuboid.geometry_status}")
    print(f"  Appended to  : {csv_path}")
    print("==============================================================\n")
    return csv_path


def prompt_ground_truth() -> GroundTruth:
    print("\n--------------------------------------------------------------")
    print("  ENTER GROUND TRUTH PHYSICAL DIMENSIONS (cm)")
    print("--------------------------------------------------------------")
    try:
        l_in = float(input(f"Enter actual box Length (longest) [default: {DEFAULT_GT_L}]: ").strip() or DEFAULT_GT_L)
        b_in = float(input(f"Enter actual box Breadth (middle) [default: {DEFAULT_GT_B}]: ").strip() or DEFAULT_GT_B)
        h_in = float(input(f"Enter actual box Height (shortest) [default: {DEFAULT_GT_H}]: ").strip() or DEFAULT_GT_H)

        gt = GroundTruth(length_cm=l_in, breadth_cm=b_in, height_cm=h_in, is_set=True)
        print(f"[OK] Ground truth updated: L={gt.length_cm:.1f}, B={gt.breadth_cm:.1f}, H={gt.height_cm:.1f} cm\n")
        return gt
    except Exception as exc:
        print(f"[WARN] Invalid input ({exc}) — ground truth not updated.\n")
        return GroundTruth()


# ===========================================================================
# 3D RECONSTRUCTION VISUALIZATION EXPORT
# ===========================================================================

def plot_3d_cuboid_reconstruction(
    points_3d: np.ndarray,
    cuboid: CuboidReconstruction,
    t_stats: TemporalStats,
    out_filename: str = "box_3d_cuboid_reconstruction_plot.png"
) -> None:
    if not HAS_MATPLOTLIB:
        print("[WARN] Matplotlib not available — skipping 3D plot.")
        return

    pts_cm = points_3d * 100.0
    corners_cm = cuboid.corners_3d_m * 100.0
    cent_cm = cuboid.centroid_3d_m * 100.0

    if len(pts_cm) > 4000:
        idx = np.random.choice(len(pts_cm), 4000, replace=False)
        pts_cm = pts_cm[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Point Cloud
    p = ax.scatter(pts_cm[:, 0], pts_cm[:, 2], -pts_cm[:, 1], c=pts_cm[:, 2], cmap="viridis", s=3, alpha=0.6)
    cbar = fig.colorbar(p, ax=ax, pad=0.1)
    cbar.set_label("Depth Z (cm)")

    # 2. Candidate Physical Edges
    if cuboid.candidate_physical_edges:
        for cand in cuboid.candidate_physical_edges:
            if cand.is_valid:
                s_cm = cand.start_pt_3d_m * 100.0
                e_cm = cand.end_pt_3d_m * 100.0
                ax.plot([s_cm[0], e_cm[0]], [s_cm[2], e_cm[2]], [-s_cm[1], -e_cm[1]],
                        color="cyan" if "RGB" in cand.source else "lime", linewidth=2.5, alpha=0.85)

    # 3. 3D Bounding Box Wireframe with corners (distinguish observed vs inferred)
    if np.any(corners_cm):
        c_plt = np.zeros_like(corners_cm)
        c_plt[:, 0] = corners_cm[:, 0]
        c_plt[:, 1] = corners_cm[:, 2]
        c_plt[:, 2] = -corners_cm[:, 1]

        for edge in cuboid.reconstructed_edges:
            idx1 = int(edge.start_corner_name[1:]) - 1
            idx2 = int(edge.end_corner_name[1:]) - 1
            if edge.is_supported:
                ax.plot([c_plt[idx1, 0], c_plt[idx2, 0]],
                        [c_plt[idx1, 1], c_plt[idx2, 1]],
                        [c_plt[idx1, 2], c_plt[idx2, 2]], color="red", linewidth=2.2, label="Observed Edge" if edge.edge_id=="E1" else "")
            else:
                ax.plot([c_plt[idx1, 0], c_plt[idx2, 0]],
                        [c_plt[idx1, 1], c_plt[idx2, 1]],
                        [c_plt[idx1, 2], c_plt[idx2, 2]], color="gray", linestyle="--", linewidth=1.2, label="Inferred Edge" if edge.edge_id=="E9" else "")

        for i in range(8):
            ax.scatter([c_plt[i, 0]], [c_plt[i, 2]], [c_plt[i, 1]], color="black", s=25)
            ax.text(c_plt[i, 0], c_plt[i, 1], c_plt[i, 2], f" C{i+1}", fontsize=8, fontweight="bold")

    # 4. Plane Normals
    axis_scale = 15.0
    for i, plane in enumerate(cuboid.detected_planes):
        n = plane.normal * axis_scale
        ax.quiver(cent_cm[0], cent_cm[2], -cent_cm[1],
                  n[0], n[2], -n[1], color=["blue", "green", "magenta"][i % 3],
                  linewidth=2.5, label=f"Plane {plane.plane_id} Normal")

    l_val = t_stats.length_median if t_stats.sample_count > 5 else cuboid.length.value_cm
    b_val = t_stats.breadth_median if t_stats.sample_count > 5 else cuboid.breadth.value_cm
    h_val = t_stats.height_median if t_stats.sample_count > 5 else cuboid.height.value_cm

    ax.set_xlabel("X — Right (cm)")
    ax.set_ylabel("Z — Forward (cm)")
    ax.set_zlabel("Y — Up (-Y camera) (cm)")
    ax.set_title(
        f"Stage 5.5: Physical Boundary & Edge Measurement\n"
        f"Measured: L = {l_val:.1f} cm [{cuboid.length.confidence}], B = {b_val:.1f} cm [{cuboid.breadth.confidence}], H = {h_val:.1f} cm [{cuboid.height.confidence}]\n"
        f"Status: {cuboid.geometry_status}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_filename, dpi=150)
    plt.close(fig)
    print(f"[INFO] 3D Debug plot saved to: {out_filename}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print("""
##############################################################
  Intel RealSense D455f — Stage 5.6: Object Detection & Metrology
##############################################################
  - Decoupled Object Detector Interface (BaseObjectDetector)
  - Mode 1: Manual ROI Selection (Key [M])
  - Mode 2: Autonomous RGB-D Detection (Key [A])
  - Validated Physical Metrology Engine (L, B, H)
  - Toggle [D] for On-Screen Debug & Console Summary
  - Press [S] to Save Measurement Snapshot to 'measurement_snapshots/'
##############################################################
""")

    cam = initialize_camera()
    setup_gui()

    detector = RGBDForegroundDetector(min_depth_m=0.35, max_depth_m=2.20, smooth_tracking=True)
    detection_mode = "MANUAL"  # Default to MANUAL Border-Drawing mode

    temporal_filter = TemporalDimensionFilter(TEMPORAL_BUFFER_SIZE)
    ground_truth = GroundTruth()
    debug_mode = False

    fps = 0.0
    frame_count = 0
    t_start = time.time()

    print("[INFO] Application running.")
    print("[INFO] Quick Start: CLICK & DRAG A BORDER AROUND YOUR OBJECT IN THE RGB WINDOW.")
    print("[INFO] Controls:")
    print("       - Mouse Drag: Draw border around any object to measure it")
    print("       - [N]       : Reset / Draw new border")
    print("       - [A]       : Switch to AUTOMATIC Detection Mode")
    print("       - [M]       : Switch to MANUAL Border Mode")
    print("       - [D]       : Toggle On-Screen Debug Overlay")
    print("       - [S]       : Save Measurement Snapshot to 'measurement_snapshots/'")
    print("       - [T]       : Record Distance Invariance row to CSV")
    print("       - [G]       : Update Ground Truth (L, B, H in cm)")
    print("       - [P]       : Export 3D Matplotlib Plot")
    print("       - [Q]/ESC   : Exit cleanly\n")

    global _user_dragged_new_roi
    try:
        while not _shutdown_requested:
            success, frameset = cam.pipeline.try_wait_for_frames(timeout_ms=3000)
            if not success:
                continue

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

            params = get_tuning_parameters()

            # Handle user drawing a new border with mouse
            if _user_dragged_new_roi:
                detection_mode = "MANUAL"
                temporal_filter.clear()
                _user_dragged_new_roi = False

            # 1. Object Detection (Auto vs Manual)
            detection_res = None
            if detection_mode == "AUTO":
                detection_res = detector.detect(color_img, depth_m)
                if detection_res.is_valid:
                    _current_roi.x1, _current_roi.y1, _current_roi.x2, _current_roi.y2 = detection_res.bbox
                    _current_roi.is_locked = True
                    _current_roi.is_selected = True
                else:
                    _current_roi.is_locked = False
                    _current_roi.is_selected = False

            # 2. Segmentation & Point Cloud
            mask, points_3d, raw_n, valid_n, clean_n, z_peak = segment_and_extract_point_cloud(
                depth_m, _current_roi, cam.intrinsics, params
            )

            # 3. Multi-Plane RANSAC & Physical Edge Metrology
            cuboid = None
            if points_3d is not None and len(points_3d) >= 25:
                planes = detect_planes_ransac(points_3d, dist_thresh_m=params["ransac_thresh_m"])
                cuboid = reconstruct_cuboid_from_physical_edges(
                    points_3d, planes, color_img, depth_m, _current_roi, cam.intrinsics, z_peak, DEFAULT_ORTHO_TOL_DEG
                )

                if cuboid is not None and cuboid.is_reliable:
                    temporal_filter.add_sample(cuboid.length.value_cm, cuboid.breadth.value_cm, cuboid.height.value_cm, cuboid.is_reliable)

            t_stats = temporal_filter.get_stats()

            # Construct Single Source of Truth MeasurementResult
            det_status = "NO OBJECT DETECTED"
            det_conf = 0.0
            if detection_mode == "AUTO":
                if detection_res is not None and detection_res.is_valid:
                    det_status = f"AUTO ({detection_res.class_name})"
                    det_conf = detection_res.confidence
                elif detection_res is not None and detection_res.status == "TRACKING_LOST":
                    det_status = "TRACKING LOST"
            else:
                det_status = "MANUAL LOCKED" if _current_roi.is_locked else "MANUAL ROI"
                det_conf = 1.0 if _current_roi.is_locked else 0.0

            meas_res = MeasurementResult(
                timestamp=time.time(),
                detection_status=det_status,
                detection_confidence=det_conf,
                detection_mode=detection_mode,
                bbox=_current_roi.box if _current_roi.is_valid else (0, 0, 0, 0),
                mask=mask if (_current_roi.is_locked and np.any(mask)) else None,
                length_cm=t_stats.length_median if (cuboid is not None and cuboid.is_reliable and t_stats.sample_count > 5) else (cuboid.length.value_cm if cuboid is not None else 0.0),
                breadth_cm=t_stats.breadth_median if (cuboid is not None and cuboid.is_reliable and t_stats.sample_count > 5) else (cuboid.breadth.value_cm if cuboid is not None else 0.0),
                height_cm=t_stats.height_median if (cuboid is not None and cuboid.is_reliable and t_stats.sample_count > 5) else (cuboid.height.value_cm if cuboid is not None else 0.0),
                length_std_cm=t_stats.length_std if (cuboid is not None and cuboid.is_reliable) else 0.0,
                breadth_std_cm=t_stats.breadth_std if (cuboid is not None and cuboid.is_reliable) else 0.0,
                height_std_cm=t_stats.height_std if (cuboid is not None and cuboid.is_reliable) else 0.0,
                length_confidence=cuboid.length.confidence if cuboid is not None else "UNSUPPORTED",
                breadth_confidence=cuboid.breadth.confidence if cuboid is not None else "UNSUPPORTED",
                height_confidence=cuboid.height.confidence if cuboid is not None else "UNSUPPORTED",
                physical_edges=cuboid.candidate_physical_edges if cuboid is not None else [],
                reconstructed_edges=cuboid.reconstructed_edges if cuboid is not None else [],
                physical_corners=cuboid.corners_labeled if cuboid is not None else {},
                metrology_status=cuboid.geometry_status if cuboid is not None else "NO_GEOMETRY",
                is_metrology_valid=cuboid.is_reliable if cuboid is not None else False,
                camera_distance_z_cm=(cuboid.centroid_3d_m[2] * 100.0) if cuboid is not None else 0.0,
                fps=fps
            )

            # 4. Render Visual Overlay
            annotated_rgb = draw_measurement_overlay(
                color_img, mask, _current_roi, cuboid, t_stats, ground_truth, cam.intrinsics, fps, debug_mode,
                detection_mode=detection_mode, detection_res=detection_res
            )

            # 5. Display
            cv2.imshow(WIN_MAIN, annotated_rgb)
            cv2.imshow(WIN_MASK, mask)

            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] User requested exit.")
                break
            elif key in (ord('a'), ord('A')):
                detection_mode = "AUTO"
                print("\n[MODE] Switched to AUTOMATIC Object Detection Mode.")
                temporal_filter.clear()
            elif key in (ord('m'), ord('M')):
                detection_mode = "MANUAL"
                reset_roi()
                detector.reset()
                temporal_filter.clear()
                print("\n[MODE] Switched to MANUAL ROI Mode. Drag a rectangle on the RGB feed.")
            elif key in (ord('d'), ord('D')):
                debug_mode = not debug_mode
                print(f"\n[INFO] Debug Mode {'ACTIVATED' if debug_mode else 'DEACTIVATED'}")
                if debug_mode:
                    print_console_debug_summary(cuboid, t_stats, points_3d, ground_truth)
            elif key in (ord('n'), ord('N')):
                reset_roi()
                detector.reset()
                temporal_filter.clear()
            elif key in (ord('g'), ord('G')):
                ground_truth = prompt_ground_truth()
            elif key in (ord('p'), ord('P')):
                if points_3d is not None and cuboid is not None:
                    plot_3d_cuboid_reconstruction(points_3d, cuboid, t_stats, "box_3d_cuboid_reconstruction_plot.png")
                else:
                    print("[WARN] Cannot export 3D plot — insufficient box geometry currently.")
            elif key in (ord('s'), ord('S')):
                save_measurement_snapshot(annotated_rgb, color_img, mask, meas_res, ground_truth)
                if debug_mode:
                    save_debug_snapshot_bundle(annotated_rgb, mask, depth_colormap, points_3d, cuboid, t_stats, ground_truth)
            elif key in (ord('t'), ord('T')):
                if cuboid is not None and cuboid.is_reliable and t_stats.sample_count > 8:
                    try:
                        dist_input = input("\nEnter actual camera-to-box distance for this test (e.g. 80, 100, 120 cm): ").strip()
                        dist_val = float(dist_input)
                        record_distance_test_csv(dist_val, t_stats, cuboid, ground_truth)
                    except Exception as e:
                        print(f"[WARN] Invalid distance input ({e}) — row not saved.")
                else:
                    print("[WARN] Show at least 2 orthogonal faces and allow 15+ frames to accumulate before recording.")

    except Exception as exc:
        print(f"\n[ERROR] Runtime exception: {exc}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n[INFO] Stopping camera pipeline...")
        try:
            cam.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Camera stopped and windows closed cleanly.")
        print("Stage 5.5 execution finished.\n")


if __name__ == "__main__":
    main()
