"""
=============================================================================
Module 2: Multi-View Point-Cloud Capture and Dataset Management
=============================================================================
Intel RealSense D455f Multi-View 3D Reconstruction Pipeline — Stage 2 / Dataset Management

Project Objective:
    Capture multiple calibrated RGB-D observations of a physical object from
    distinct viewpoints and organize each observation into an independent,
    structured dataset ready for subsequent multi-view registration and fusion.

Architecture & Design Principles:
    1. Independent Viewpoint Representation:
       Every captured view is stored in its OWN camera-centric coordinate frame:
         +X : Right
         +Y : Down
         +Z : Forward (along camera optical axis)
       No inter-view registration or spatial alignment is assumed at this stage.

    2. Structured Session Hierarchy:
       datasets/
         session_001/
           metadata.json
           view_001/
             rgb.png
             depth.npy
             pointcloud.ply
             pointcloud.csv
             metadata.json
           view_002/
             ...

    3. Pre-Capture Validation Quality Gate:
       Guarantees that a view is saved only if:
         - Frame synchronization is healthy
         - Valid depth coverage >= minimum threshold
         - Depth statistics (Min, Max, Median, Mean Z) are finite and physically valid

    4. Zero Transformation / Filtering Guarantee:
       Raw metric 3D point cloud data is exported strictly in METERS (m).
       No ICP, smoothing, voxel fusion, plane fitting, segmentation, or measurement
       is performed in this module.

Controls:
    [C]       : Capture current view (View 001, 002, ...)
    [F]       : Finish current session & generate dataset validation report
    [R]       : Reset / start a new session (creates session_002, ...)
    [P]       : Pause / Resume live feed
    [Q] / ESC : Exit application cleanly
=============================================================================
"""

import os
import sys
import time
import json
import argparse
import glob
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2

# Import hardware & deprojection classes from Module 1
from module1_pointcloud_acquisition import (
    RealSenseCamera,
    PointCloudGenerator,
    PointCloudVisualizer,
    PointCloudData,
    CameraIntrinsics,
    DeviceInfo,
    DEFAULT_STREAM_WIDTH,
    DEFAULT_STREAM_HEIGHT,
    DEFAULT_FPS,
    WIN_RGB,
    WIN_DEPTH,
    WIN_3D_VIZ
)


# ===========================================================================
# CONFIGURATION CONSTANTS
# ===========================================================================
DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
MIN_VALID_POINTS_THRESHOLD = 5000       # Minimum valid 3D points required to accept a view
MIN_DEPTH_COVERAGE_PCT = 0.5            # Minimum 0.5% valid depth pixels required


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================
@dataclass
class ViewValidationResult:
    """Stores the outcome of quality-gate checks on a candidate view."""
    is_valid: bool
    rejection_reason: str
    total_pixels: int
    valid_pixels: int
    coverage_pct: float
    valid_3d_points: int
    min_z_m: float
    max_z_m: float
    mean_z_m: float
    median_z_m: float


@dataclass
class ViewMetadata:
    """Complete metadata record for an individual captured viewpoint."""
    view_id: str
    session_id: str
    capture_order: int
    timestamp_iso: str
    unix_timestamp: float
    camera_name: str
    camera_serial: str
    firmware: str
    usb_type: str
    depth_scale: float
    resolution: Tuple[int, int]
    fps: float
    intrinsics_fx: float
    intrinsics_fy: float
    intrinsics_cx: float
    intrinsics_cy: float
    distortion_model: str
    distortion_coeffs: List[float]
    valid_3d_points: int
    depth_coverage_pct: float
    min_z_m: float
    max_z_m: float
    mean_z_m: float
    median_z_m: float
    units: str = "meters"
    coordinate_frame: str = "RealSense Optical (+X Right, +Y Down, +Z Forward)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# CLASS: MultiViewDatasetManager
# ===========================================================================
class MultiViewDatasetManager:
    """
    Manages session directory allocation, automated view indexing, quality-gate
    validation, file persistence (RGB, Depth, PLY, CSV, JSON), and dataset reporting.
    """

    def __init__(self, root_dir: str = DEFAULT_DATASET_ROOT):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.current_session_id: Optional[str] = None
        self.current_session_dir: Optional[str] = None
        self.captured_views: List[ViewMetadata] = []
        self.start_new_session()

    def _get_next_session_id(self) -> str:
        """Finds the next sequential session ID (session_001, session_002, ...)."""
        existing = glob.glob(os.path.join(self.root_dir, "session_*"))
        indices = []
        for p in existing:
            base = os.path.basename(p)
            parts = base.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                indices.append(int(parts[1]))

        next_idx = max(indices, default=0) + 1
        return f"session_{next_idx:03d}"

    def start_new_session(self) -> str:
        """Initializes a new capture session directory."""
        self.current_session_id = self._get_next_session_id()
        self.current_session_dir = os.path.join(self.root_dir, self.current_session_id)
        os.makedirs(self.current_session_dir, exist_ok=True)
        self.captured_views = []

        print(f"\n[SESSION] Created New Multi-View Session: {self.current_session_id}")
        print(f"          Directory: {self.current_session_dir}")
        return self.current_session_id

    @staticmethod
    def validate_candidate_view(
        color_bgr: Optional[np.ndarray],
        depth_m: Optional[np.ndarray],
        pcd: Optional[PointCloudData]
    ) -> ViewValidationResult:
        """
        Applies rigorous quality gating to reject corrupt, missing, or sparse depth frames.
        """
        if color_bgr is None or depth_m is None or pcd is None:
            return ViewValidationResult(
                is_valid=False,
                rejection_reason="Frame stream unavailable or None",
                total_pixels=0, valid_pixels=0, coverage_pct=0.0,
                valid_3d_points=0, min_z_m=0.0, max_z_m=0.0, mean_z_m=0.0, median_z_m=0.0
            )

        total_pixels = depth_m.size
        valid_depth_mask = (depth_m > 0.1) & (depth_m < 8.0) & np.isfinite(depth_m)
        valid_pixels = int(np.count_nonzero(valid_depth_mask))
        coverage_pct = (valid_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        if valid_pixels == 0:
            return ViewValidationResult(
                is_valid=False,
                rejection_reason="Zero valid depth pixels detected in frame",
                total_pixels=total_pixels, valid_pixels=0, coverage_pct=0.0,
                valid_3d_points=0, min_z_m=0.0, max_z_m=0.0, mean_z_m=0.0, median_z_m=0.0
            )

        valid_z = depth_m[valid_depth_mask]
        min_z = float(np.min(valid_z))
        max_z = float(np.max(valid_z))
        mean_z = float(np.mean(valid_z))
        median_z = float(np.median(valid_z))

        valid_3d_points = pcd.valid_count

        if valid_3d_points < MIN_VALID_POINTS_THRESHOLD:
            return ViewValidationResult(
                is_valid=False,
                rejection_reason=f"Insufficient 3D points: {valid_3d_points:,} < {MIN_VALID_POINTS_THRESHOLD:,}",
                total_pixels=total_pixels, valid_pixels=valid_pixels, coverage_pct=coverage_pct,
                valid_3d_points=valid_3d_points, min_z_m=min_z, max_z_m=max_z, mean_z_m=mean_z, median_z_m=median_z
            )

        if coverage_pct < MIN_DEPTH_COVERAGE_PCT:
            return ViewValidationResult(
                is_valid=False,
                rejection_reason=f"Low depth coverage: {coverage_pct:.2f}% < {MIN_DEPTH_COVERAGE_PCT:.2f}%",
                total_pixels=total_pixels, valid_pixels=valid_pixels, coverage_pct=coverage_pct,
                valid_3d_points=valid_3d_points, min_z_m=min_z, max_z_m=max_z, mean_z_m=mean_z, median_z_m=median_z
            )

        return ViewValidationResult(
            is_valid=True,
            rejection_reason="",
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            coverage_pct=coverage_pct,
            valid_3d_points=valid_3d_points,
            min_z_m=min_z,
            max_z_m=max_z,
            mean_z_m=mean_z,
            median_z_m=median_z
        )

    def save_view(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        pcd: PointCloudData,
        dev_info: DeviceInfo,
        intrinsics: CameraIntrinsics,
        fps: float,
        val: ViewValidationResult
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Saves a validated viewpoint observation bundle into its dedicated folder:
          view_XXX/
            rgb.png
            depth.npy
            pointcloud.ply
            pointcloud.csv
            metadata.json
        """
        if not val.is_valid:
            return False, f"View rejected: {val.rejection_reason}", None

        view_index = len(self.captured_views) + 1
        view_id = f"view_{view_index:03d}"
        view_dir = os.path.join(self.current_session_dir, view_id)
        os.makedirs(view_dir, exist_ok=True)

        iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        unix_ts = time.time()

        # 1. Save RGB Image (.png)
        rgb_path = os.path.join(view_dir, "rgb.png")
        cv2.imwrite(rgb_path, color_bgr)

        # 2. Save Metric Depth Map (.npy in float32 meters)
        depth_npy_path = os.path.join(view_dir, "depth.npy")
        np.save(depth_npy_path, depth_m.astype(np.float32))

        # 3. Save Metric Point Cloud (.ply - standard ASCII format in meters)
        ply_path = os.path.join(view_dir, "pointcloud.ply")
        num_points = len(pcd.points)
        ply_header = (
            "ply\n"
            "format ascii 1.0\n"
            f"comment RealSense D455f Multi-View Observation [{self.current_session_id} / {view_id}]\n"
            f"comment Metric Units: METERS\n"
            f"comment Timestamp: {iso_timestamp}\n"
            f"comment Camera Serial: {dev_info.serial}\n"
            f"comment Intrinsics: fx={intrinsics.fx:.4f}, fy={intrinsics.fy:.4f}, cx={intrinsics.cx:.4f}, cy={intrinsics.cy:.4f}\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(ply_path, "w") as f:
            f.write(ply_header)
            for i in range(num_points):
                x, y, z = pcd.points[i]
                r, g, b = pcd.colors[i]
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")

        # 4. Save Point Cloud CSV (x_m,y_m,z_m,r,g,b in meters)
        csv_path = os.path.join(view_dir, "pointcloud.csv")
        with open(csv_path, "w") as f:
            f.write("x_m,y_m,z_m,r,g,b\n")
            for i in range(num_points):
                x, y, z = pcd.points[i]
                r, g, b = pcd.colors[i]
                f.write(f"{x:.6f},{y:.6f},{z:.6f},{r},{g},{b}\n")

        # 5. Save View Metadata (.json)
        meta = ViewMetadata(
            view_id=view_id,
            session_id=self.current_session_id,
            capture_order=view_index,
            timestamp_iso=iso_timestamp,
            unix_timestamp=unix_ts,
            camera_name=dev_info.name,
            camera_serial=dev_info.serial,
            firmware=dev_info.firmware,
            usb_type=dev_info.usb_type,
            depth_scale=dev_info.depth_scale,
            resolution=(intrinsics.width, intrinsics.height),
            fps=fps,
            intrinsics_fx=intrinsics.fx,
            intrinsics_fy=intrinsics.fy,
            intrinsics_cx=intrinsics.cx,
            intrinsics_cy=intrinsics.cy,
            distortion_model=intrinsics.model,
            distortion_coeffs=intrinsics.coeffs,
            valid_3d_points=val.valid_3d_points,
            depth_coverage_pct=val.coverage_pct,
            min_z_m=val.min_z_m,
            max_z_m=val.max_z_m,
            mean_z_m=val.mean_z_m,
            median_z_m=val.median_z_m
        )

        meta_path = os.path.join(view_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        self.captured_views.append(meta)
        self._update_session_metadata()

        log_msg = f"Captured {view_id} successfully ({val.valid_3d_points:,} pts, Z_med={val.median_z_m:.2f}m)"
        print(f"[CAPTURE SUCCESS] {log_msg}")
        return True, log_msg, view_dir

    def _update_session_metadata(self) -> None:
        """Updates root session metadata JSON file with view inventory."""
        session_meta_path = os.path.join(self.current_session_dir, "metadata.json")
        data = {
            "session_id": self.current_session_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_views": len(self.captured_views),
            "units": "meters",
            "views": [v.to_dict() for v in self.captured_views]
        }
        with open(session_meta_path, "w") as f:
            json.dump(data, f, indent=2)

    def print_dataset_validation_report(self) -> None:
        """Prints a comprehensive tabular validation summary of the capture session."""
        print("\n" + "=" * 78)
        print(f"  DATASET VALIDATION REPORT : {self.current_session_id.upper()}")
        print("=" * 78)
        print(f"  Session Directory   : {self.current_session_dir}")
        print(f"  Total Views Captured: {len(self.captured_views)}")
        print("-" * 78)
        print(f"  {'VIEW ID':<10} | {'3D POINTS':<12} | {'COVERAGE %':<11} | {'Z MEDIAN':<10} | {'Z MIN-MAX (m)':<15} | {'STATUS'}")
        print("-" * 78)

        if not self.captured_views:
            print("  (No views captured in this session)")
        else:
            for v in self.captured_views:
                z_range = f"[{v.min_z_m:.2f}, {v.max_z_m:.2f}]"
                print(f"  {v.view_id:<10} | {v.valid_3d_points:<12,d} | {v.depth_coverage_pct:<10.2f}% | {v.median_z_m:<8.3f} m | {z_range:<15} | VALID [OK]")

        print("-" * 78)
        print("  FILES SAVED PER VIEW: rgb.png, depth.npy, pointcloud.ply, pointcloud.csv, metadata.json")
        print("=" * 78 + "\n")


# ===========================================================================
# CLASS: MultiViewCaptureUI
# ===========================================================================
class MultiViewCaptureUI:
    """
    Renders live multi-view telemetry, capture notification banners, session
    status counters, and 3D preview views.
    """

    def __init__(self, visualizer: PointCloudVisualizer):
        self.visualizer = visualizer
        self.notification_text: str = ""
        self.notification_color: Tuple[int, int, int] = (0, 255, 0)
        self.notification_expiry: float = 0.0

    def set_notification(self, text: str, success: bool = True, duration_sec: float = 3.0) -> None:
        self.notification_text = text
        self.notification_color = (0, 255, 0) if success else (0, 0, 255)
        self.notification_expiry = time.time() + duration_sec

    def draw_multiview_hud(
        self,
        canvas: np.ndarray,
        session_id: str,
        view_count: int,
        pcd: PointCloudData,
        val: ViewValidationResult,
        dev_info: DeviceInfo,
        intrinsics: CameraIntrinsics,
        fps: float,
        paused: bool
    ) -> None:
        """Renders comprehensive multi-view capture HUD overlay."""
        h, w = canvas.shape[:2]

        # Top-Left Telemetry Panel
        overlay = canvas.copy()
        panel_w = 480
        panel_h = 270
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (panel_w, panel_h), (0, 255, 200), 1)

        # Crosshair
        cx_px, cy_px = w // 2, h // 2
        cv2.line(canvas, (cx_px - 15, cy_px), (cx_px + 15, cy_px), (0, 255, 0), 1)
        cv2.line(canvas, (cx_px, cy_px - 15), (cx_px, cy_px + 15), (0, 255, 0), 1)
        cv2.circle(canvas, (cx_px, cy_px), 4, (0, 255, 0), 1)

        lines = [
            f"SESSION: {session_id} | CAPTURED VIEWS: {view_count}",
            f"Device: {dev_info.name} ({dev_info.usb_type})",
            f"Serial: {dev_info.serial} | FW: {dev_info.firmware}",
            f"Resolution: {intrinsics.width}x{intrinsics.height} @ {fps:.1f} FPS",
            f"Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}",
            f"Depth Scale: {dev_info.depth_scale:.6f} m/unit",
            f"CURRENT VALID 3D POINTS: {pcd.valid_count:,} ({val.coverage_pct:.1f}%)",
            f"DEPTH METRICS: Med Z={val.median_z_m:.3f}m | Mean Z={val.mean_z_m:.3f}m",
            f"Z RANGE: [{val.min_z_m:.2f}m, {val.max_z_m:.2f}m]",
            f"CAPTURE STATUS: {'READY TO CAPTURE [C]' if val.is_valid else 'INSUFFICIENT DEPTH'}"
        ]

        y_offset = 32
        for i, text in enumerate(lines):
            if i == 0:
                color = (0, 255, 255)
            elif i == 6 or i == 7:
                color = (0, 255, 0)
            elif i == 9:
                color = (0, 255, 0) if val.is_valid else (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.putText(canvas, text, (20, y_offset + i * 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

        # Temporary Notification Banner
        if time.time() < self.notification_expiry:
            banner_y = h - 60
            cv2.rectangle(canvas, (10, banner_y - 25), (w - 10, banner_y + 15), (0, 0, 0), -1)
            cv2.rectangle(canvas, (10, banner_y - 25), (w - 10, banner_y + 15), self.notification_color, 2)
            cv2.putText(canvas, self.notification_text, (25, banner_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.notification_color, 2, cv2.LINE_AA)

        # Interactive Controls Footer Banner
        footer = "[C] Capture View   [F] Finish Session   [R] Reset/New Session   [P] Pause   [Q/ESC] Exit"
        cv2.putText(canvas, footer, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        if paused:
            cv2.putText(canvas, "PAUSED", (w // 2 - 80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)


# ===========================================================================
# AUTOMATED SELF-TEST FOR MODULE 2
# ===========================================================================
def run_module2_self_test() -> bool:
    """
    Executes automated self-test on Module 2 multi-view capture & dataset generation:
    1. Initializes RealSense D455f pipeline.
    2. Collects synchronized RGB-D frames.
    3. Executes quality-gate validation.
    4. Simulates a 2-view capture session (view_001, view_002).
    5. Verifies filesystem hierarchy (rgb.png, depth.npy, pointcloud.ply, pointcloud.csv, metadata.json).
    6. Confirms root session metadata.json is well-formed.
    7. Prints dataset validation report.
    """
    print("\n" + "=" * 70)
    print("  RUNNING MODULE 2 AUTOMATED DIAGNOSTIC SELF-TEST")
    print("=" * 70)

    dataset_manager = MultiViewDatasetManager()
    cam = RealSenseCamera(width=1280, height=720, fps=30)
    if not cam.start(warmup=True):
        print("[FAIL] RealSense device initialization failed.")
        return False

    generator = PointCloudGenerator(cam.color_intrinsics)

    print("\n[TEST 1/4] Capturing simulated View 001...")
    success, color_img, depth_m = cam.get_aligned_frames()
    if not success or color_img is None or depth_m is None:
        print("[FAIL] Frame acquisition failed.")
        cam.stop()
        return False

    pcd1 = generator.deproject_to_point_cloud(color_img, depth_m)
    val1 = dataset_manager.validate_candidate_view(color_img, depth_m, pcd1)
    if not val1.is_valid:
        print(f"[FAIL] Quality gate validation rejected view 1: {val1.rejection_reason}")
        cam.stop()
        return False

    saved1, msg1, view1_dir = dataset_manager.save_view(
        color_img, depth_m, pcd1, cam.device_info, cam.color_intrinsics, 30.0, val1
    )
    if not saved1:
        print(f"[FAIL] Failed to save view 1: {msg1}")
        cam.stop()
        return False
    print(f"  --> Saved View 1: {view1_dir}")
    print("  [PASS] View 1 captured and written to disk.")

    time.sleep(0.5)

    print("\n[TEST 2/4] Capturing simulated View 002...")
    success, color_img2, depth_m2 = cam.get_aligned_frames()
    pcd2 = generator.deproject_to_point_cloud(color_img2, depth_m2)
    val2 = dataset_manager.validate_candidate_view(color_img2, depth_m2, pcd2)
    saved2, msg2, view2_dir = dataset_manager.save_view(
        color_img2, depth_m2, pcd2, cam.device_info, cam.color_intrinsics, 30.0, val2
    )
    if not saved2:
        print(f"[FAIL] Failed to save view 2: {msg2}")
        cam.stop()
        return False
    print(f"  --> Saved View 2: {view2_dir}")
    print("  [PASS] View 2 captured and written to disk.")

    print("\n[TEST 3/4] Verifying dataset directory contents & file formats...")
    expected_files = ["rgb.png", "depth.npy", "pointcloud.ply", "pointcloud.csv", "metadata.json"]
    for view_dir in [view1_dir, view2_dir]:
        for fname in expected_files:
            fpath = os.path.join(view_dir, fname)
            if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
                print(f"[FAIL] Missing or empty file: {fpath}")
                cam.stop()
                return False
            print(f"  --> Verified {fname} ({os.path.getsize(fpath):,} bytes)")

    # Verify session root metadata
    session_json = os.path.join(dataset_manager.current_session_dir, "metadata.json")
    if not os.path.isfile(session_json):
        print("[FAIL] Missing session root metadata.json")
        cam.stop()
        return False
    with open(session_json, "r") as f:
        meta_data = json.load(f)
        if meta_data.get("total_views") != 2:
            print("[FAIL] Session metadata view count mismatch.")
            cam.stop()
            return False
    print("  [PASS] Directory hierarchy, point clouds, images, and JSON metadata verified.")

    print("\n[TEST 4/4] Printing final Dataset Validation Report...")
    dataset_manager.print_dataset_validation_report()

    cam.stop()
    print("=" * 70)
    print("  MODULE 2 SELF-TEST COMPLETED SUCCESSFULLY [4/4]")
    print("=" * 70 + "\n")
    return True


# ===========================================================================
# MAIN INTERACTIVE MULTI-VIEW CAPTURE LOOP
# ===========================================================================
def run_multiview_capture(width: int = DEFAULT_STREAM_WIDTH, height: int = DEFAULT_STREAM_HEIGHT, fps: int = DEFAULT_FPS) -> None:
    """Runs interactive multi-view capture session."""
    dataset_manager = MultiViewDatasetManager()
    cam = RealSenseCamera(width=width, height=height, fps=fps)
    if not cam.start(warmup=True):
        sys.exit(1)

    generator = PointCloudGenerator(cam.color_intrinsics)
    visualizer = PointCloudVisualizer(width=640, height=480)
    ui = MultiViewCaptureUI(visualizer)

    cv2.namedWindow(WIN_RGB, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_DEPTH, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_3D_VIZ, cv2.WINDOW_AUTOSIZE)

    paused = False
    last_color_img = None
    last_depth_m = None
    last_pcd = None

    frame_count = 0
    t_start = time.time()
    current_fps = float(fps)

    print("\n" + "=" * 70)
    print("  MULTI-VIEW 3D CAPTURE SESSION ACTIVE")
    print("=" * 70)
    print(f"  Active Session : {dataset_manager.current_session_id}")
    print("  Workflow:")
    print("    1. Position object / camera at View 1 -> Press [C] to Capture")
    print("    2. Reposition object / camera to View 2 -> Press [C] to Capture")
    print("    3. Repeat for desired number of viewpoints (e.g. 4-8 views)")
    print("    4. Press [F] to Finish session and print dataset summary")
    print("    5. Press [R] to start a brand new session")
    print("    6. Press [Q] or [ESC] to Exit\n")

    try:
        while True:
            if not paused:
                success, color_img, depth_m = cam.get_aligned_frames()
                if not success or color_img is None or depth_m is None:
                    continue

                pcd = generator.deproject_to_point_cloud(color_img, depth_m)

                last_color_img = color_img.copy()
                last_depth_m = depth_m.copy()
                last_pcd = pcd

                frame_count += 1
                t_elapsed = time.time() - t_start
                if t_elapsed >= 1.0:
                    current_fps = frame_count / t_elapsed
                    frame_count = 0
                    t_start = time.time()
            else:
                color_img = last_color_img.copy()
                depth_m = last_depth_m.copy()
                pcd = last_pcd

            # Validate current candidate frame
            val = dataset_manager.validate_candidate_view(color_img, depth_m, pcd)

            # Draw HUD
            display_rgb = color_img.copy()
            ui.draw_multiview_hud(
                display_rgb,
                session_id=dataset_manager.current_session_id,
                view_count=len(dataset_manager.captured_views),
                pcd=pcd,
                val=val,
                dev_info=cam.device_info,
                intrinsics=cam.color_intrinsics,
                fps=current_fps,
                paused=paused
            )

            # Depth Colormap
            depth_colormap = visualizer.create_depth_colormap(depth_m)

            # 3D Orthographic projection view
            view_3d = visualizer.render_3d_ortho_projection(pcd, downsample_factor=10)

            scaled_rgb = cv2.resize(display_rgb, (854, 480))
            scaled_depth = cv2.resize(depth_colormap, (640, 480))

            cv2.imshow(WIN_RGB, scaled_rgb)
            cv2.imshow(WIN_DEPTH, scaled_depth)
            cv2.imshow(WIN_3D_VIZ, view_3d)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # Quit
                print("[INFO] Exiting multi-view capture application...")
                break

            elif key in [ord('p'), ord('P')]:     # Pause toggle
                paused = not paused
                state_str = "PAUSED" if paused else "RESUMED"
                print(f"[INFO] Feed {state_str}")

            elif key in [ord('c'), ord('C')]:     # Capture View
                saved, msg, view_dir = dataset_manager.save_view(
                    color_img, depth_m, pcd, cam.device_info, cam.color_intrinsics, current_fps, val
                )
                if saved:
                    ui.set_notification(f"CAPTURED {os.path.basename(view_dir).upper()} ({val.valid_3d_points:,} pts)", success=True)
                else:
                    ui.set_notification(f"CAPTURE REJECTED: {val.rejection_reason}", success=False)
                    print(f"[WARN] {msg}")

            elif key in [ord('f'), ord('F')]:     # Finish Session
                print("\n[ACTION] Finishing current capture session...")
                dataset_manager.print_dataset_validation_report()
                ui.set_notification(f"SESSION {dataset_manager.current_session_id} FINISHED ({len(dataset_manager.captured_views)} views)", success=True)

            elif key in [ord('r'), ord('R')]:     # Reset / Start New Session
                old_session = dataset_manager.current_session_id
                dataset_manager.print_dataset_validation_report()
                new_session = dataset_manager.start_new_session()
                ui.set_notification(f"STARTED NEW SESSION: {new_session}", success=True)

    finally:
        dataset_manager.print_dataset_validation_report()
        cam.stop()
        cv2.destroyAllWindows()


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 2: RealSense D455f Multi-View Point-Cloud Capture and Dataset Management"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run non-interactive automated diagnostic self-test and exit."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_STREAM_WIDTH,
        help=f"Stream width (default: {DEFAULT_STREAM_WIDTH})"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_STREAM_HEIGHT,
        help=f"Stream height (default: {DEFAULT_STREAM_HEIGHT})"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Stream frame rate (default: {DEFAULT_FPS})"
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_module2_self_test()
        sys.exit(0 if success else 1)
    else:
        run_multiview_capture(width=args.width, height=args.height, fps=args.fps)
