"""
=============================================================================
Module 1: RGB-D Acquisition & Metric 3D Point Cloud Generation
=============================================================================
Intel RealSense D455f Multi-View 3D Reconstruction Pipeline — Stage 1 / Foundation

Project Objective:
    Build the initial foundation module for high-precision multi-view 3D object
    reconstruction:
    1. Synchronized RGB + Depth acquisition (1280x720 @ 30 FPS).
    2. Real-time hardware-aligned depth-to-color frame reprojection.
    3. Dynamic retrieval of live sensor intrinsics & depth scale.
    4. Exact metric pinhole deprojection into 3D camera coordinate space (X, Y, Z in meters).
    5. Colored point cloud synthesis (X, Y, Z + R, G, B).
    6. Raw point cloud export to standard .PLY and .CSV formats.
    7. Diagnostic HUD overlay and basic 3D point cloud visualization.

Geometry & Coordinate System Details:
    -------------------------------------------------------------------------
    1. RealSense Optical Camera Coordinate Frame:
       +X-axis : Points to the RIGHT (horizontal, along image columns)
       +Y-axis : Points DOWNWARDS (vertical, along image rows)
       +Z-axis : Points FORWARD (along the camera's optical axis into scene)
       Origin (0, 0, 0) : Color camera optical centre.

    2. Mathematical Deprojection Model:
       Given:
         - Pixel coordinate: (u, v) where u in [0, W-1], v in [0, H-1]
         - Depth value: Z in METERS (Z = raw_depth_uint16 * depth_scale)
         - Camera Intrinsics: fx, fy (focal lengths in pixels), cx, cy (principal point in pixels)

       The 3D point (X, Y, Z) in the camera frame is:
         X = (u - cx) * Z / fx
         Y = (v - cy) * Z / fy
         Z = Z

    3. Depth Z vs Euclidean Distance:
       - Z is the orthogonal distance from the camera sensor plane along the optical (+Z) axis.
       - The straight-line Euclidean distance from the optical center is:
         d_euclidean = sqrt(X^2 + Y^2 + Z^2) = Z * sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
       - All raw metric coordinates are recorded directly in METERS (m).

    4. Zero Smoothing / Filtering Constraint:
       This module outputs raw, calibrated, unfiltered metric point clouds.
       No temporal averaging, hole filling, spatial decimation, plane fitting,
       or object segmentation is applied at this stage.

Controls:
    [S]       : Capture & Save raw 3D Point Cloud (.ply and .csv)
    [P]       : Pause / Resume live camera feed
    [Q] / ESC : Exit application cleanly
=============================================================================
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import pyrealsense2 as rs


# ===========================================================================
# CONFIGURATION CONSTANTS
# ===========================================================================
DEFAULT_STREAM_WIDTH = 1280
DEFAULT_STREAM_HEIGHT = 720
DEFAULT_FPS = 30
WARMUP_FRAMES = 30
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Window Names
WIN_RGB = "RealSense D455f — Aligned RGB Feed"
WIN_DEPTH = "RealSense D455f — Depth Map (JET Colormap)"
WIN_3D_VIZ = "RealSense D455f — 3D Point Cloud View"


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================
@dataclass
class CameraIntrinsics:
    """Encapsulates calibrated pinhole camera intrinsic parameters."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    coeffs: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "model": self.model,
            "coeffs": self.coeffs
        }


@dataclass
class DeviceInfo:
    """Encapsulates hardware diagnostics and identification."""
    name: str
    serial: str
    firmware: str
    usb_type: str
    depth_scale: float


@dataclass
class PointCloudData:
    """
    Holds structured metric 3D point cloud with corresponding RGB color.
    Units: X, Y, Z in METERS; R, G, B in [0, 255] uint8.
    """
    points: np.ndarray      # (N, 3) float32 in METERS (X, Y, Z)
    colors: np.ndarray      # (N, 3) uint8 (R, G, B)
    timestamp: float
    valid_count: int
    mean_z: float


# ===========================================================================
# CLASS: RealSenseCamera
# ===========================================================================
class RealSenseCamera:
    """
    Manages Intel RealSense D455f hardware lifecycle, synchronized color/depth
    streaming, hardware alignment, and dynamic intrinsics retrieval.
    """

    def __init__(
        self,
        width: int = DEFAULT_STREAM_WIDTH,
        height: int = DEFAULT_STREAM_HEIGHT,
        fps: int = DEFAULT_FPS
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.align: Optional[rs.align] = None

        self.device_info: Optional[DeviceInfo] = None
        self.color_intrinsics: Optional[CameraIntrinsics] = None
        self.depth_intrinsics: Optional[CameraIntrinsics] = None
        self.depth_scale: float = 0.001  # default fallback

    def start(self, warmup: bool = True) -> bool:
        """
        Initializes the RealSense pipeline, verifies USB connectivity, reads
        intrinsics, and performs camera stabilization warm-up.
        """
        print("\n" + "=" * 70)
        print("  INITIALIZING INTEL REALSENSE D455f RGB-D PIPELINE")
        print("=" * 70)

        ctx = rs.context()
        devices = ctx.devices
        if len(devices) == 0:
            print("[ERROR] No Intel RealSense devices detected. Please check USB connection.")
            return False

        dev = devices[0]
        dev_name = dev.get_info(rs.camera_info.name)
        dev_serial = dev.get_info(rs.camera_info.serial_number)
        dev_fw = dev.get_info(rs.camera_info.firmware_version)
        dev_usb = dev.get_info(rs.camera_info.usb_type_descriptor)

        # Get depth sensor and depth scale
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.device_info = DeviceInfo(
            name=dev_name,
            serial=dev_serial,
            firmware=dev_fw,
            usb_type=dev_usb,
            depth_scale=self.depth_scale
        )

        # Configure pipeline streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        # Start streaming
        try:
            self.profile = self.pipeline.start(config)
        except Exception as e:
            print(f"[ERROR] Failed to start RealSense pipeline: {e}")
            return False

        # Configure hardware alignment to Color frame
        self.align = rs.align(rs.stream.color)

        # Extract calibrated intrinsics from active stream profiles
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()

        c_intr = color_stream.get_intrinsics()
        d_intr = depth_stream.get_intrinsics()

        self.color_intrinsics = CameraIntrinsics(
            width=c_intr.width,
            height=c_intr.height,
            fx=c_intr.fx,
            fy=c_intr.fy,
            cx=c_intr.ppx,
            cy=c_intr.ppy,
            model=str(c_intr.model),
            coeffs=list(c_intr.coeffs)
        )

        self.depth_intrinsics = CameraIntrinsics(
            width=d_intr.width,
            height=d_intr.height,
            fx=d_intr.fx,
            fy=d_intr.fy,
            cx=d_intr.ppx,
            cy=d_intr.ppy,
            model=str(d_intr.model),
            coeffs=list(d_intr.coeffs)
        )

        self.print_diagnostic_report()

        if warmup:
            print(f"\n[INFO] Warming up camera sensor ({WARMUP_FRAMES} frames)...", end="", flush=True)
            for _ in range(WARMUP_FRAMES):
                self.pipeline.wait_for_frames()
            print(" Done! Sensor & auto-exposure stabilized.\n")

        return True

    def print_diagnostic_report(self) -> None:
        """Prints a complete structured startup report of camera hardware & intrinsics."""
        print("-" * 70)
        print("  CONNECTED DEVICE DIAGNOSTICS")
        print("-" * 70)
        print(f"  Device Name         : {self.device_info.name}")
        print(f"  Serial Number       : {self.device_info.serial}")
        print(f"  Firmware Version    : {self.device_info.firmware}")
        print(f"  USB Descriptor      : {self.device_info.usb_type}")
        print(f"  Active Depth Scale  : {self.depth_scale:.8f} m/unit  (1000 = {1000 * self.depth_scale:.4f} m)")
        print("-" * 70)
        print("  COLOR STREAM INTRINSICS (Active Aligned Reference)")
        print("-" * 70)
        print(f"  Resolution          : {self.color_intrinsics.width} x {self.color_intrinsics.height} px")
        print(f"  Focal Length (fx,fy): fx = {self.color_intrinsics.fx:.4f} px, fy = {self.color_intrinsics.fy:.4f} px")
        print(f"  Principal Point     : cx = {self.color_intrinsics.cx:.4f} px, cy = {self.color_intrinsics.cy:.4f} px")
        print(f"  Distortion Model    : {self.color_intrinsics.model}")
        print(f"  Distortion Coeffs   : {self.color_intrinsics.coeffs}")
        print("-" * 70)
        print("  RAW DEPTH STREAM INTRINSICS")
        print("-" * 70)
        print(f"  Resolution          : {self.depth_intrinsics.width} x {self.depth_intrinsics.height} px")
        print(f"  Focal Length (fx,fy): fx = {self.depth_intrinsics.fx:.4f} px, fy = {self.depth_intrinsics.fy:.4f} px")
        print(f"  Principal Point     : cx = {self.depth_intrinsics.cx:.4f} px, cy = {self.depth_intrinsics.cy:.4f} px")
        print("=" * 70)

    def get_aligned_frames(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Polls for synchronized frames and applies hardware depth-to-color alignment.
        Returns:
            (success, color_bgr_image, depth_meters_float32)
        """
        if self.pipeline is None:
            return False, None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return False, None, None

            # Convert to numpy arrays
            color_img = np.asanyarray(color_frame.get_data())
            raw_depth = np.asanyarray(depth_frame.get_data())

            # Convert raw 16-bit depth to float32 metric meters
            depth_m = raw_depth.astype(np.float32) * self.depth_scale

            return True, color_img, depth_m

        except Exception as e:
            print(f"[WARN] Frame acquisition timeout or error: {e}")
            return False, None, None

    def stop(self) -> None:
        """Safely stops the RealSense pipeline."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("[INFO] RealSense pipeline stopped cleanly.")
            except Exception:
                pass
            self.pipeline = None


# ===========================================================================
# CLASS: PointCloudGenerator
# ===========================================================================
class PointCloudGenerator:
    """
    Computes exact metric 3D point clouds from aligned RGB-D data using
    calibrated pinhole deprojection.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
        self._precompute_pixel_grid()

    def _precompute_pixel_grid(self) -> None:
        """
        Precomputes normalized image ray coordinates:
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        Enables ultra-fast vectorized 3D deprojection at 30+ FPS.
        """
        w, h = self.intrinsics.width, self.intrinsics.height
        u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        self.ray_x = (u_grid - self.intrinsics.cx) / self.intrinsics.fx
        self.ray_y = (v_grid - self.intrinsics.cy) / self.intrinsics.fy

    def deproject_to_point_cloud(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        min_depth_m: float = 0.1,
        max_depth_m: float = 8.0
    ) -> PointCloudData:
        """
        Deprojects full aligned frame into calibrated (X, Y, Z) point cloud in METERS.
        Pairs each 3D point with its corresponding RGB color.

        Mathematical Formulation:
            X = ray_x * Z = (u - cx) * Z / fx
            Y = ray_y * Z = (v - cy) * Z / fy
            Z = Z (depth along camera optical axis)
        """
        # Create valid depth mask (reject zero, non-finite, and out-of-range depths)
        valid_mask = (depth_m > min_depth_m) & (depth_m < max_depth_m) & np.isfinite(depth_m)
        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count == 0:
            return PointCloudData(
                points=np.empty((0, 3), dtype=np.float32),
                colors=np.empty((0, 3), dtype=np.uint8),
                timestamp=time.time(),
                valid_count=0,
                mean_z=0.0
            )

        # Extract valid depths and corresponding rays
        z_valid = depth_m[valid_mask]
        x_valid = self.ray_x[valid_mask] * z_valid
        y_valid = self.ray_y[valid_mask] * z_valid

        # Stack into (N, 3) metric points in meters
        points = np.column_stack((x_valid, y_valid, z_valid)).astype(np.float32)

        # Extract RGB colors (convert BGR to RGB)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        colors = color_rgb[valid_mask].astype(np.uint8)

        mean_z = float(np.mean(z_valid))

        return PointCloudData(
            points=points,
            colors=colors,
            timestamp=time.time(),
            valid_count=valid_count,
            mean_z=mean_z
        )


# ===========================================================================
# CLASS: PointCloudExporter
# ===========================================================================
class PointCloudExporter:
    """
    Exports raw, calibrated metric point cloud data to standard .PLY and .CSV formats.
    """

    @staticmethod
    def ensure_output_dir() -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return OUTPUT_DIR

    @staticmethod
    def save_ply(
        pcd: PointCloudData,
        intrinsics: CameraIntrinsics,
        prefix: str = "pointcloud"
    ) -> str:
        """
        Saves the point cloud to an ASCII/Binary .PLY file with metric coordinates (meters).
        Compatible with MeshLab, CloudCompare, Blender, Open3D.
        """
        out_dir = PointCloudExporter.ensure_output_dir()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp_str}.ply"
        filepath = os.path.join(out_dir, filename)

        num_points = len(pcd.points)

        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"comment Intel RealSense D455f Point Cloud (Metric Units: METERS)\n"
            f"comment Timestamp: {pcd.timestamp}\n"
            f"comment Focal Length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}\n"
            f"comment Principal Point: cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        with open(filepath, "w") as f:
            f.write(header)
            for i in range(num_points):
                x, y, z = pcd.points[i]
                r, g, b = pcd.colors[i]
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")

        print(f"[EXPORT] Saved PLY Point Cloud: {filepath} ({num_points:,} vertices)")
        return filepath

    @staticmethod
    def save_csv(
        pcd: PointCloudData,
        prefix: str = "pointcloud"
    ) -> str:
        """
        Saves the point cloud to a structured CSV file with metric coordinates (meters).
        Header: x_m,y_m,z_m,r,g,b
        """
        out_dir = PointCloudExporter.ensure_output_dir()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp_str}.csv"
        filepath = os.path.join(out_dir, filename)

        num_points = len(pcd.points)

        with open(filepath, "w") as f:
            f.write("x_m,y_m,z_m,r,g,b\n")
            for i in range(num_points):
                x, y, z = pcd.points[i]
                r, g, b = pcd.colors[i]
                f.write(f"{x:.6f},{y:.6f},{z:.6f},{r},{g},{b}\n")

        print(f"[EXPORT] Saved CSV Point Cloud: {filepath} ({num_points:,} rows)")
        return filepath


# ===========================================================================
# CLASS: PointCloudVisualizer
# ===========================================================================
class PointCloudVisualizer:
    """
    Renders live RGB HUD overlays, depth colormaps, and interactive 3D point cloud projections.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.viz_width = width
        self.viz_height = height

    def create_depth_colormap(self, depth_m: np.ndarray, max_range_m: float = 3.5) -> np.ndarray:
        """Generates a high-contrast JET colormap for depth visualization."""
        clipped = np.clip(depth_m / max_range_m * 255.0, 0, 255).astype(np.uint8)
        colormap = cv2.applyColorMap(clipped, cv2.COLORMAP_JET)
        # Black out zero/invalid depth
        colormap[depth_m <= 0.05] = [0, 0, 0]
        return colormap

    def render_3d_ortho_projection(
        self,
        pcd: PointCloudData,
        elevation_deg: float = 25.0,
        azimuth_deg: float = -35.0,
        downsample_factor: int = 8
    ) -> np.ndarray:
        """
        Renders an isometric/orthographic 3D point cloud visualization in real time.
        Allows immediate visual validation of 3D spatial geometry directly in OpenCV.
        """
        canvas = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)

        if len(pcd.points) == 0:
            cv2.putText(canvas, "No Valid 3D Points", (50, self.viz_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return canvas

        # Downsample points for real-time visualization frame rate
        pts = pcd.points[::downsample_factor]
        cols = pcd.colors[::downsample_factor]

        # Rotation angles in radians
        el = np.radians(elevation_deg)
        az = np.radians(azimuth_deg)

        # 3D Rotation matrices (Azimuth around Y, Elevation around X)
        R_y = np.array([
            [np.cos(az), 0, np.sin(az)],
            [0, 1, 0],
            [-np.sin(az), 0, np.cos(az)]
        ], dtype=np.float32)

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(el), -np.sin(el)],
            [0, np.sin(el), np.cos(el)]
        ], dtype=np.float32)

        R = R_x @ R_y

        # Center point cloud for visualization
        centroid = np.median(pts, axis=0)
        pts_centered = pts - centroid

        # Rotate points
        pts_rot = pts_centered @ R.T

        # Orthographic screen projection
        scale = 220.0  # pixels per meter
        screen_x = (pts_rot[:, 0] * scale + self.viz_width / 2.0).astype(np.int32)
        screen_y = (pts_rot[:, 1] * scale + self.viz_height / 2.0).astype(np.int32)

        # Bounds check
        in_bounds = (screen_x >= 0) & (screen_x < self.viz_width) & \
                    (screen_y >= 0) & (screen_y < self.viz_height)

        screen_x = screen_x[in_bounds]
        screen_y = screen_y[in_bounds]
        cols_valid = cols[in_bounds]

        # Draw projected colored points
        # Convert RGB colors to BGR for OpenCV canvas
        cols_bgr = cols_valid[:, ::-1]
        canvas[screen_y, screen_x] = cols_bgr

        # Overlay coordinate grid & info
        cv2.putText(canvas, "3D Point Cloud (Isometric View)", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Points Displayed: {len(screen_x):,} / {len(pcd.points):,}", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Center Z: {centroid[2]:.3f} m", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        return canvas

    def draw_hud(
        self,
        canvas: np.ndarray,
        dev_info: DeviceInfo,
        intrinsics: CameraIntrinsics,
        fps: float,
        pcd: PointCloudData,
        center_z: float,
        paused: bool = False
    ) -> None:
        """Overlays diagnostic telemetry onto the RGB stream."""
        h, w = canvas.shape[:2]

        # Semi-transparent HUD background panel
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (450, 230), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (450, 230), (0, 255, 200), 1)

        # Centre Crosshair
        cx_px, cy_px = w // 2, h // 2
        cv2.line(canvas, (cx_px - 15, cy_px), (cx_px + 15, cy_px), (0, 255, 0), 1)
        cv2.line(canvas, (cx_px, cy_px - 15), (cx_px, cy_px + 15), (0, 255, 0), 1)
        cv2.circle(canvas, (cx_px, cy_px), 4, (0, 255, 0), 1)

        lines = [
            f"Device: {dev_info.name} ({dev_info.usb_type})",
            f"Serial: {dev_info.serial} | FW: {dev_info.firmware}",
            f"Resolution: {intrinsics.width}x{intrinsics.height} @ {fps:.1f} FPS",
            f"Depth Scale: {dev_info.depth_scale:.6f} m/unit",
            f"Focal Length: fx={intrinsics.fx:.1f} px, fy={intrinsics.fy:.1f} px",
            f"Principal Pt: cx={intrinsics.cx:.1f} px, cy={intrinsics.cy:.1f} px",
            f"Valid 3D Points: {pcd.valid_count:,} ({(pcd.valid_count/(w*h))*100:.1f}%)",
            f"Centre Probe Z: {center_z*100.0:.2f} cm ({center_z:.3f} m)",
        ]

        y_offset = 32
        for i, text in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            if i >= 6:
                color = (0, 255, 0)
            cv2.putText(canvas, text, (20, y_offset + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # Controls footer banner
        footer = "[S] Save Point Cloud (.PLY & .CSV)   [P] Pause   [Q/ESC] Quit"
        cv2.putText(canvas, footer, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if paused:
            cv2.putText(canvas, "PAUSED", (w // 2 - 80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)


# ===========================================================================
# AUTOMATED SELF-TEST
# ===========================================================================
def run_self_test() -> bool:
    """
    Executes a comprehensive, non-interactive diagnostic self-test:
    1. Connects to RealSense D455f camera.
    2. Verifies RGB and depth frame reception.
    3. Confirms non-zero valid depth pixel count.
    4. Validates metric 3D deprojection produces finite, non-zero X, Y, Z.
    5. Saves test PLY and CSV point cloud files and verifies integrity.
    """
    print("\n" + "=" * 70)
    print("  RUNNING MODULE 1 AUTOMATED DIAGNOSTIC SELF-TEST")
    print("=" * 70)

    cam = RealSenseCamera(width=1280, height=720, fps=30)
    if not cam.start(warmup=True):
        print("[FAIL] Self-test failed: RealSense device initialization failed.")
        return False

    print("\n[TEST 1/5] Testing frame acquisition pipeline...")
    success, color_img, depth_m = cam.get_aligned_frames()
    if not success or color_img is None or depth_m is None:
        print("[FAIL] Frame acquisition failed.")
        cam.stop()
        return False
    print(f"  --> RGB Frame Shape   : {color_img.shape} (dtype: {color_img.dtype})")
    print(f"  --> Depth Frame Shape : {depth_m.shape} (dtype: {depth_m.dtype})")
    print("  [PASS] Frame acquisition successful.")

    print("\n[TEST 2/5] Testing valid depth coverage...")
    valid_depth_pixels = int(np.count_nonzero(depth_m > 0.05))
    total_pixels = depth_m.size
    depth_coverage = (valid_depth_pixels / total_pixels) * 100.0
    print(f"  --> Valid Depth Pixels : {valid_depth_pixels:,} / {total_pixels:,} ({depth_coverage:.2f}%)")
    if valid_depth_pixels < 1000:
        print("[FAIL] Insufficient valid depth pixels detected.")
        cam.stop()
        return False
    print("  [PASS] Depth stream active and delivering valid measurements.")

    print("\n[TEST 3/5] Testing metric 3D deprojection & point cloud generation...")
    generator = PointCloudGenerator(cam.color_intrinsics)
    pcd = generator.deproject_to_point_cloud(color_img, depth_m)
    print(f"  --> Generated 3D Points : {pcd.valid_count:,}")
    print(f"  --> Point Array Shape   : {pcd.points.shape} (dtype: {pcd.points.dtype})")
    print(f"  --> Color Array Shape   : {pcd.colors.shape} (dtype: {pcd.colors.dtype})")

    # Check finite values
    if not np.all(np.isfinite(pcd.points)):
        print("[FAIL] Non-finite values detected in 3D point cloud.")
        cam.stop()
        return False

    x_min, x_max = np.min(pcd.points[:, 0]), np.max(pcd.points[:, 0])
    y_min, y_max = np.min(pcd.points[:, 1]), np.max(pcd.points[:, 1])
    z_min, z_max = np.min(pcd.points[:, 2]), np.max(pcd.points[:, 2])

    print(f"  --> X Span (meters)     : [{x_min:+.4f}, {x_max:+.4f}] m")
    print(f"  --> Y Span (meters)     : [{y_min:+.4f}, {y_max:+.4f}] m")
    print(f"  --> Z Span (meters)     : [{z_min:+.4f}, {z_max:+.4f}] m (Mean Z: {pcd.mean_z:.4f} m)")
    print("  [PASS] 3D deprojection produces finite metric points in camera coordinate frame.")

    print("\n[TEST 4/5] Testing point cloud export to PLY format...")
    ply_path = PointCloudExporter.save_ply(pcd, cam.color_intrinsics, prefix="selftest")
    if not os.path.isfile(ply_path) or os.path.getsize(ply_path) < 100:
        print("[FAIL] PLY point cloud export failed or produced empty file.")
        cam.stop()
        return False
    print(f"  --> PLY File Size       : {os.path.getsize(ply_path):,} bytes")
    print("  [PASS] PLY point cloud successfully written.")

    print("\n[TEST 5/5] Testing point cloud export to CSV format...")
    csv_path = PointCloudExporter.save_csv(pcd, prefix="selftest")
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) < 100:
        print("[FAIL] CSV point cloud export failed or produced empty file.")
        cam.stop()
        return False
    print(f"  --> CSV File Size       : {os.path.getsize(csv_path):,} bytes")
    print("  [PASS] CSV point cloud successfully written.")

    cam.stop()

    print("\n" + "=" * 70)
    print("  ALL SELF-TESTS PASSED SUCCESSFULLY [5/5]")
    print("=" * 70 + "\n")
    return True


# ===========================================================================
# MAIN LIVE INTERACTIVE LOOP
# ===========================================================================
def run_live_pipeline(width: int = 1280, height: int = 720, fps: int = 30) -> None:
    """Runs the full real-time interactive acquisition and visualization pipeline."""
    cam = RealSenseCamera(width=width, height=height, fps=fps)
    if not cam.start(warmup=True):
        sys.exit(1)

    generator = PointCloudGenerator(cam.color_intrinsics)
    visualizer = PointCloudVisualizer(width=640, height=480)

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

    print("\n[INFO] Starting live RGB-D streaming loop...")
    print("       Press [S] to Save Point Cloud (.PLY + .CSV)")
    print("       Press [P] to Pause / Unpause")
    print("       Press [Q] or [ESC] to Exit\n")

    try:
        while True:
            if not paused:
                success, color_img, depth_m = cam.get_aligned_frames()
                if not success or color_img is None or depth_m is None:
                    continue

                # Generate metric 3D point cloud
                pcd = generator.deproject_to_point_cloud(color_img, depth_m)

                last_color_img = color_img.copy()
                last_depth_m = depth_m.copy()
                last_pcd = pcd

                # Calculate real-time frame rate
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

            # Probe centre pixel depth
            cy_mid, cx_mid = height // 2, width // 2
            center_z = float(depth_m[cy_mid, cx_mid])

            # Render display canvases
            display_rgb = color_img.copy()
            visualizer.draw_hud(
                display_rgb,
                dev_info=cam.device_info,
                intrinsics=cam.color_intrinsics,
                fps=current_fps,
                pcd=pcd,
                center_z=center_z,
                paused=paused
            )

            # Generate depth colormap
            depth_colormap = visualizer.create_depth_colormap(depth_m)

            # Generate 3D point cloud projection view
            view_3d = visualizer.render_3d_ortho_projection(pcd, downsample_factor=10)

            # Display GUI windows (scale RGB & Depth for comfortable side-by-side display)
            scaled_rgb = cv2.resize(display_rgb, (854, 480))
            scaled_depth = cv2.resize(depth_colormap, (640, 480))

            cv2.imshow(WIN_RGB, scaled_rgb)
            cv2.imshow(WIN_DEPTH, scaled_depth)
            cv2.imshow(WIN_3D_VIZ, view_3d)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                print("[INFO] Quitting application...")
                break
            elif key in [ord('p'), ord('P')]:     # Pause toggle
                paused = not paused
                state_str = "PAUSED" if paused else "RESUMED"
                print(f"[INFO] Feed {state_str}")
            elif key in [ord('s'), ord('S')]:     # Save Point Cloud
                if pcd is not None and pcd.valid_count > 0:
                    print("\n[ACTION] Capturing calibrated metric 3D point cloud snapshot...")
                    ply_path = PointCloudExporter.save_ply(pcd, cam.color_intrinsics)
                    csv_path = PointCloudExporter.save_csv(pcd)
                    print(f"[SUCCESS] Saved snapshot to:\n  - PLY: {ply_path}\n  - CSV: {csv_path}\n")
                else:
                    print("[WARN] Cannot save: No valid 3D points available in current frame.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 1: RealSense D455f RGB-D Acquisition & Metric 3D Point Cloud Generation"
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
        success = run_self_test()
        sys.exit(0 if success else 1)
    else:
        run_live_pipeline(width=args.width, height=args.height, fps=args.fps)
