"""
depth_to_3d.py
==============
Stage 2 — Metric 3D Coordinate Recovery from RealSense D455f Depth Pixels

Purpose:
    Demonstrate and validate the conversion of a 2D depth-image pixel (u, v)
    with its associated depth value Z into a calibrated metric 3D point
    (X, Y, Z) in the RealSense camera coordinate system.

    No object detection, segmentation, or dimension measurement is performed.
    This stage exists purely to prove that our depth-to-3D pipeline is
    numerically correct and ready to serve future measurement stages.

RealSense Camera Coordinate System
-----------------------------------
    +X  →  right   (along sensor width)
    +Y  ↓  down    (along sensor height)
    +Z  →  forward (out of the lens, toward the scene)

    The origin is the depth sensor optical centre.
    All coordinates are in metres.

Deprojection formula (pinhole model)
--------------------------------------
    Z_m  = raw_depth × depth_scale                (raw uint16 → metres)
    X_m  = (u - cx) × Z_m / fx
    Y_m  = (v - cy) × Z_m / fy

    In code we call the official SDK function:
        rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], Z_m)
    which implements the same formula and additionally handles the camera's
    calibrated distortion model stored in the intrinsics object.

Usage:
    python depth_to_3d.py

    Move the mouse over the depth window to inspect any pixel.
    Press Q / ESC / Ctrl-C to quit.

Author : (your name)
Date   : 2026-09-21
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import sys
import signal
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import pyrealsense2 as rs
import numpy as np
import cv2

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Depth stream (D455f native resolution)
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30

# Colouriser preset  0=Jet  1=Classic  2=WhiteToBlack  3=BlackToWhite
COLORIZER_PRESET  = 0
# Colouriser clip range (metres) — only affects visualisation, NOT raw data
DEPTH_VIS_MIN_M   = 0.1
DEPTH_VIS_MAX_M   = 4.0

# Warm-up frames to discard (camera auto-exposure stabilisation)
WARMUP_FRAMES = 15

# Marker appearance
MARKER_RADIUS    = 6
MARKER_COLOR     = (0, 255, 255)   # cyan
MARKER_THICKNESS = 2

# HUD text properties
HUD_FONT       = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE      = 0.55
HUD_COLOR_OK   = (0, 255, 0)      # green  — valid depth
HUD_COLOR_ERR  = (0, 60, 255)     # red-ish — invalid depth
HUD_THICKNESS  = 1
HUD_LINE_AA    = cv2.LINE_AA
HUD_LINE_H     = 22               # pixels between HUD text lines

# Window name — used by cv2.setMouseCallback
WINDOW_NAME = "D455f — Depth 3D Inspector"


# ===========================================================================
# Graceful Ctrl-C shutdown
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
    """
    Holds the depth stream intrinsics retrieved live from the camera.
    Never hard-coded; always sourced from the active pipeline profile.
    """
    width:  int
    height: int
    fx:     float   # focal length x  (pixels)
    fy:     float   # focal length y  (pixels)
    cx:     float   # principal point x  (pixels)
    cy:     float   # principal point y  (pixels)
    rs_intr: rs.intrinsics   # raw SDK object (kept for rs2_deproject call)


@dataclass
class Point3D:
    """Metric 3D point in the RealSense camera coordinate system."""
    X: float   # metres, +right
    Y: float   # metres, +down
    Z: float   # metres, +forward


@dataclass
class PixelProbe:
    """Everything known about the currently inspected pixel."""
    u:         int
    v:         int
    raw_depth: int             # raw uint16 z16 value
    depth_m:   float           # Z in metres  (0.0 if invalid)
    point3d:   Optional[Point3D]   # None if depth is invalid
    valid:     bool


# ===========================================================================
# Camera initialisation
# ===========================================================================

def initialize_camera() -> Tuple[rs.pipeline, rs.pipeline_profile]:
    """
    Detect the D455f, configure and start a depth-only pipeline.

    Returns
    -------
    pipeline : rs.pipeline
    profile  : rs.pipeline_profile

    Raises
    ------
    RuntimeError if no camera is found or the pipeline fails to start.
    """
    ctx     = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No Intel RealSense device detected.\n"
            "  * Ensure the D455f is connected via USB 3.x.\n"
            "  * Close any other application that may have opened the camera\n"
            "    (e.g., RealSense Viewer, camera_validation.py)."
        )

    device = devices[0]
    _print_device_banner(device)

    pipeline = rs.pipeline()
    cfg      = rs.config()

    # Depth stream only for this stage — we will add colour later
    cfg.enable_stream(
        rs.stream.depth,
        DEPTH_WIDTH, DEPTH_HEIGHT,
        rs.format.z16,
        DEPTH_FPS
    )

    try:
        profile = pipeline.start(cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to start RealSense pipeline:\n  {exc}\n\n"
            "  * Check that no other process is using the camera.\n"
            "  * Try a different USB 3.x port / cable."
        )

    return pipeline, profile


def _print_device_banner(device: rs.device) -> None:
    """Print device hardware info to stdout."""
    print("\n" + "=" * 60)
    print("  DEVICE")
    print("=" * 60)
    for label, field in [
        ("Name",             rs.camera_info.name),
        ("Serial",           rs.camera_info.serial_number),
        ("Firmware",         rs.camera_info.firmware_version),
        ("Product line",     rs.camera_info.product_line),
        ("USB type",         rs.camera_info.usb_type_descriptor),
    ]:
        try:
            if device.supports(field):
                print(f"  {label:<16}: {device.get_info(field)}")
        except Exception:
            pass
    print("=" * 60 + "\n")


# ===========================================================================
# Intrinsics and depth scale
# ===========================================================================

def get_depth_intrinsics(profile: rs.pipeline_profile) -> DepthIntrinsics:
    """
    Retrieve depth stream intrinsics dynamically from the active profile.

    IMPORTANT: Values are NOT hard-coded. They come from the connected
    camera's calibration data, so they are correct for this specific device
    and resolution combination.

    Parameters
    ----------
    profile : rs.pipeline_profile

    Returns
    -------
    DepthIntrinsics dataclass
    """
    stream_profile = profile.get_stream(rs.stream.depth)
    vsp            = stream_profile.as_video_stream_profile()
    intr           = vsp.get_intrinsics()   # rs.intrinsics object

    return DepthIntrinsics(
        width    = intr.width,
        height   = intr.height,
        fx       = intr.fx,
        fy       = intr.fy,
        cx       = intr.ppx,   # ppx == cx  (principal point x)
        cy       = intr.ppy,   # ppy == cy  (principal point y)
        rs_intr  = intr,
    )


def get_depth_scale(profile: rs.pipeline_profile) -> float:
    """
    Return the depth scale in metres per raw depth unit.

    depth_metres = raw_z16_value * depth_scale

    Default for D455f : 0.001  (1 mm per unit → 1000 units = 1 m)
    """
    depth_sensor = profile.get_device().first_depth_sensor()
    return depth_sensor.get_depth_scale()


def print_intrinsics_report(di: DepthIntrinsics,
                            depth_scale: float) -> None:
    """Print a formatted intrinsics table to the terminal."""
    sep = "-" * 60
    fov = rs.rs2_fov(di.rs_intr)

    print(sep)
    print("  DEPTH STREAM INTRINSICS  (sourced live from camera)")
    print(sep)
    print(f"  Resolution   : {di.width} x {di.height} px")
    print(f"  Focal length : fx = {di.fx:.4f} px   fy = {di.fy:.4f} px")
    print(f"  Principal pt : cx = {di.cx:.4f} px   cy = {di.cy:.4f} px")
    print(f"  Field of view: H = {fov[0]:.2f} deg   V = {fov[1]:.2f} deg")
    print(f"  Distortion   : model={di.rs_intr.model}")
    print(f"                 coeffs={[round(c,6) for c in di.rs_intr.coeffs]}")
    print()
    print(f"  Depth scale  : {depth_scale:.8f} m / depth-unit")
    print(f"  (raw 1000 -> {1000 * depth_scale:.4f} m = "
          f"{1000 * depth_scale * 100:.2f} cm)")
    print(sep + "\n")

    print("  COORDINATE SYSTEM")
    print(sep)
    print("  +X  →  right    (along sensor width)")
    print("  +Y  ↓  down     (along sensor height)")
    print("  +Z  →  forward  (out of lens, toward scene)")
    print("  Origin at depth sensor optical centre.")
    print("  All coordinates in metres.")
    print(sep + "\n")


# ===========================================================================
# Deprojection
# ===========================================================================

def deproject_pixel(u: int, v: int,
                    raw_depth: int,
                    depth_scale: float,
                    di: DepthIntrinsics) -> PixelProbe:
    """
    Convert a depth-image pixel (u, v) and its raw z16 depth value into a
    calibrated metric 3D point using the official SDK deprojection function.

    Deprojection formula (pinhole + distortion via SDK):
        Z_m  = raw_depth * depth_scale
        [X, Y, Z] = rs2_deproject_pixel_to_point(intrinsics, [u, v], Z_m)

    Parameters
    ----------
    u, v        : pixel coordinates (column, row) — integers
    raw_depth   : raw uint16 value from the z16 depth array
    depth_scale : metres per depth unit
    di          : DepthIntrinsics from the active profile

    Returns
    -------
    PixelProbe dataclass
    """
    if raw_depth == 0:
        # Zero means no valid depth return for this pixel
        return PixelProbe(u=u, v=v, raw_depth=0,
                          depth_m=0.0, point3d=None, valid=False)

    depth_m = float(raw_depth) * depth_scale

    # Official SDK deprojection — accounts for the camera's distortion model
    point = rs.rs2_deproject_pixel_to_point(
        di.rs_intr,
        [float(u), float(v)],
        depth_m
    )
    # Returns [X, Y, Z] in metres, RealSense camera coordinate system

    return PixelProbe(
        u=u, v=v,
        raw_depth=raw_depth,
        depth_m=depth_m,
        point3d=Point3D(X=point[0], Y=point[1], Z=point[2]),
        valid=True
    )


# ===========================================================================
# Numerical validation (terminal print)
# ===========================================================================

def print_probe_terminal(probe: PixelProbe, depth_scale: float) -> None:
    """
    Print a structured numerical validation report to the terminal.
    Called once per mouse-click or on first frame.
    """
    print()
    print("=" * 50)
    print("  PIXEL PROBE")
    print("=" * 50)
    print(f"  Pixel       : ({probe.u}, {probe.v})")
    print(f"  Raw depth   : {probe.raw_depth}  (z16 uint16)")
    print(f"  Depth scale : {depth_scale:.8f} m/unit")

    if probe.valid and probe.point3d is not None:
        p = probe.point3d
        print(f"  Z (depth)   : {probe.depth_m:.4f} m  = "
              f"{probe.depth_m * 100:.2f} cm")
        print(f"  --- 3D Point (camera frame) ---")
        print(f"  X           : {p.X:+.4f} m  = {p.X * 100:+.2f} cm  (+right)")
        print(f"  Y           : {p.Y:+.4f} m  = {p.Y * 100:+.2f} cm  (+down)")
        print(f"  Z           : {p.Z:+.4f} m  = {p.Z * 100:+.2f} cm  (+forward)")
    else:
        print("  Depth       : INVALID (raw=0, no return)")
    print("=" * 50)


# ===========================================================================
# Mouse callback state
# ===========================================================================

class MouseState:
    """Shared mutable state updated by the OpenCV mouse callback."""
    def __init__(self, init_u: int, init_v: int):
        self.u: int = init_u
        self.v: int = init_v
        self.clicked: bool = False   # True if user clicked (triggers terminal print)


def make_mouse_callback(state: MouseState, depth_array_ref: list,
                        depth_scale: float, di: DepthIntrinsics):
    """
    Factory that returns an OpenCV mouse callback closure.

    Uses a list reference (depth_array_ref[0]) so the callback always reads
    the latest depth frame without needing a global variable.

    Left-click  → prints full probe report to terminal.
    Mouse move  → updates HUD coordinates in real time.
    """
    def callback(event, x, y, flags, param):
        state.u = x
        state.v = y

        if event == cv2.EVENT_LBUTTONDOWN:
            state.clicked = True
            depth_arr = depth_array_ref[0]
            if depth_arr is not None:
                h, w = depth_arr.shape
                if 0 <= y < h and 0 <= x < w:
                    raw = int(depth_arr[y, x])
                    probe = deproject_pixel(x, y, raw, depth_scale, di)
                    print_probe_terminal(probe, depth_scale)

    return callback


# ===========================================================================
# HUD overlay
# ===========================================================================

def draw_hud(image: np.ndarray, probe: PixelProbe) -> None:
    """
    Render pixel coordinates, raw depth, Z, and 3D XYZ onto the image.

    Layout (top-left origin):
        Line 1 : Pixel: (u, v)   Raw: XXXX
        Line 2 : Z = X.XXX m  (XXX.XX cm)
        Line 3 : X = ±X.XXX m   Y = ±X.XXX m
        Line 4 : [instruction]
    """
    color  = HUD_COLOR_OK if probe.valid else HUD_COLOR_ERR
    lh     = HUD_LINE_H
    x0, y0 = 10, 28

    def put(line_idx: int, text: str, c=None) -> None:
        cv2.putText(image, text,
                    (x0, y0 + line_idx * lh),
                    HUD_FONT, HUD_SCALE, c or color,
                    HUD_THICKNESS, HUD_LINE_AA)

    put(0, f"Pixel: ({probe.u}, {probe.v})   Raw depth: {probe.raw_depth}")

    if probe.valid and probe.point3d is not None:
        p = probe.point3d
        put(1, f"Z = {p.Z:.4f} m  ({p.Z*100:.2f} cm)  [ +forward ]")
        put(2, f"X = {p.X:+.4f} m  ({p.X*100:+.2f} cm)  [ +right ]")
        put(3, f"Y = {p.Y:+.4f} m  ({p.Y*100:+.2f} cm)  [ +down  ]")
    else:
        put(1, "Depth UNAVAILABLE  (invalid / out of range)")

    # instruction line at bottom
    h = image.shape[0]
    cv2.putText(image,
                "Move mouse to inspect | Left-click to log to terminal | Q/ESC to quit",
                (10, h - 10),
                HUD_FONT, 0.45, (180, 180, 180), 1, HUD_LINE_AA)


def draw_marker(image: np.ndarray, u: int, v: int,
                valid: bool) -> None:
    """Draw a crosshair + circle at the probe pixel."""
    h, w = image.shape[:2]
    u = max(MARKER_RADIUS, min(w - MARKER_RADIUS - 1, u))
    v = max(MARKER_RADIUS, min(h - MARKER_RADIUS - 1, v))

    col = MARKER_COLOR if valid else (0, 60, 255)
    cv2.circle(image, (u, v), MARKER_RADIUS, col, MARKER_THICKNESS,
               cv2.LINE_AA)
    cv2.line(image, (u - MARKER_RADIUS - 4, v),
                    (u + MARKER_RADIUS + 4, v), col, 1, cv2.LINE_AA)
    cv2.line(image, (u, v - MARKER_RADIUS - 4),
                    (u, v + MARKER_RADIUS + 4), col, 1, cv2.LINE_AA)


# ===========================================================================
# Main display loop
# ===========================================================================

def display_depth(pipeline: rs.pipeline,
                  di: DepthIntrinsics,
                  depth_scale: float) -> None:
    """
    Main loop: acquire depth frames, run deprojection on mouse position,
    draw HUD, display window.

    Parameters
    ----------
    pipeline    : rs.pipeline  (already started)
    di          : DepthIntrinsics  (from active profile)
    depth_scale : float
    """
    global _shutdown_requested

    # Colouriser for depth visualisation
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, COLORIZER_PRESET)
    colorizer.set_option(rs.option.min_distance, DEPTH_VIS_MIN_M)
    colorizer.set_option(rs.option.max_distance, DEPTH_VIS_MAX_M)

    # Mouse state — starts at image centre
    init_u   = di.width  // 2
    init_v   = di.height // 2
    mouse    = MouseState(init_u, init_v)

    # Mutable list so the callback can always read the latest raw depth array
    depth_array_ref: list = [None]

    # Warm-up
    print(f"[INFO] Warming up — discarding {WARMUP_FRAMES} frames ...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()
    print("[INFO] Warm-up complete. Live window opening ...\n")
    print("  Move the mouse over the depth window to see 3D coordinates.")
    print("  Left-click to print a full validation report to this terminal.")
    print("  Press Q or ESC to quit.\n")

    # Create window and bind mouse callback
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(
        WINDOW_NAME,
        make_mouse_callback(mouse, depth_array_ref, depth_scale, di)
    )

    # Print initial centre-pixel probe immediately (before user interaction)
    first_print_done = False

    try:
        while not _shutdown_requested:
            frameset    = pipeline.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                continue

            # Raw depth array (uint16) — shared with mouse callback
            depth_arr            = np.asanyarray(depth_frame.get_data())
            depth_array_ref[0]   = depth_arr

            # Colourised depth image for display (BGR uint8)
            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data()
            )

            # ---- Probe at mouse position ----------------------------------
            u = np.clip(mouse.u, 0, di.width  - 1)
            v = np.clip(mouse.v, 0, di.height - 1)

            raw_depth = int(depth_arr[v, u])
            probe     = deproject_pixel(u, v, raw_depth, depth_scale, di)

            # First frame — auto-print the centre-pixel probe to terminal
            if not first_print_done:
                print("[INFO] Auto-probe at image centre on first frame:")
                print_probe_terminal(probe, depth_scale)
                first_print_done = True

            # ---- Draw overlay --------------------------------------------
            display = depth_colormap.copy()
            draw_marker(display, u, v, probe.valid)
            draw_hud(display, probe)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Quit key pressed — shutting down ...")
                break

    except Exception as exc:
        print(f"\n[ERROR] Capture loop error:\n  {exc}")
        raise

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped. Validation complete.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Program entry point."""
    print("\n" + "#" * 60)
    print("  RealSense D455f — Stage 2: Depth → 3D Coordinate Probe")
    print("#" * 60 + "\n")

    try:
        pipeline, profile = initialize_camera()
        di          = get_depth_intrinsics(profile)
        depth_scale = get_depth_scale(profile)

        print_intrinsics_report(di, depth_scale)
        display_depth(pipeline, di, depth_scale)

    except RuntimeError as err:
        print(f"\n[FATAL] {err}")
        sys.exit(1)
    except Exception as err:
        print(f"\n[FATAL] Unexpected error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
