"""
camera_validation.py
====================
Stage 1 — Intel RealSense D455f Camera Validation Script

Purpose:
    Verify that the Intel RealSense D455f RGB-D camera is correctly detected,
    initialised, and streaming valid colour + depth data.

    This script is the foundation of the RealSense Object Dimension Measurement
    project. No object detection or measurement is performed here.

What it does:
    1. Detects and connects to the first available RealSense device.
    2. Configures and starts colour (RGB) and depth streams.
    3. Aligns the depth frame to the colour frame coordinate system.
    4. Retrieves and prints key camera intrinsics and depth scale.
    5. Displays a live colour window and a live depth-colourised window.
    6. Accepts 'q' / Escape / Ctrl-C to quit cleanly.

Author : (your name)
Date   : 2026-09-21
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import sys
import signal

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import pyrealsense2 as rs
import numpy as np
import cv2

# ===========================================================================
# CONFIGURATION — adjust these if your D455f does not support these modes
# ===========================================================================

# Colour stream settings
COLOR_WIDTH  = 1280
COLOR_HEIGHT = 720
COLOR_FPS    = 30

# Depth stream settings  (D455f supports 1280x720 depth natively)
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30

# Colouriser visual preset (0=Jet, 1=Classic, 2=WhiteToBlack, 3=BlackToWhite,
#                           4=Bio, 5=Cold, 6=Warm, 7=Quantized, 8=Pattern)
COLORIZER_PRESET = 0

# Depth clip range for the colouriser visualisation (metres).
# Values outside this range are clipped (does NOT affect raw depth data).
DEPTH_MIN_VIS_M = 0.1   # 10 cm
DEPTH_MAX_VIS_M = 4.0   # 4 m


# ===========================================================================
# Helper — graceful shutdown via Ctrl-C
# ===========================================================================

_shutdown_requested = False

def _signal_handler(sig, frame):
    """Handle SIGINT (Ctrl-C) without a traceback."""
    global _shutdown_requested
    print("\n[INFO] Ctrl-C received — shutting down ...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _signal_handler)


# ===========================================================================
# Camera initialisation
# ===========================================================================

def detect_device(ctx: rs.context) -> rs.device:
    """
    Return the first connected RealSense device or raise RuntimeError.

    Parameters
    ----------
    ctx : rs.context
        An active RealSense context.

    Returns
    -------
    rs.device
        The first detected RealSense device.

    Raises
    ------
    RuntimeError
        If no device is connected.
    """
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No Intel RealSense device detected.\n"
            "  * Check that the D455f is plugged in via USB 3.x.\n"
            "  * Verify the USB cable and port.\n"
            "  * On Linux, ensure udev rules are installed (realsense-rules).\n"
            "  * Try running Intel RealSense Viewer to confirm hardware is OK."
        )
    return devices[0]


def print_device_info(device: rs.device) -> None:
    """Print basic hardware information for the connected device."""
    print("\n" + "=" * 60)
    print("  CONNECTED DEVICE")
    print("=" * 60)

    info_fields = [
        ("Name",                rs.camera_info.name),
        ("Serial Number",       rs.camera_info.serial_number),
        ("Firmware Version",    rs.camera_info.firmware_version),
        ("Product Line",        rs.camera_info.product_line),
        ("USB Type",            rs.camera_info.usb_type_descriptor),
    ]

    for label, field in info_fields:
        try:
            if device.supports(field):
                print(f"  {label:<22}: {device.get_info(field)}")
        except Exception:
            pass   # silently skip unsupported info fields

    print("=" * 60 + "\n")


def build_pipeline() -> tuple:
    """
    Create, configure, and start the RealSense pipeline.

    Streams enabled
    ---------------
    * Colour  -- BGR8 at COLOR_WIDTH x COLOR_HEIGHT @ COLOR_FPS
    * Depth   -- Z16  at DEPTH_WIDTH x DEPTH_HEIGHT @ DEPTH_FPS

    Returns
    -------
    pipeline : rs.pipeline
        The running pipeline.
    profile  : rs.pipeline_profile
        The active pipeline profile (used to query stream parameters).
    """
    pipeline = rs.pipeline()
    cfg      = rs.config()

    # ---- Colour stream -------------------------------------------------------
    cfg.enable_stream(
        rs.stream.color,
        COLOR_WIDTH, COLOR_HEIGHT,
        rs.format.bgr8,          # OpenCV-native BGR order
        COLOR_FPS
    )

    # ---- Depth stream --------------------------------------------------------
    cfg.enable_stream(
        rs.stream.depth,
        DEPTH_WIDTH, DEPTH_HEIGHT,
        rs.format.z16,            # 16-bit unsigned depth in device depth units
        DEPTH_FPS
    )

    try:
        profile = pipeline.start(cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to start RealSense pipeline:\n  {exc}\n\n"
            "Possible causes:\n"
            "  * Resolution/FPS combination not supported by this device.\n"
            "  * Device is in use by another process (e.g. RealSense Viewer).\n"
            "  * USB bandwidth insufficient — try a different USB 3.x port."
        )

    return pipeline, profile


# ===========================================================================
# Intrinsics & depth-scale retrieval
# ===========================================================================

def get_stream_intrinsics(profile: rs.pipeline_profile,
                          stream: rs.stream) -> rs.intrinsics:
    """
    Extract the video intrinsics for the specified stream type.

    The intrinsics object contains:
        width, height -- frame resolution in pixels
        fx, fy        -- focal lengths in pixels
        ppx, ppy      -- principal point (optical centre) in pixels
        model         -- distortion model identifier
        coeffs        -- distortion coefficients [k1, k2, p1, p2, k3]

    Parameters
    ----------
    profile : rs.pipeline_profile
    stream  : rs.stream
        e.g. rs.stream.color, rs.stream.depth

    Returns
    -------
    rs.intrinsics
    """
    stream_profile = profile.get_stream(stream)
    # Cast to video_stream_profile to access get_intrinsics()
    video_profile  = stream_profile.as_video_stream_profile()
    return video_profile.get_intrinsics()


def get_depth_scale(profile: rs.pipeline_profile) -> float:
    """
    Return the depth scale factor in metres per depth unit.

    The D455f raw depth frame stores 16-bit integers (z16 format).
    Each integer value represents one "depth unit."

        depth_metres = raw_z16_value * depth_scale

    For the D455f the default depth scale is typically 0.001 (1 mm per unit),
    so a raw value of 1000 corresponds to 1.000 m.

    Parameters
    ----------
    profile : rs.pipeline_profile

    Returns
    -------
    float
        Metres per depth unit (e.g. 0.001 for the D455f default).
    """
    device       = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    return depth_sensor.get_depth_scale()


def print_intrinsics_report(color_intr: rs.intrinsics,
                            depth_intr: rs.intrinsics,
                            depth_scale: float) -> None:
    """Pretty-print stream intrinsics and depth-scale to stdout."""

    sep = "-" * 60

    print("\n" + sep)
    print("  COLOUR STREAM INTRINSICS")
    print(sep)
    print(f"  Resolution   : {color_intr.width} x {color_intr.height} px")
    print(f"  Focal length : fx = {color_intr.fx:.4f} px,  "
          f"fy = {color_intr.fy:.4f} px")
    print(f"  Principal pt : cx = {color_intr.ppx:.4f} px,  "
          f"cy = {color_intr.ppy:.4f} px")
    print(f"  Distortion   : model={color_intr.model},  "
          f"coeffs={[round(c, 6) for c in color_intr.coeffs]}")

    print("\n" + sep)
    print("  DEPTH STREAM INTRINSICS")
    print(sep)
    print(f"  Resolution   : {depth_intr.width} x {depth_intr.height} px")
    print(f"  Focal length : fx = {depth_intr.fx:.4f} px,  "
          f"fy = {depth_intr.fy:.4f} px")
    print(f"  Principal pt : cx = {depth_intr.ppx:.4f} px,  "
          f"cy = {depth_intr.ppy:.4f} px")
    print(f"  Distortion   : model={depth_intr.model},  "
          f"coeffs={[round(c, 6) for c in depth_intr.coeffs]}")

    print("\n" + sep)
    print("  DEPTH SCALE")
    print(sep)
    print(f"  Depth scale  : {depth_scale} m / depth-unit")
    print(f"  (Raw value 1000 -> {1000 * depth_scale:.4f} m = "
          f"{1000 * depth_scale * 100:.2f} cm)")
    print(sep + "\n")


# ===========================================================================
# Depth-frame statistics (for a central patch)
# ===========================================================================

def sample_centre_depth(depth_frame: rs.depth_frame,
                        depth_scale: float,
                        patch_radius: int = 20) -> dict:
    """
    Sample depth values in a small square patch at the frame centre.

    Uses numpy directly on the raw depth array so we can inspect raw
    z16 integers alongside the converted metric values.

    Parameters
    ----------
    depth_frame   : rs.depth_frame
        The aligned depth frame.
    depth_scale   : float
        Metres per depth unit.
    patch_radius  : int
        Half-side of the sampling square (pixels).

    Returns
    -------
    dict with keys: cx, cy, raw_mean, raw_std, valid_ratio,
                    mean_m, mean_cm
    """
    depth_image = np.asanyarray(depth_frame.get_data())   # uint16 (z16)
    h, w        = depth_image.shape
    cx, cy      = w // 2, h // 2

    patch = depth_image[
        max(0, cy - patch_radius) : min(h, cy + patch_radius),
        max(0, cx - patch_radius) : min(w, cx + patch_radius)
    ]

    valid_mask  = patch > 0                # 0 = invalid / no return
    valid_ratio = valid_mask.sum() / patch.size if patch.size > 0 else 0.0

    if valid_mask.sum() > 0:
        raw_mean = float(patch[valid_mask].mean())
        raw_std  = float(patch[valid_mask].std())
        mean_m   = raw_mean * depth_scale
        mean_cm  = mean_m * 100.0
    else:
        raw_mean = raw_std = mean_m = mean_cm = float("nan")

    return dict(
        cx=cx, cy=cy,
        raw_mean=raw_mean, raw_std=raw_std,
        valid_ratio=valid_ratio,
        mean_m=mean_m, mean_cm=mean_cm
    )


# ===========================================================================
# Overlay helpers
# ===========================================================================

def draw_crosshair(image: np.ndarray, cx: int, cy: int,
                   colour=(0, 255, 0), size: int = 20,
                   thickness: int = 1) -> None:
    """Draw a small crosshair at (cx, cy)."""
    cv2.line(image, (cx - size, cy), (cx + size, cy), colour, thickness)
    cv2.line(image, (cx, cy - size), (cx, cy + size), colour, thickness)


def overlay_depth_text(image: np.ndarray, stats: dict) -> None:
    """Overlay depth statistics near the top-left of the frame."""
    if np.isnan(stats["mean_m"]):
        text = "Centre depth: NO VALID DATA"
    else:
        text = (f"Centre depth: {stats['mean_cm']:.1f} cm  "
                f"(raw={stats['raw_mean']:.0f}, "
                f"valid={stats['valid_ratio']*100:.0f}%)")

    cv2.putText(image, text,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)


def overlay_instruction(image: np.ndarray) -> None:
    """Overlay quit instructions at the bottom of the frame."""
    h = image.shape[0]
    cv2.putText(image, "Press 'Q' or ESC to quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ===========================================================================
# Main loop
# ===========================================================================

def run_validation() -> None:
    """
    Main validation routine.

    Steps
    -----
    1. Detect camera and print hardware info.
    2. Start pipeline and retrieve intrinsics / depth scale.
    3. Enter the live display loop until the user quits.
    4. Stop the pipeline cleanly on exit.
    """
    global _shutdown_requested

    # ------------------------------------------------------------------
    # 1. Context and device detection
    # ------------------------------------------------------------------
    ctx = rs.context()
    device = detect_device(ctx)
    print_device_info(device)

    # ------------------------------------------------------------------
    # 2. Pipeline startup
    # ------------------------------------------------------------------
    print("[INFO] Starting pipeline ...")
    pipeline, profile = build_pipeline()
    print("[INFO] Pipeline started successfully.")

    # ------------------------------------------------------------------
    # 3. Intrinsics and depth scale
    # ------------------------------------------------------------------
    color_intr  = get_stream_intrinsics(profile, rs.stream.color)
    depth_intr  = get_stream_intrinsics(profile, rs.stream.depth)
    depth_scale = get_depth_scale(profile)

    print_intrinsics_report(color_intr, depth_intr, depth_scale)

    # ------------------------------------------------------------------
    # 4. Post-processing helpers
    # ------------------------------------------------------------------
    # Aligner: reprojects depth pixels into the colour-camera coordinate frame
    # so that each colour pixel (u, v) has a directly corresponding depth value.
    align      = rs.align(rs.stream.color)

    # Colouriser: maps raw z16 depth values to a false-colour BGR image
    # for human-friendly visualisation.
    colorizer  = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme,    COLORIZER_PRESET)
    colorizer.set_option(rs.option.min_distance,    DEPTH_MIN_VIS_M)
    colorizer.set_option(rs.option.max_distance,    DEPTH_MAX_VIS_M)

    # ------------------------------------------------------------------
    # 5. Warm-up: discard the first few frames while the camera stabilises
    # ------------------------------------------------------------------
    WARMUP_FRAMES = 10
    print(f"[INFO] Warming up — discarding {WARMUP_FRAMES} frames ...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()
    print("[INFO] Warm-up complete. Displaying live feed.\n")
    print("       Press 'Q' or ESC in either window to quit.\n")

    # ------------------------------------------------------------------
    # 6. Live display loop
    # ------------------------------------------------------------------
    try:
        while not _shutdown_requested:

            # ---- Acquire aligned frameset --------------------------------
            frameset        = pipeline.wait_for_frames()
            aligned_frames  = align.process(frameset)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # Skip if either frame is missing (transient drop)
            if not color_frame or not depth_frame:
                continue

            # ---- Convert to numpy arrays ---------------------------------
            color_image  = np.asanyarray(color_frame.get_data())   # BGR uint8
            depth_data   = np.asanyarray(depth_frame.get_data())   # uint16 z16

            # ---- Depth colourised visualisation --------------------------
            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data()
            )

            # ---- Centre-patch depth statistics ---------------------------
            stats = sample_centre_depth(depth_frame, depth_scale)
            cx, cy = stats["cx"], stats["cy"]

            # ---- Annotate colour frame -----------------------------------
            color_display = color_image.copy()
            draw_crosshair(color_display, cx, cy, colour=(0, 255, 0))
            overlay_depth_text(color_display, stats)
            overlay_instruction(color_display)

            # ---- Annotate depth frame ------------------------------------
            depth_display = depth_colormap.copy()
            draw_crosshair(depth_display, cx, cy, colour=(255, 255, 255))
            overlay_instruction(depth_display)

            # ---- Show windows -------------------------------------------
            cv2.imshow("D455f — RGB (aligned)", color_display)
            cv2.imshow("D455f — Depth (colourised)", depth_display)

            # ---- Key handling -------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # q, Q, or ESC
                print("\n[INFO] Quit key pressed — shutting down ...")
                break

    except Exception as exc:
        print(f"\n[ERROR] Unexpected error in capture loop:\n  {exc}")
        raise

    finally:
        # ------------------------------------------------------------------
        # 7. Clean shutdown — always executed
        # ------------------------------------------------------------------
        print("[INFO] Stopping pipeline ...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped. Windows closed.")
        print("[INFO] Validation complete — camera is operational.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    try:
        run_validation()
    except RuntimeError as err:
        print(f"\n[FATAL] {err}")
        sys.exit(1)
    except Exception as err:
        print(f"\n[FATAL] Unexpected error: {err}")
        sys.exit(1)
