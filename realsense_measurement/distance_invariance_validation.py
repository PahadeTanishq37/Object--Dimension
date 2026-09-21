"""
distance_invariance_validation.py
==================================
Stage 5.6 / Prompt 10 — Distance-Invariance Validation Suite
Intel RealSense D455f RGB-D Camera

Validates whether the Stage 5.5 physical boundary and edge reconstruction
geometry is invariant across camera-to-object distances (80 cm, 100 cm, 120 cm).

Features:
    1. Direct integration with frozen Stage 5.5 geometry from box_measurement.py
    2. Multi-sample temporal statistics collection (15-30 valid frames)
    3. Ground-truth error analysis (Absolute Error, Percentage Error, MAE, RMSE)
    4. Distance variation analysis (Min, Max, Range, Coefficient of Variation)
    5. Separation of Temporal Noise (std) vs Distance-Dependent Systematic Error
    6. CSV logging to 'distance_invariance_validation.csv'
    7. Multi-panel validation plot export to 'distance_invariance_plot.png'

Author : (your name)
Date   : 2026-09-21
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import cv2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import frozen pipeline & data structures from box_measurement
from box_measurement import (
    initialize_camera,
    setup_gui,
    get_tuning_parameters,
    segment_and_extract_point_cloud,
    detect_planes_ransac,
    reconstruct_cuboid_from_physical_edges,
    draw_measurement_overlay,
    print_console_debug_summary,
    TemporalDimensionFilter,
    GroundTruth,
    ROIState,
    _current_roi,
    WIN_MAIN,
    WIN_MASK,
    DEFAULT_GT_L,
    DEFAULT_GT_B,
    DEFAULT_GT_H,
    DEFAULT_ORTHO_TOL_DEG,
    STREAM_HEIGHT,
    STREAM_WIDTH
)

CSV_FILENAME = "distance_invariance_validation.csv"
PLOT_FILENAME = "distance_invariance_plot.png"


# ===========================================================================
# CSV RECORDING & SUMMARY CALCULATIONS
# ===========================================================================

def load_validation_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(csv_path):
        return []

    records = []
    with open(csv_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return []

    header = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().replace('"', '') for p in line.split(",")]
        if len(parts) >= 16:
            records.append({
                "distance_cm": float(parts[0]),
                "length_cm": float(parts[1]),
                "breadth_cm": float(parts[2]),
                "height_cm": float(parts[3]),
                "length_std_cm": float(parts[4]),
                "breadth_std_cm": float(parts[5]),
                "height_std_cm": float(parts[6]),
                "length_abs_error_cm": float(parts[7]),
                "breadth_abs_error_cm": float(parts[8]),
                "height_abs_error_cm": float(parts[9]),
                "length_percent_error": float(parts[10]),
                "breadth_percent_error": float(parts[11]),
                "height_percent_error": float(parts[12]),
                "confidence_length": parts[13],
                "confidence_breadth": parts[14],
                "confidence_height": parts[15],
            })
    return records


def record_validation_entry(
    distance_label_cm: float,
    l_med: float, b_med: float, h_med: float,
    l_std: float, b_std: float, h_std: float,
    conf_l: str, conf_b: str, conf_h: str,
    gt_l: float = DEFAULT_GT_L,
    gt_b: float = DEFAULT_GT_B,
    gt_h: float = DEFAULT_GT_H,
    csv_filename: str = CSV_FILENAME
) -> str:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
    file_exists = os.path.isfile(csv_path)

    err_l = abs(l_med - gt_l)
    err_b = abs(b_med - gt_b)
    err_h = abs(h_med - gt_h)

    pct_l = (err_l / gt_l) * 100.0 if gt_l > 0 else 0.0
    pct_b = (err_b / gt_b) * 100.0 if gt_b > 0 else 0.0
    pct_h = (err_h / gt_h) * 100.0 if gt_h > 0 else 0.0

    header = (
        "distance_cm,length_cm,breadth_cm,height_cm,"
        "length_std_cm,breadth_std_cm,height_std_cm,"
        "length_abs_error_cm,breadth_abs_error_cm,height_abs_error_cm,"
        "length_percent_error,breadth_percent_error,height_percent_error,"
        "confidence_length,confidence_breadth,confidence_height\n"
    )

    row = (
        f"{distance_label_cm:.1f},"
        f"{l_med:.2f},{b_med:.2f},{h_med:.2f},"
        f"{l_std:.2f},{b_std:.2f},{h_std:.2f},"
        f"{err_l:.2f},{err_b:.2f},{err_h:.2f},"
        f"{pct_l:.2f},{pct_b:.2f},{pct_h:.2f},"
        f"\"{conf_l}\",\"{conf_b}\",\"{conf_h}\"\n"
    )

    with open(csv_path, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(row)

    return csv_path


def compute_and_print_overall_summary(
    records: List[Dict[str, Any]],
    gt_l: float = DEFAULT_GT_L,
    gt_b: float = DEFAULT_GT_B,
    gt_h: float = DEFAULT_GT_H
) -> None:
    if not records:
        return

    print("\n==================================================")
    print("  DISTANCE-INVARIANCE VALIDATION SUMMARY")
    print("==================================================")
    print(f"Ground Truth:")
    print(f"  Length  (L) = {gt_l:.1f} cm")
    print(f"  Breadth (B) = {gt_b:.1f} cm")
    print(f"  Height  (H) = {gt_h:.1f} cm\n")

    distances = []
    l_vals, b_vals, h_vals = [], [], []
    l_errs, b_errs, h_errs = [], [], []
    l_pcts, b_pcts, h_pcts = [], [], []

    for r in records:
        d = r["distance_cm"]
        distances.append(d)
        l, b, h = r["length_cm"], r["breadth_cm"], r["height_cm"]
        l_s, b_s, h_s = r["length_std_cm"], r["breadth_std_cm"], r["height_std_cm"]
        c_l, c_b, c_h = r["confidence_length"], r["confidence_breadth"], r["confidence_height"]

        l_vals.append(l)
        b_vals.append(b)
        h_vals.append(h)

        e_l = abs(l - gt_l)
        e_b = abs(b - gt_b)
        e_h = abs(h - gt_h)

        p_l = (e_l / gt_l) * 100.0
        p_b = (e_b / gt_b) * 100.0
        p_h = (e_h / gt_h) * 100.0

        l_errs.append(e_l)
        b_errs.append(e_b)
        h_errs.append(e_h)

        l_pcts.append(p_l)
        b_pcts.append(p_b)
        h_pcts.append(p_h)

        print(f"Distance: {d:.0f} cm")
        print(f"  L = {l:.2f} cm (std: {l_s:.2f} cm) [{c_l}] | Error: {e_l:.2f} cm ({p_l:.1f}%)")
        print(f"  B = {b:.2f} cm (std: {b_s:.2f} cm) [{c_b}] | Error: {e_b:.2f} cm ({p_b:.1f}%)")
        print(f"  H = {h:.2f} cm (std: {h_s:.2f} cm) [{c_h}] | Error: {e_h:.2f} cm ({p_h:.1f}%)\n")

    # Range & Variations
    l_range = max(l_vals) - min(l_vals)
    b_range = max(b_vals) - min(b_vals)
    h_range = max(h_vals) - min(h_vals)

    print("DISTANCE VARIATION (Invariance Metric):")
    print(f"  L: min = {min(l_vals):.2f} cm, max = {max(l_vals):.2f} cm, range = {l_range:.2f} cm")
    print(f"  B: min = {min(b_vals):.2f} cm, max = {max(b_vals):.2f} cm, range = {b_range:.2f} cm")
    print(f"  H: min = {min(h_vals):.2f} cm, max = {max(h_vals):.2f} cm, range = {h_range:.2f} cm\n")

    # MAE & RMSE per dimension
    l_mae, b_mae, h_mae = float(np.mean(l_errs)), float(np.mean(b_errs)), float(np.mean(h_errs))
    l_rmse = float(np.sqrt(np.mean(np.array(l_errs)**2)))
    b_rmse = float(np.sqrt(np.mean(np.array(b_errs)**2)))
    h_rmse = float(np.sqrt(np.mean(np.array(h_errs)**2)))

    all_errs = l_errs + b_errs + h_errs
    overall_mae = float(np.mean(all_errs))
    overall_rmse = float(np.sqrt(np.mean(np.array(all_errs)**2)))

    print("ACCURACY METRICS:")
    print(f"  Length  : MAE = {l_mae:.2f} cm | RMSE = {l_rmse:.2f} cm | Mean % Err = {np.mean(l_pcts):.1f}% | Max % Err = {max(l_pcts):.1f}%")
    print(f"  Breadth : MAE = {b_mae:.2f} cm | RMSE = {b_rmse:.2f} cm | Mean % Err = {np.mean(b_pcts):.1f}% | Max % Err = {max(b_pcts):.1f}%")
    print(f"  Height  : MAE = {h_mae:.2f} cm | RMSE = {h_rmse:.2f} cm | Mean % Err = {np.mean(h_pcts):.1f}% | Max % Err = {max(h_pcts):.1f}%\n")

    print("OVERALL METROLOGY PERFORMANCE:")
    print(f"  Overall MAE  = {overall_mae:.2f} cm")
    print(f"  Overall RMSE = {overall_rmse:.2f} cm")
    print("==================================================\n")


# ===========================================================================
# MATPLOTLIB VALIDATION PLOT
# ===========================================================================

def generate_validation_plot(
    records: List[Dict[str, Any]],
    gt_l: float = DEFAULT_GT_L,
    gt_b: float = DEFAULT_GT_B,
    gt_h: float = DEFAULT_GT_H,
    out_path: str = PLOT_FILENAME
) -> None:
    if not HAS_MATPLOTLIB or not records:
        return

    # Sort records by distance
    sorted_recs = sorted(records, key=lambda r: r["distance_cm"])

    dists = [r["distance_cm"] for r in sorted_recs]
    l_vals = [r["length_cm"] for r in sorted_recs]
    l_stds = [r["length_std_cm"] for r in sorted_recs]
    b_vals = [r["breadth_cm"] for r in sorted_recs]
    b_stds = [r["breadth_std_cm"] for r in sorted_recs]
    h_vals = [r["height_cm"] for r in sorted_recs]
    h_stds = [r["height_std_cm"] for r in sorted_recs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    # 1. Length Subplot
    axes[0].errorbar(dists, l_vals, yerr=l_stds, fmt='o-', color='#1f77b4', ecolor='#aec7e8', elinewidth=2, capsize=5, label='Measured L')
    axes[0].axhline(gt_l, color='black', linestyle='--', linewidth=1.5, label=f'GT L = {gt_l} cm')
    axes[0].set_title("Length (L) vs Distance", fontweight="bold")
    axes[0].set_xlabel("Camera Distance (cm)")
    axes[0].set_ylabel("Length (cm)")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(min(l_vals + [gt_l]) - 2.0, max(l_vals + [gt_l]) + 2.0)

    # 2. Breadth Subplot
    axes[1].errorbar(dists, b_vals, yerr=b_stds, fmt='s-', color='#2ca02c', ecolor='#98df8a', elinewidth=2, capsize=5, label='Measured B')
    axes[1].axhline(gt_b, color='black', linestyle='--', linewidth=1.5, label=f'GT B = {gt_b} cm')
    axes[1].set_title("Breadth (B) vs Distance", fontweight="bold")
    axes[1].set_xlabel("Camera Distance (cm)")
    axes[1].set_ylabel("Breadth (cm)")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(min(b_vals + [gt_b]) - 2.0, max(b_vals + [gt_b]) + 2.0)

    # 3. Height Subplot
    axes[2].errorbar(dists, h_vals, yerr=h_stds, fmt='^-', color='#ff7f0e', ecolor='#ffbb78', elinewidth=2, capsize=5, label='Measured H')
    axes[2].axhline(gt_h, color='black', linestyle='--', linewidth=1.5, label=f'GT H = {gt_h} cm')
    axes[2].set_title("Height (H) vs Distance", fontweight="bold")
    axes[2].set_xlabel("Camera Distance (cm)")
    axes[2].set_ylabel("Height (cm)")
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right')
    axes[2].set_ylim(min(h_vals + [gt_h]) - 2.0, max(h_vals + [gt_h]) + 2.0)

    plt.suptitle("Prompt 10: 3D Dimension Distance-Invariance Validation (D455f)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Validation plot saved to: {out_path}")


# ===========================================================================
# SINGLE-DISTANCE TEST EXECUTION
# ===========================================================================

def run_single_distance_test(
    target_distance_cm: float = 80.0,
    required_frames: int = 20,
    gt_l: float = DEFAULT_GT_L,
    gt_b: float = DEFAULT_GT_B,
    gt_h: float = DEFAULT_GT_H
) -> Optional[Dict[str, Any]]:
    print(f"""
##############################################################
  DISTANCE-INVARIANCE TEST: {target_distance_cm:.0f} cm TARGET
##############################################################
  1. Position the box at approximately {target_distance_cm:.0f} cm from the D455f.
  2. Ensure 2 orthogonal faces (e.g. Front + Top) are visible.
  3. Drag a bounding rectangle (ROI) around the box on the RGB feed.
  4. The test will automatically collect {required_frames} stable frames.
  5. Press [Q] to abort.
##############################################################
""")

    cam = initialize_camera()
    setup_gui()
    temporal_filter = TemporalDimensionFilter(45)
    ground_truth = GroundTruth(length_cm=gt_l, breadth_cm=gt_b, height_cm=gt_h, is_set=True)

    fps = 0.0
    frame_count = 0
    t_start = time.time()
    collected_samples = 0
    last_cuboid = None

    try:
        while collected_samples < required_frames:
            success, frameset = cam.pipeline.try_wait_for_frames(timeout_ms=3000)
            if not success:
                continue

            aligned = cam.align.process(frameset)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m   = depth_raw.astype(np.float32) * cam.depth_scale

            params = get_tuning_parameters()
            mask, points_3d, raw_n, valid_n, clean_n, z_peak = segment_and_extract_point_cloud(
                depth_m, _current_roi, cam.intrinsics, params
            )

            cuboid = None
            if points_3d is not None and len(points_3d) >= 60:
                planes = detect_planes_ransac(points_3d, dist_thresh_m=params["ransac_thresh_m"])
                cuboid = reconstruct_cuboid_from_physical_edges(
                    points_3d, planes, color_img, depth_m, _current_roi, cam.intrinsics, z_peak, DEFAULT_ORTHO_TOL_DEG
                )

                if cuboid is not None and cuboid.is_reliable:
                    temporal_filter.add_sample(cuboid.length.value_cm, cuboid.breadth.value_cm, cuboid.height.value_cm, cuboid.is_reliable)
                    collected_samples += 1
                    last_cuboid = cuboid

            t_stats = temporal_filter.get_stats()

            # Progress overlay
            annotated = draw_measurement_overlay(
                color_img, mask, _current_roi, cuboid, t_stats, ground_truth, cam.intrinsics, fps, debug_mode=True
            )
            prog_str = f"PROGRESS: Collecting {collected_samples}/{required_frames} valid frames at {target_distance_cm:.0f} cm"
            cv2.putText(annotated, prog_str, (20, STREAM_HEIGHT - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WIN_MAIN, annotated)
            cv2.imshow(WIN_MASK, mask)

            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[WARN] Test aborted by user.")
                return None

        # Finished collecting required frames
        stats = temporal_filter.get_stats()
        conf_l = last_cuboid.length.confidence if last_cuboid else "UNKNOWN"
        conf_b = last_cuboid.breadth.confidence if last_cuboid else "UNKNOWN"
        conf_h = last_cuboid.height.confidence if last_cuboid else "UNKNOWN"

        csv_path = record_validation_entry(
            target_distance_cm,
            stats.length_median, stats.breadth_median, stats.height_median,
            stats.length_std, stats.breadth_std, stats.height_std,
            conf_l, conf_b, conf_h,
            gt_l, gt_b, gt_h
        )

        all_records = load_validation_csv(csv_path)
        generate_validation_plot(all_records, gt_l, gt_b, gt_h)

        err_l = abs(stats.length_median - gt_l)
        err_b = abs(stats.breadth_median - gt_b)
        err_h = abs(stats.height_median - gt_h)

        pct_l = (err_l / gt_l) * 100.0
        pct_b = (err_b / gt_b) * 100.0
        pct_h = (err_h / gt_h) * 100.0

        print("\n==================================================")
        print(f"  DISTANCE TEST RESULT: {target_distance_cm:.0f} cm")
        print("==================================================")
        print(f"Distance: {target_distance_cm:.0f} cm")
        print(f"L: {stats.length_median:.2f} cm")
        print(f"B: {stats.breadth_median:.2f} cm")
        print(f"H: {stats.height_median:.2f} cm\n")

        print("L standard deviation: {:.2f} cm".format(stats.length_std))
        print("B standard deviation: {:.2f} cm".format(stats.breadth_std))
        print("H standard deviation: {:.2f} cm\n".format(stats.height_std))

        print("Confidence:")
        print(f"L: {conf_l}")
        print(f"B: {conf_b}")
        print(f"H: {conf_h}\n")

        print(f"Ground Truth Error (L={gt_l:.1f}, B={gt_b:.1f}, H={gt_h:.1f} cm):")
        print(f"{target_distance_cm:.0f} cm:")
        print(f"L error = {err_l:.2f} cm ({pct_l:.1f}%)")
        print(f"B error = {err_b:.2f} cm ({pct_b:.1f}%)")
        print(f"H error = {err_h:.2f} cm ({pct_h:.1f}%)\n")

        print(f"Saved row to: {csv_path}")
        print("==================================================\n")

        return {
            "distance_cm": target_distance_cm,
            "length_cm": stats.length_median,
            "breadth_cm": stats.breadth_median,
            "height_cm": stats.height_median,
            "length_std_cm": stats.length_std,
            "breadth_std_cm": stats.breadth_std,
            "height_std_cm": stats.height_std,
            "conf_l": conf_l, "conf_b": conf_b, "conf_h": conf_h,
            "err_l": err_l, "err_b": err_b, "err_h": err_h,
            "pct_l": pct_l, "pct_b": pct_b, "pct_h": pct_h
        }

    finally:
        try:
            cam.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Distance-Invariance Validation Suite")
    parser.add_argument("--distance", type=float, default=80.0, help="Target camera distance in cm (e.g. 80, 100, 120)")
    parser.add_argument("--frames", type=int, default=20, help="Number of stable temporal frames to collect")
    parser.add_argument("--summary-only", action="store_true", help="Only compute and print summary from existing CSV")
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILENAME)

    if args.summary_only:
        records = load_validation_csv(csv_path)
        if records:
            compute_and_print_overall_summary(records)
            generate_validation_plot(records)
        else:
            print(f"[WARN] No records found in {csv_path}.")
        return

    run_single_distance_test(
        target_distance_cm=args.distance,
        required_frames=args.frames
    )


if __name__ == "__main__":
    main()
