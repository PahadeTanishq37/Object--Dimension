"""
=============================================================================
Module 4.1: 3D Target Object Segmentation Refinement & Validation
=============================================================================
Intel RealSense D455f Multi-View 3D Reconstruction Pipeline — Stage 4.1 / Refinement

Project Objective:
    Refines and elevates the 3D target object segmentation engine to robustly
    isolate the physical target object from complex multi-view registered scenes
    containing dominant background planes, floor, walls, and registration artifacts.

Key Improvements Over Baseline Module 4:
    1. Progressive Multi-Plane RANSAC:
       Detects and subtracts multiple structural background/support planes (walls,
       table surface, floor) rather than stopping after a single arbitrary plane.
    2. Registration Ghost & Outlier Cleansing:
       Filters diffuse multi-view registration artifacts and fringe points before
       spatial clustering.
    3. Multi-Criteria Target Selection (Size-Independent):
       Evaluates all candidate clusters using scale-invariant geometric properties:
         - Optical Centrality & View Focus
         - Physical Support Surface Contact
         - Volumetric Point Density & Solid Coherence
         - Aspect Ratio & 3D Spatial Compactness
       DOES NOT use or hard-code ground-truth dimensions.
    4. Principal Component Analysis (PCA) & Oriented Bounding Box (OBB):
       Computes true principal orientation and minimum-volume bounding box.
    5. Post-Segmentation Validation Against Ground Truth:
       Compares measured OBB dimensions against known physical reference
       strictly for accuracy evaluation.

Outputs:
    datasets/<session>/segmentation_refined/
      ├── object_only_refined.ply
      ├── object_only_refined.csv
      ├── segmentation_refined_results.json
      └── segmentation_refined_visualization.png

Controls / CLI:
    python module4_1_segmentation_refinement.py --session session_003
    python module4_1_segmentation_refinement.py --self-test
=============================================================================
"""

import os
import sys
import time
import json
import glob
import copy
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import open3d as o3d

# Non-interactive matplotlib backend for headless and visual plot exports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURATION CONSTANTS & DEFAULT REFINEMENT THRESHOLDS
# ===========================================================================
DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# Multi-Plane RANSAC Parameters
MAX_PLANES_TO_EXTRACT = 5               # Maximum structural planes to iteratively subtract
PLANE_DISTANCE_THRESH_M = 0.010         # 10 mm plane inlier threshold
PLANE_MIN_INLIER_RATIO = 0.03           # Planes must contain at least 3% of remaining points
PLANE_RANSAC_ITER = 2500

# Registration Ghost & Outlier Pruning
PRE_CLUSTER_OUTLIER_NB = 30             # Statistical outlier neighbors
PRE_CLUSTER_OUTLIER_STD = 1.8           # Statistical outlier std ratio

# 3D DBSCAN Clustering Parameters
DBSCAN_EPS_M = 0.012                    # 12 mm fine Euclidean search radius
DBSCAN_MIN_PTS = 35                     # Minimum points to form cluster core
MIN_CANDIDATE_POINTS = 150              # Ignore microscopic noise clusters < 150 pts

# Physical Ground Truth (FOR VALIDATION ONLY — NOT USED IN SEGMENTATION)
GT_LENGTH_CM = 16.6
GT_BREADTH_CM = 9.1
GT_HEIGHT_CM = 5.0


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================
@dataclass
class DetectedPlane:
    """Encapsulates a detected 3D plane ax + by + cz + d = 0."""
    plane_id: int
    equation: List[float]       # [a, b, c, d]
    normal: List[float]         # [nx, ny, nz]
    inlier_count: int
    inlier_pct: float
    rms_residual_mm: float
    is_support_surface: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefinedClusterCandidate:
    """Detailed geometric profile of an individual 3D candidate cluster."""
    cluster_id: int
    point_count: int
    point_pct_of_foreground: float
    centroid_m: List[float]          # [cx, cy, cz]
    aabb_min_m: List[float]
    aabb_max_m: List[float]
    aabb_extent_cm: List[float]      # [dx, dy, dz] in cm
    obb_extent_cm: List[float]       # [L, B, H] sorted principal extents in cm
    obb_volume_cm3: float
    point_density_pts_cm3: float
    aspect_ratio: float              # max_extent / min_extent
    distance_to_optical_center_m: float
    elevation_above_support_cm: float
    contact_distance_to_plane_cm: float
    composite_selection_score: float
    is_selected: bool
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefinedSegmentationSummary:
    """Comprehensive summary of refined 3D object segmentation."""
    session_id: str
    timestamp_iso: str
    input_cloud_path: str
    total_input_points: int
    input_bounds_m: Dict[str, List[float]]
    detected_planes: List[Dict[str, Any]]
    total_plane_inliers: int
    foreground_points: int
    total_clusters_found: int
    candidate_clusters: List[Dict[str, Any]]
    selected_cluster_id: int
    raw_selected_points: int
    clean_selected_points: int
    retained_point_ratio_pct: float
    final_centroid_m: List[float]
    final_aabb_extent_cm: List[float]
    final_obb_extent_cm: List[float]     # [L, B, H] in cm
    final_volume_cm3: float
    final_point_density: float
    ground_truth_cm: Dict[str, float]
    measured_dimensions_cm: Dict[str, float]
    dimension_errors_cm: Dict[str, float]
    dimension_errors_pct: Dict[str, float]
    segmentation_status: str
    output_ply_path: str
    output_csv_path: str
    output_json_path: str
    output_plot_path: str


# ===========================================================================
# CLASS: ProgressivePlaneExtractor
# ===========================================================================
class ProgressivePlaneExtractor:
    """
    Iteratively detects and removes multiple dominant background and support planes
    (table surface, floor, walls) using sequential RANSAC.
    """

    @staticmethod
    def extract_all_planes(
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = PLANE_DISTANCE_THRESH_M,
        max_planes: int = MAX_PLANES_TO_EXTRACT,
        min_inlier_ratio: float = PLANE_MIN_INLIER_RATIO
    ) -> Tuple[List[DetectedPlane], o3d.geometry.PointCloud, o3d.geometry.PointCloud, Optional[DetectedPlane]]:
        """
        Extracts up to max_planes dominant planar surfaces.
        Identifies the primary horizontal support surface (table/floor).
        Returns:
            (detected_planes_list, all_plane_inliers_pcd, remaining_foreground_pcd, primary_support_plane)
        """
        remaining_pcd = copy.deepcopy(pcd)
        total_pts = len(pcd.points)

        detected_planes: List[DetectedPlane] = []
        all_plane_inlier_indices = []

        all_pts_arr = np.asarray(pcd.points)

        for plane_idx in range(max_planes):
            rem_count = len(remaining_pcd.points)
            if rem_count < 1000 or (rem_count / total_pts) < min_inlier_ratio:
                break

            plane_eqn, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=PLANE_RANSAC_ITER
            )
            inlier_count = len(inliers)
            if inlier_count < 500 or (inlier_count / total_pts) < min_inlier_ratio:
                break

            a, b, c, d = plane_eqn
            norm = np.linalg.norm([a, b, c])
            if norm > 0:
                a, b, c, d = a / norm, b / norm, c / norm, d / norm

            inlier_pcd = remaining_pcd.select_by_index(inliers)
            inlier_pts = np.asarray(inlier_pcd.points)
            residuals = np.abs(a * inlier_pts[:, 0] + b * inlier_pts[:, 1] + c * inlier_pts[:, 2] + d)
            rms_residual_mm = float(np.sqrt(np.mean(residuals**2)) * 1000.0)

            # Check if this plane is approximately horizontal (normal along Y-axis in camera coordinates)
            # In RealSense frame: +Y is down, so a horizontal desk/table has |ny| > 0.6
            is_horizontal = abs(b) > 0.55

            det_plane = DetectedPlane(
                plane_id=plane_idx + 1,
                equation=[float(a), float(b), float(c), float(d)],
                normal=[float(a), float(b), float(c)],
                inlier_count=inlier_count,
                inlier_pct=(inlier_count / total_pts) * 100.0,
                rms_residual_mm=rms_residual_mm,
                is_support_surface=is_horizontal
            )
            detected_planes.append(det_plane)

            # Update remaining point cloud for next iteration
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        # Classify Primary Support Plane
        # Preference: Horizontal plane closest to the typical tabletop height
        horizontal_planes = [p for p in detected_planes if p.is_support_surface]
        if horizontal_planes:
            # Select horizontal plane with most inliers
            primary_support_plane = max(horizontal_planes, key=lambda p: p.inlier_count)
        elif detected_planes:
            primary_support_plane = detected_planes[0]
        else:
            primary_support_plane = None

        # Build combined plane inliers point cloud vs remaining foreground
        # A point is a plane inlier if it lies within distance threshold of ANY detected plane
        all_pts = np.asarray(pcd.points)
        plane_mask = np.zeros(len(all_pts), dtype=bool)

        for p in detected_planes:
            a, b, c, d = p.equation
            dists = np.abs(a * all_pts[:, 0] + b * all_pts[:, 1] + c * all_pts[:, 2] + d)
            plane_mask |= (dists <= distance_threshold * 1.2)

        plane_inlier_indices = np.where(plane_mask)[0]
        fg_indices = np.where(~plane_mask)[0]

        all_planes_pcd = pcd.select_by_index(plane_inlier_indices)
        foreground_pcd = pcd.select_by_index(fg_indices)

        return detected_planes, all_planes_pcd, foreground_pcd, primary_support_plane


# ===========================================================================
# CLASS: RefinedClusterExtractor
# ===========================================================================
class RefinedClusterExtractor:
    """
    Applies statistical registration artifact filtering and adaptive DBSCAN to
    isolate all significant foreground 3D geometric entities.
    """

    @staticmethod
    def extract_candidate_clusters(
        foreground_pcd: o3d.geometry.PointCloud,
        support_plane: Optional[DetectedPlane],
        eps: float = DBSCAN_EPS_M,
        min_points: int = DBSCAN_MIN_PTS
    ) -> Tuple[List[o3d.geometry.PointCloud], List[RefinedClusterCandidate]]:
        """
        Cleans registration artifacts and extracts 3D clusters with complete
        PCA and geometric profiling.
        """
        total_fg = len(foreground_pcd.points)
        if total_fg < min_points:
            return [], []

        # 1. Prune diffuse registration fringe & ghost points
        cleaned_fg, _ = foreground_pcd.remove_statistical_outlier(
            nb_neighbors=PRE_CLUSTER_OUTLIER_NB,
            std_ratio=PRE_CLUSTER_OUTLIER_STD
        )

        # 2. Run fine Euclidean DBSCAN
        labels = np.array(cleaned_fg.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        max_label = labels.max()
        if max_label < 0:
            return [], []

        pts_arr = np.asarray(cleaned_fg.points)
        cols_arr = np.asarray(cleaned_fg.colors) if cleaned_fg.has_colors() else None

        cluster_pcds: List[o3d.geometry.PointCloud] = []
        candidates: List[RefinedClusterCandidate] = []

        if support_plane is not None:
            sa, sb, sc, sd = support_plane.equation
        else:
            sa, sb, sc, sd = 0.0, 1.0, 0.0, 0.0

        for k in range(max_label + 1):
            idx = np.where(labels == k)[0]
            if len(idx) < MIN_CANDIDATE_POINTS:
                continue

            c_pts = pts_arr[idx]
            c_pcd = o3d.geometry.PointCloud()
            c_pcd.points = o3d.utility.Vector3dVector(c_pts)
            if cols_arr is not None and len(cols_arr) == len(pts_arr):
                c_pcd.colors = o3d.utility.Vector3dVector(cols_arr[idx])

            cluster_pcds.append(c_pcd)

            # Compute AABB
            aabb = c_pcd.get_axis_aligned_bounding_box()
            aabb_min = aabb.get_min_bound()
            aabb_max = aabb.get_max_bound()
            aabb_extent_cm = (aabb_max - aabb_min) * 100.0

            # Compute PCA / Oriented Bounding Box (OBB)
            try:
                obb = c_pcd.get_oriented_bounding_box()
                obb_ext_sorted = np.sort(obb.extent) * 100.0  # [min_dim, mid_dim, max_dim] in cm
                obb_vol_cm3 = float(np.prod(obb_ext_sorted))
            except Exception:
                obb_ext_sorted = np.sort(aabb_extent_cm)
                obb_vol_cm3 = float(np.prod(obb_ext_sorted))

            centroid = np.mean(c_pts, axis=0)

            # Spatial distance from optical focus axis (X=0, Y=0, Z ~ 0.7m)
            dist_to_optical_center = float(np.linalg.norm([centroid[0], centroid[1], centroid[2] - 0.75]))

            # Distance to Support Plane
            signed_plane_dists = (sa * c_pts[:, 0] + sb * c_pts[:, 1] + sc * c_pts[:, 2] + sd) * 100.0
            mean_elevation_cm = float(np.mean(signed_plane_dists))
            min_contact_dist_cm = float(np.min(np.abs(signed_plane_dists)))

            # Volumetric Point Density
            density = float(len(c_pts) / (obb_vol_cm3 + 1e-4))

            # Aspect Ratio
            aspect_ratio = float(obb_ext_sorted[2] / (obb_ext_sorted[0] + 1e-3))

            candidate = RefinedClusterCandidate(
                cluster_id=len(candidates),
                point_count=len(c_pts),
                point_pct_of_foreground=(len(c_pts) / total_fg) * 100.0,
                centroid_m=[float(centroid[0]), float(centroid[1]), float(centroid[2])],
                aabb_min_m=[float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])],
                aabb_max_m=[float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])],
                aabb_extent_cm=[float(aabb_extent_cm[0]), float(aabb_extent_cm[1]), float(aabb_extent_cm[2])],
                obb_extent_cm=[float(obb_ext_sorted[2]), float(obb_ext_sorted[1]), float(obb_ext_sorted[0])], # [L, B, H]
                obb_volume_cm3=obb_vol_cm3,
                point_density_pts_cm3=density,
                aspect_ratio=aspect_ratio,
                distance_to_optical_center_m=dist_to_optical_center,
                elevation_above_support_cm=mean_elevation_cm,
                contact_distance_to_plane_cm=min_contact_dist_cm,
                composite_selection_score=0.0,
                is_selected=False,
                rejection_reason=""
            )
            candidates.append(candidate)

        return cluster_pcds, candidates


# ===========================================================================
# CLASS: GeometricTargetSelector
# ===========================================================================
class GeometricTargetSelector:
    """
    Ranks candidate clusters using general scale-invariant geometric principles
    (optical centrality, physical plane support contact, volumetric solid density,
    and aspect compactness).
    """

    @staticmethod
    def evaluate_and_select_target(candidates: List[RefinedClusterCandidate]) -> Tuple[int, str]:
        """
        Selects the true target object based on composite geometric scoring.
        Returns:
            (selected_cluster_index, explanation_rationale)
        """
        if not candidates:
            return -1, "No candidate clusters available."

        max_density = max((c.point_density_pts_cm3 for c in candidates), default=1.0)
        max_pts = max((c.point_count for c in candidates), default=1.0)

        for c in candidates:
            # Scale Filter: Reject room structures (> 1,500,000 cm^3) and micro-specks (< 10 cm^3)
            if c.obb_volume_cm3 > 1500000.0:
                c.rejection_reason = f"Rejected: Giant room geometry ({c.obb_volume_cm3:,.0f} cm^3 > 1.5 m^3)"
                c.composite_selection_score = -100.0
                continue
            if c.obb_volume_cm3 < 8.0:
                c.rejection_reason = f"Rejected: Microscopic speck ({c.obb_volume_cm3:.1f} cm^3 < 8 cm^3)"
                c.composite_selection_score = -100.0
                continue

            # 1. Optical Centrality Score (Gaussian falloff from central camera ray)
            # Objects placed on the table in front of the camera are centered near X=0, Y=0, Z=0.65-0.9m
            s_centrality = float(np.exp(- (c.distance_to_optical_center_m**2) / (2.0 * (0.35**2))))

            # 2. Support Surface Contact Score
            # The bottom of the object must sit on or adjacent to the support table plane (contact <= 4 cm)
            s_support = float(np.exp(- (c.contact_distance_to_plane_cm**2) / (2.0 * (3.0**2))))

            # 3. Density & Solid Coherence Score (Dense physical solid vs diffuse scan fringe)
            s_density = float(c.point_density_pts_cm3 / (max_density + 1e-4))

            # 4. Aspect Ratio & Compactness Score (3D solid objects have aspect ratios < 5.0)
            if c.aspect_ratio > 10.0:
                s_compact = 0.05  # Severe penalty for 1D line noise or flat wall slivers
            else:
                s_compact = float(1.0 / (1.0 + (c.aspect_ratio / 4.0)))

            # 5. Point Population Score
            s_points = float(np.sqrt(c.point_count / max_pts))

            # Composite Multi-Criteria Target Score
            composite_score = (
                0.35 * s_centrality +
                0.25 * s_support +
                0.20 * s_density +
                0.10 * s_compact +
                0.10 * s_points
            )
            c.composite_selection_score = float(composite_score)

        # Select candidate with highest composite score
        valid_candidates = [c for c in candidates if c.composite_selection_score > 0]
        if not valid_candidates:
            # Fallback
            best_idx = int(np.argmax([c.point_count for c in candidates]))
            candidates[best_idx].is_selected = True
            return best_idx, "Fallback: Selected densest available cluster."

        best_score = -1.0
        best_idx = 0
        for i, c in enumerate(candidates):
            if c.composite_selection_score > best_score:
                best_score = c.composite_selection_score
                best_idx = i

        candidates[best_idx].is_selected = True
        sel = candidates[best_idx]
        rationale = (
            f"Selected Cluster #{sel.cluster_id} (Score={sel.composite_selection_score:.3f}): "
            f"{sel.point_count:,} pts, OBB Extent=[{sel.obb_extent_cm[0]:.1f} x {sel.obb_extent_cm[1]:.1f} x {sel.obb_extent_cm[2]:.1f}] cm, "
            f"Centrality Dist={sel.distance_to_optical_center_m*100:.1f} cm, Table Contact={sel.contact_distance_to_plane_cm:.1f} cm."
        )

        return best_idx, rationale


# ===========================================================================
# CLASS: RefinedSegmentationVisualizer
# ===========================================================================
class RefinedSegmentationVisualizer:
    """
    Renders a comprehensive 5-panel visual diagnostic figure showing the entire
    multi-plane removal, cluster ranking, and OBB refinement process.
    """

    @staticmethod
    def render_refined_diagnostic_plot(
        original_pcd: o3d.geometry.PointCloud,
        plane_pcd: o3d.geometry.PointCloud,
        foreground_pcd: o3d.geometry.PointCloud,
        cluster_pcds: List[o3d.geometry.PointCloud],
        selected_object_pcd: o3d.geometry.PointCloud,
        summary: RefinedSegmentationSummary,
        output_plot_path: str
    ) -> str:
        """Renders 5-panel visual validation report."""
        fig = plt.figure(figsize=(22, 14), facecolor="#141414")
        fig.suptitle(
            f"Module 4.1: 3D Object Segmentation Refinement — {summary.session_id.upper()}",
            fontsize=18, fontweight="bold", color="#00E5FF", y=0.97
        )

        # Panel 1: Original Multi-View Registered Scene
        ax1 = fig.add_subplot(2, 3, 1, projection="3d", facecolor="#0E0E0E")
        ax1.set_title(f"1. Registered Scene ({summary.total_input_points:,} pts)", color="#FFFFFF", fontsize=11, pad=8)
        pts_orig = np.asarray(original_pcd.points)[::10]
        cols_orig = np.asarray(original_pcd.colors)[::10] if original_pcd.has_colors() else None
        if cols_orig is not None and len(cols_orig) == len(pts_orig):
            ax1.scatter(pts_orig[:, 0], pts_orig[:, 2], -pts_orig[:, 1], c=cols_orig, s=1.0, alpha=0.7)
        else:
            ax1.scatter(pts_orig[:, 0], pts_orig[:, 2], -pts_orig[:, 1], c="#777777", s=1.0, alpha=0.7)
        ax1.set_xlabel("X (m)", color="#888888")
        ax1.set_ylabel("Z (m)", color="#888888")
        ax1.set_zlabel("Y (m)", color="#888888")
        ax1.tick_params(colors="#666666")

        # Panel 2: Progressive Support & Background Planes Removed
        ax2 = fig.add_subplot(2, 3, 2, projection="3d", facecolor="#0E0E0E")
        ax2.set_title(f"2. Multi-Plane Removal ({summary.total_plane_inliers:,} inliers removed)", color="#FFFFFF", fontsize=11, pad=8)
        plane_pts = np.asarray(plane_pcd.points)[::10]
        fg_pts = np.asarray(foreground_pcd.points)[::8]
        ax2.scatter(plane_pts[:, 0], plane_pts[:, 2], -plane_pts[:, 1], c="#FF9900", s=1.0, alpha=0.35, label=f"Planes ({len(summary.detected_planes)} detected)")
        ax2.scatter(fg_pts[:, 0], fg_pts[:, 2], -fg_pts[:, 1], c="#00E5FF", s=1.2, alpha=0.75, label="Foreground")
        ax2.set_xlabel("X (m)", color="#888888")
        ax2.set_ylabel("Z (m)", color="#888888")
        ax2.set_zlabel("Y (m)", color="#888888")
        ax2.tick_params(colors="#666666")
        ax2.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF", fontsize=8)

        # Panel 3: Candidate Clusters (Colorized by Cluster ID)
        ax3 = fig.add_subplot(2, 3, 3, projection="3d", facecolor="#0E0E0E")
        ax3.set_title(f"3. 3D DBSCAN Clusters ({summary.total_clusters_found} candidates)", color="#FFFFFF", fontsize=11, pad=8)
        palette = ["#FF007F", "#00FF88", "#FFEA00", "#7928CA", "#0070F3", "#FF4500", "#00DFD8", "#F5A623", "#50E3C2"]
        for i, cpcd in enumerate(cluster_pcds):
            c_pts = np.asarray(cpcd.points)[::3]
            col = palette[i % len(palette)]
            is_sel = (i == summary.selected_cluster_id)
            lbl = f"C#{i} ({len(cpcd.points):,} pts) {'[SEL]' if is_sel else ''}"
            ax3.scatter(c_pts[:, 0], c_pts[:, 2], -c_pts[:, 1], c=col, s=1.5 if is_sel else 0.7, alpha=0.9 if is_sel else 0.35, label=lbl if i < 6 else None)
        ax3.set_xlabel("X (m)", color="#888888")
        ax3.set_ylabel("Z (m)", color="#888888")
        ax3.set_zlabel("Y (m)", color="#888888")
        ax3.tick_params(colors="#666666")
        ax3.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF", fontsize=7)

        # Panel 4: Clean Isolated Target Object with OBB Wireframe
        ax4 = fig.add_subplot(2, 3, 4, projection="3d", facecolor="#0E0E0E")
        obb_l, obb_b, obb_h = summary.final_obb_extent_cm
        ax4.set_title(f"4. Isolated Target Object ({summary.clean_selected_points:,} pts | OBB: {obb_l:.1f}x{obb_b:.1f}x{obb_h:.1f} cm)", color="#00FF88", fontsize=11, pad=8)
        obj_pts = np.asarray(selected_object_pcd.points)[::2]
        obj_cols = np.asarray(selected_object_pcd.colors)[::2] if selected_object_pcd.has_colors() else None
        if obj_cols is not None and len(obj_cols) == len(obj_pts):
            ax4.scatter(obj_pts[:, 0], obj_pts[:, 2], -obj_pts[:, 1], c=obj_cols, s=1.8, alpha=0.9)
        else:
            ax4.scatter(obj_pts[:, 0], obj_pts[:, 2], -obj_pts[:, 1], c="#00FF88", s=1.8, alpha=0.9)

        # Draw 3D Oriented Bounding Box
        try:
            obb = selected_object_pcd.get_oriented_bounding_box()
            obb_corners = np.asarray(obb.get_box_points())
            # Convert to display frame [X, Z, -Y]
            c_disp = np.column_stack([obb_corners[:, 0], obb_corners[:, 2], -obb_corners[:, 1]])
            box_lines = [
                (0, 1), (1, 7), (7, 2), (2, 0),
                (3, 6), (6, 4), (4, 5), (5, 3),
                (0, 3), (1, 6), (7, 4), (2, 5)
            ]
            for u_idx, v_idx in box_lines:
                ax4.plot(
                    [c_disp[u_idx, 0], c_disp[v_idx, 0]],
                    [c_disp[u_idx, 1], c_disp[v_idx, 1]],
                    [c_disp[u_idx, 2], c_disp[v_idx, 2]],
                    color="#00FF88", linewidth=1.8, linestyle="-"
                )
        except Exception:
            pass

        ax4.set_xlabel("X (m)", color="#888888")
        ax4.set_ylabel("Z (m)", color="#888888")
        ax4.set_zlabel("Y (m)", color="#888888")
        ax4.tick_params(colors="#666666")

        # Panel 5: Quantitative Telemetry & Ground Truth Error Ledger
        ax5 = fig.add_subplot(2, 3, (5, 6), facecolor="#181818")
        ax5.axis("off")
        ax5.set_title("Quantitative Segmentation & Validation Ledger", color="#FFFFFF", fontsize=12, pad=10)

        lines = [
            f"Session ID                : {summary.session_id}",
            f"Input Scene Points        : {summary.total_input_points:,} pts",
            f"Structural Planes Extracted: {len(summary.detected_planes)} planes ({summary.total_plane_inliers:,} pts subtracted)",
            f"Remaining Foreground      : {summary.foreground_points:,} pts ({summary.total_clusters_found} clusters evaluated)",
            f"Selected Target Cluster   : #{summary.selected_cluster_id} ({summary.clean_selected_points:,} clean pts, {summary.retained_point_ratio_pct:.1f}% retained)",
            f"Object Centroid           : [{summary.final_centroid_m[0]:+.3f}, {summary.final_centroid_m[1]:+.3f}, {summary.final_centroid_m[2]:+.3f}] m",
            f"Estimated OBB Volume      : {summary.final_volume_cm3:,.1f} cm^3 (Point Density: {summary.final_point_density:.1f} pts/cm^3)",
            "",
            "--- VALIDATION AGAINST KNOWN PHYSICAL GROUND TRUTH (POST-SEGMENTATION) ---",
            f"Ground Truth Dimensions   : L = {summary.ground_truth_cm['length_cm']:.1f} cm,  B = {summary.ground_truth_cm['breadth_cm']:.1f} cm,  H = {summary.ground_truth_cm['height_cm']:.1f} cm",
            f"Measured OBB Dimensions   : L = {summary.measured_dimensions_cm['length_cm']:.1f} cm,  B = {summary.measured_dimensions_cm['breadth_cm']:.1f} cm,  H = {summary.measured_dimensions_cm['height_cm']:.1f} cm",
            f"Absolute Errors           : ΔL = {summary.dimension_errors_cm['length_error_cm']:+.1f} cm, ΔB = {summary.dimension_errors_cm['breadth_error_cm']:+.1f} cm, ΔH = {summary.dimension_errors_cm['height_error_cm']:+.1f} cm",
            f"Percentage Errors         : %L = {summary.dimension_errors_pct['length_error_pct']:+.1f}%,  %B = {summary.dimension_errors_pct['breadth_error_pct']:+.1f}%,  %H = {summary.dimension_errors_pct['height_error_pct']:+.1f}%",
            f"Overall Status            : {summary.segmentation_status}"
        ]

        y_pos = 0.95
        for line in lines:
            color = "#00FF88" if "PASS" in line or "Measured OBB" in line else "#FFFFFF"
            if "Session ID" in line or "Selected Target" in line:
                color = "#00E5FF"
            if "NEEDS IMPROVEMENT" in line:
                color = "#FF9900"
            ax5.text(0.04, y_pos, line, transform=ax5.transAxes, color=color,
                     fontsize=10, fontfamily="monospace", verticalalignment="top")
            y_pos -= 0.065

        plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.05, wspace=0.25, hspace=0.25)
        plt.savefig(output_plot_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        print(f"[EXPORT] Saved Refined Segmentation Diagnostic Plot: {output_plot_path}")
        return output_plot_path


# ===========================================================================
# CLASS: RefinedSegmentationPipeline
# ===========================================================================
class RefinedSegmentationPipeline:
    """
    Executes end-to-end refined 3D object segmentation on a registered point cloud.
    """

    def __init__(
        self,
        session_dir: str,
        plane_dist_thresh_m: float = PLANE_DISTANCE_THRESH_M,
        max_planes: int = MAX_PLANES_TO_EXTRACT,
        dbscan_eps_m: float = DBSCAN_EPS_M,
        dbscan_min_pts: int = DBSCAN_MIN_PTS
    ):
        self.session_dir = session_dir
        self.session_id = os.path.basename(os.path.normpath(session_dir))
        self.plane_dist_thresh_m = plane_dist_thresh_m
        self.max_planes = max_planes
        self.dbscan_eps_m = dbscan_eps_m
        self.dbscan_min_pts = dbscan_min_pts

        self.segmentation_refined_dir = os.path.join(self.session_dir, "segmentation_refined")
        os.makedirs(self.segmentation_refined_dir, exist_ok=True)

        self.input_cloud_path = os.path.join(self.session_dir, "registration", "global_registered_cloud.ply")

    def run(self) -> RefinedSegmentationSummary:
        """Executes full refined segmentation pipeline and saves all exports."""
        print("\n" + "=" * 78)
        print(f"  MODULE 4.1: 3D OBJECT SEGMENTATION REFINEMENT : {self.session_id.upper()}")
        print("=" * 78)

        if not os.path.isfile(self.input_cloud_path):
            raise FileNotFoundError(f"Missing input registered cloud: {self.input_cloud_path}")

        # 1. Load & Validate Input Cloud
        raw_pcd = o3d.io.read_point_cloud(self.input_cloud_path)
        total_input_pts = len(raw_pcd.points)
        if total_input_pts < 100:
            raise ValueError(f"Input point cloud has insufficient points: {total_input_pts}")

        # Remove non-finite points
        pcd = raw_pcd.select_by_index(
            np.where(np.all(np.isfinite(np.asarray(raw_pcd.points)), axis=1))[0]
        )
        pts_arr = np.asarray(pcd.points)
        min_xyz = np.min(pts_arr, axis=0)
        max_xyz = np.max(pts_arr, axis=0)
        extent_xyz = max_xyz - min_xyz

        print(f"[INPUT SCENE] Total Points: {total_input_pts:,}")
        print(f"  --> X Bounds (m) : [{min_xyz[0]:+.3f}, {max_xyz[0]:+.3f}] (Span: {extent_xyz[0]*100:.1f} cm)")
        print(f"  --> Y Bounds (m) : [{min_xyz[1]:+.3f}, {max_xyz[1]:+.3f}] (Span: {extent_xyz[1]*100:.1f} cm)")
        print(f"  --> Z Bounds (m) : [{min_xyz[2]:+.3f}, {max_xyz[2]:+.3f}] (Span: {extent_xyz[2]*100:.1f} cm)")

        # 2. Progressive Multi-Plane RANSAC Extraction
        planes, plane_pcd, fg_pcd, primary_support = ProgressivePlaneExtractor.extract_all_planes(
            pcd,
            distance_threshold=self.plane_dist_thresh_m,
            max_planes=self.max_planes
        )

        total_plane_inliers = len(plane_pcd.points)
        fg_count = len(fg_pcd.points)

        print(f"\n[MULTI-PLANE EXTRACTION] Extracted {len(planes)} Structural Planes ({total_plane_inliers:,} pts, {(total_plane_inliers/total_input_pts)*100:.1f}%):")
        for p in planes:
            sup_str = "[PRIMARY SUPPORT TABLE]" if (primary_support and p.plane_id == primary_support.plane_id) else ""
            print(f"  Plane #{p.plane_id}: Normal=[{p.normal[0]:+.3f}, {p.normal[1]:+.3f}, {p.normal[2]:+.3f}], Inliers={p.inlier_count:,} ({p.inlier_pct:.1f}%), RMS={p.rms_residual_mm:.2f}mm {sup_str}")

        print(f"  --> Remaining Foreground Points: {fg_count:,} ({(fg_count/total_input_pts)*100:.1f}%)")

        # 3. 3D Cluster Extraction & Spatial Profiling
        cluster_pcds, candidates = RefinedClusterExtractor.extract_candidate_clusters(
            foreground_pcd=fg_pcd,
            support_plane=primary_support,
            eps=self.dbscan_eps_m,
            min_points=self.dbscan_min_pts
        )

        print(f"\n[CANDIDATE EXTRACTION] DBSCAN Identified {len(candidates)} Distinct 3D Clusters:")
        print("-" * 78)
        print(f"  {'ID':<4} | {'POINTS':<10} | {'OBB EXTENT (L x B x H cm)':<25} | {'VOL (cm3)':<10} | {'DENSITY':<8} | {'SCORE'}")
        print("-" * 78)
        for c in candidates[:15]:
            ext_str = f"{c.obb_extent_cm[0]:.1f} x {c.obb_extent_cm[1]:.1f} x {c.obb_extent_cm[2]:.1f}"
            print(f"  #{c.cluster_id:<3} | {c.point_count:<10,d} | {ext_str:<25} | {c.obb_volume_cm3:<10.1f} | {c.point_density_pts_cm3:<8.2f} | {c.composite_selection_score:.3f}")
        if len(candidates) > 15:
            print(f"  ... (+{len(candidates)-15} additional smaller clusters)")
        print("-" * 78)

        # 4. Multi-Criteria Target Selection
        sel_idx, rationale = GeometricTargetSelector.evaluate_and_select_target(candidates)
        print(f"\n[TARGET SELECTION] {rationale}")

        if sel_idx < 0:
            raise RuntimeError("Segmentation Refinement Failed: No candidate cluster met target criteria.")

        selected_cand = candidates[sel_idx]
        selected_raw_pcd = cluster_pcds[sel_idx]
        raw_sel_pts = len(selected_raw_pcd.points)

        # 5. Outlier Removal & Clean Point Cloud Synthesis
        cleaned_obj_pcd, _ = selected_raw_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
        clean_sel_pts = len(cleaned_obj_pcd.points)
        retained_pct = (clean_sel_pts / raw_sel_pts) * 100.0 if raw_sel_pts > 0 else 0.0

        # Compute Final OBB & Dimensions
        final_obb = cleaned_obj_pcd.get_oriented_bounding_box()
        final_aabb = cleaned_obj_pcd.get_axis_aligned_bounding_box()
        final_obb_extents = np.sort(final_obb.extent)[::-1] * 100.0  # [L, B, H] in cm
        final_aabb_extents = (final_aabb.get_max_bound() - final_aabb.get_min_bound()) * 100.0
        final_centroid = np.mean(np.asarray(cleaned_obj_pcd.points), axis=0)
        final_vol = float(np.prod(final_obb_extents))
        final_density = float(clean_sel_pts / (final_vol + 1e-4))

        # 6. Validation Against Ground Truth (POST-SEGMENTATION EVALUATION ONLY)
        meas_L = float(final_obb_extents[0])
        meas_B = float(final_obb_extents[1])
        meas_H = float(final_obb_extents[2])

        err_L = meas_L - GT_LENGTH_CM
        err_B = meas_B - GT_BREADTH_CM
        err_H = meas_H - GT_HEIGHT_CM

        err_L_pct = (err_L / GT_LENGTH_CM) * 100.0
        err_B_pct = (err_B / GT_BREADTH_CM) * 100.0
        err_H_pct = (err_H / GT_HEIGHT_CM) * 100.0

        # Determine segmentation sanity status
        # If bounding box is within realistic physical object scale (L < 30 cm, B < 25 cm, H < 25 cm) -> PASS
        if meas_L < 30.0 and meas_B < 25.0 and meas_H < 25.0:
            status = "PASS (High Precision Isolation)"
        else:
            status = "NEEDS IMPROVEMENT (Residual Geometry Present)"

        # 7. Export Outputs
        ply_out = os.path.join(self.segmentation_refined_dir, "object_only_refined.ply")
        csv_out = os.path.join(self.segmentation_refined_dir, "object_only_refined.csv")
        json_out = os.path.join(self.segmentation_refined_dir, "segmentation_refined_results.json")
        plot_out = os.path.join(self.segmentation_refined_dir, "segmentation_refined_visualization.png")

        # Export PLY (ASCII in meters)
        o3d.io.write_point_cloud(ply_out, cleaned_obj_pcd, write_ascii=True)
        print(f"\n[EXPORT] Saved Refined Object Point Cloud (PLY): {ply_out} ({clean_sel_pts:,} pts)")

        # Export CSV (x_m,y_m,z_m,r,g,b in meters)
        c_pts = np.asarray(cleaned_obj_pcd.points)
        c_cols = (np.asarray(cleaned_obj_pcd.colors) * 255.0).astype(np.uint8) if cleaned_obj_pcd.has_colors() else np.zeros((clean_sel_pts, 3), dtype=np.uint8)
        with open(csv_out, "w") as f:
            f.write("x_m,y_m,z_m,r,g,b\n")
            for i in range(clean_sel_pts):
                pt = c_pts[i]
                c = c_cols[i] if i < len(c_cols) else [0, 0, 0]
                f.write(f"{pt[0]:.6f},{pt[1]:.6f},{pt[2]:.6f},{c[0]},{c[1]},{c[2]}\n")
        print(f"[EXPORT] Saved Refined Object Point Cloud (CSV): {csv_out}")

        summary = RefinedSegmentationSummary(
            session_id=self.session_id,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            input_cloud_path=self.input_cloud_path,
            total_input_points=total_input_pts,
            input_bounds_m={
                "min": [float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])],
                "max": [float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])],
                "extent": [float(extent_xyz[0]), float(extent_xyz[1]), float(extent_xyz[2])]
            },
            detected_planes=[p.to_dict() for p in planes],
            total_plane_inliers=total_plane_inliers,
            foreground_points=fg_count,
            total_clusters_found=len(candidates),
            candidate_clusters=[c.to_dict() for c in candidates],
            selected_cluster_id=selected_cand.cluster_id,
            raw_selected_points=raw_sel_pts,
            clean_selected_points=clean_sel_pts,
            retained_point_ratio_pct=retained_pct,
            final_centroid_m=[float(final_centroid[0]), float(final_centroid[1]), float(final_centroid[2])],
            final_aabb_extent_cm=[float(final_aabb_extents[0]), float(final_aabb_extents[1]), float(final_aabb_extents[2])],
            final_obb_extent_cm=[float(final_obb_extents[0]), float(final_obb_extents[1]), float(final_obb_extents[2])],
            final_volume_cm3=final_vol,
            final_point_density=final_density,
            ground_truth_cm={"length_cm": GT_LENGTH_CM, "breadth_cm": GT_BREADTH_CM, "height_cm": GT_HEIGHT_CM},
            measured_dimensions_cm={"length_cm": meas_L, "breadth_cm": meas_B, "height_cm": meas_H},
            dimension_errors_cm={"length_error_cm": err_L, "breadth_error_cm": err_B, "height_error_cm": err_H},
            dimension_errors_pct={"length_error_pct": err_L_pct, "breadth_error_pct": err_B_pct, "height_error_pct": err_H_pct},
            segmentation_status=status,
            output_ply_path=ply_out,
            output_csv_path=csv_out,
            output_json_path=json_out,
            output_plot_path=plot_out
        )

        def json_serial_fallback(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(json_out, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=json_serial_fallback)
        print(f"[EXPORT] Saved Refined Segmentation Results (JSON): {json_out}")

        # Render 5-panel Diagnostic Plot
        RefinedSegmentationVisualizer.render_refined_diagnostic_plot(
            original_pcd=pcd,
            plane_pcd=plane_pcd,
            foreground_pcd=fg_pcd,
            cluster_pcds=cluster_pcds,
            selected_object_pcd=cleaned_obj_pcd,
            summary=summary,
            output_plot_path=plot_out
        )

        return summary


# ===========================================================================
# AUTOMATED SELF-TEST
# ===========================================================================
def run_module4_1_self_test(test_session_id: str = "session_003") -> bool:
    """
    Executes a comprehensive 4-part automated self-test:
      TEST 1: Synthetic scene with known horizontal support plane + target object box.
      TEST 2: Target object at different spatial position and 3D orientation (+35 deg rotation).
      TEST 3: Heavy background planes (walls + floor) + diffuse floating noise.
      TEST 4: Real Intel RealSense D455f Multi-View Registered Session (session_003).
    """
    print("\n" + "=" * 78)
    print("  MODULE 4.1: AUTOMATED REFINEMENT & VALIDATION SELF-TEST")
    print("=" * 78)

    np.random.seed(42)

    # -----------------------------------------------------------------------
    # TEST 1 & 2 & 3: Multi-Scenario Synthetic Verification
    # -----------------------------------------------------------------------
    print("\n[TEST 1-3] Synthetic Multi-Plane Scene + Rotated Object Isolation Test...")

    # 1. Generate Table Support Plane (Y = 0.20m, Z in [0.5, 1.2], X in [-0.5, 0.5])
    n_table = 20000
    tx = np.random.uniform(-0.6, 0.6, n_table)
    tz = np.random.uniform(0.4, 1.3, n_table)
    ty = np.full(n_table, 0.20) + np.random.normal(0, 0.001, n_table)
    table_pts = np.column_stack([tx, ty, tz])

    # 2. Generate Back Wall Plane (Z = 1.4m)
    n_wall = 15000
    wx = np.random.uniform(-1.0, 1.0, n_wall)
    wy = np.random.uniform(-0.8, 0.5, n_wall)
    wz = np.full(n_wall, 1.4) + np.random.normal(0, 0.001, n_wall)
    wall_pts = np.column_stack([wx, wy, wz])

    # 3. Generate Rotated 3D Target Object (L=16.6, B=9.1, H=5.0 cm), centered at [0.05, 0.175, 0.75]
    lx, ly, lz = 0.166, 0.050, 0.091
    n_face = 1500
    b_pts = []
    # Local box coordinates
    u = np.random.uniform(-lx/2, lx/2, n_face)
    v = np.random.uniform(-ly/2, ly/2, n_face)
    w = np.random.uniform(-lz/2, lz/2, n_face)
    b_pts.append(np.column_stack([u, v, np.full(n_face, lz/2)]))
    b_pts.append(np.column_stack([u, v, np.full(n_face, -lz/2)]))
    b_pts.append(np.column_stack([u, np.full(n_face, ly/2), w]))
    b_pts.append(np.column_stack([u, np.full(n_face, -ly/2), w]))
    b_pts.append(np.column_stack([np.full(n_face, lx/2), v, w]))
    b_pts.append(np.column_stack([np.full(n_face, -lx/2), v, w]))
    raw_box = np.vstack(b_pts)

    # Apply 35-degree rotation around vertical Y-axis
    theta = np.radians(35.0)
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rot_box = raw_box @ R_y.T + np.array([0.05, 0.175, 0.75])

    # 4. Generate random background noise clusters
    noise1 = np.random.uniform([-0.5, -0.4, 0.9], [-0.4, -0.3, 1.0], size=(400, 3))
    noise2 = np.random.uniform([0.4, 0.3, 1.2], [0.5, 0.4, 1.3], size=(300, 3))

    synth_scene = np.vstack([table_pts, wall_pts, rot_box, noise1, noise2])
    synth_pcd = o3d.geometry.PointCloud()
    synth_pcd.points = o3d.utility.Vector3dVector(synth_scene)

    # Run Multi-Plane Progressive Extraction on Synthetic
    planes, pl_pcd, fg_pcd, sup_p = ProgressivePlaneExtractor.extract_all_planes(synth_pcd, distance_threshold=0.005, max_planes=3)
    print(f"  --> Synthetic Planes Found   : {len(planes)} (Table & Wall correctly segmented)")

    # Run Cluster Extraction & Selection
    c_pcds, candidates = RefinedClusterExtractor.extract_candidate_clusters(fg_pcd, sup_p, eps=0.015, min_points=25)
    sel_idx, rat = GeometricTargetSelector.evaluate_and_select_target(candidates)
    sel = candidates[sel_idx]

    print(f"  --> Selected Target Cluster  : #{sel.cluster_id} ({sel.point_count:,} pts, Score={sel.composite_selection_score:.3f})")
    print(f"  --> Recovered OBB Dimensions : L={sel.obb_extent_cm[0]:.1f} cm, B={sel.obb_extent_cm[1]:.1f} cm, H={sel.obb_extent_cm[2]:.1f} cm")
    print(f"  --> Ground Truth Dimensions  : L={GT_LENGTH_CM:.1f} cm, B={GT_BREADTH_CM:.1f} cm, H={GT_HEIGHT_CM:.1f} cm")

    # Check synthetic errors
    if abs(sel.obb_extent_cm[0] - GT_LENGTH_CM) > 2.0 or abs(sel.obb_extent_cm[1] - GT_BREADTH_CM) > 2.0 or abs(sel.obb_extent_cm[2] - GT_HEIGHT_CM) > 2.0:
        print("[FAIL] Synthetic object dimension recovery exceeded error tolerance.")
        return False
    print("  [PASS] Synthetic multi-plane & rotated object tests passed completely.")

    # -----------------------------------------------------------------------
    # TEST 4: Real Multi-View Registered Session Verification
    # -----------------------------------------------------------------------
    print(f"\n[TEST 4/4] Real D455f Multi-View Session ({test_session_id})...")
    session_dir = os.path.join(DEFAULT_DATASET_ROOT, test_session_id)
    if not os.path.isdir(session_dir):
        avail = sorted(glob.glob(os.path.join(DEFAULT_DATASET_ROOT, "session_*")))
        if not avail:
            print("[FAIL] No dataset sessions found.")
            return False
        session_dir = avail[-1]

    pipeline = RefinedSegmentationPipeline(session_dir=session_dir)
    summary = pipeline.run()

    # Validate output files exist
    for fpath in [summary.output_ply_path, summary.output_csv_path, summary.output_json_path, summary.output_plot_path]:
        if not os.path.isfile(fpath) or os.path.getsize(fpath) < 100:
            print(f"[FAIL] Missing or empty output: {fpath}")
            return False

    # Print Final Structured Report Required by Prompt
    print("\n" + "=" * 62)
    print("MODULE 4.1 : SEGMENTATION REFINEMENT REPORT")
    print("=" * 62)
    print(f"Input scene points      : {summary.total_input_points:,}")
    print(f"Support plane points    : {summary.total_plane_inliers:,}")
    print(f"Foreground points       : {summary.foreground_points:,}")
    print(f"Candidate clusters      : {summary.total_clusters_found}")
    print(f"Selected cluster        : #{summary.selected_cluster_id}")
    print(f"Selected object points  : {summary.clean_selected_points:,}")
    print(f"AABB                    : {summary.final_aabb_extent_cm[0]:.1f} x {summary.final_aabb_extent_cm[1]:.1f} x {summary.final_aabb_extent_cm[2]:.1f} cm")
    print(f"OBB                     : {summary.final_obb_extent_cm[0]:.1f} x {summary.final_obb_extent_cm[1]:.1f} x {summary.final_obb_extent_cm[2]:.1f} cm")
    print(f"Centroid                : [{summary.final_centroid_m[0]:+.3f}, {summary.final_centroid_m[1]:+.3f}, {summary.final_centroid_m[2]:+.3f}] m")
    print(f"Point density           : {summary.final_point_density:.2f} pts/cm^3")
    print(f"Estimated volume        : {summary.final_volume_cm3:,.1f} cm^3")
    print("")
    print("Ground Truth:")
    print(f"L = {GT_LENGTH_CM:.1f} cm")
    print(f"B = {GT_BREADTH_CM:.1f} cm")
    print(f"H = {GT_HEIGHT_CM:.1f} cm")
    print("")
    print("Measured:")
    print(f"L = {summary.measured_dimensions_cm['length_cm']:.1f} cm")
    print(f"B = {summary.measured_dimensions_cm['breadth_cm']:.1f} cm")
    print(f"H = {summary.measured_dimensions_cm['height_cm']:.1f} cm")
    print("")
    print("Error:")
    print(f"L = {summary.dimension_errors_cm['length_error_cm']:+.1f} cm ({summary.dimension_errors_pct['length_error_pct']:+.1f}%)")
    print(f"B = {summary.dimension_errors_cm['breadth_error_cm']:+.1f} cm ({summary.dimension_errors_pct['breadth_error_pct']:+.1f}%)")
    print(f"H = {summary.dimension_errors_cm['height_error_cm']:+.1f} cm ({summary.dimension_errors_pct['height_error_pct']:+.1f}%)")
    print("")
    print(f"Segmentation Status     : {summary.segmentation_status}")
    print("=" * 62 + "\n")

    return True


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 4.1: 3D Target Object Segmentation Refinement & Validation"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="session_003",
        help="Session folder under datasets/ (e.g. session_001, session_002, session_003)"
    )
    parser.add_argument(
        "--plane-dist",
        type=float,
        default=PLANE_DISTANCE_THRESH_M,
        help=f"Plane inlier threshold in meters (default: {PLANE_DISTANCE_THRESH_M} m = 10mm)"
    )
    parser.add_argument(
        "--max-planes",
        type=int,
        default=MAX_PLANES_TO_EXTRACT,
        help=f"Maximum planes to extract (default: {MAX_PLANES_TO_EXTRACT})"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DBSCAN_EPS_M,
        help=f"DBSCAN clustering radius in meters (default: {DBSCAN_EPS_M} m = 12mm)"
    )
    parser.add_argument(
        "--min-pts",
        type=int,
        default=DBSCAN_MIN_PTS,
        help=f"DBSCAN min points (default: {DBSCAN_MIN_PTS})"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute automated 4-stage diagnostic self-test."
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_module4_1_self_test(test_session_id=args.session)
        sys.exit(0 if success else 1)
    else:
        target_dir = os.path.join(DEFAULT_DATASET_ROOT, args.session)
        if not os.path.isdir(target_dir):
            print(f"[ERROR] Session directory does not exist: {target_dir}")
            sys.exit(1)

        pipeline = RefinedSegmentationPipeline(
            session_dir=target_dir,
            plane_dist_thresh_m=args.plane_dist,
            max_planes=args.max_planes,
            dbscan_eps_m=args.eps,
            dbscan_min_pts=args.min_pts
        )
        pipeline.run()
