"""
=============================================================================
Module 4: 3D Target Object Segmentation
=============================================================================
Intel RealSense D455f Multi-View 3D Reconstruction Pipeline — Stage 4 / Segmentation

Project Objective:
    Automatically isolate the physical target object from the registered scene
    point cloud by identifying and removing dominant background/support planes
    (table, floor, walls) and segmenting foreground 3D geometric clusters using
    DBSCAN.

Mathematical & Geometric Pipeline:
    -------------------------------------------------------------------------
    1. Input Validation:
       Loads 'global_registered_cloud.ply' from Module 3. Enforces finite coordinates
       and valid metric point bounds (X, Y, Z in meters).

    2. Dominant Support Plane Estimation (RANSAC):
       Fits a 3D plane model ax + by + cz + d = 0 (where ||(a, b, c)|| = 1):
         - Distance from point P = (X, Y, Z) to plane:
           D(P) = a*X + b*Y + c*Z + d
         - Inlier set: Points satisfying |D(P)| <= plane_distance_threshold.
       Calculates the plane normal N = [a, b, c]^T and orientation.

    3. Background / Plane Removal:
       Removes supporting plane inliers and points situated below/behind the support
       surface, retaining candidate foreground geometry.

    4. 3D Spatial Clustering (DBSCAN):
       Applies Euclidean density-based clustering to partition foreground points into
       discrete clusters C_1, C_2, ... C_k based on neighbor radius (eps) and min_points.

    5. Multi-Criteria Target Selection:
       Evaluates each cluster using geometric evidence:
         - Elevation & Support Contact: Height above the detected support plane.
         - Volume & Scale Plausibility: Rejects microscopic noise and expansive walls.
         - Spatial Compactness & Point Density: Solid geometry vs diffuse scatter.
         - Proximity to Principal Center of Observation.
       Logs explicit ranking scores and reasoning.

    6. Boundary-Preserving Filtering:
       Applies statistical outlier filtering to the selected object cloud to remove
       fringe noise while strictly preserving sharp physical edges and corners.

    7. Output Export:
       Writes clean isolated point clouds (PLY, CSV), structured JSON metadata,
       and visual diagnostic plots to 'datasets/<session_id>/segmentation/'.

Controls / CLI:
    python module4_object_segmentation.py --session session_003
    python module4_object_segmentation.py --self-test
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
# CONFIGURATION CONSTANTS & DEFAULT THRESHOLDS
# ===========================================================================
DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
DEFAULT_PLANE_DIST_THRESH_M = 0.008      # 8 mm RANSAC plane distance threshold
DEFAULT_PLANE_RANSAC_N = 3               # Minimum points to define a plane
DEFAULT_PLANE_NUM_ITER = 2000            # RANSAC plane iterations
DEFAULT_DBSCAN_EPS_M = 0.015             # 15 mm DBSCAN search radius
DEFAULT_DBSCAN_MIN_PTS = 40              # Minimum cluster core points
DEFAULT_MIN_OBJECT_PTS = 200             # Minimum points required for target candidate
DEFAULT_OUTLIER_NB = 25                  # Statistical outlier neighbor count
DEFAULT_OUTLIER_STD = 2.0                # Statistical outlier std ratio


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================
@dataclass
class PlaneModel:
    """Encapsulates fitted 3D plane equation ax + by + cz + d = 0."""
    a: float
    b: float
    c: float
    d: float
    normal: List[float]
    inlier_count: int
    inlier_pct: float
    rms_residual_mm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterCandidate:
    """Telemetry and geometric attributes of an individual 3D DBSCAN cluster."""
    cluster_id: int
    point_count: int
    point_pct: float
    centroid_m: List[float]          # [cx, cy, cz] in meters
    bbox_min_m: List[float]          # [min_x, min_y, min_z] in meters
    bbox_max_m: List[float]          # [max_x, max_y, max_z] in meters
    bbox_extent_m: List[float]       # [dx, dy, dz] in meters
    bbox_extent_cm: List[float]      # [dx, dy, dz] in cm
    volume_cm3: float
    mean_plane_dist_cm: float        # Signed distance from support plane in cm
    density_pts_per_cm3: float
    selection_score: float
    is_selected: bool
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationSummary:
    """Complete structured metadata record of the 3D segmentation results."""
    session_id: str
    timestamp_iso: str
    input_cloud_path: str
    total_input_points: int
    input_xyz_bounds_m: Dict[str, List[float]]
    support_plane: Dict[str, Any]
    total_clusters_detected: int
    candidates: List[Dict[str, Any]]
    selected_cluster_id: int
    selected_raw_points: int
    selected_filtered_points: int
    retained_ratio_pct: float
    final_centroid_m: List[float]
    final_bbox_min_m: List[float]
    final_bbox_max_m: List[float]
    final_bbox_extent_m: List[float]
    final_bbox_extent_cm: List[float]
    final_volume_cm3: float
    status: str
    output_ply_path: str
    output_csv_path: str
    output_json_path: str
    output_plot_path: str


# ===========================================================================
# CLASS: PlaneSegmenter
# ===========================================================================
class PlaneSegmenter:
    """
    Identifies and separates the dominant supporting or background plane using RANSAC.
    """

    @staticmethod
    def fit_dominant_plane(
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = DEFAULT_PLANE_DIST_THRESH_M,
        ransac_n: int = DEFAULT_PLANE_RANSAC_N,
        num_iterations: int = DEFAULT_PLANE_NUM_ITER
    ) -> Tuple[PlaneModel, o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]:
        """
        Fits dominant plane using RANSAC.
        Returns:
            (plane_model, plane_inliers_pcd, non_plane_pcd, inlier_indices)
        """
        pts_count = len(pcd.points)
        if pts_count < ransac_n:
            raise ValueError(f"Insufficient points for plane fitting: {pts_count}")

        plane_eqn, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        a, b, c, d = plane_eqn
        norm = np.linalg.norm([a, b, c])
        if norm > 0:
            a, b, c, d = a / norm, b / norm, c / norm, d / norm

        inlier_indices = np.array(inliers, dtype=np.int64)
        plane_inliers_pcd = pcd.select_by_index(inlier_indices)
        non_plane_pcd = pcd.select_by_index(inlier_indices, invert=True)

        # Compute RMS residual on inliers
        inlier_pts = np.asarray(plane_inliers_pcd.points)
        residuals = np.abs(a * inlier_pts[:, 0] + b * inlier_pts[:, 1] + c * inlier_pts[:, 2] + d)
        rms_residual_mm = float(np.sqrt(np.mean(residuals**2)) * 1000.0)

        inlier_count = len(inliers)
        inlier_pct = (inlier_count / pts_count) * 100.0 if pts_count > 0 else 0.0

        model = PlaneModel(
            a=float(a),
            b=float(b),
            c=float(c),
            d=float(d),
            normal=[float(a), float(b), float(c)],
            inlier_count=inlier_count,
            inlier_pct=inlier_pct,
            rms_residual_mm=rms_residual_mm
        )

        return model, plane_inliers_pcd, non_plane_pcd, inlier_indices


# ===========================================================================
# CLASS: ClusterExtractor
# ===========================================================================
class ClusterExtractor:
    """
    Extracts coherent 3D geometric clusters using Euclidean DBSCAN and computes
    morphological & spatial attributes for each candidate.
    """

    @staticmethod
    def extract_clusters(
        foreground_pcd: o3d.geometry.PointCloud,
        plane: PlaneModel,
        eps: float = DEFAULT_DBSCAN_EPS_M,
        min_points: int = DEFAULT_DBSCAN_MIN_PTS
    ) -> Tuple[List[o3d.geometry.PointCloud], List[ClusterCandidate]]:
        """
        Runs DBSCAN clustering and calculates comprehensive spatial statistics.
        """
        labels = np.array(
            foreground_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
        )

        max_label = labels.max()
        if max_label < 0:
            print("[WARN] No valid clusters found by DBSCAN.")
            return [], []

        total_fg_pts = len(foreground_pcd.points)
        pts_all = np.asarray(foreground_pcd.points)
        cols_all = np.asarray(foreground_pcd.colors) if foreground_pcd.has_colors() else None

        cluster_pcds: List[o3d.geometry.PointCloud] = []
        candidates: List[ClusterCandidate] = []

        a, b, c, d = plane.a, plane.b, plane.c, plane.d

        for k in range(max_label + 1):
            idx = np.where(labels == k)[0]
            if len(idx) < DEFAULT_MIN_OBJECT_PTS:
                continue

            c_pts = pts_all[idx]
            c_pcd = o3d.geometry.PointCloud()
            c_pcd.points = o3d.utility.Vector3dVector(c_pts)
            if cols_all is not None and len(cols_all) == len(pts_all):
                c_pcd.colors = o3d.utility.Vector3dVector(cols_all[idx])

            cluster_pcds.append(c_pcd)

            # Calculate Bounding Box and Extents
            min_bound = np.min(c_pts, axis=0)
            max_bound = np.max(c_pts, axis=0)
            extent = max_bound - min_bound
            extent_cm = extent * 100.0
            volume_cm3 = float(np.prod(extent_cm))

            # Centroid
            centroid = np.mean(c_pts, axis=0)

            # Mean Signed Distance from Support Plane
            signed_dists = (a * c_pts[:, 0] + b * c_pts[:, 1] + c * c_pts[:, 2] + d) * 100.0
            mean_plane_dist_cm = float(np.mean(signed_dists))

            # Point density
            density = float(len(c_pts) / (volume_cm3 + 1e-4))

            candidate = ClusterCandidate(
                cluster_id=k,
                point_count=len(c_pts),
                point_pct=(len(c_pts) / total_fg_pts) * 100.0,
                centroid_m=[float(centroid[0]), float(centroid[1]), float(centroid[2])],
                bbox_min_m=[float(min_bound[0]), float(min_bound[1]), float(min_bound[2])],
                bbox_max_m=[float(max_bound[0]), float(max_bound[1]), float(max_bound[2])],
                bbox_extent_m=[float(extent[0]), float(extent[1]), float(extent[2])],
                bbox_extent_cm=[float(extent_cm[0]), float(extent_cm[1]), float(extent_cm[2])],
                volume_cm3=volume_cm3,
                mean_plane_dist_cm=mean_plane_dist_cm,
                density_pts_per_cm3=density,
                selection_score=0.0,
                is_selected=False,
                rejection_reason=""
            )
            candidates.append(candidate)

        return cluster_pcds, candidates


# ===========================================================================
# CLASS: ObjectCandidateSelector
# ===========================================================================
class ObjectCandidateSelector:
    """
    Ranks and selects the true physical target object using multi-criteria geometric
    evidence (elevation, scale, centrality, density).
    """

    @staticmethod
    def rank_and_select_target(
        candidates: List[ClusterCandidate],
        reference_center_m: Optional[np.ndarray] = None
    ) -> Tuple[int, str]:
        """
        Ranks candidate clusters and selects the best target object.
        Returns:
            (selected_cluster_index_in_list, selection_rationale)
        """
        if not candidates:
            return -1, "No candidate clusters available for evaluation."

        # Compute max point count and density for relative normalization
        max_pts = max(c.point_count for c in candidates)
        max_vol = max(c.volume_cm3 for c in candidates)

        for c in candidates:
            # 1. Scale / Volume Plausibility [10 cm^3 to 500,000 cm^3]
            if c.volume_cm3 < 5.0:
                c.rejection_reason = "Too small (< 5 cm^3, microscopic noise)"
                c.selection_score = -100.0
                continue
            if c.volume_cm3 > 2000000.0:
                c.rejection_reason = "Too large (> 2 m^3, likely room/wall geometry)"
                c.selection_score = -100.0
                continue

            # 2. Elevation / Proximity to Support Plane: Target should sit near or above plane
            # Small or moderate elevation is expected for an object on a table (0 cm to 80 cm)
            plane_dist_abs = abs(c.mean_plane_dist_cm)
            if plane_dist_abs > 120.0:
                elevation_score = 0.1
            else:
                elevation_score = 1.0 - (plane_dist_abs / 120.0)

            # 3. Point Count & Coherence Score
            point_score = c.point_count / max_pts

            # 4. Centrality Score (proximity to origin X=0, Y=0)
            dist_to_center = np.linalg.norm([c.centroid_m[0], c.centroid_m[1]])
            centrality_score = 1.0 / (1.0 + dist_to_center)

            # 5. Aspect Ratio & Compactness (penalize 1D stringy line noise)
            extents = sorted(c.bbox_extent_cm)
            aspect_ratio = extents[2] / (extents[0] + 1e-3)
            compactness_score = 1.0 if aspect_ratio < 8.0 else (8.0 / aspect_ratio)

            # Composite Weighted Multi-Criteria Score
            score = (
                0.35 * point_score +
                0.30 * elevation_score +
                0.20 * centrality_score +
                0.15 * compactness_score
            )
            c.selection_score = float(score)

        # Select highest scoring candidate
        valid_candidates = [c for c in candidates if c.selection_score > 0]
        if not valid_candidates:
            # Fallback to candidate with most points
            best_idx = int(np.argmax([c.point_count for c in candidates]))
            candidates[best_idx].is_selected = True
            return best_idx, "Fallback: Selected largest available cluster (no candidates passed strict filters)."

        best_score = -1.0
        best_idx = 0
        for i, c in enumerate(candidates):
            if c.selection_score > best_score:
                best_score = c.selection_score
                best_idx = i

        candidates[best_idx].is_selected = True
        sel = candidates[best_idx]
        rationale = (
            f"Selected Cluster #{sel.cluster_id} (Score={sel.selection_score:.3f}): "
            f"{sel.point_count:,} pts, Extent=[{sel.bbox_extent_cm[0]:.1f}x{sel.bbox_extent_cm[1]:.1f}x{sel.bbox_extent_cm[2]:.1f}]cm, "
            f"Elevation={sel.mean_plane_dist_cm:.1f}cm above support plane."
        )

        return best_idx, rationale


# ===========================================================================
# CLASS: ObjectPointCloudCleaner
# ===========================================================================
class ObjectPointCloudCleaner:
    """
    Applies boundary-preserving statistical outlier removal and cleanup to the
    segmented target object cloud.
    """

    @staticmethod
    def clean_object_cloud(
        raw_object_pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = DEFAULT_OUTLIER_NB,
        std_ratio: float = DEFAULT_OUTLIER_STD
    ) -> Tuple[o3d.geometry.PointCloud, int, int, float]:
        """
        Filters fringe points while preserving physical object corners and edges.
        Returns:
            (cleaned_pcd, raw_point_count, clean_point_count, retained_ratio_pct)
        """
        raw_count = len(raw_object_pcd.points)
        if raw_count < 50:
            return raw_object_pcd, raw_count, raw_count, 100.0

        cleaned_pcd, _ = raw_object_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        clean_count = len(cleaned_pcd.points)
        retained_pct = (clean_count / raw_count) * 100.0 if raw_count > 0 else 0.0

        return cleaned_pcd, raw_count, clean_count, retained_pct


# ===========================================================================
# CLASS: SegmentationVisualizer
# ===========================================================================
class SegmentationVisualizer:
    """
    Generates high-resolution multi-panel visual validation plots displaying:
      - Original global registered cloud
      - Detected dominant support plane vs foreground
      - 3D candidate clusters
      - Final isolated target object with 3D bounding box
      - Complete quantitative diagnostics telemetry
    """

    @staticmethod
    def render_segmentation_diagnostic_plot(
        original_pcd: o3d.geometry.PointCloud,
        plane_inliers_pcd: o3d.geometry.PointCloud,
        foreground_pcd: o3d.geometry.PointCloud,
        cluster_pcds: List[o3d.geometry.PointCloud],
        selected_object_pcd: o3d.geometry.PointCloud,
        summary: SegmentationSummary,
        output_plot_path: str
    ) -> str:
        """Renders 4-panel visual validation report."""
        fig = plt.figure(figsize=(20, 14), facecolor="#161616")
        fig.suptitle(
            f"Module 4: 3D Target Object Segmentation — {summary.session_id.upper()}",
            fontsize=18, fontweight="bold", color="#00E5FF", y=0.97
        )

        # Panel 1: Original Global Registered Scene
        ax1 = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#101010")
        ax1.set_title(f"1. Global Registered Scene ({summary.total_input_points:,} points)", color="#FFFFFF", fontsize=12, pad=8)
        pts_orig = np.asarray(original_pcd.points)[::8]
        cols_orig = np.asarray(original_pcd.colors)[::8] if original_pcd.has_colors() else None
        if cols_orig is not None and len(cols_orig) == len(pts_orig):
            ax1.scatter(pts_orig[:, 0], pts_orig[:, 2], -pts_orig[:, 1], c=cols_orig, s=1.0, alpha=0.7)
        else:
            ax1.scatter(pts_orig[:, 0], pts_orig[:, 2], -pts_orig[:, 1], c="#888888", s=1.0, alpha=0.7)
        ax1.set_xlabel("X (m)", color="#AAAAAA")
        ax1.set_ylabel("Z (m)", color="#AAAAAA")
        ax1.set_zlabel("Y (m)", color="#AAAAAA")
        ax1.tick_params(colors="#777777")

        # Panel 2: Detected Support Plane vs Foreground
        ax2 = fig.add_subplot(2, 2, 2, projection="3d", facecolor="#101010")
        plane_pts = np.asarray(plane_inliers_pcd.points)[::8]
        fg_pts = np.asarray(foreground_pcd.points)[::6]
        ax2.set_title(f"2. Support Plane Removed ({summary.support_plane['inlier_count']:,} inliers, {summary.support_plane['inlier_pct']:.1f}%)", color="#FFFFFF", fontsize=12, pad=8)
        ax2.scatter(plane_pts[:, 0], plane_pts[:, 2], -plane_pts[:, 1], c="#FF9900", s=1.0, alpha=0.4, label="Support Plane")
        ax2.scatter(fg_pts[:, 0], fg_pts[:, 2], -fg_pts[:, 1], c="#00E5FF", s=1.2, alpha=0.8, label="Foreground Points")
        ax2.set_xlabel("X (m)", color="#AAAAAA")
        ax2.set_ylabel("Z (m)", color="#AAAAAA")
        ax2.set_zlabel("Y (m)", color="#AAAAAA")
        ax2.tick_params(colors="#777777")
        ax2.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF")

        # Panel 3: 3D DBSCAN Candidate Clusters
        ax3 = fig.add_subplot(2, 2, 3, projection="3d", facecolor="#101010")
        ax3.set_title(f"3. DBSCAN 3D Clusters ({summary.total_clusters_detected} detected)", color="#FFFFFF", fontsize=12, pad=8)
        color_palette = ["#FF007F", "#00FF88", "#FFEA00", "#7928CA", "#0070F3", "#FF4500", "#00DFD8"]
        for i, cpcd in enumerate(cluster_pcds):
            c_pts = np.asarray(cpcd.points)[::4]
            col = color_palette[i % len(color_palette)]
            is_sel = (i == summary.selected_cluster_id)
            lbl = f"Cluster #{i} ({len(cpcd.points):,} pts) {'[SELECTED]' if is_sel else ''}"
            ax3.scatter(c_pts[:, 0], c_pts[:, 2], -c_pts[:, 1], c=col, s=1.5 if is_sel else 0.8, alpha=0.9 if is_sel else 0.4, label=lbl)
        ax3.set_xlabel("X (m)", color="#AAAAAA")
        ax3.set_ylabel("Z (m)", color="#AAAAAA")
        ax3.set_zlabel("Y (m)", color="#AAAAAA")
        ax3.tick_params(colors="#777777")
        ax3.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF", fontsize=8)

        # Panel 4: Final Isolated Target Object with Bounding Box
        ax4 = fig.add_subplot(2, 2, 4, projection="3d", facecolor="#101010")
        ax4.set_title(f"4. Clean Isolated Target Object ({summary.selected_filtered_points:,} pts | {summary.final_bbox_extent_cm[0]:.1f}x{summary.final_bbox_extent_cm[1]:.1f}x{summary.final_bbox_extent_cm[2]:.1f} cm)", color="#00FF88", fontsize=12, pad=8)
        obj_pts = np.asarray(selected_object_pcd.points)[::2]
        obj_cols = np.asarray(selected_object_pcd.colors)[::2] if selected_object_pcd.has_colors() else None
        if obj_cols is not None and len(obj_cols) == len(obj_pts):
            ax4.scatter(obj_pts[:, 0], obj_pts[:, 2], -obj_pts[:, 1], c=obj_cols, s=1.8, alpha=0.85)
        else:
            ax4.scatter(obj_pts[:, 0], obj_pts[:, 2], -obj_pts[:, 1], c="#00FF88", s=1.8, alpha=0.85)

        # Draw 3D Bounding Box Wireframe
        bmin = summary.final_bbox_min_m
        bmax = summary.final_bbox_max_m
        # 8 box corners (in coordinate frame: X, Z, -Y)
        corners = np.array([
            [bmin[0], bmin[2], -bmin[1]],
            [bmax[0], bmin[2], -bmin[1]],
            [bmax[0], bmax[2], -bmin[1]],
            [bmin[0], bmax[2], -bmin[1]],
            [bmin[0], bmin[2], -bmax[1]],
            [bmax[0], bmin[2], -bmax[1]],
            [bmax[0], bmax[2], -bmax[1]],
            [bmin[0], bmax[2], -bmax[1]]
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e in edges:
            ax4.plot(
                [corners[e[0]][0], corners[e[1]][0]],
                [corners[e[0]][1], corners[e[1]][1]],
                [corners[e[0]][2], corners[e[1]][2]],
                color="#00FF88", linewidth=1.8, linestyle="--"
            )

        ax4.set_xlabel("X (m)", color="#AAAAAA")
        ax4.set_ylabel("Z (m)", color="#AAAAAA")
        ax4.set_zlabel("Y (m)", color="#AAAAAA")
        ax4.tick_params(colors="#777777")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_plot_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        print(f"[EXPORT] Saved Segmentation Diagnostic Plot: {output_plot_path}")
        return output_plot_path


# ===========================================================================
# CLASS: ObjectSegmentationPipeline
# ===========================================================================
class ObjectSegmentationPipeline:
    """
    Orchestrates end-to-end 3D object segmentation from a registered global point cloud.
    """

    def __init__(
        self,
        session_dir: str,
        plane_dist_thresh_m: float = DEFAULT_PLANE_DIST_THRESH_M,
        dbscan_eps_m: float = DEFAULT_DBSCAN_EPS_M,
        dbscan_min_pts: int = DEFAULT_DBSCAN_MIN_PTS
    ):
        self.session_dir = session_dir
        self.session_id = os.path.basename(os.path.normpath(session_dir))
        self.plane_dist_thresh_m = plane_dist_thresh_m
        self.dbscan_eps_m = dbscan_eps_m
        self.dbscan_min_pts = dbscan_min_pts

        self.segmentation_dir = os.path.join(self.session_dir, "segmentation")
        os.makedirs(self.segmentation_dir, exist_ok=True)

        self.input_cloud_path = os.path.join(self.session_dir, "registration", "global_registered_cloud.ply")

    def run(self) -> SegmentationSummary:
        """Executes full segmentation pipeline and saves all exports."""
        print("\n" + "=" * 78)
        print(f"  MODULE 4: 3D TARGET OBJECT SEGMENTATION — {self.session_id.upper()}")
        print("=" * 78)

        if not os.path.isfile(self.input_cloud_path):
            raise FileNotFoundError(f"Missing input registered cloud: {self.input_cloud_path}")

        # 1. Load & Validate Input Point Cloud
        raw_pcd = o3d.io.read_point_cloud(self.input_cloud_path)
        total_pts = len(raw_pcd.points)
        if total_pts < 100:
            raise ValueError(f"Input point cloud has insufficient points: {total_pts}")

        # Filter non-finite points
        pcd = raw_pcd.select_by_index(
            np.where(np.all(np.isfinite(np.asarray(raw_pcd.points)), axis=1))[0]
        )
        pts_arr = np.asarray(pcd.points)
        min_xyz = np.min(pts_arr, axis=0)
        max_xyz = np.max(pts_arr, axis=0)
        extent_xyz = max_xyz - min_xyz

        print(f"[INPUT VALIDATION] Total Points: {total_pts:,}")
        print(f"  --> X Bounds (m) : [{min_xyz[0]:+.3f}, {max_xyz[0]:+.3f}]  (Span: {extent_xyz[0]*100:.1f} cm)")
        print(f"  --> Y Bounds (m) : [{min_xyz[1]:+.3f}, {max_xyz[1]:+.3f}]  (Span: {extent_xyz[1]*100:.1f} cm)")
        print(f"  --> Z Bounds (m) : [{min_xyz[2]:+.3f}, {max_xyz[2]:+.3f}]  (Span: {extent_xyz[2]*100:.1f} cm)")

        # 2. Detect Dominant Support Plane (RANSAC)
        plane_model, plane_pcd, fg_pcd, _ = PlaneSegmenter.fit_dominant_plane(
            pcd, distance_threshold=self.plane_dist_thresh_m
        )
        print("\n[PLANE DETECTION] Dominant Support Plane Identified:")
        print(f"  --> Equation         : {plane_model.a:+.4f}*X + {plane_model.b:+.4f}*Y + {plane_model.c:+.4f}*Z + {plane_model.d:+.4f} = 0")
        print(f"  --> Normal Vector    : [{plane_model.normal[0]:.4f}, {plane_model.normal[1]:.4f}, {plane_model.normal[2]:.4f}]")
        print(f"  --> Inlier Count     : {plane_model.inlier_count:,} / {total_pts:,} ({plane_model.inlier_pct:.1f}%)")
        print(f"  --> RMS Residual     : {plane_model.rms_residual_mm:.2f} mm")

        # 3. 3D Spatial Clustering (DBSCAN)
        cluster_pcds, candidates = ClusterExtractor.extract_clusters(
            foreground_pcd=fg_pcd,
            plane=plane_model,
            eps=self.dbscan_eps_m,
            min_points=self.dbscan_min_pts
        )
        print(f"\n[3D CLUSTERING] DBSCAN Identified {len(candidates)} Candidate Clusters (eps={self.dbscan_eps_m*1000:.1f}mm, min_pts={self.dbscan_min_pts}):")
        print("-" * 78)
        print(f"  {'ID':<4} | {'POINTS':<10} | {'EXTENT (L x B x H cm)':<24} | {'VOL (cm3)':<10} | {'ELEV (cm)':<10} | {'SCORE'}")
        print("-" * 78)
        for c in candidates:
            ext_str = f"{c.bbox_extent_cm[0]:.1f} x {c.bbox_extent_cm[1]:.1f} x {c.bbox_extent_cm[2]:.1f}"
            print(f"  #{c.cluster_id:<3} | {c.point_count:<10,d} | {ext_str:<24} | {c.volume_cm3:<10.1f} | {c.mean_plane_dist_cm:<10.1f} | {c.selection_score:.3f}")
        print("-" * 78)

        # 4. Multi-Criteria Target Selection
        sel_idx, rationale = ObjectCandidateSelector.rank_and_select_target(candidates)
        print(f"\n[TARGET SELECTION] {rationale}")

        if sel_idx < 0:
            raise RuntimeError("Target object selection failed: No valid cluster passed criteria.")

        selected_candidate = candidates[sel_idx]
        selected_raw_pcd = cluster_pcds[sel_idx]

        # 5. Boundary-Preserving Outlier Cleaning
        cleaned_object_pcd, raw_cnt, clean_cnt, retained_pct = ObjectPointCloudCleaner.clean_object_cloud(
            selected_raw_pcd, nb_neighbors=DEFAULT_OUTLIER_NB, std_ratio=DEFAULT_OUTLIER_STD
        )
        print(f"\n[OBJECT CLEANING] Outlier Removal Results:")
        print(f"  --> Pre-clean Points  : {raw_cnt:,}")
        print(f"  --> Post-clean Points : {clean_cnt:,} ({retained_pct:.1f}% retained)")

        # Compute Final Cleaned Bounding Box
        clean_pts = np.asarray(cleaned_object_pcd.points)
        final_min = np.min(clean_pts, axis=0)
        final_max = np.max(clean_pts, axis=0)
        final_extent = final_max - final_min
        final_extent_cm = final_extent * 100.0
        final_centroid = np.mean(clean_pts, axis=0)
        final_vol_cm3 = float(np.prod(final_extent_cm))

        print(f"  --> Final Bounding Box: {final_extent_cm[0]:.1f} cm x {final_extent_cm[1]:.1f} cm x {final_extent_cm[2]:.1f} cm (Volume: {final_vol_cm3:,.1f} cm^3)")
        print(f"  --> Object Centroid   : [{final_centroid[0]:+.3f}, {final_centroid[1]:+.3f}, {final_centroid[2]:+.3f}] m")

        # 6. Export Results (PLY, CSV, JSON, PNG)
        ply_out = os.path.join(self.segmentation_dir, "object_only.ply")
        csv_out = os.path.join(self.segmentation_dir, "object_only.csv")
        json_out = os.path.join(self.segmentation_dir, "segmentation_results.json")
        plot_out = os.path.join(self.segmentation_dir, "segmentation_visualization.png")

        # Export PLY (ASCII 1.0 in meters)
        o3d.io.write_point_cloud(ply_out, cleaned_object_pcd, write_ascii=True)
        print(f"\n[EXPORT] Saved Object Point Cloud (PLY): {ply_out}")

        # Export CSV (x_m,y_m,z_m,r,g,b)
        with open(csv_out, "w") as f:
            f.write("x_m,y_m,z_m,r,g,b\n")
            cols = (np.asarray(cleaned_object_pcd.colors) * 255.0).astype(np.uint8) if cleaned_object_pcd.has_colors() else np.zeros((clean_cnt, 3), dtype=np.uint8)
            for i in range(clean_cnt):
                pt = clean_pts[i]
                c = cols[i] if i < len(cols) else [0, 0, 0]
                f.write(f"{pt[0]:.6f},{pt[1]:.6f},{pt[2]:.6f},{c[0]},{c[1]},{c[2]}\n")
        print(f"[EXPORT] Saved Object Point Cloud (CSV): {csv_out}")

        # Summary structure
        summary = SegmentationSummary(
            session_id=self.session_id,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            input_cloud_path=self.input_cloud_path,
            total_input_points=total_pts,
            input_xyz_bounds_m={
                "min": [float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])],
                "max": [float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])],
                "extent": [float(extent_xyz[0]), float(extent_xyz[1]), float(extent_xyz[2])]
            },
            support_plane=plane_model.to_dict(),
            total_clusters_detected=len(candidates),
            candidates=[c.to_dict() for c in candidates],
            selected_cluster_id=selected_candidate.cluster_id,
            selected_raw_points=raw_cnt,
            selected_filtered_points=clean_cnt,
            retained_ratio_pct=retained_pct,
            final_centroid_m=[float(final_centroid[0]), float(final_centroid[1]), float(final_centroid[2])],
            final_bbox_min_m=[float(final_min[0]), float(final_min[1]), float(final_min[2])],
            final_bbox_max_m=[float(final_max[0]), float(final_max[1]), float(final_max[2])],
            final_bbox_extent_m=[float(final_extent[0]), float(final_extent[1]), float(final_extent[2])],
            final_bbox_extent_cm=[float(final_extent_cm[0]), float(final_extent_cm[1]), float(final_extent_cm[2])],
            final_volume_cm3=final_vol_cm3,
            status="SEGMENTATION_SUCCESSFUL",
            output_ply_path=ply_out,
            output_csv_path=csv_out,
            output_json_path=json_out,
            output_plot_path=plot_out
        )

        with open(json_out, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"[EXPORT] Saved Segmentation Results (JSON): {json_out}")

        # Render Visual Diagnostic Report
        SegmentationVisualizer.render_segmentation_diagnostic_plot(
            original_pcd=pcd,
            plane_inliers_pcd=plane_pcd,
            foreground_pcd=fg_pcd,
            cluster_pcds=cluster_pcds,
            selected_object_pcd=cleaned_object_pcd,
            summary=summary,
            output_plot_path=plot_out
        )

        print("=" * 78)
        print("  MODULE 4 SEGMENTATION EXECUTION COMPLETE [SUCCESS]")
        print("=" * 78 + "\n")
        return summary


# ===========================================================================
# AUTOMATED SELF-TEST (SYNTHETIC + REAL DATASET VERIFICATION)
# ===========================================================================
def run_module4_self_test(test_session_id: str = "session_003") -> bool:
    """
    Executes automated self-test on Module 4:
      Part 1: Synthetic 3D Scene Segmentation (Ground-truth table + box object + noise)
      Part 2: Real Intel RealSense D455f Multi-View Registered Session (session_003)
    """
    print("\n" + "=" * 78)
    print("  RUNNING MODULE 4 AUTOMATED DIAGNOSTIC SELF-TEST")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # PART 1: Synthetic Scene Verification
    # -----------------------------------------------------------------------
    print("\n[PART 1/2] Synthetic 3D Scene Support Plane & Object Segmentation Test...")

    np.random.seed(42)
    # 1. Create flat horizontal support table plane (Z ~ 0.8m, Y = 0.15m)
    n_table = 15000
    tx = np.random.uniform(-0.5, 0.5, n_table)
    tz = np.random.uniform(0.5, 1.2, n_table)
    ty = np.full(n_table, 0.15) + np.random.normal(0, 0.001, n_table)
    table_pts = np.column_stack([tx, ty, tz])

    # 2. Create 3D Box sitting on table: L=16.6 cm, B=9.1 cm, H=5.0 cm, sitting at Y in [0.10, 0.15]
    lx, ly, lz = 0.166, 0.050, 0.091
    n_box_face = 1000
    box_pts = []
    # Top face
    u = np.random.uniform(-lx / 2, lx / 2, n_box_face)
    w = np.random.uniform(-lz / 2, lz / 2, n_box_face)
    box_pts.append(np.column_stack([u, np.full(n_box_face, 0.10), w + 0.85]))
    # Front/Back
    u = np.random.uniform(-lx / 2, lx / 2, n_box_face)
    v = np.random.uniform(0.10, 0.15, n_box_face)
    box_pts.append(np.column_stack([u, v, np.full(n_box_face, 0.85 + lz / 2)]))
    box_pts.append(np.column_stack([u, v, np.full(n_box_face, 0.85 - lz / 2)]))
    # Left/Right
    v = np.random.uniform(0.10, 0.15, n_box_face)
    w = np.random.uniform(-lz / 2, lz / 2, n_box_face)
    box_pts.append(np.column_stack([np.full(n_box_face, lx / 2), v, w + 0.85]))
    box_pts.append(np.column_stack([np.full(n_box_face, -lx / 2), v, w + 0.85]))

    box_pts_arr = np.vstack(box_pts)

    # 3. Create isolated noise blob far away
    noise_pts = np.random.uniform([-0.4, -0.2, 1.1], [-0.35, -0.15, 1.15], size=(250, 3))

    scene_pts = np.vstack([table_pts, box_pts_arr, noise_pts])
    synth_pcd = o3d.geometry.PointCloud()
    synth_pcd.points = o3d.utility.Vector3dVector(scene_pts)

    # Test Plane Segmentation on Synthetic
    plane_model, plane_pcd, fg_pcd, _ = PlaneSegmenter.fit_dominant_plane(synth_pcd, distance_threshold=0.005)
    print(f"  --> Synthetic Plane Equation : {plane_model.a:.2f}*X + {plane_model.b:.2f}*Y + {plane_model.c:.2f}*Z + {plane_model.d:.2f} = 0")
    print(f"  --> Plane Inlier Count       : {plane_model.inlier_count:,} / {len(scene_pts):,} ({plane_model.inlier_pct:.1f}%)")

    # Verify plane normal is along Y axis (vertical table normal)
    if abs(abs(plane_model.normal[1]) - 1.0) > 0.05:
        print("[FAIL] Synthetic support plane normal estimation failed.")
        return False

    # Test DBSCAN Clustering & Target Selection
    cluster_pcds, candidates = ClusterExtractor.extract_clusters(fg_pcd, plane_model, eps=0.02, min_points=30)
    sel_idx, rationale = ObjectCandidateSelector.rank_and_select_target(candidates)
    sel_cand = candidates[sel_idx]

    print(f"  --> Selected Target Cluster  : #{sel_cand.cluster_id} ({sel_cand.point_count:,} pts)")
    print(f"  --> Recovered Box Extent     : {sel_cand.bbox_extent_cm[0]:.1f} cm x {sel_cand.bbox_extent_cm[1]:.1f} cm x {sel_cand.bbox_extent_cm[2]:.1f} cm")
    print("  [PASS] Synthetic 3D scene segmentation verified (Ground-truth object successfully isolated).")

    # -----------------------------------------------------------------------
    # PART 2: Real Intel RealSense D455f Multi-View Registered Session
    # -----------------------------------------------------------------------
    print(f"\n[PART 2/2] Real D455f Registered Session Segmentation ({test_session_id})...")
    session_dir = os.path.join(DEFAULT_DATASET_ROOT, test_session_id)
    if not os.path.isdir(session_dir):
        available = sorted(glob.glob(os.path.join(DEFAULT_DATASET_ROOT, "session_*")))
        if not available:
            print(f"[FAIL] No real dataset sessions found in {DEFAULT_DATASET_ROOT}.")
            return False
        session_dir = available[-1]
        test_session_id = os.path.basename(session_dir)

    pipeline = ObjectSegmentationPipeline(session_dir=session_dir)
    summary = pipeline.run()

    # Validate Export Files
    if not os.path.isfile(summary.output_ply_path) or os.path.getsize(summary.output_ply_path) < 1000:
        print("[FAIL] Output object_only.ply missing or empty.")
        return False

    if not os.path.isfile(summary.output_csv_path) or os.path.getsize(summary.output_csv_path) < 100:
        print("[FAIL] Output object_only.csv missing or empty.")
        return False

    if not os.path.isfile(summary.output_json_path) or os.path.getsize(summary.output_json_path) < 100:
        print("[FAIL] Output segmentation_results.json missing or empty.")
        return False

    if not os.path.isfile(summary.output_plot_path) or os.path.getsize(summary.output_plot_path) < 1000:
        print("[FAIL] Output segmentation_visualization.png missing or empty.")
        return False

    print("\n" + "=" * 78)
    print("  FINAL SEGMENTATION VALIDATION REPORT")
    print("=" * 78)
    print(f"  Session ID           : {summary.session_id}")
    print(f"  Input Scene Points   : {summary.total_input_points:,}")
    print(f"  Plane Inliers        : {summary.support_plane['inlier_count']:,} ({summary.support_plane['inlier_pct']:.1f}%)")
    print(f"  Total Clusters       : {summary.total_clusters_detected}")
    print(f"  Selected Object Pts  : {summary.selected_filtered_points:,} (Retained: {summary.retained_ratio_pct:.1f}%)")
    print(f"  Object Extent (cm)   : {summary.final_bbox_extent_cm[0]:.1f} cm x {summary.final_bbox_extent_cm[1]:.1f} cm x {summary.final_bbox_extent_cm[2]:.1f} cm")
    print(f"  Object Centroid      : [{summary.final_centroid_m[0]:+.3f}, {summary.final_centroid_m[1]:+.3f}, {summary.final_centroid_m[2]:+.3f}] m")
    print(f"  Output PLY           : {summary.output_ply_path}")
    print(f"  Output CSV           : {summary.output_csv_path}")
    print(f"  Output JSON          : {summary.output_json_path}")
    print(f"  Output Plot          : {summary.output_plot_path}")
    print("=" * 78)
    print("  MODULE 4 SEGMENTATION SELF-TEST COMPLETED SUCCESSFULLY [PASS]")
    print("=" * 78 + "\n")
    return True


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 4: 3D Target Object Segmentation from Registered Point Cloud"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="session_003",
        help="Session folder name under datasets/ (e.g., session_001, session_002, session_003)"
    )
    parser.add_argument(
        "--plane-dist",
        type=float,
        default=DEFAULT_PLANE_DIST_THRESH_M,
        help=f"RANSAC plane distance threshold in meters (default: {DEFAULT_PLANE_DIST_THRESH_M} m = 8mm)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_DBSCAN_EPS_M,
        help=f"DBSCAN clustering search radius in meters (default: {DEFAULT_DBSCAN_EPS_M} m = 15mm)"
    )
    parser.add_argument(
        "--min-pts",
        type=int,
        default=DEFAULT_DBSCAN_MIN_PTS,
        help=f"DBSCAN minimum points per cluster (default: {DEFAULT_DBSCAN_MIN_PTS})"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute automated diagnostic self-test (Synthetic ground truth + Real registered session)."
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_module4_self_test(test_session_id=args.session)
        sys.exit(0 if success else 1)
    else:
        target_session_dir = os.path.join(DEFAULT_DATASET_ROOT, args.session)
        if not os.path.isdir(target_session_dir):
            print(f"[ERROR] Session directory does not exist: {target_session_dir}")
            sys.exit(1)

        pipeline = ObjectSegmentationPipeline(
            session_dir=target_session_dir,
            plane_dist_thresh_m=args.plane_dist,
            dbscan_eps_m=args.eps,
            dbscan_min_pts=args.min_pts
        )
        pipeline.run()
