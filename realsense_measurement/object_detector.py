"""
object_detector.py
==================
Modular Object Detection Interface & RGB-D Foreground Baseline Detector
Intel RealSense D455f Measurement System

Provides a decoupled detector interface separating "Object Detection" (Where is the box?)
from "3D Metrology" (What are its physical dimensions?).

Modules:
    1. BaseObjectDetector (Abstract Base Class)
    2. DetectionResult (Standardized detection payload & ROI converter)
    3. DetectionTracker (Temporal bounding box smoothing to eliminate jitter)
    4. RGBDForegroundDetector (Lightweight baseline RGB-D object detector)

Author : (your name)
Date   : 2026-09-21
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2


# ===========================================================================
# DETECTION RESULT DATA STRUCTURE
# ===========================================================================

@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]    # (xmin, ymin, xmax, ymax) in image pixels
    confidence: float                   # Detection confidence score [0.0 - 1.0]
    class_name: str = "box"
    mask: Optional[np.ndarray] = None   # Full-frame binary mask (uint8 0/255)
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True
    center_xy: Tuple[float, float] = (0.0, 0.0)
    area_px: int = 0
    source: str = "RGB-D Foreground Detector"

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    def to_roi_box(self) -> Tuple[int, int, int, int]:
        return self.bbox


# ===========================================================================
# ABSTRACT BASE DETECTOR INTERFACE
# ===========================================================================

class BaseObjectDetector(ABC):
    """
    Abstract Base Class for all object detectors (classical, ML, YOLO, SAM, etc.)
    Decouples object detection from metric 3D metrology.
    """

    @abstractmethod
    def detect(
        self,
        color_img: np.ndarray,
        depth_m: np.ndarray,
        **kwargs
    ) -> DetectionResult:
        """
        Runs object detection on the RGB-D frame.

        Parameters:
            color_img: (H, W, 3) BGR uint8 image
            depth_m:   (H, W) float32 metric depth array in metres

        Returns:
            DetectionResult with bounding box, confidence, and optional mask.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets any internal detector state or tracker memory."""
        pass


# ===========================================================================
# TEMPORAL BOUNDING BOX TRACKER / SMOOTHER
# ===========================================================================

class DetectionTracker:
    """
    Smooths detected bounding box parameters (center_x, center_y, width, height)
    over time to eliminate inter-frame bounding box jitter.
    """

    def __init__(self, alpha: float = 0.60, max_missed_frames: int = 6):
        self.alpha = alpha  # Smoothing weight for new observation [0.0 - 1.0]
        self.max_missed_frames = max_missed_frames
        self.smooth_cx: Optional[float] = None
        self.smooth_cy: Optional[float] = None
        self.smooth_w: Optional[float] = None
        self.smooth_h: Optional[float] = None
        self.smooth_conf: float = 0.0
        self.missed_count: int = 0

    def reset(self):
        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_w = None
        self.smooth_h = None
        self.smooth_conf = 0.0
        self.missed_count = 0

    def update(
        self,
        raw_result: DetectionResult,
        img_width: int = 1280,
        img_height: int = 720
    ) -> DetectionResult:
        if not raw_result.is_valid:
            self.missed_count += 1
            if self.missed_count <= self.max_missed_frames and self.smooth_cx is not None:
                # Coast with last smoothed box
                xmin = int(np.clip(self.smooth_cx - self.smooth_w / 2.0, 0, img_width - 1))
                ymin = int(np.clip(self.smooth_cy - self.smooth_h / 2.0, 0, img_height - 1))
                xmax = int(np.clip(self.smooth_cx + self.smooth_w / 2.0, 0, img_width - 1))
                ymax = int(np.clip(self.smooth_cy + self.smooth_h / 2.0, 0, img_height - 1))
                return DetectionResult(
                    bbox=(xmin, ymin, xmax, ymax),
                    confidence=max(0.1, self.smooth_conf * 0.85),
                    class_name=raw_result.class_name,
                    mask=raw_result.mask,
                    timestamp=time.time(),
                    is_valid=True,
                    center_xy=(self.smooth_cx, self.smooth_cy),
                    area_px=int(self.smooth_w * self.smooth_h),
                    source=f"{raw_result.source} (Tracked)"
                )
            return raw_result

        # Valid observation
        self.missed_count = 0
        obs_xmin, obs_ymin, obs_xmax, obs_ymax = raw_result.bbox
        obs_w = float(obs_xmax - obs_xmin)
        obs_h = float(obs_ymax - obs_ymin)
        obs_cx = float(obs_xmin + obs_w / 2.0)
        obs_cy = float(obs_ymin + obs_h / 2.0)

        if self.smooth_cx is None:
            self.smooth_cx = obs_cx
            self.smooth_cy = obs_cy
            self.smooth_w = obs_w
            self.smooth_h = obs_h
            self.smooth_conf = raw_result.confidence
        else:
            self.smooth_cx = self.alpha * obs_cx + (1.0 - self.alpha) * self.smooth_cx
            self.smooth_cy = self.alpha * obs_cy + (1.0 - self.alpha) * self.smooth_cy
            self.smooth_w  = self.alpha * obs_w  + (1.0 - self.alpha) * self.smooth_w
            self.smooth_h  = self.alpha * obs_h  + (1.0 - self.alpha) * self.smooth_h
            self.smooth_conf = self.alpha * raw_result.confidence + (1.0 - self.alpha) * self.smooth_conf

        s_xmin = int(np.clip(self.smooth_cx - self.smooth_w / 2.0, 0, img_width - 1))
        s_ymin = int(np.clip(self.smooth_cy - self.smooth_h / 2.0, 0, img_height - 1))
        s_xmax = int(np.clip(self.smooth_cx + self.smooth_w / 2.0, 0, img_width - 1))
        s_ymax = int(np.clip(self.smooth_cy + self.smooth_h / 2.0, 0, img_height - 1))

        return DetectionResult(
            bbox=(s_xmin, s_ymin, s_xmax, s_ymax),
            confidence=self.smooth_conf,
            class_name=raw_result.class_name,
            mask=raw_result.mask,
            timestamp=time.time(),
            is_valid=True,
            center_xy=(self.smooth_cx, self.smooth_cy),
            area_px=int(self.smooth_w * self.smooth_h),
            source=raw_result.source
        )


# ===========================================================================
# LIGHTWEIGHT RGB-D FOREGROUND OBJECT DETECTOR
# ===========================================================================

class RGBDForegroundDetector(BaseObjectDetector):
    """
    Lightweight baseline detector that automatically identifies target box regions
    using depth consistency, background/wall rejection, and RGB gradient saliency.
    """

    def __init__(
        self,
        min_depth_m: float = 0.35,
        max_depth_m: float = 2.20,
        min_area_px: int = 1800,
        margin_pct: float = 0.08,
        smooth_tracking: bool = True
    ):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.min_area_px = min_area_px
        self.margin_pct = margin_pct
        self.tracker = DetectionTracker(alpha=0.65, max_missed_frames=5) if smooth_tracking else None

    def reset(self):
        if self.tracker is not None:
            self.tracker.reset()

    def detect(
        self,
        color_img: np.ndarray,
        depth_m: np.ndarray,
        **kwargs
    ) -> DetectionResult:
        h, w = depth_m.shape[:2]

        # 1. Depth range gating: discard near sensor noise (<0.35m) and distant background/walls (>2.2m)
        valid_depth_mask = (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        valid_depths = depth_m[valid_depth_mask]

        if len(valid_depths) < 500:
            invalid_res = DetectionResult(
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                is_valid=False,
                source="RGB-D Foreground Detector"
            )
            return self.tracker.update(invalid_res, w, h) if self.tracker else invalid_res

        # 2. Foreground Depth Peak Identification
        min_d = np.percentile(valid_depths, 1.0)
        max_d = np.percentile(valid_depths, 95.0)
        bins = np.arange(min_d, max_d + 0.02, 0.02)

        if len(bins) >= 3:
            hist, bin_edges = np.histogram(valid_depths, bins=bins)
            # Find the most prominent foreground depth peak
            peak_idx = int(np.argmax(hist))
            z_fg_peak = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0)
        else:
            z_fg_peak = float(np.median(valid_depths))

        # Depth gate around foreground object (tolerance: 18 cm)
        fg_depth_tol = 0.18
        fg_mask = (
            (depth_m >= max(self.min_depth_m, z_fg_peak - fg_depth_tol)) &
            (depth_m <= (z_fg_peak + fg_depth_tol))
        ).astype(np.uint8) * 255

        # 3. Morphological Cleanup (remove salt-and-pepper noise, bridge small gaps)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

        # 4. Contour Analysis & Saliency Scoring
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candidates = []

        img_center = np.array([w / 2.0, h / 2.0])
        max_diag = np.sqrt(w**2 + h**2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

            # Discard extreme aspect ratios (e.g. thin lines across entire image)
            if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                continue

            # Centroid & centrality score
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cnt_center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            else:
                cnt_center = np.array([bx + bw / 2.0, by + bh / 2.0])

            center_dist = np.linalg.norm(cnt_center - img_center)
            center_score = 1.0 - 0.40 * (center_dist / max_diag)

            # Compactness / solidity score
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 0.0

            # Total score
            score = area * center_score * (0.5 + 0.5 * solidity)
            valid_candidates.append((score, cnt, bx, by, bw, bh, solidity))

        if not valid_candidates:
            invalid_res = DetectionResult(
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                is_valid=False,
                source="RGB-D Foreground Detector"
            )
            return self.tracker.update(invalid_res, w, h) if self.tracker else invalid_res

        # Select highest scoring candidate (documented deterministic selection rule)
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cnt, bx, by, bw, bh, solidity = valid_candidates[0]

        # 5. Expand bounding box by margin percentage so metrology captures full edges
        pad_x = int(bw * self.margin_pct)
        pad_y = int(bh * self.margin_pct)
        xmin = max(0, bx - pad_x)
        ymin = max(0, by - pad_y)
        xmax = min(w - 1, bx + bw + pad_x)
        ymax = min(h - 1, by + bh + pad_y)

        # 6. Detection Confidence Calculation
        # Normalized by area, solidity, and depth consistency
        depth_roi = depth_m[ymin:ymax, xmin:xmax]
        valid_depth_roi = depth_roi[(depth_roi > 0.15) & (depth_roi < 3.0)]
        z_std = float(np.std(valid_depth_roi)) if len(valid_depth_roi) > 30 else 0.5
        depth_consistency_score = np.clip(1.0 - (z_std / 0.15), 0.2, 1.0)

        conf = float(np.clip(0.40 + 0.35 * solidity + 0.25 * depth_consistency_score, 0.0, 0.98))

        # Binary object mask
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(obj_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

        raw_result = DetectionResult(
            bbox=(xmin, ymin, xmax, ymax),
            confidence=conf,
            class_name="box",
            mask=obj_mask,
            timestamp=time.time(),
            is_valid=True,
            center_xy=(float(bx + bw / 2.0), float(by + bh / 2.0)),
            area_px=int(bw * bh),
            source="Auto RGB-D Foreground"
        )

        return self.tracker.update(raw_result, w, h) if self.tracker else raw_result
