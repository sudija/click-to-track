"""
Core tracker module implementing:
- KLT optical flow feature tracking (primary)
- Template matching / NCC correlation (secondary verifier)
- Kalman filter prediction (smoothing + coasting)
- Feature refresh and state machine

Optimized for Raspberry Pi 5 with ROI-based processing.
"""

import cv2
import numpy as np
import logging
from enum import Enum, auto
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

import config
from imu import IMUProvider, IMUSample, PixelShift

logger = logging.getLogger(__name__)


class TrackerState(Enum):
    """Tracker state machine states."""
    IDLE = auto()      # No target selected
    TRACKING = auto()  # Actively tracking with good confidence
    COASTING = auto()  # Lost measurements, predicting position
    LOST = auto()      # Target lost, waiting for reselection


@dataclass
class BBox:
    """Bounding box in center form."""
    cx: float  # Center X
    cy: float  # Center Y
    w: float   # Width
    h: float   # Height

    def to_rect(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) rectangle format."""
        x = int(self.cx - self.w / 2)
        y = int(self.cy - self.h / 2)
        return (x, y, int(self.w), int(self.h))

    def to_corners(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert to corner points ((x1, y1), (x2, y2))."""
        x1 = int(self.cx - self.w / 2)
        y1 = int(self.cy - self.h / 2)
        x2 = int(self.cx + self.w / 2)
        y2 = int(self.cy + self.h / 2)
        return ((x1, y1), (x2, y2))

    @classmethod
    def from_rect(cls, x: int, y: int, w: int, h: int) -> 'BBox':
        """Create from (x, y, w, h) rectangle."""
        return cls(x + w / 2, y + h / 2, w, h)

    @classmethod
    def from_corners(cls, x1: int, y1: int, x2: int, y2: int) -> 'BBox':
        """Create from corner points."""
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        cx = min(x1, x2) + w / 2
        cy = min(y1, y2) + h / 2
        return cls(cx, cy, w, h)

    def clamp(self, frame_w: int, frame_h: int) -> 'BBox':
        """Clamp bbox to frame bounds."""
        half_w = self.w / 2
        half_h = self.h / 2
        cx = max(half_w, min(frame_w - half_w, self.cx))
        cy = max(half_h, min(frame_h - half_h, self.cy))
        return BBox(cx, cy, self.w, self.h)

    def is_inside_frame(self, frame_w: int, frame_h: int, 
                        tolerance: float = 0.5) -> bool:
        """Check if bbox is mostly inside frame."""
        corners = self.to_corners()
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        
        # Calculate overlap
        visible_x1 = max(0, x1)
        visible_y1 = max(0, y1)
        visible_x2 = min(frame_w, x2)
        visible_y2 = min(frame_h, y2)
        
        if visible_x2 <= visible_x1 or visible_y2 <= visible_y1:
            return False
            
        visible_area = (visible_x2 - visible_x1) * (visible_y2 - visible_y1)
        total_area = self.w * self.h
        
        return (visible_area / total_area) >= tolerance


@dataclass
class TrackingMetrics:
    """Metrics from current frame tracking."""
    klt_inliers: int = 0
    klt_total: int = 0
    klt_quality: float = 0.0
    ncc_score: float = 0.0
    ncc_position: Tuple[float, float] = (0.0, 0.0)
    measurement_valid: bool = False


class KalmanTracker:
    """
    Kalman filter for position and velocity estimation.
    State: [cx, cy, vx, vy]
    Measurement: [cx, cy]
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state dims, 2 measurement dims
        self._initialized = False

    def initialize(self, cx: float, cy: float):
        """Initialize Kalman filter at position."""
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # cx' = cx + vx*dt (dt=1 initially)
            [0, 1, 0, 1],  # cy' = cy + vy*dt
            [0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 1]   # vy' = vy
        ], dtype=np.float32)

        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.KALMAN_PROCESS_NOISE

        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.KALMAN_MEASUREMENT_NOISE

        # Initial error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * config.KALMAN_INITIAL_ERROR

        # Initial state
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        
        self._initialized = True

    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        """
        Predict next position.
        
        Args:
            dt: Time delta since last update
            
        Returns:
            Predicted (cx, cy)
        """
        if not self._initialized:
            return (0.0, 0.0)
            
        # Update transition matrix with dt
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        
        prediction = self.kf.predict()
        return (float(prediction[0]), float(prediction[1]))

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        """
        Update with measurement.
        
        Args:
            cx, cy: Measured center position
            
        Returns:
            Filtered (cx, cy)
        """
        if not self._initialized:
            return (cx, cy)
            
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        return (float(corrected[0]), float(corrected[1]))

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        if not self._initialized:
            return (0.0, 0.0)
        return (float(self.kf.statePost[2]), float(self.kf.statePost[3]))

    def add_imu_compensation(self, dx: float, dy: float):
        """
        Add IMU-based pixel shift to state prediction.
        Call after predict() to incorporate gyro data.
        """
        if not self._initialized:
            return
        self.kf.statePost[0] += dx
        self.kf.statePost[1] += dy


class Tracker:
    """
    Main tracker class implementing KLT + NCC + Kalman fusion.
    """

    def __init__(self, imu_provider: Optional[IMUProvider] = None):
        """
        Initialize tracker.
        
        Args:
            imu_provider: Optional IMU provider for gyro compensation
        """
        self.imu = imu_provider
        
        # State
        self.state = TrackerState.IDLE
        self.bbox: Optional[BBox] = None
        self.confidence = 0.0
        
        # Kalman filter
        self.kalman = KalmanTracker()
        
        # Template
        self.template: Optional[np.ndarray] = None
        self.template_size = config.TEMPLATE_SIZE
        
        # Feature points
        self.features: Optional[np.ndarray] = None  # Nx1x2 float32
        
        # Previous frame data
        self.last_gray: Optional[np.ndarray] = None
        self.last_timestamp: float = 0.0
        
        # Counters
        self.weak_frames = 0
        self.coast_frames = 0
        self.out_of_frame_frames = 0
        
        # Frame dimensions
        self.frame_w = 0
        self.frame_h = 0
        
        # Feature refresh tracking
        self._features_just_refreshed = False
        
        # LK optical flow parameters
        self.lk_params = dict(
            winSize=config.LK_WIN_SIZE,
            maxLevel=config.LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     config.LK_CRITERIA[1], config.LK_CRITERIA[2])
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=config.MAX_FEATURES,
            qualityLevel=config.FEATURE_QUALITY_LEVEL,
            minDistance=config.FEATURE_MIN_DISTANCE,
            blockSize=config.FEATURE_BLOCK_SIZE
        )

        # Metrics from last update
        self.last_metrics = TrackingMetrics()

    def initialize(self, frame_gray: np.ndarray, bbox: BBox, timestamp: float):
        """
        Initialize tracking on a target.
        
        Args:
            frame_gray: Grayscale frame
            bbox: Initial bounding box
            timestamp: Frame timestamp
        """
        self.frame_h, self.frame_w = frame_gray.shape[:2]
        self.bbox = bbox.clamp(self.frame_w, self.frame_h)
        self.last_gray = frame_gray.copy()
        self.last_timestamp = timestamp
        
        # Initialize Kalman
        self.kalman.initialize(self.bbox.cx, self.bbox.cy)
        
        # Extract template
        self._update_template(frame_gray, force=True)
        
        # Detect initial features
        self._refresh_features(frame_gray)
        
        # Reset counters
        self.confidence = 1.0
        self.weak_frames = 0
        self.coast_frames = 0
        self.out_of_frame_frames = 0
        self._features_just_refreshed = True  # Skip first KLT since features are fresh
        
        # Set state
        self._set_state(TrackerState.TRACKING)
        
        logger.info(f"Tracker initialized: bbox={bbox.to_rect()}, "
                   f"features={len(self.features) if self.features is not None else 0}")

    def update(self, frame_gray: np.ndarray, timestamp: float,
               imu_sample: Optional[IMUSample] = None) -> Optional[BBox]:
        """
        Update tracker with new frame.
        
        Args:
            frame_gray: Grayscale frame
            timestamp: Frame timestamp
            imu_sample: Optional IMU sample for compensation
            
        Returns:
            Updated bounding box, or None if lost
        """
        if self.state == TrackerState.IDLE or self.state == TrackerState.LOST:
            return None
            
        if self.bbox is None or self.last_gray is None:
            return None

        self.frame_h, self.frame_w = frame_gray.shape[:2]
        dt = max(0.001, timestamp - self.last_timestamp)  # Avoid div by zero
        
        # === Step 1: Kalman Predict + IMU Compensation ===
        pred_cx, pred_cy = self.kalman.predict(dt)
        
        # Get IMU-based motion estimate for fast motion compensation
        imu_shift = (0.0, 0.0)
        if imu_sample is not None and self.imu is not None:
            pixel_shift = self.imu.compute_pixel_shift(imu_sample)
            if pixel_shift.valid:
                imu_shift = (pixel_shift.dx, pixel_shift.dy)
                self.kalman.add_imu_compensation(pixel_shift.dx, pixel_shift.dy)
                pred_cx += pixel_shift.dx
                pred_cy += pixel_shift.dy
                print(f"[IMU] Pixel shift: dx={pixel_shift.dx:.1f}, dy={pixel_shift.dy:.1f}")

        # Create predicted bbox
        pred_bbox = BBox(pred_cx, pred_cy, self.bbox.w, self.bbox.h)
        
        # === Step 2: Define Search ROI ===
        roi, roi_offset = self._get_search_roi(pred_bbox)
        
        # === Step 3: KLT Optical Flow Tracking ===
        # Pass IMU shift to help with fast motion
        klt_result = self._track_klt(frame_gray, roi_offset, imu_shift)
        
        # === Step 4: NCC Template Matching ===
        ncc_result = self._track_ncc(frame_gray, pred_bbox)
        
        # === Step 5: Fuse Measurements ===
        meas_cx, meas_cy, meas_valid = self._fuse_measurements(
            pred_bbox, klt_result, ncc_result
        )
        
        print(f"[UPDATE] Fused measurement: ({meas_cx:.1f}, {meas_cy:.1f}), valid={meas_valid}")
        
        # Store metrics
        self.last_metrics = TrackingMetrics(
            klt_inliers=klt_result[0],
            klt_total=klt_result[1],
            klt_quality=klt_result[2],
            ncc_score=ncc_result[0],
            ncc_position=ncc_result[1],
            measurement_valid=meas_valid
        )
        
        # === Step 6: Kalman Update and Bbox Update ===
        old_cx, old_cy = self.bbox.cx, self.bbox.cy
        if meas_valid:
            filtered_cx, filtered_cy = self.kalman.update(meas_cx, meas_cy)
            self.bbox = BBox(filtered_cx, filtered_cy, self.bbox.w, self.bbox.h)
        else:
            # Use prediction only
            self.bbox = pred_bbox
        
        print(f"[UPDATE] Bbox before clamp: ({self.bbox.cx:.1f}, {self.bbox.cy:.1f})")
        
        # Clamp to frame
        self.bbox = self.bbox.clamp(self.frame_w, self.frame_h)
        
        print(f"[UPDATE] Bbox after clamp: ({self.bbox.cx:.1f}, {self.bbox.cy:.1f}), frame=({self.frame_w}, {self.frame_h})")
        
        # === Step 7: State Machine Update ===
        self._update_state_machine(meas_valid, klt_result[2], ncc_result[0])
        
        # === Step 8: Out-of-frame Check ===
        self._check_out_of_frame()
        
        # === Step 9: Template Update (slow) ===
        if self.state == TrackerState.TRACKING and self.confidence > 0.7:
            self._update_template(frame_gray, force=False)
        
        # === Step 10: Feature Refresh ===
        # Only refresh if features are critically low
        # IMPORTANT: After refresh, we must NOT use these features until next frame
        # because they were detected on current frame, not last_gray
        current_feature_count = len(self.features) if self.features is not None else 0
        needs_refresh = current_feature_count < config.MIN_FEATURES
        
        if needs_refresh:
            logger.debug(f"Feature count critically low ({current_feature_count}), refreshing...")
            self._refresh_features(frame_gray)
            # Mark that we need to skip KLT on next frame since features are fresh
            self._features_just_refreshed = True
        else:
            self._features_just_refreshed = False
        
        # === Step 11: Store frame for next iteration ===
        self.last_gray = frame_gray.copy()
        self.last_timestamp = timestamp
        
        if self.state == TrackerState.LOST:
            return None
            
        return self.bbox

    def _get_search_roi(self, pred_bbox: BBox) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
        """
        Calculate search ROI around predicted position.
        
        Returns:
            (roi as (x, y, w, h), offset as (x, y))
        """
        margin = config.ROI_EXPAND_MARGIN
        if self.state == TrackerState.COASTING:
            margin = int(margin * config.ROI_EXPAND_COASTING_FACTOR)
        
        x1 = int(max(0, pred_bbox.cx - pred_bbox.w/2 - margin))
        y1 = int(max(0, pred_bbox.cy - pred_bbox.h/2 - margin))
        x2 = int(min(self.frame_w, pred_bbox.cx + pred_bbox.w/2 + margin))
        y2 = int(min(self.frame_h, pred_bbox.cy + pred_bbox.h/2 + margin))
        
        return ((x1, y1, x2-x1, y2-y1), (x1, y1))

    def _track_klt(self, frame_gray: np.ndarray, 
                   roi_offset: Tuple[int, int],
                   imu_shift: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, int, float, float, float]:
        """
        Track using KLT optical flow.
        
        Args:
            frame_gray: Current grayscale frame
            roi_offset: ROI offset (unused currently)
            imu_shift: (dx, dy) predicted shift from IMU for fast motion compensation
        
        Returns:
            (inliers, total, quality, dx, dy) where dx,dy is displacement from current bbox
        """
        # Skip if features were just refreshed - they're from current frame, 
        # so there's nothing to track yet
        if getattr(self, '_features_just_refreshed', False):
            print(f"[KLT] Skipping - features just refreshed")
            return (0, 0, 0.0, 0.0, 0.0)
        
        if self.features is None or len(self.features) == 0:
            return (0, 0, 0.0, 0.0, 0.0)
            
        if self.last_gray is None:
            return (0, 0, 0.0, 0.0, 0.0)

        # Store original features for displacement calculation
        p0 = self.features.copy()
        
        # If IMU provides motion estimate, use it as initial guess for optical flow
        # This helps when motion is fast and features would be outside search window
        imu_dx, imu_dy = imu_shift
        if abs(imu_dx) > 0.5 or abs(imu_dy) > 0.5:
            # Create initial guess by shifting features by IMU prediction
            p1_init = p0.copy()
            p1_init[:, 0, 0] += imu_dx
            p1_init[:, 0, 1] += imu_dy
            print(f"[KLT] Using IMU prediction: dx={imu_dx:.1f}, dy={imu_dy:.1f}")
        else:
            p1_init = None
        
        print(f"[KLT] Tracking {len(p0)} features")

        # Forward tracking (with optional initial guess from IMU)
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
            self.last_gray, frame_gray, p0, p1_init, **self.lk_params
        )
        
        if p1 is None:
            return (0, 0, 0.0, 0.0, 0.0)
        
        st1 = st1.flatten()
        
        # Forward-backward check for robustness
        if config.FB_CHECK_ENABLED:
            p0_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(
                frame_gray, self.last_gray, p1, None, **self.lk_params
            )
            
            if p0_back is not None:
                st_back = st_back.flatten()
                fb_error = np.linalg.norm(
                    p0.reshape(-1, 2) - p0_back.reshape(-1, 2),
                    axis=1
                )
                fb_good = fb_error < config.FB_ERROR_THRESHOLD
                st1 = st1 & st_back & fb_good

        # Filter good points
        good_mask = st1 == 1
        good_old = p0[good_mask]
        good_new = p1[good_mask]
        
        inliers = len(good_new)
        total = len(p0)
        
        print(f"[KLT] Inliers: {inliers}/{total}")
        
        if inliers < config.MIN_FEATURES:
            self.features = None
            return (inliers, total, 0.0, 0.0, 0.0)
        
        # Compute displacement from point movements
        displacements = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
        dx = float(np.median(displacements[:, 0]))
        dy = float(np.median(displacements[:, 1]))
        
        print(f"[KLT] Median displacement: dx={dx:.2f}, dy={dy:.2f}")
        
        # Quality based on inlier ratio
        quality = inliers / max(1, total)
        
        # Consistency check
        if len(displacements) > 3:
            displacement_std = np.std(displacements, axis=0)
            consistency = 1.0 / (1.0 + np.mean(displacement_std))
            quality *= consistency
        
        # Update features to new positions
        self.features = good_new.reshape(-1, 1, 2).astype(np.float32)
        
        print(f"[KLT] Quality: {quality:.2f}, Features remaining: {len(self.features)}")
        
        return (inliers, total, quality, dx, dy)

    def _track_ncc(self, frame_gray: np.ndarray, 
                   pred_bbox: BBox) -> Tuple[float, Tuple[float, float]]:
        """
        Track using NCC template matching.
        
        Returns:
            (score, (cx, cy))
        """
        if self.template is None:
            return (0.0, (pred_bbox.cx, pred_bbox.cy))
        
        # Define search window
        margin = config.NCC_SEARCH_MARGIN
        if self.state == TrackerState.COASTING:
            margin = int(margin * config.ROI_EXPAND_COASTING_FACTOR)
            
        th, tw = self.template.shape[:2]
        
        # Search region bounds
        x1 = int(max(0, pred_bbox.cx - pred_bbox.w/2 - margin))
        y1 = int(max(0, pred_bbox.cy - pred_bbox.h/2 - margin))
        x2 = int(min(self.frame_w, pred_bbox.cx + pred_bbox.w/2 + margin))
        y2 = int(min(self.frame_h, pred_bbox.cy + pred_bbox.h/2 + margin))
        
        # Ensure search region is large enough for template
        if (x2 - x1) < tw or (y2 - y1) < th:
            return (0.0, (pred_bbox.cx, pred_bbox.cy))
        
        search_region = frame_gray[y1:y2, x1:x2]
        
        # Template matching
        try:
            result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Convert to center coordinates in frame space
            match_x = x1 + max_loc[0] + tw / 2
            match_y = y1 + max_loc[1] + th / 2
            
            return (max_val, (match_x, match_y))
        except cv2.error as e:
            logger.debug(f"NCC matching error: {e}")
            return (0.0, (pred_bbox.cx, pred_bbox.cy))

    def _fuse_measurements(self, pred_bbox: BBox,
                          klt_result: Tuple[int, int, float, float, float],
                          ncc_result: Tuple[float, Tuple[float, float]]
                          ) -> Tuple[float, float, bool]:
        """
        Fuse KLT and NCC measurements.
        
        Returns:
            (cx, cy, valid)
        """
        klt_inliers, klt_total, klt_quality, klt_dx, klt_dy = klt_result
        ncc_score, (ncc_cx, ncc_cy) = ncc_result
        
        # PRIMARY METHOD: Use feature centroid directly
        # This is the most reliable - bbox should be where the features ARE
        if self.features is not None and len(self.features) >= config.MIN_FEATURES:
            centroid_x = float(np.mean(self.features[:, 0, 0]))
            centroid_y = float(np.mean(self.features[:, 0, 1]))
            
            # STABILITY CHECK: Only move if centroid shifted significantly
            # This prevents "dancing" from small tracking noise
            distance_moved = np.sqrt((centroid_x - self.bbox.cx)**2 + 
                                    (centroid_y - self.bbox.cy)**2)
            
            stability_threshold = getattr(config, 'BBOX_STABILITY_THRESHOLD', 3.0)
            
            if distance_moved < stability_threshold:
                # Not enough movement - keep current position
                print(f"[FUSE] Stable (moved {distance_moved:.1f}px < {stability_threshold}px threshold)")
                return (self.bbox.cx, self.bbox.cy, True)
            else:
                print(f"[FUSE] Moving to centroid: ({centroid_x:.1f}, {centroid_y:.1f}), moved {distance_moved:.1f}px")
                return (centroid_x, centroid_y, True)
        
        # FALLBACK: If features are low/missing, try NCC
        if ncc_score >= config.NCC_THRESHOLD_HIGH:
            print(f"[FUSE] Using NCC (features low): ({ncc_cx:.1f}, {ncc_cy:.1f}), score={ncc_score:.2f}")
            return (ncc_cx, ncc_cy, True)
        
        # LAST RESORT: Use Kalman prediction
        print(f"[FUSE] No valid measurement, using prediction: ({pred_bbox.cx:.1f}, {pred_bbox.cy:.1f})")
        return (pred_bbox.cx, pred_bbox.cy, False)

    def _refresh_features(self, frame_gray: np.ndarray):
        """Detect new features strictly inside current bbox."""
        if self.bbox is None:
            return
        
        # Get bbox bounds (no expansion - we want features ON the target only)
        x1 = int(max(0, self.bbox.cx - self.bbox.w/2))
        y1 = int(max(0, self.bbox.cy - self.bbox.h/2))
        x2 = int(min(self.frame_w, self.bbox.cx + self.bbox.w/2))
        y2 = int(min(self.frame_h, self.bbox.cy + self.bbox.h/2))
        
        if x2 <= x1 or y2 <= y1:
            self.features = None
            return
        
        # Extract ROI for feature detection
        roi = frame_gray[y1:y2, x1:x2]
        
        # Detect features in ROI
        features_roi = cv2.goodFeaturesToTrack(
            roi, 
            maxCorners=config.MAX_FEATURES,
            qualityLevel=config.FEATURE_QUALITY_LEVEL,
            minDistance=config.FEATURE_MIN_DISTANCE,
            blockSize=config.FEATURE_BLOCK_SIZE
        )
        
        if features_roi is not None and len(features_roi) > 0:
            # Convert ROI coordinates to frame coordinates
            features_roi = features_roi.reshape(-1, 2)
            features_roi[:, 0] += x1
            features_roi[:, 1] += y1
            self.features = features_roi.reshape(-1, 1, 2).astype(np.float32)
            logger.debug(f"Features refreshed: {len(self.features)} points inside bbox")
        else:
            self.features = None
            logger.debug("No features detected in bbox - target may lack texture")

    def _update_template(self, frame_gray: np.ndarray, force: bool = False):
        """Update template patch."""
        if self.bbox is None:
            return
            
        # Extract ROI
        rect = self.bbox.to_rect()
        x, y, w, h = rect
        
        # Clamp to frame
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_w, x + w)
        y2 = min(self.frame_h, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return
            
        roi = frame_gray[y1:y2, x1:x2]
        
        # Resize to fixed template size
        new_template = cv2.resize(roi, self.template_size, interpolation=cv2.INTER_LINEAR)
        
        if force or self.template is None:
            # Store as uint8 for matchTemplate compatibility
            self.template = new_template.astype(np.uint8)
        else:
            # Slow update - blend and convert back to uint8
            alpha = config.TEMPLATE_UPDATE_ALPHA
            blended = ((1 - alpha) * self.template.astype(np.float32) + 
                      alpha * new_template.astype(np.float32))
            self.template = blended.astype(np.uint8)

    def _update_state_machine(self, meas_valid: bool, 
                             klt_quality: float, ncc_score: float):
        """Update tracker state based on measurements."""
        strong = (klt_quality >= config.KLT_INLIER_RATIO_STRONG or 
                 ncc_score >= config.NCC_THRESHOLD_HIGH)
        weak = (klt_quality < config.KLT_INLIER_RATIO_WEAK and 
               ncc_score < config.NCC_THRESHOLD_LOW)
        
        if self.state == TrackerState.TRACKING:
            if meas_valid and strong:
                # Strong tracking
                self.weak_frames = 0
                self.confidence = min(1.0, self.confidence + config.CONFIDENCE_BOOST)
            elif meas_valid:
                # Acceptable tracking
                self.weak_frames = 0
                # Slight decay
                self.confidence = max(0, self.confidence - config.CONFIDENCE_DECAY_TRACKING / 2)
            else:
                # Weak tracking
                self.weak_frames += 1
                self.confidence = max(0, self.confidence - config.CONFIDENCE_DECAY_TRACKING)
                
                if self.weak_frames >= config.COAST_TRIGGER_FRAMES:
                    self._set_state(TrackerState.COASTING)
                    self.coast_frames = 0
                    
        elif self.state == TrackerState.COASTING:
            self.coast_frames += 1
            self.confidence = max(0, self.confidence - config.CONFIDENCE_DECAY_COASTING)
            
            if meas_valid and strong:
                # Reacquired
                self._set_state(TrackerState.TRACKING)
                self.weak_frames = 0
            elif self.coast_frames >= config.MAX_COAST_FRAMES:
                # Coast timeout
                self._set_state(TrackerState.LOST)
            elif self.confidence < config.CONFIDENCE_LOST_THRESHOLD:
                # Confidence too low
                self._set_state(TrackerState.LOST)

    def _check_out_of_frame(self):
        """Check if target has left the frame."""
        if self.bbox is None:
            return
            
        in_frame = self.bbox.is_inside_frame(self.frame_w, self.frame_h, tolerance=0.3)
        
        if not in_frame:
            self.out_of_frame_frames += 1
            if self.state == TrackerState.TRACKING:
                self._set_state(TrackerState.COASTING)
            if self.out_of_frame_frames >= config.OUT_OF_FRAME_TOLERANCE:
                self._set_state(TrackerState.LOST)
        else:
            self.out_of_frame_frames = 0

    def _set_state(self, new_state: TrackerState):
        """Set tracker state with logging."""
        if new_state != self.state:
            if config.LOG_STATE_TRANSITIONS:
                logger.info(f"State transition: {self.state.name} -> {new_state.name}")
            self.state = new_state

    def reset(self):
        """Reset tracker to idle state."""
        self._set_state(TrackerState.IDLE)
        self.bbox = None
        self.template = None
        self.features = None
        self.confidence = 0.0
        self.weak_frames = 0
        self.coast_frames = 0
        self.out_of_frame_frames = 0
        self.last_gray = None

    def get_state(self) -> TrackerState:
        """Get current tracker state."""
        return self.state

    def get_confidence(self) -> float:
        """Get current confidence value."""
        return self.confidence

    def get_features(self) -> Optional[np.ndarray]:
        """Get current feature points."""
        return self.features

    def get_metrics(self) -> TrackingMetrics:
        """Get metrics from last update."""
        return self.last_metrics
