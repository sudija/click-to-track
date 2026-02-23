"""
Object tracker using OpenCV's built-in correlation-based trackers.
Designed for tracking moving objects on static backgrounds (drone use case).

Supports: CSRT (accurate), KCF (fast), MOSSE (fastest)
"""

import cv2
import numpy as np
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)


class TrackerType(Enum):
    """Available tracker types."""
    CSRT = auto()    # Most accurate, handles scale/rotation
    KCF = auto()     # Fast, good balance
    MOSSE = auto()   # Fastest, less accurate


class TrackerState(Enum):
    """Tracker state."""
    IDLE = auto()
    TRACKING = auto()
    LOST = auto()


@dataclass
class BBox:
    """Bounding box representation."""
    cx: float  # Center x
    cy: float  # Center y
    w: float   # Width
    h: float   # Height

    def to_rect(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) format for OpenCV."""
        x = int(self.cx - self.w / 2)
        y = int(self.cy - self.h / 2)
        return (x, y, int(self.w), int(self.h))

    def to_corners(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert to corner points ((x1,y1), (x2,y2))."""
        x1 = int(self.cx - self.w / 2)
        y1 = int(self.cy - self.h / 2)
        x2 = int(self.cx + self.w / 2)
        y2 = int(self.cy + self.h / 2)
        return ((x1, y1), (x2, y2))

    @classmethod
    def from_rect(cls, x: int, y: int, w: int, h: int) -> 'BBox':
        """Create from (x, y, w, h) format."""
        return cls(cx=x + w/2, cy=y + h/2, w=float(w), h=float(h))

    @classmethod
    def from_corners(cls, x1: int, y1: int, x2: int, y2: int) -> 'BBox':
        """Create from corner points."""
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        cx = min(x1, x2) + w / 2
        cy = min(y1, y2) + h / 2
        return cls(cx=cx, cy=cy, w=float(w), h=float(h))

    def clamp(self, frame_w: int, frame_h: int) -> 'BBox':
        """Clamp bbox to frame boundaries."""
        half_w = self.w / 2
        half_h = self.h / 2
        cx = max(half_w, min(frame_w - half_w, self.cx))
        cy = max(half_h, min(frame_h - half_h, self.cy))
        return BBox(cx, cy, self.w, self.h)


@dataclass
class TrackingMetrics:
    """Tracking quality metrics."""
    confidence: float = 0.0
    tracker_type: str = ""


class Tracker:
    """
    Object tracker using OpenCV's correlation-based trackers.
    
    These trackers learn the target appearance and find it in each frame,
    which works well for objects moving across static backgrounds.
    """

    def __init__(self, tracker_type: TrackerType = None):
        """
        Initialize tracker.
        
        Args:
            tracker_type: Type of tracker to use (default from config)
        """
        # Get tracker type from config or parameter
        if tracker_type is None:
            type_name = getattr(config, 'TRACKER_TYPE', 'CSRT').upper()
            tracker_type = TrackerType[type_name]
        
        self.tracker_type = tracker_type
        self.cv_tracker = None
        
        # State
        self.state = TrackerState.IDLE
        self.bbox: Optional[BBox] = None
        self.confidence = 0.0
        
        # Frame info
        self.frame_w = 0
        self.frame_h = 0
        self.last_timestamp = 0.0
        
        # Lost frame counter
        self.lost_frames = 0
        self.max_lost_frames = getattr(config, 'MAX_LOST_FRAMES', 30)
        
        # Re-detection template (for recovery)
        self.template = None
        self.template_bbox = None
        
        # Metrics
        self.last_metrics = TrackingMetrics()
        
        logger.info(f"Tracker initialized with {tracker_type.name}")

    def _create_tracker(self) -> cv2.Tracker:
        """Create OpenCV tracker instance."""
        if self.tracker_type == TrackerType.CSRT:
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == TrackerType.KCF:
            return cv2.TrackerKCF_create()
        elif self.tracker_type == TrackerType.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
        # Default to CSRT
        return cv2.TrackerCSRT_create()

    def initialize(self, frame: np.ndarray, bbox: BBox, timestamp: float = 0.0):
        """
        Initialize tracker with target.
        
        Args:
            frame: BGR or grayscale frame
            bbox: Initial bounding box
            timestamp: Frame timestamp
        """
        # Ensure frame is color (trackers need BGR)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        self.frame_h, self.frame_w = frame.shape[:2]
        self.last_timestamp = timestamp
        
        # Clamp bbox to frame
        self.bbox = bbox.clamp(self.frame_w, self.frame_h)
        
        # Ensure bbox has valid size
        if self.bbox.w < 10 or self.bbox.h < 10:
            logger.warning("Bbox too small, expanding to minimum size")
            self.bbox = BBox(self.bbox.cx, self.bbox.cy, 
                           max(20, self.bbox.w), max(20, self.bbox.h))
        
        # Create new tracker instance
        self.cv_tracker = self._create_tracker()
        
        # Initialize with bbox in (x, y, w, h) format
        rect = self.bbox.to_rect()
        
        # Ensure rect is within frame bounds
        x, y, w, h = rect
        x = max(0, min(x, self.frame_w - w))
        y = max(0, min(y, self.frame_h - h))
        rect = (x, y, w, h)
        
        try:
            success = self.cv_tracker.init(frame, rect)
        except Exception as e:
            logger.error(f"Tracker init exception: {e}")
            success = False
        
        if success:
            self.state = TrackerState.TRACKING
            self.confidence = 1.0
            self.lost_frames = 0
            
            # Store template for potential recovery
            self._store_template(frame)
            
            logger.info(f"Tracker initialized: {rect}")
        else:
            self.state = TrackerState.LOST
            self.confidence = 0.0
            logger.warning(f"Tracker initialization failed for rect: {rect}")
        
        self._update_metrics()

    def update(self, frame: np.ndarray, timestamp: float = 0.0, 
               imu_sample=None) -> Optional[BBox]:
        """
        Update tracker with new frame.
        
        Args:
            frame: BGR or grayscale frame
            timestamp: Frame timestamp
            imu_sample: IMU data (unused, kept for API compatibility)
            
        Returns:
            Updated bounding box, or None if lost
        """
        if self.state == TrackerState.IDLE:
            return None
        
        if self.cv_tracker is None:
            return None
        
        # Ensure frame is color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        self.frame_h, self.frame_w = frame.shape[:2]
        self.last_timestamp = timestamp
        
        # Run tracker
        success, rect = self.cv_tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in rect]
            self.bbox = BBox.from_rect(x, y, w, h)
            self.bbox = self.bbox.clamp(self.frame_w, self.frame_h)
            
            self.state = TrackerState.TRACKING
            self.confidence = min(1.0, self.confidence + 0.1)
            self.lost_frames = 0
            
            # Update template periodically
            if self.confidence > 0.8:
                self._store_template(frame)
        else:
            # Tracking failed
            self.lost_frames += 1
            self.confidence = max(0.0, self.confidence - 0.15)
            
            if self.lost_frames > self.max_lost_frames:
                self.state = TrackerState.LOST
                logger.info("Target lost")
            else:
                # Try to recover using template matching
                recovered = self._try_recover(frame)
                if not recovered:
                    self.state = TrackerState.LOST
        
        self._update_metrics()
        
        if self.state == TrackerState.LOST:
            return None
        
        return self.bbox

    def _store_template(self, frame: np.ndarray):
        """Store current target as template for recovery."""
        if self.bbox is None:
            return
        
        x, y, w, h = self.bbox.to_rect()
        
        # Clamp to frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(self.frame_w, x + w)
        y2 = min(self.frame_h, y + h)
        
        if x2 > x and y2 > y:
            self.template = frame[y:y2, x:x2].copy()
            self.template_bbox = self.bbox

    def _try_recover(self, frame: np.ndarray) -> bool:
        """Try to recover tracking using template matching."""
        if self.template is None or self.bbox is None:
            return False
        
        try:
            # Search in expanded region around last known position
            search_margin = getattr(config, 'RECOVERY_SEARCH_MARGIN', 100)
            
            x, y, w, h = self.bbox.to_rect()
            
            # Define search region
            sx1 = max(0, x - search_margin)
            sy1 = max(0, y - search_margin)
            sx2 = min(self.frame_w, x + w + search_margin)
            sy2 = min(self.frame_h, y + h + search_margin)
            
            search_region = frame[sy1:sy2, sx1:sx2]
            
            if search_region.shape[0] < self.template.shape[0] or \
               search_region.shape[1] < self.template.shape[1]:
                return False
            
            # Template matching
            result = cv2.matchTemplate(search_region, self.template, 
                                       cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # If good match found, reinitialize tracker
            if max_val > getattr(config, 'RECOVERY_THRESHOLD', 0.6):
                new_x = sx1 + max_loc[0]
                new_y = sy1 + max_loc[1]
                new_bbox = BBox.from_rect(new_x, new_y, 
                                          self.template.shape[1],
                                          self.template.shape[0])
                
                # Reinitialize tracker at new position
                self.cv_tracker = self._create_tracker()
                rect = new_bbox.to_rect()
                success = self.cv_tracker.init(frame, rect)
                
                if success:
                    self.bbox = new_bbox
                    self.state = TrackerState.TRACKING
                    self.confidence = max_val
                    logger.info(f"Tracker recovered at {rect}")
                    return True
        
        except Exception as e:
            logger.debug(f"Recovery failed: {e}")
        
        return False

    def _update_metrics(self):
        """Update tracking metrics."""
        self.last_metrics = TrackingMetrics(
            confidence=self.confidence,
            tracker_type=self.tracker_type.name
        )

    def reset(self):
        """Reset tracker to idle state."""
        self.cv_tracker = None
        self.state = TrackerState.IDLE
        self.bbox = None
        self.confidence = 0.0
        self.lost_frames = 0
        self.template = None
        self.last_timestamp = 0.0
        logger.info("Tracker reset")

    def get_state(self) -> TrackerState:
        """Get current tracker state."""
        return self.state

    def get_confidence(self) -> float:
        """Get tracking confidence."""
        return self.confidence

    def get_features(self) -> Optional[np.ndarray]:
        """Get tracked features (not applicable for correlation trackers)."""
        return None

    def get_metrics(self) -> TrackingMetrics:
        """Get tracking metrics."""
        return self.last_metrics
