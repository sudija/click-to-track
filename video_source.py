"""
Video source abstraction for different backends.
Supports: Picamera2, OpenCV VideoCapture
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Optional, Union
from abc import ABC, abstractmethod

import config

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Container for frame data."""
    image: np.ndarray       # BGR image
    gray: np.ndarray        # Grayscale image
    timestamp: float        # Timestamp in seconds
    frame_number: int       # Frame counter


class VideoBackend(ABC):
    """Abstract base class for video backends."""
    
    @abstractmethod
    def open(self) -> bool:
        """Open the video source."""
        pass
    
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read a frame (BGR format)."""
        pass
    
    @abstractmethod
    def release(self):
        """Release the video source."""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second."""
        pass
    
    def trigger_autofocus(self):
        """Trigger autofocus if supported."""
        pass
    
    def set_manual_focus(self, position: float):
        """Set manual focus position."""
        pass
    
    def set_camera_settings(self, auto_exposure: bool = True, exposure_time: int = 33000,
                           analog_gain: float = 1.0, brightness: float = 0.0):
        """Set camera exposure and gain settings."""
        pass


class OpenCVBackend(VideoBackend):
    """OpenCV VideoCapture backend."""
    
    def __init__(self, source: Union[int, str], target_width: int = 640):
        self.source = source
        self.target_width = target_width
        self.cap = None
        self.fps = 30.0
    
    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.target_width * 3 / 4))
        
        # Get actual FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
        
        logger.info(f"OpenCV backend opened: {self.source}, FPS: {self.fps}")
        return True
    
    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_fps(self) -> float:
        return self.fps


class Picamera2Backend(VideoBackend):
    """Picamera2 backend for Raspberry Pi cameras."""
    
    def __init__(self, target_width: int = 640):
        self.target_width = target_width
        self.picam2 = None
        self.fps = 30.0
    
    def open(self) -> bool:
        try:
            from picamera2 import Picamera2
            
            self.picam2 = Picamera2()
            
            # Configure for video
            config_cam = self.picam2.create_video_configuration(
                main={"size": (self.target_width, int(self.target_width * 3 / 4)),
                      "format": "RGB888"},
                controls={"FrameRate": config.TARGET_FPS}
            )
            self.picam2.configure(config_cam)
            self.picam2.start()
            
            # Apply camera controls
            self._apply_camera_controls()
            
            self.fps = config.TARGET_FPS
            logger.info(f"Picamera2 backend opened, resolution: {self.target_width}x{int(self.target_width * 3/4)}")
            return True
            
        except ImportError:
            logger.error("Picamera2 not available")
            return False
        except Exception as e:
            logger.error(f"Failed to open Picamera2: {e}")
            return False
    
    def _apply_camera_controls(self):
        """Apply camera control settings from config."""
        if self.picam2 is None:
            return
        
        try:
            ctrl = {}
            
            if config.CAMERA_AUTO_EXPOSURE:
                ctrl["AeEnable"] = True
            else:
                ctrl["AeEnable"] = False
                ctrl["ExposureTime"] = config.CAMERA_EXPOSURE_TIME
            
            ctrl["AnalogueGain"] = config.CAMERA_ANALOG_GAIN
            ctrl["Brightness"] = config.CAMERA_BRIGHTNESS
            ctrl["Contrast"] = config.CAMERA_CONTRAST
            
            self.picam2.set_controls(ctrl)
            
            # Try to enable autofocus
            if config.CAMERA_AUTOFOCUS:
                self.trigger_autofocus()
                
        except Exception as e:
            logger.warning(f"Failed to apply camera controls: {e}")
    
    def trigger_autofocus(self):
        """Trigger autofocus."""
        if self.picam2 is None:
            return
        
        try:
            # Try continuous autofocus first
            self.picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
            logger.info("Autofocus triggered (continuous mode)")
        except Exception:
            try:
                # Try single-shot autofocus
                self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
                logger.info("Autofocus triggered (single-shot mode)")
            except Exception as e:
                logger.debug(f"Autofocus not available: {e}")
    
    def set_manual_focus(self, position: float):
        """Set manual focus position."""
        if self.picam2 is None:
            return
        
        try:
            self.picam2.set_controls({"AfMode": 0})  # Manual mode
            self.picam2.set_controls({"LensPosition": position})
            logger.info(f"Manual focus set to position {position}")
        except Exception as e:
            logger.warning(f"Manual focus failed: {e}")
    
    def set_camera_settings(self, auto_exposure: bool = True, exposure_time: int = 33000,
                           analog_gain: float = 1.0, brightness: float = 0.0):
        """Set camera exposure and gain settings."""
        if self.picam2 is None:
            return
        
        try:
            ctrl = {}
            
            if auto_exposure:
                ctrl["AeEnable"] = True
            else:
                ctrl["AeEnable"] = False
                ctrl["ExposureTime"] = exposure_time
            
            ctrl["AnalogueGain"] = analog_gain
            ctrl["Brightness"] = brightness
            
            self.picam2.set_controls(ctrl)
            logger.info(f"Camera settings: AE={auto_exposure}, gain={analog_gain}, brightness={brightness}")
            
        except Exception as e:
            logger.warning(f"Failed to set camera settings: {e}")
    
    def read(self) -> Optional[np.ndarray]:
        if self.picam2 is None:
            return None
        
        try:
            # Capture frame (RGB format)
            frame = self.picam2.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def release(self):
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
    
    def get_fps(self) -> float:
        return self.fps


class VideoSource:
    """
    High-level video source that auto-selects backend.
    """
    
    def __init__(self, source: Union[int, str] = 0, target_width: int = 640):
        self.source = source
        self.target_width = target_width
        self.backend: Optional[VideoBackend] = None
        self.frame_number = 0
        self.start_time = 0.0
    
    def open(self) -> bool:
        """Open video source with appropriate backend."""
        
        # Try Picamera2 first for Raspberry Pi
        if isinstance(self.source, int) and self.source == 0:
            try:
                self.backend = Picamera2Backend(self.target_width)
                if self.backend.open():
                    self.start_time = time.time()
                    return True
            except Exception:
                pass
        
        # Fall back to OpenCV
        self.backend = OpenCVBackend(self.source, self.target_width)
        if self.backend.open():
            self.start_time = time.time()
            return True
        
        return False
    
    def read(self) -> Optional[Frame]:
        """Read next frame."""
        if self.backend is None:
            return None
        
        image = self.backend.read()
        if image is None:
            return None
        
        # Resize if needed
        h, w = image.shape[:2]
        if w != self.target_width:
            scale = self.target_width / w
            new_h = int(h * scale)
            image = cv2.resize(image, (self.target_width, new_h))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create frame
        timestamp = time.time() - self.start_time
        frame = Frame(
            image=image,
            gray=gray,
            timestamp=timestamp,
            frame_number=self.frame_number
        )
        
        self.frame_number += 1
        return frame
    
    def release(self):
        """Release video source."""
        if self.backend is not None:
            self.backend.release()
            self.backend = None
    
    def get_fps(self) -> float:
        """Get FPS."""
        if self.backend is not None:
            return self.backend.get_fps()
        return 30.0
    
    def trigger_autofocus(self):
        """Trigger autofocus."""
        if self.backend is not None:
            self.backend.trigger_autofocus()
    
    def set_manual_focus(self, position: float):
        """Set manual focus position."""
        if self.backend is not None and hasattr(self.backend, 'set_manual_focus'):
            self.backend.set_manual_focus(position)
    
    def set_camera_settings(self, auto_exposure: bool = True, exposure_time: int = 33000,
                           analog_gain: float = 1.0, brightness: float = 0.0):
        """Set camera exposure and gain settings."""
        if self.backend is not None and hasattr(self.backend, 'set_camera_settings'):
            self.backend.set_camera_settings(auto_exposure, exposure_time, analog_gain, brightness)
