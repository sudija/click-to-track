"""
Video source module supporting webcam/CSI and RTSP streams.
Handles frame capture with timestamps and resizing for Pi 5 optimization.

CSI Camera Support:
- Uses picamera2 library for Pi CSI cameras (recommended for Pi 5)
- Falls back to OpenCV with libcamera pipeline if picamera2 unavailable
"""

import cv2
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import config

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Container for a captured frame with metadata."""
    image: any  # numpy array BGR
    gray: any  # numpy array grayscale
    timestamp: float  # capture time
    width: int
    height: int
    frame_number: int


class CaptureBackend(ABC):
    """Abstract base class for capture backends."""
    
    @abstractmethod
    def open(self) -> bool:
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, any]:
        pass
    
    @abstractmethod
    def release(self):
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        pass
    
    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        pass


class OpenCVBackend(CaptureBackend):
    """Standard OpenCV VideoCapture backend."""
    
    def __init__(self, source, target_width: int):
        self.source = source
        self.target_width = target_width
        self.cap = None
        self._size = (0, 0)
    
    def open(self) -> bool:
        if isinstance(self.source, int):
            # Try different backends for local camera
            backends = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "AUTO"),
            ]
            
            for backend, name in backends:
                logger.info(f"Trying OpenCV backend: {name}")
                self.cap = cv2.VideoCapture(self.source, backend)
                if self.cap.isOpened():
                    logger.info(f"Opened with backend: {name}")
                    break
                self.cap.release()
        else:
            # RTSP or file
            if str(self.source).startswith("rtsp://"):
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
        
        # Configure
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.target_width * 3 / 4))
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        
        # Warm up
        time.sleep(0.3)
        for _ in range(5):
            self.cap.read()
        
        # Get actual size
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._size = (frame.shape[1], frame.shape[0])
        else:
            self._size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        
        return True
    
    def read(self) -> Tuple[bool, any]:
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def get_size(self) -> Tuple[int, int]:
        return self._size


class LibcameraBackend(CaptureBackend):
    """OpenCV with libcamera pipeline for CSI cameras."""
    
    def __init__(self, camera_id: int, target_width: int):
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = int(target_width * 3 / 4)
        self.cap = None
        self._size = (target_width, self.target_height)
    
    def open(self) -> bool:
        # Build libcamera pipeline for OpenCV
        # This works on Pi OS with libcamera stack
        pipeline = (
            f"libcamerasrc camera-name=/base/soc/i2c0mux/i2c@1/imx219@10 "
            f"! video/x-raw,width={self.target_width},height={self.target_height},framerate={config.TARGET_FPS}/1 "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=1"
        )
        
        # Simpler pipeline that auto-detects camera
        simple_pipeline = (
            f"libcamerasrc "
            f"! video/x-raw,width={self.target_width},height={self.target_height},framerate={config.TARGET_FPS}/1 "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=1"
        )
        
        logger.info(f"Trying libcamera GStreamer pipeline...")
        self.cap = cv2.VideoCapture(simple_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            logger.warning("Simple libcamera pipeline failed, trying alternative...")
            # Try with explicit camera index
            alt_pipeline = (
                f"libcamerasrc camera-name={self.camera_id} "
                f"! video/x-raw,width={self.target_width},height={self.target_height} "
                f"! videoconvert ! video/x-raw,format=BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(alt_pipeline, cv2.CAP_GSTREAMER)
        
        if self.cap.isOpened():
            time.sleep(0.5)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._size = (frame.shape[1], frame.shape[0])
            return True
        
        return False
    
    def read(self) -> Tuple[bool, any]:
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def get_size(self) -> Tuple[int, int]:
        return self._size


class Picamera2Backend(CaptureBackend):
    """Native picamera2 backend for Pi CSI cameras (recommended)."""
    
    def __init__(self, camera_id: int, target_width: int):
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = int(target_width * 3 / 4)
        self.picam2 = None
        self._size = (target_width, self.target_height)
    
    def open(self) -> bool:
        try:
            from picamera2 import Picamera2
            
            logger.info(f"Initializing Picamera2 for camera {self.camera_id}...")
            self.picam2 = Picamera2(camera_num=self.camera_id)
            
            # Configure for video capture — BGR888 so OpenCV gets it natively
            video_config = self.picam2.create_video_configuration(
                main={"size": (self.target_width, self.target_height), "format": "BGR888"},
                buffer_count=2
            )
            self.picam2.configure(video_config)
            self.picam2.start()
            
            # Apply camera controls for exposure/brightness
            self._apply_camera_controls()
            
            # Warm up
            time.sleep(0.5)
            
            # Get actual size from a test frame
            frame = self.picam2.capture_array("main")
            if frame is not None:
                self._size = (frame.shape[1], frame.shape[0])
                print(f"Picamera2 frame shape: {frame.shape}  dtype: {frame.dtype}")
            
            logger.info(f"Picamera2 initialized: {self._size}")
            return True
            
        except ImportError:
            logger.warning("picamera2 not installed")
            return False
        except Exception as e:
            logger.warning(f"Picamera2 initialization failed: {e}")
            return False
    
    def _apply_camera_controls(self):
        """Apply exposure, gain, and brightness controls from config."""
        if self.picam2 is None:
            return
        
        try:
            from libcamera import controls
            
            ctrl = {}
            
            # Autofocus (if camera supports it)
            if getattr(config, 'CAMERA_AUTOFOCUS', True):
                try:
                    ctrl["AfMode"] = controls.AfModeEnum.Continuous
                    logger.info("Camera: Continuous autofocus enabled")
                except:
                    try:
                        ctrl["AfMode"] = 2  # Continuous = 2
                        logger.info("Camera: Continuous autofocus enabled (mode 2)")
                    except:
                        logger.warning("Camera: Autofocus not supported")
            
            # Exposure mode
            if getattr(config, 'CAMERA_AUTO_EXPOSURE', True):
                ctrl["AeEnable"] = True
                logger.info("Camera: Auto exposure enabled")
            else:
                ctrl["AeEnable"] = False
                exposure = getattr(config, 'CAMERA_EXPOSURE_TIME', 33000)
                ctrl["ExposureTime"] = exposure
                logger.info(f"Camera: Manual exposure = {exposure}µs")
            
            # Analog gain (higher = more sensitive to light)
            gain = getattr(config, 'CAMERA_ANALOG_GAIN', 1.0)
            ctrl["AnalogueGain"] = gain
            logger.info(f"Camera: Analog gain = {gain}")
            
            # Brightness
            brightness = getattr(config, 'CAMERA_BRIGHTNESS', 0.0)
            ctrl["Brightness"] = brightness
            logger.info(f"Camera: Brightness = {brightness}")
            
            # Contrast
            contrast = getattr(config, 'CAMERA_CONTRAST', 1.0)
            ctrl["Contrast"] = contrast
            logger.info(f"Camera: Contrast = {contrast}")
            
            # Apply all controls
            self.picam2.set_controls(ctrl)
            
        except Exception as e:
            logger.warning(f"Could not apply camera controls: {e}")
    
    def read(self) -> Tuple[bool, any]:
        if self.picam2 is None:
            return False, None
        try:
            frame = self.picam2.capture_array("main")
            if frame is None:
                return False, None
            # Ensure exactly 3 channels — drop alpha if present
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return True, frame
        except Exception as e:
            logger.debug(f"Picamera2 capture error: {e}")
            return False, None
    
    def trigger_autofocus(self):
        """Trigger a single autofocus cycle."""
        if self.picam2 is None:
            return
        
        try:
            from libcamera import controls
            
            # Try standard Pi Camera 3 method first
            try:
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
                self.picam2.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
                logger.info("Autofocus triggered (standard mode)")
                return
            except:
                pass
            
            # Try Arducam method - they sometimes use different control names
            try:
                # Arducam 64MP autofocus
                self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
                logger.info("Autofocus triggered (Arducam mode 1)")
                return
            except:
                pass
            
            # Try setting lens position to trigger focus search
            try:
                # This triggers a focus sweep on some cameras
                self.picam2.set_controls({"AfMode": 2})  # Continuous
                logger.info("Autofocus set to continuous mode")
                return
            except:
                pass
            
            # Last resort - try LensPosition control (manual focus)
            try:
                # Get current metadata to find focus range
                metadata = self.picam2.capture_metadata()
                logger.info(f"Camera metadata: {metadata}")
                logger.warning("Auto autofocus not available - camera may need manual focus")
            except Exception as e:
                logger.warning(f"Could not get camera metadata: {e}")
                
        except Exception as e:
            logger.warning(f"Autofocus trigger failed: {e}")
    
    def set_manual_focus(self, position: float):
        """
        Set manual focus position.
        
        Args:
            position: Focus position (0.0 = infinity, larger = closer)
                     Typical range is 0.0 to 10.0 or higher
        """
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
        """
        Set camera exposure and gain settings.
        
        Args:
            auto_exposure: Enable auto exposure
            exposure_time: Exposure time in microseconds (if auto_exposure=False)
            analog_gain: Analog gain (1.0 - 16.0)
            brightness: Brightness adjustment (-1.0 to 1.0)
        """
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
    
    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None
    
    def is_opened(self) -> bool:
        return self.picam2 is not None
    
    def get_size(self) -> Tuple[int, int]:
        return self._size


class VideoSource:
    """
    Video capture wrapper supporting local cameras and RTSP streams.
    Handles resizing for performance optimization on Pi 5.
    
    For CSI cameras, tries backends in order:
    1. picamera2 (native, recommended)
    2. libcamera via GStreamer
    3. OpenCV V4L2
    """

    def __init__(self, source=None, target_width: int = None):
        """
        Initialize video source.
        
        Args:
            source: Camera index (int) or RTSP URL (str). None uses config default.
                   For CSI cameras, use 0 or 1 for the camera port.
            target_width: Target width for resizing. None uses config default.
        """
        self.source = source if source is not None else config.VIDEO_SOURCE
        self.target_width = target_width or config.TARGET_WIDTH
        self.backend: Optional[CaptureBackend] = None
        self.frame_number = 0
        self.scale_factor = 1.0
        self.original_size = (0, 0)
        self.scaled_size = (0, 0)
        self._consecutive_failures = 0

    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if successfully opened, False otherwise.
        """
        logger.info(f"Opening video source: {self.source}")
        
        if isinstance(self.source, int):
            # Camera index - try CSI backends first, then USB
            backends_to_try = [
                ("Picamera2", lambda: Picamera2Backend(self.source, self.target_width)),
                ("Libcamera", lambda: LibcameraBackend(self.source, self.target_width)),
                ("OpenCV", lambda: OpenCVBackend(self.source, self.target_width)),
            ]
            
            for name, create_backend in backends_to_try:
                logger.info(f"Trying {name} backend...")
                try:
                    self.backend = create_backend()
                    if self.backend.open():
                        logger.info(f"Successfully opened with {name}")
                        break
                    else:
                        self.backend = None
                except Exception as e:
                    logger.warning(f"{name} backend failed: {e}")
                    self.backend = None
        else:
            # RTSP or file - use OpenCV
            self.backend = OpenCVBackend(self.source, self.target_width)
            if not self.backend.open():
                logger.error(f"Failed to open: {self.source}")
                return False

        if self.backend is None or not self.backend.is_opened():
            logger.error("All backends failed to open video source")
            logger.error("Troubleshooting for CSI camera:")
            logger.error("  1. Check camera is properly connected to CSI port")
            logger.error("  2. Enable camera: sudo raspi-config -> Interface -> Camera")
            logger.error("  3. Install picamera2: sudo apt install python3-picamera2")
            logger.error("  4. Test camera: libcamera-hello")
            return False
        
        # Get dimensions
        self.original_size = self.backend.get_size()
        self.scaled_size = self.original_size  # Backend handles scaling
        self.scale_factor = 1.0  # Already at target size
        
        logger.info(f"Video source opened: {self.original_size}")
        return True

    def read(self) -> Optional[Frame]:
        """
        Read and process the next frame.
        
        Returns:
            Frame object with image data, or None if read failed.
        """
        if self.backend is None or not self.backend.is_opened():
            return None

        # Try to read
        ret, frame = self.backend.read()
        
        if not ret or frame is None:
            self._consecutive_failures += 1
            
            if self._consecutive_failures <= 3:
                return None
            elif self._consecutive_failures <= 10:
                logger.warning(f"Failed to read frame (attempt {self._consecutive_failures})")
                return None
            else:
                logger.error("Too many consecutive frame failures")
                return None
        
        # Reset failure counter on success
        self._consecutive_failures = 0
        timestamp = time.time()
        self.frame_number += 1

        # Resize if needed (backend may not match exact target)
        if frame.shape[1] != self.target_width:
            scale = self.target_width / frame.shape[1]
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (self.target_width, new_height), 
                             interpolation=cv2.INTER_LINEAR)
        
        # Update size
        self.scaled_size = (frame.shape[1], frame.shape[0])

        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return Frame(
            image=frame,
            gray=gray,
            timestamp=timestamp,
            width=self.scaled_size[0],
            height=self.scaled_size[1],
            frame_number=self.frame_number
        )

    def release(self):
        """Release the video capture resource."""
        if self.backend is not None:
            self.backend.release()
            self.backend = None
            logger.info("Video source released")

    def get_fps(self) -> float:
        """Get the frame rate of the video source."""
        return config.TARGET_FPS

    def is_opened(self) -> bool:
        """Check if video source is open."""
        return self.backend is not None and self.backend.is_opened()

    def trigger_autofocus(self):
        """Trigger a single autofocus cycle."""
        if self.backend is not None and hasattr(self.backend, 'trigger_autofocus'):
            self.backend.trigger_autofocus()

    def set_manual_focus(self, position: float):
        """Set manual focus position (0=infinity, higher=closer)."""
        if self.backend is not None and hasattr(self.backend, 'set_manual_focus'):
            self.backend.set_manual_focus(position)

    def set_camera_settings(self, auto_exposure: bool = True, exposure_time: int = 33000,
                           analog_gain: float = 1.0, brightness: float = 0.0):
        """Set camera exposure and gain settings."""
        if self.backend is not None and hasattr(self.backend, 'set_camera_settings'):
            self.backend.set_camera_settings(auto_exposure, exposure_time, analog_gain, brightness)

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
