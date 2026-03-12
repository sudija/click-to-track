#!/usr/bin/env python3
"""
Click-to-Track System for Raspberry Pi 5

A real-time visual tracking system using:
- KLT optical flow (primary tracker)
- NCC template matching (secondary verifier)
- Kalman filter (prediction and smoothing)
- Optional IMU integration (gyro compensation)

Usage:
    python main.py [--source SOURCE] [--width WIDTH] [--imu]

Keys:
    M - Toggle selection mode (click vs drag)
    R - Reset tracker
    Q/ESC - Quit
"""

import sys
import time
import logging
import argparse
from typing import Optional

import cv2
import numpy as np

import config
from video_source import VideoSource, Frame
from tracker import Tracker, BBox, TrackerState
from ui import UI, LightMode
from imu import create_imu_provider, IMUProvider

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger('picamera2').setLevel(logging.WARNING)
logging.getLogger('libcamera').setLevel(logging.WARNING)


class ClickToTrackApp:
    """Main application class."""

    def __init__(self, source=None, target_width: int = None, 
                 use_imu: bool = None):
        """
        Initialize application.
        
        Args:
            source: Video source (camera index or RTSP URL)
            target_width: Target frame width for processing
            use_imu: Enable IMU integration (None = use config setting)
        """
        # Video source
        self.video = VideoSource(source, target_width)
        
        # IMU provider - use config setting if not explicitly specified
        if use_imu is None:
            use_imu = config.IMU_ENABLED
        
        self.imu: Optional[IMUProvider] = None
        if use_imu:
            self.imu = create_imu_provider()
            if self.imu.is_available():
                logger.info("IMU enabled and available")
            else:
                logger.warning("IMU enabled but hardware not available")
        else:
            logger.info("IMU disabled")
        
        # Tracker
        self.tracker = Tracker(self.imu)
        
        # UI with callbacks
        self.ui = UI(
            window_name="Click-to-Track",
            on_bbox_selected=self._on_bbox_selected,
            on_light_mode_changed=self._on_light_mode_changed,
            on_focus_change=self._on_focus_change,
            on_autofocus=self._on_autofocus
        )
        
        # State
        self.running = False
        self.pending_init: Optional[BBox] = None
        self.current_frame: Optional[Frame] = None
        
        # Performance monitoring
        self.fps_counter = FPSCounter()

    def _on_bbox_selected(self, bbox: BBox):
        """Callback when user selects a bbox."""
        self.pending_init = bbox
        logger.info(f"Bbox selected, will initialize on next frame")

    def _on_light_mode_changed(self, mode: LightMode):
        """Callback when light mode changes."""
        if mode == LightMode.LOW:
            # Bright daylight
            self.video.set_camera_settings(
                auto_exposure=True,
                analog_gain=1.0,
                brightness=0.0
            )
        elif mode == LightMode.MID:
            # Indoor / normal
            self.video.set_camera_settings(
                auto_exposure=True,
                analog_gain=4.0,
                brightness=0.1
            )
        elif mode == LightMode.HIGH:
            # Low light / night
            self.video.set_camera_settings(
                auto_exposure=False,
                exposure_time=100000,
                analog_gain=16.0,
                brightness=0.3
            )
        logger.info(f"Light mode changed to {mode.name}")

    def _on_focus_change(self, position: float):
        """Callback when focus is adjusted."""
        self.video.set_manual_focus(position)

    def _on_autofocus(self):
        """Callback to trigger autofocus."""
        self.video.trigger_autofocus()

    def run(self):
        """Main application loop."""
        # Open video source
        if not self.video.open():
            logger.error("Failed to open video source")
            return 1

        # Wait for first valid frame
        logger.info("Waiting for first frame...")
        first_frame = None
        for attempt in range(30):  # Try for up to 3 seconds
            first_frame = self.video.read()
            if first_frame is not None:
                logger.info(f"First frame received after {attempt + 1} attempts")
                logger.info(f"Frame size: {first_frame.width}x{first_frame.height}")
                break
            time.sleep(0.1)
        
        if first_frame is None:
            logger.error("Could not get any frames from video source")
            logger.error("Troubleshooting tips:")
            logger.error("  1. Check camera connection")
            logger.error("  2. Try a different camera index: --source 1")
            logger.error("  3. Check camera permissions: ls -la /dev/video*")
            logger.error("  4. Test camera with: ffplay /dev/video0")
            self.cleanup()
            return 1

        logger.info("Starting main loop")
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                frame = self.video.read()
                if frame is None:
                    # Brief pause to avoid spinning
                    time.sleep(0.01)
                    continue
                
                self.current_frame = frame
                
                # Handle pending initialization
                if self.pending_init is not None:
                    self.tracker.initialize(
                        frame.gray, 
                        self.pending_init, 
                        frame.timestamp
                    )
                    self.pending_init = None
                
                # Update tracker
                bbox = None
                if self.tracker.get_state() != TrackerState.IDLE:
                    # Get IMU sample if available
                    imu_sample = None
                    if self.imu is not None and self.imu.is_available():
                        imu_sample = self.imu.get_sample(
                            self.tracker.last_timestamp,
                            frame.timestamp
                        )
                    
                    # Update tracker
                    bbox = self.tracker.update(
                        frame.gray,
                        frame.timestamp,
                        imu_sample
                    )
                
                # Draw overlay
                display = self.ui.draw_overlay(
                    frame.image,
                    bbox,
                    self.tracker.get_state(),
                    self.tracker.get_confidence(),
                    self.tracker.get_features(),
                    self.tracker.get_metrics()
                )
                
                # Draw FPS
                fps = self.fps_counter.update()
                cv2.putText(display, f"FPS: {fps:.1f}", 
                           (display.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Show frame
                key = self.ui.show(display)
                
                # Handle keys
                if key == ord('q') or key == 27:  # Q or ESC
                    self.running = False
                elif key == ord('m') or key == ord('M'):
                    self.ui.toggle_mode()
                elif key == ord('r') or key == ord('R'):
                    self.tracker.reset()
                    logger.info("Tracker reset")
                elif key == ord('d') or key == ord('D'):
                    # Toggle debug drawing
                    config.DRAW_FEATURES = not config.DRAW_FEATURES
                    logger.info(f"Feature drawing: {config.DRAW_FEATURES}")
                elif key == ord('a') or key == ord('A'):
                    # Trigger autofocus
                    self.video.trigger_autofocus()
                    logger.info("Autofocus triggered")
                elif key == ord('+') or key == ord('='):
                    # Focus closer (increase lens position)
                    self.focus_position = getattr(self, 'focus_position', 1.0) + 0.5
                    self.video.set_manual_focus(self.focus_position)
                    logger.info(f"Focus position: {self.focus_position} (closer)")
                elif key == ord('-') or key == ord('_'):
                    # Focus farther (decrease lens position)
                    self.focus_position = max(0.0, getattr(self, 'focus_position', 1.0) - 0.5)
                    self.video.set_manual_focus(self.focus_position)
                    logger.info(f"Focus position: {self.focus_position} (farther)")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
        
        return 0

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.video.release()
        self.ui.destroy()
        cv2.destroyAllWindows()


class FPSCounter:
    """Simple FPS counter using moving average."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
        
    def update(self) -> float:
        """Update counter and return current FPS."""
        now = time.time()
        self.timestamps.append(now)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
        
        # Calculate FPS
        if len(self.timestamps) < 2:
            return 0.0
            
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
            
        return (len(self.timestamps) - 1) / duration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Click-to-Track System for Raspberry Pi 5'
    )
    parser.add_argument(
        '--source', '-s',
        default=None,
        help='Video source: camera index (0,1,...) or RTSP URL. '
             'Default uses config.VIDEO_SOURCE'
    )
    parser.add_argument(
        '--width', '-w',
        type=int,
        default=None,
        help='Target frame width for processing. Default uses config.TARGET_WIDTH'
    )
    parser.add_argument(
        '--imu',
        action='store_true',
        help='Enable IMU integration for gyro compensation'
    )
    parser.add_argument(
        '--drag',
        action='store_true',
        help='Start in click-drag selection mode (default is click-center)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    
    # Update config based on args
    if args.debug:
        config.LOG_LEVEL = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.drag:
        config.CLICK_DRAG_MODE = True
    
    # Parse source - could be int or string
    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string (RTSP URL)
    
    # Create and run app
    app = ClickToTrackApp(
        source=source,
        target_width=args.width,
        use_imu=args.imu
    )
    
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
