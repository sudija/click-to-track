"""
Configuration for Click-to-Track system.
"""

# =============================================================================
# VIDEO SOURCE SETTINGS
# =============================================================================
VIDEO_SOURCE = 0  # 0 for default camera, or path to video file
TARGET_WIDTH = 640  # Resize frame width (maintains aspect ratio)
TARGET_FPS = 30

# =============================================================================
# TRACKER SETTINGS
# =============================================================================
# Tracker type: CSRT (accurate), KCF (fast), MOSSE (fastest)
TRACKER_TYPE = "CSRT"

# Recovery settings (when tracking is lost)
MAX_LOST_FRAMES = 30  # Frames before declaring lost
RECOVERY_SEARCH_MARGIN = 100  # Pixels to search around last position
RECOVERY_THRESHOLD = 0.5  # Template match threshold for recovery

# =============================================================================
# BBOX SELECTION SETTINGS
# =============================================================================
DEFAULT_BBOX_WIDTH = 80  # Default bbox width for click mode
DEFAULT_BBOX_HEIGHT = 80  # Default bbox height for click mode
CLICK_DRAG_MODE = False  # False = click centers bbox, True = click-drag defines corners

# =============================================================================
# IMU SETTINGS (MPU6050)
# =============================================================================
IMU_ENABLED = False  # Set True if MPU6050 is connected
IMU_I2C_BUS = 1
IMU_I2C_ADDRESS = 0x68
IMU_PX_PER_RAD = 500  # Pixels per radian - calibrate for your camera FOV

# =============================================================================
# CAMERA CONTROL SETTINGS (Picamera2)
# =============================================================================
CAMERA_AUTO_EXPOSURE = True
CAMERA_EXPOSURE_TIME = 33000  # Microseconds (33ms = 30fps max)
CAMERA_ANALOG_GAIN = 4.0  # 1.0 - 16.0
CAMERA_BRIGHTNESS = 0.0  # -1.0 to 1.0
CAMERA_CONTRAST = 1.0
CAMERA_AUTOFOCUS = True  # Enable continuous autofocus if available

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
DRAW_FEATURES = False  # Not used with CSRT/KCF trackers
BBOX_COLOR_TRACKING = (0, 255, 0)  # Green
BBOX_COLOR_LOST = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
FEATURE_COLOR = (255, 0, 255)  # Magenta
