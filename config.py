"""
Configuration constants for the click-to-track system.
Tunable parameters for Raspberry Pi 5 optimization.
"""

# =============================================================================
# VIDEO CAPTURE SETTINGS
# =============================================================================
VIDEO_SOURCE = 0  # 0 for local camera, or RTSP URL string
TARGET_WIDTH = 640  # Downscale to this width for processing
TARGET_FPS = 30  # Target frame rate

# Camera exposure settings (configured for NIGHT / LOW LIGHT)
CAMERA_AUTO_EXPOSURE = False  # Manual control for night
CAMERA_EXPOSURE_TIME = 100000  # 100ms exposure (long for low light)
CAMERA_ANALOG_GAIN = 16.0  # Maximum gain for night
CAMERA_BRIGHTNESS = 0.3  # Boost brightness
CAMERA_CONTRAST = 1.0  # Normal contrast
CAMERA_AUTOFOCUS = True  # Enable continuous autofocus

# =============================================================================
# SERVO SETTINGS
# =============================================================================
SERVO_PAN_PIN = 18           # GPIO pin for pan servo (BCM numbering)
SERVO_SIMULATE = False       # Set False when servo is connected
SERVO_INVERT = False         # Flip pan direction if servo moves the wrong way
SERVO_MIN_ANGLE = -90        # Minimum pan angle
SERVO_MAX_ANGLE = 90         # Maximum pan angle

# =============================================================================
# TRACKER SETTINGS
# =============================================================================
# Tracker type: CSRT (accurate), KCF (fast), MOSSE (fastest)
TRACKER_TYPE = "KCF"

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
BBOX_STABILITY_THRESHOLD = 3.0  # Pixels - box won't move unless centroid moves more than this

# =============================================================================
# KLT OPTICAL FLOW SETTINGS
# =============================================================================
MAX_FEATURES = 200  # Maximum features to track (balance accuracy vs speed)
MIN_FEATURES = 15  # Minimum features before tracking is considered weak
FEATURE_REFRESH_THRESHOLD = 50  # Refresh when below this count
FEATURE_QUALITY_LEVEL = 0.01  # goodFeaturesToTrack quality level
FEATURE_MIN_DISTANCE = 5  # Minimum distance between features
FEATURE_BLOCK_SIZE = 7  # Block size for corner detection

# LK optical flow parameters
LK_WIN_SIZE = (21, 21)  # Window size for Lucas-Kanade
LK_MAX_LEVEL = 3  # Pyramid levels
LK_CRITERIA = (3, 10, 0.03)  # (type, maxCount, epsilon) - cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT

# Forward-backward check
FB_CHECK_ENABLED = True  # Enable forward-backward consistency check
FB_ERROR_THRESHOLD = 1.5  # Maximum pixel error for FB check

# =============================================================================
# TEMPLATE MATCHING SETTINGS
# =============================================================================
TEMPLATE_SIZE = (64, 64)  # Template patch size (w, h)
TEMPLATE_UPDATE_ALPHA = 0.02  # Template update rate (slow adaptation)
NCC_SEARCH_MARGIN = 30  # Search margin around predicted position
NCC_THRESHOLD_HIGH = 0.7  # High confidence NCC score
NCC_THRESHOLD_LOW = 0.4  # Below this, NCC is unreliable

# =============================================================================
# KALMAN FILTER SETTINGS
# =============================================================================
KALMAN_PROCESS_NOISE = 1e-2  # Process noise covariance
KALMAN_MEASUREMENT_NOISE = 1e-1  # Measurement noise covariance
KALMAN_INITIAL_ERROR = 1.0  # Initial error covariance

# =============================================================================
# STATE MACHINE SETTINGS
# =============================================================================
COAST_TRIGGER_FRAMES = 3  # Frames of weak tracking before coasting
MAX_COAST_FRAMES = 30  # Maximum frames to coast before LOST
OUT_OF_FRAME_TOLERANCE = 5  # Frames out of frame before LOST

# Confidence thresholds
CONFIDENCE_LOST_THRESHOLD = 0.2  # Below this, declare LOST
CONFIDENCE_DECAY_TRACKING = 0.02  # Decay per weak frame in TRACKING
CONFIDENCE_DECAY_COASTING = 0.05  # Decay per frame in COASTING
CONFIDENCE_BOOST = 0.05  # Boost per strong frame

# KLT quality thresholds
KLT_INLIER_RATIO_STRONG = 0.6  # Above this, KLT is strong
KLT_INLIER_RATIO_WEAK = 0.3  # Below this, KLT is weak

# =============================================================================
# ROI SETTINGS
# =============================================================================
ROI_EXPAND_MARGIN = 50  # Pixels to expand bbox for ROI
ROI_EXPAND_COASTING_FACTOR = 1.5  # Additional expansion during coasting

# =============================================================================
# IMU SETTINGS (MPU6050)
# =============================================================================
IMU_ENABLED = True  # Enable IMU integration
IMU_PX_PER_RAD = 500  # Approximate pixels per radian (calibration factor)
IMU_I2C_BUS = 1  # I2C bus number
IMU_I2C_ADDRESS = 0x68  # MPU6050 I2C address

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
DRAW_FEATURES = True  # Draw feature points
DRAW_SEARCH_ROI = False  # Draw search ROI rectangle
BBOX_COLOR_TRACKING = (0, 255, 0)  # Green
BBOX_COLOR_COASTING = (0, 255, 255)  # Yellow
BBOX_COLOR_LOST = (0, 0, 255)  # Red
FEATURE_COLOR = (255, 0, 255)  # Magenta
TEXT_COLOR = (255, 255, 255)  # White

# =============================================================================
# LOGGING SETTINGS
# =============================================================================
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
LOG_STATE_TRANSITIONS = True  # Log tracker state changes
