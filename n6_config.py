# =============================================================================
# OpenMV N6 Tracker — Configuration
# n6-phase1 / Step 1: RGB camera only
# =============================================================================

# --- Camera ---
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
# Supported framesizes: csi.QQVGA (160x120), csi.QVGA (320x240),
#                       csi.VGA (640x480), csi.SVGA (800x600)
FRAME_SIZE   = "QVGA"   # resolved to csi constant in main.py

# White balance / gain — disable for consistent blob detection
AUTO_WHITEBALANCE = False
AUTO_GAIN         = False
AUTO_EXPOSURE     = True   # Keep AE on for now; disable once outdoors

# --- Blob Detection (drone candidate) ---
# Thresholds are (min, max) pairs in grayscale (L channel of LAB)
# Start broad; tune on actual hardware with the threshold tool in OpenMV IDE
BLOB_THRESHOLDS  = [(200, 255)]  # Bright target against sky
BLOB_MIN_PIXELS  = 10            # Ignore noise smaller than this
BLOB_MAX_PIXELS  = 5000          # Ignore large background patches
BLOB_MARGIN      = 20            # Extra pixels around blob for bbox

# --- Tracker state machine ---
COAST_TRIGGER_FRAMES    = 3     # Weak frames before switching to COASTING
MAX_COAST_FRAMES        = 30    # COASTING frames before LOST
CONFIDENCE_BOOST        = 0.05
CONFIDENCE_DECAY_TRACK  = 0.02
CONFIDENCE_DECAY_COAST  = 0.05
CONFIDENCE_LOST_THRESH  = 0.20

# --- Kalman filter (scalar, centre-form) ---
KALMAN_Q = 1e-2   # Process noise  (increase = trust motion more)
KALMAN_R = 1e-1   # Measurement noise (increase = smooth more)

# --- Servo ---
SERVO_PIN        = 1      # OpenMV Servo pin (P7 = Servo 1)
SERVO_DEAD_ZONE  = 10     # px — hold still inside this band
SERVO_MAX_SPEED  = 2.0    # deg/frame at full error
SERVO_MIN_SPEED  = 0.2    # deg/frame at dead-zone edge
SERVO_RAMP_PX    = 150    # px error where max speed is reached
SERVO_MIN_ANGLE  = -90
SERVO_MAX_ANGLE  =  90
SERVO_INVERT     = False  # Flip if servo tracks the wrong way

# --- IMU (MPU6050) — disabled until wired ---
IMU_ENABLED    = False
IMU_I2C_BUS    = 2        # OpenMV I2C bus 2 (P4=SDA, P5=SCL)
IMU_I2C_ADDR   = 0x68
IMU_PX_PER_RAD = 500      # Calibration: pixels per radian

# --- Debug / Display ---
DRAW_BLOBS     = True    # Draw blob boxes in IDE frame buffer
DRAW_FPS       = True    # Print FPS to console
