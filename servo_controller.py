"""
Servo controller for camera pan — hardware PWM via sysfs.

Uses the same sysfs PWM approach proven in servo_test.py.
No pigpio, no gpiozero, no jitter.

Hardware setup (one-time):
  /boot/firmware/config.txt must contain:
    dtoverlay=pwm,pin=18,func=PWM0_CHAN2

  Verify:
    pinctrl get 18   → should show: a3 // GPIO18 = PWM0_CHAN2

  Permissions (if needed):
    sudo chmod -R a+rw /sys/class/pwm/

Wiring:
  Signal (orange/yellow) → GPIO 18 (Pin 12)
  Power  (red)           → 5V     (Pin 2 or 4)
  Ground (brown/black)   → GND    (Pin 6)
"""

import os
import math
import time
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── sysfs PWM constants ────────────────────────────────────────
PWM_CHIP    = 0
PWM_CHANNEL = 2
PWM_BASE    = f"/sys/class/pwm/pwmchip{PWM_CHIP}"
PWM_PATH    = f"{PWM_BASE}/pwm{PWM_CHANNEL}"

PERIOD_NS   = 20_000_000   # 20ms → 50Hz
PULSE_MIN   =    500_000   # 0.5ms → -90°
PULSE_CTR   =  1_500_000   # 1.5ms →   0°
PULSE_MAX   =  2_500_000   # 2.5ms → +90°

ANGLE_MIN   = -90.0
ANGLE_MAX   =  90.0


# ── sysfs helpers ──────────────────────────────────────────────
def _pwm_write(filename: str, value):
    with open(os.path.join(PWM_PATH, filename), 'w') as f:
        f.write(str(value))


def _pwm_export():
    if not os.path.exists(PWM_PATH):
        with open(os.path.join(PWM_BASE, "export"), 'w') as f:
            f.write(str(PWM_CHANNEL))
        time.sleep(0.1)


def _pwm_unexport():
    try:
        with open(os.path.join(PWM_BASE, "unexport"), 'w') as f:
            f.write(str(PWM_CHANNEL))
    except Exception:
        pass


def _angle_to_ns(angle: float) -> int:
    """Convert -90..+90 degrees to pulse width in nanoseconds."""
    angle = max(ANGLE_MIN, min(ANGLE_MAX, angle))
    return int(PULSE_CTR + (angle / 90.0) * (PULSE_MAX - PULSE_CTR))


# ── Servo tracking controller ──────────────────────────────────
@dataclass
class PIDConfig:
    dead_zone: int   = 10    # Pixels — hold still inside this
    max_speed: float = 2.0   # Degrees/frame at max error
    min_speed: float = 0.2   # Degrees/frame at dead zone edge
    ramp_pixels: int = 150   # Error in pixels where max_speed is reached


class PIDController:

    def __init__(self, cfg: PIDConfig = None):
        self.cfg        = cfg or PIDConfig()
        self.last_error = 0.0

    def reset(self):
        self.last_error = 0.0

    def update(self, error: float) -> float:
        """
        Proportional control — speed scales linearly from min_speed
        at the dead zone edge up to max_speed at ramp_pixels.
        Direction is always toward centre.
        """
        abs_error = abs(error)
        self.last_error = error

        if abs_error < self.cfg.dead_zone:
            return 0.0

        # Linear ramp: min_speed at dead_zone edge, max_speed at ramp_pixels
        t = min(1.0, (abs_error - self.cfg.dead_zone) /
                     (self.cfg.ramp_pixels - self.cfg.dead_zone))
        speed = self.cfg.min_speed + t * (self.cfg.max_speed - self.cfg.min_speed)

        return math.copysign(speed, error)


# ── Servo controller ───────────────────────────────────────────
class ServoController:
    """
    Hardware PWM servo controller.

    Usage:
        servo = ServoController()
        servo.connect()          # Enable PWM

        # In tracking loop:
        servo.update(target_x=bbox.cx, frame_width=frame.shape[1])

        servo.disconnect()       # Disable PWM cleanly
    """

    def __init__(self, pid_cfg: PIDConfig = None, simulate: bool = False, invert: bool = False):
        self.pid       = PIDController(pid_cfg or PIDConfig())
        self.simulate  = simulate
        self.invert    = invert
        self.connected = False
        self.detached  = False
        self._angle    = 0.0
        self._sweeping = False
        self._frame_count = 0  # For throttling debug output

    # ── Connection ─────────────────────────────────────────────
    def connect(self) -> bool:
        """Initialise sysfs PWM, move to centre, then run startup sweep. Returns True on success."""
        if self.simulate:
            print("ServoController: simulation mode (no hardware)")
            self.connected = True
            threading.Thread(target=self._startup_sweep, daemon=True).start()
            return True

        print(f"Servo: connecting via {PWM_PATH} ...")
        try:
            _pwm_export()
            print("Servo: exported PWM")
            _pwm_write("period", PERIOD_NS)
            print(f"Servo: period set to {PERIOD_NS}ns")
            _pwm_write("duty_cycle", PULSE_CTR)
            print(f"Servo: duty_cycle set to {PULSE_CTR}ns (centre)")
            _pwm_write("enable", 1)
            print("Servo: enabled — starting sweep")
            self.connected = True
            self.detached  = False
            self._angle    = 0.0
            threading.Thread(target=self._startup_sweep, daemon=True).start()
            return True
        except PermissionError as e:
            print(f"Servo: PERMISSION DENIED — run: sudo chmod -R a+rw /sys/class/pwm/")
        except FileNotFoundError as e:
            print(f"Servo: PWM PATH NOT FOUND — {e}")
            print(f"  Is the overlay set? /boot/firmware/config.txt should contain:")
            print(f"  dtoverlay=pwm,pin=18,func=PWM0_CHAN2")
            print(f"  Verify with: pinctrl get 18")
        except Exception as e:
            print(f"Servo: connect error — {type(e).__name__}: {e}")
        return False

    def _startup_sweep(self):
        """Sweep -90 → +90 → 0 so the operator knows the servo is alive."""
        print("Servo: sweep starting...")
        self._sweeping = True
        time.sleep(0.3)   # Let PWM settle before moving

        # Sweep to -90°
        for angle in range(0, -91, -3):
            self._write_angle(angle)
            time.sleep(0.02)

        # Sweep to +90°
        for angle in range(-90, 91, 3):
            self._write_angle(angle)
            time.sleep(0.02)

        # Return to centre
        for angle in range(90, -1, -3):
            self._write_angle(angle)
            time.sleep(0.02)

        self._write_angle(0.0)
        self.pid.reset()
        self._sweeping = False
        print("Servo: sweep complete — ready for tracking")

    def disconnect(self):
        """Disable PWM output cleanly."""
        if self.simulate:
            self.connected = False
            return
        try:
            _pwm_write("enable", 0)
            _pwm_unexport()
        except Exception:
            pass
        self.connected = False
        logger.info("Servo disconnected")

    def detach(self):
        """Cut PWM signal — servo relaxes, no jitter at rest."""
        if self.connected and not self.simulate:
            try:
                _pwm_write("enable", 0)
                self.detached = True
                logger.debug("Servo detached")
            except Exception as e:
                logger.warning(f"Servo detach error: {e}")

    def reattach(self):
        """Re-enable PWM and restore last angle."""
        if self.connected and self.detached and not self.simulate:
            try:
                _pwm_write("enable", 1)
                self.detached = False
                self._write_angle(self._angle)
            except Exception as e:
                logger.warning(f"Servo reattach error: {e}")

    # ── Angle control ───────────────────────────────────────────
    def _write_angle(self, angle: float):
        """Write angle to hardware (or log if simulating)."""
        angle = max(ANGLE_MIN, min(ANGLE_MAX, angle))
        self._angle = angle
        if self.simulate:
            return
        try:
            _pwm_write("duty_cycle", _angle_to_ns(angle))
        except Exception as e:
            logger.warning(f"Servo write error: {e}")

    def set_angle(self, angle: float):
        """Manually set servo to a specific angle. Resets PID."""
        self.pid.reset()
        if self.detached:
            self.reattach()
        self._write_angle(angle)

    def center(self):
        """Return servo to 0° and reset PID."""
        self.set_angle(0.0)

    def get_angle(self) -> float:
        return self._angle

    # ── Tracking update ─────────────────────────────────────────
    def update(self, target_x: float, frame_width: int):
        """
        Move servo to keep target_x centred in the frame.

        Call once per frame while tracking.

        Args:
            target_x:    Horizontal centre of the tracked bounding box (pixels).
            frame_width: Width of the video frame (pixels).
        """
        if not self.connected or self._sweeping:
            return

        if self.detached:
            self.reattach()

        error = target_x - frame_width / 2.0
        delta = self.pid.update(error)

        if self.invert:
            delta = -delta

        self._frame_count += 1
        if self._frame_count % 10 == 0:
            trend = "converging" if abs(error) < abs(self.pid.last_error) else "DIVERGING"
            print(f"Servo: error={error:+.0f}px  delta={delta:+.3f}°  angle={self._angle:.1f}°  {trend}")

        if delta != 0.0:
            self._write_angle(self._angle + delta)
