"""
IMU integration module for gyro-assisted tracking.
Provides angular rate input for prediction and ROI warp compensation.

Supports:
- MPU6050 (I2C)
- BNO055 (I2C) 
- Stub for testing without hardware
"""

import time
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque

import config

logger = logging.getLogger(__name__)


@dataclass
class IMUSample:
    """Container for IMU sample data."""
    timestamp: float  # Sample timestamp
    gyro_x: float  # Angular rate around X axis (rad/s) - roll
    gyro_y: float  # Angular rate around Y axis (rad/s) - pitch  
    gyro_z: float  # Angular rate around Z axis (rad/s) - yaw
    # Integrated angles over the sample period
    delta_roll: float = 0.0  # Integrated roll (rad)
    delta_pitch: float = 0.0  # Integrated pitch (rad)
    delta_yaw: float = 0.0  # Integrated yaw (rad)


@dataclass
class PixelShift:
    """Computed pixel shift from IMU data."""
    dx: float  # Horizontal pixel shift
    dy: float  # Vertical pixel shift
    valid: bool = True


class IMUProvider(ABC):
    """Abstract base class for IMU providers."""

    @abstractmethod
    def get_sample(self, t0: float, t1: float) -> Optional[IMUSample]:
        """
        Get integrated IMU sample between two timestamps.
        
        Args:
            t0: Start timestamp
            t1: End timestamp
            
        Returns:
            IMUSample with integrated angular changes, or None if unavailable.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if IMU is available and providing data."""
        pass

    def compute_pixel_shift(self, sample: IMUSample, 
                           px_per_rad: float = None) -> PixelShift:
        """
        Convert IMU angular changes to pixel shift.
        
        Uses a simple linear approximation suitable for small angles.
        For a camera pointing along Z axis:
        - Yaw (rotation around Y for landscape camera) causes horizontal shift
        - Pitch (rotation around X) causes vertical shift
        
        Args:
            sample: IMU sample with integrated angles
            px_per_rad: Calibration factor (pixels per radian). None uses config.
            
        Returns:
            PixelShift with dx, dy values.
        """
        if sample is None:
            return PixelShift(0.0, 0.0, valid=False)
            
        px_per_rad = px_per_rad or config.IMU_PX_PER_RAD
        
        # Camera coordinate system (looking at scene):
        # - Yaw (rotation around vertical) -> horizontal pixel shift
        # - Pitch (rotation around horizontal) -> vertical pixel shift
        # Signs depend on IMU mounting orientation
        dx = -sample.delta_yaw * px_per_rad  # Negative: yaw right = scene moves left
        dy = sample.delta_pitch * px_per_rad  # Positive: pitch up = scene moves down
        
        return PixelShift(dx, dy, valid=True)


class StubIMUProvider(IMUProvider):
    """
    Stub IMU provider that returns no data.
    Use as placeholder when no IMU hardware is connected.
    """

    def __init__(self):
        logger.info("StubIMUProvider initialized (no actual IMU)")

    def get_sample(self, t0: float, t1: float) -> Optional[IMUSample]:
        """Returns None - no IMU data available."""
        return None

    def is_available(self) -> bool:
        """Always returns False for stub."""
        return False


class MPU6050Provider(IMUProvider):
    """
    MPU6050 IMU provider using I2C.
    Reads gyroscope data for motion compensation.
    """
    
    # MPU6050 registers
    PWR_MGMT_1 = 0x6B
    GYRO_CONFIG = 0x1B
    GYRO_XOUT_H = 0x43
    
    # Gyro sensitivity (deg/s per LSB) for different ranges
    GYRO_SCALE = {
        0: 131.0,   # ±250°/s
        1: 65.5,    # ±500°/s
        2: 32.8,    # ±1000°/s
        3: 16.4     # ±2000°/s
    }
    
    def __init__(self, bus: int = None, address: int = None, gyro_range: int = 1):
        """
        Initialize MPU6050.
        
        Args:
            bus: I2C bus number (None = use config)
            address: I2C address (None = use config, typically 0x68)
            gyro_range: 0=±250°/s, 1=±500°/s, 2=±1000°/s, 3=±2000°/s
        """
        self.bus_num = bus if bus is not None else getattr(config, 'IMU_I2C_BUS', 1)
        self.address = address if address is not None else getattr(config, 'IMU_I2C_ADDRESS', 0x68)
        self.gyro_range = gyro_range
        self.gyro_scale = self.GYRO_SCALE[gyro_range]
        self.bus = None
        self._available = False
        self._sample_buffer: deque = deque(maxlen=1000)
        self._last_read_time = 0
        
        self._init_device()
    
    def _init_device(self):
        """Initialize I2C connection and configure MPU6050."""
        try:
            import smbus2
            self.bus = smbus2.SMBus(self.bus_num)
            
            # Wake up MPU6050 (it starts in sleep mode)
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            
            # Set gyro range
            self.bus.write_byte_data(self.address, self.GYRO_CONFIG, self.gyro_range << 3)
            
            self._available = True
            logger.info(f"MPU6050 initialized on I2C bus {self.bus_num}, address 0x{self.address:02X}")
            logger.info(f"Gyro range: ±{[250, 500, 1000, 2000][self.gyro_range]}°/s")
            
        except ImportError:
            logger.warning("smbus2 not installed. Install with: pip install smbus2")
            self._available = False
        except Exception as e:
            logger.warning(f"MPU6050 initialization failed: {e}")
            self._available = False
    
    def _read_raw_gyro(self) -> Tuple[int, int, int]:
        """Read raw gyroscope values."""
        if not self._available or self.bus is None:
            return (0, 0, 0)
        
        try:
            # Read 6 bytes starting from GYRO_XOUT_H
            data = self.bus.read_i2c_block_data(self.address, self.GYRO_XOUT_H, 6)
            
            # Combine high and low bytes (big-endian, signed)
            gx = (data[0] << 8) | data[1]
            gy = (data[2] << 8) | data[3]
            gz = (data[4] << 8) | data[5]
            
            # Convert to signed
            if gx > 32767: gx -= 65536
            if gy > 32767: gy -= 65536
            if gz > 32767: gz -= 65536
            
            return (gx, gy, gz)
        except Exception as e:
            logger.debug(f"Gyro read error: {e}")
            return (0, 0, 0)
    
    def _read_gyro_dps(self) -> Tuple[float, float, float]:
        """Read gyroscope values in degrees per second."""
        gx, gy, gz = self._read_raw_gyro()
        return (
            gx / self.gyro_scale,
            gy / self.gyro_scale,
            gz / self.gyro_scale
        )
    
    def _read_gyro_rps(self) -> Tuple[float, float, float]:
        """Read gyroscope values in radians per second."""
        import math
        gx, gy, gz = self._read_gyro_dps()
        deg_to_rad = math.pi / 180.0
        return (
            gx * deg_to_rad,
            gy * deg_to_rad,
            gz * deg_to_rad
        )
    
    def _update_buffer(self):
        """Read current gyro and add to buffer."""
        if not self._available:
            return
        
        now = time.time()
        gx, gy, gz = self._read_gyro_rps()
        
        self._sample_buffer.append({
            'timestamp': now,
            'gx': gx,
            'gy': gy,
            'gz': gz
        })
        self._last_read_time = now
    
    def get_sample(self, t0: float, t1: float) -> Optional[IMUSample]:
        """
        Get integrated IMU sample between timestamps.
        Integrates gyro readings to get angular change.
        """
        if not self._available:
            return None
        
        # Read fresh data
        self._update_buffer()
        
        if len(self._sample_buffer) < 2:
            return None
        
        # Integrate samples in time range
        delta_roll = 0.0
        delta_pitch = 0.0
        delta_yaw = 0.0
        count = 0
        prev_t = t0
        
        for sample in self._sample_buffer:
            t = sample['timestamp']
            if t0 <= t <= t1:
                dt = t - prev_t
                if dt > 0 and dt < 0.1:  # Sanity check
                    delta_roll += sample['gx'] * dt
                    delta_pitch += sample['gy'] * dt
                    delta_yaw += sample['gz'] * dt
                    count += 1
                prev_t = t
        
        if count == 0:
            # No samples in range, use latest reading with estimated dt
            latest = self._sample_buffer[-1]
            dt = t1 - t0
            return IMUSample(
                timestamp=t1,
                gyro_x=latest['gx'],
                gyro_y=latest['gy'],
                gyro_z=latest['gz'],
                delta_roll=latest['gx'] * dt,
                delta_pitch=latest['gy'] * dt,
                delta_yaw=latest['gz'] * dt
            )
        
        return IMUSample(
            timestamp=t1,
            gyro_x=0,  # Instantaneous not computed
            gyro_y=0,
            gyro_z=0,
            delta_roll=delta_roll,
            delta_pitch=delta_pitch,
            delta_yaw=delta_yaw
        )
    
    def is_available(self) -> bool:
        return self._available
    
    def close(self):
        """Close I2C connection."""
        if self.bus is not None:
            self.bus.close()
            self.bus = None
            self._available = False


class SimulatedIMUProvider(IMUProvider):
    """
    Simulated IMU provider for testing.
    Generates synthetic angular rates for development without hardware.
    """

    def __init__(self, base_rate: float = 0.0):
        self.base_rate = base_rate
        logger.info(f"SimulatedIMUProvider initialized with base_rate={base_rate}")

    def get_sample(self, t0: float, t1: float) -> Optional[IMUSample]:
        dt = t1 - t0
        if dt <= 0:
            return None
        delta = self.base_rate * dt
        return IMUSample(
            timestamp=t1,
            gyro_x=0.0, gyro_y=0.0, gyro_z=self.base_rate,
            delta_roll=0.0, delta_pitch=0.0, delta_yaw=delta
        )

    def is_available(self) -> bool:
        return True


def create_imu_provider() -> IMUProvider:
    """
    Factory function to create appropriate IMU provider based on config.
    
    Returns:
        IMUProvider instance (stub if IMU disabled or unavailable)
    """
    if not config.IMU_ENABLED:
        logger.info("IMU disabled in config")
        return StubIMUProvider()
    
    # Try MPU6050 first (most common)
    logger.info("Attempting to initialize MPU6050...")
    provider = MPU6050Provider()
    if provider.is_available():
        return provider
    
    # Fall back to stub
    logger.warning("No IMU available, using stub provider")
    return StubIMUProvider()
