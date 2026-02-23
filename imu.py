"""
IMU support for motion compensation (optional).
Supports MPU6050 via I2C.
"""

import logging
from typing import Optional
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class IMUSample:
    """IMU sample data."""
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    dt: float = 0.0


class IMUProvider:
    """Base IMU provider."""
    
    def is_available(self) -> bool:
        return False
    
    def get_sample(self, start_time: float, end_time: float) -> Optional[IMUSample]:
        return None


class DummyIMUProvider(IMUProvider):
    """Dummy IMU that returns no data."""
    pass


def create_imu_provider() -> IMUProvider:
    """Create appropriate IMU provider."""
    if not config.IMU_ENABLED:
        return DummyIMUProvider()
    
    try:
        import smbus2
        # Try to create MPU6050 provider
        # (Full implementation would go here)
        logger.info("IMU support not fully implemented, using dummy")
        return DummyIMUProvider()
    except ImportError:
        logger.warning("smbus2 not available, IMU disabled")
        return DummyIMUProvider()
