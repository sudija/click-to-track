#!/usr/bin/env python3
"""Check which OpenCV trackers are available."""

import cv2
print(f"OpenCV version: {cv2.__version__}")

# Check for trackers
trackers = [
    ('cv2.TrackerCSRT_create', hasattr(cv2, 'TrackerCSRT_create')),
    ('cv2.TrackerKCF_create', hasattr(cv2, 'TrackerKCF_create')),
    ('cv2.TrackerMOSSE_create', hasattr(cv2, 'TrackerMOSSE_create')),
    ('cv2.legacy.TrackerCSRT_create', hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create')),
    ('cv2.legacy.TrackerKCF_create', hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create')),
    ('cv2.legacy.TrackerMOSSE_create', hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create')),
]

print("\nAvailable trackers:")
for name, available in trackers:
    status = "✓" if available else "✗"
    print(f"  {status} {name}")

# Check if legacy module exists
if hasattr(cv2, 'legacy'):
    print("\ncv2.legacy contents:")
    for item in dir(cv2.legacy):
        if 'Tracker' in item:
            print(f"  - {item}")
else:
    print("\ncv2.legacy module not found")

# Check contrib
print("\nTo install tracking module, run:")
print("  sudo apt install python3-opencv")
print("  # or")
print("  pip install opencv-contrib-python")
