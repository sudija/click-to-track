# OpenMV N6 Multispectral Thermal Camera Module — Working Setup Guide

## Hardware
- OpenMV N6 board (firmware v4.8.1)
- Multispectral module with RGB (PAG7936) + FLIR Lepton thermal sensor

---

## Key Learnings (after extensive troubleshooting)

1. **Do NOT** use the old `sensor` module or `fir` module for multispectral
2. **Do NOT** try to configure radiometry mode — causes `"invalid argument"` error
3. The Lepton on this module does **not** support radiometry (temperature measurement)

---

## Working Code Pattern

```python
import csi
import image
import math

# RGB camera (default CSI)
csi0 = csi.CSI()
csi0.reset(hard=True)
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.QVGA)

# Thermal camera (CRITICAL: use cid=csi.LEPTON)
csi1 = csi.CSI(cid=csi.LEPTON)  # ← This is the key!
csi1.reset(hard=False)
csi1.pixformat(csi.GRAYSCALE)
csi1.framesize(csi.QVGA)

# Create thermal buffer
img1 = image.Image(csi1.width(), csi1.height(), csi1.pixformat())

# Capture loop
while True:
    img0 = csi0.snapshot()                          # RGB
    csi1.snapshot(blocking=False, image=img1)       # Thermal

    # Overlay thermal on RGB with alignment offset
    img0.draw_image(img1, x_offset, y_offset,
                   color_palette=image.PALETTE_IRONBOW,
                   alpha_palette=alpha_pal,
                   hint=image.BILINEAR)
```

---

## Alignment

- Thermal needs **+10px right, -10px up** offset for proper alignment
- Use `x_scale` and `y_scale` to stretch thermal if needed

---

## Reference Scripts

- Working example: `/mnt/user-data/outputs/n6_multispectral_working.py`
- With alignment controls: `/mnt/user-data/outputs/n6_multispectral_alignment.py`

---

## What Doesn't Work

| Approach | Failure |
|---|---|
| `fir.init()` | Fails to detect sensor on multispectral module |
| `IOCTL_LEPTON_SET_MODE` | Fails with `"invalid argument"` |
| `sensor` module instead of `csi` | Not supported |
| `CSI(0)` and `CSI(1)` for dual cameras | Does not work |
