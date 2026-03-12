"""
UI module for mouse interaction and visualization.
Handles bbox selection and overlay drawing.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto

import config
from tracker import BBox, TrackerState, TrackingMetrics

logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    """Bbox selection mode."""
    CLICK_CENTER = auto()  # Click to center default-sized bbox
    CLICK_DRAG = auto()    # Click and drag to define corners


class LightMode(Enum):
    """Camera light sensitivity presets."""
    LOW = auto()     # Bright conditions
    MID = auto()     # Indoor/normal
    HIGH = auto()    # Low light/night


@dataclass
class SelectionState:
    """State for bbox selection UI."""
    active: bool = False
    start_x: int = 0
    start_y: int = 0
    end_x: int = 0
    end_y: int = 0


@dataclass 
class Button:
    """Simple button for UI."""
    x: int
    y: int
    w: int
    h: int
    label: str
    active: bool = False
    
    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = None):
        if color is None:
            color = (0, 200, 0) if self.active else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 1)
        
        # Center text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        (tw, th), _ = cv2.getTextSize(self.label, font, font_scale, 1)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(frame, self.label, (tx, ty), font, font_scale, (255, 255, 255), 1)


class UI:
    """
    UI handler for click-to-track system.
    Manages mouse callbacks and visualization.
    """

    def __init__(self, window_name: str = "Click-to-Track",
                 on_bbox_selected: Optional[Callable[[BBox], None]] = None,
                 on_light_mode_changed: Optional[Callable[[LightMode], None]] = None,
                 on_focus_change: Optional[Callable[[float], None]] = None,
                 on_autofocus: Optional[Callable[[], None]] = None):
        """
        Initialize UI.
        
        Args:
            window_name: OpenCV window name
            on_bbox_selected: Callback when bbox is finalized
            on_light_mode_changed: Callback when light mode changes
            on_focus_change: Callback when focus adjusted (delta value)
            on_autofocus: Callback to trigger autofocus
        """
        self.window_name = window_name
        self.on_bbox_selected = on_bbox_selected
        self.on_light_mode_changed = on_light_mode_changed
        self.on_focus_change = on_focus_change
        self.on_autofocus = on_autofocus
        
        # Selection state
        self.selection = SelectionState()
        self.mode = (SelectionMode.CLICK_DRAG if config.CLICK_DRAG_MODE 
                    else SelectionMode.CLICK_CENTER)
        
        # Light mode
        self.light_mode = LightMode.MID
        
        # Focus position
        self.focus_position = 1.0
        
        # Window created flag
        self._window_created = False
        
        # Current display frame (for selection preview)
        self.current_frame: Optional[np.ndarray] = None
        
        # Buttons (will be positioned in draw_overlay based on frame size)
        self.buttons = {}
        self._buttons_initialized = False

    def _init_buttons(self, frame_w: int, frame_h: int):
        """Initialize button positions based on frame size."""
        btn_w, btn_h = 50, 25
        margin = 5
        right_x = frame_w - btn_w - margin
        
        # Light mode buttons (top right)
        self.buttons['light_low'] = Button(right_x - 2*(btn_w + margin), margin, btn_w, btn_h, "LOW")
        self.buttons['light_mid'] = Button(right_x - (btn_w + margin), margin, btn_w, btn_h, "MID")
        self.buttons['light_high'] = Button(right_x, margin, btn_w, btn_h, "HIGH")
        
        # Focus buttons (below light buttons)
        btn_y = margin + btn_h + margin
        self.buttons['focus_minus'] = Button(right_x - 2*(btn_w + margin), btn_y, btn_w, btn_h, "F -")
        self.buttons['focus_auto'] = Button(right_x - (btn_w + margin), btn_y, btn_w, btn_h, "AF")
        self.buttons['focus_plus'] = Button(right_x, btn_y, btn_w, btn_h, "F +")
        
        # Set initial active state
        self.buttons['light_mid'].active = True
        
        self._buttons_initialized = True

    def create_window(self):
        """Create OpenCV window and set up mouse callback."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._window_created = True
        logger.info(f"Window '{self.window_name}' created, mode: {self.mode.name}")

    def _mouse_callback(self, event: int, x: int, y: int, 
                        flags: int, param: any):
        """Handle mouse events."""
        
        # Check button clicks first
        if event == cv2.EVENT_LBUTTONDOWN and self._buttons_initialized:
            for name, btn in self.buttons.items():
                if btn.contains(x, y):
                    self._handle_button_click(name)
                    return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start selection
            self.selection.active = True
            self.selection.start_x = x
            self.selection.start_y = y
            self.selection.end_x = x
            self.selection.end_y = y
            logger.debug(f"Selection started at ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selection.active:
            # Update selection (for drag mode)
            if self.mode == SelectionMode.CLICK_DRAG:
                self.selection.end_x = x
                self.selection.end_y = y
                
        elif event == cv2.EVENT_LBUTTONUP:
            if not self.selection.active:
                return
                
            self.selection.active = False
            
            if self.mode == SelectionMode.CLICK_CENTER:
                # Create bbox centered at click
                bbox = BBox(
                    cx=float(self.selection.start_x),
                    cy=float(self.selection.start_y),
                    w=float(config.DEFAULT_BBOX_WIDTH),
                    h=float(config.DEFAULT_BBOX_HEIGHT)
                )
            else:
                # Create bbox from corners
                self.selection.end_x = x
                self.selection.end_y = y
                
                # Calculate dimensions
                w = abs(self.selection.end_x - self.selection.start_x)
                h = abs(self.selection.end_y - self.selection.start_y)
                
                # Minimum size check
                if w < 10 or h < 10:
                    # Too small, treat as click
                    bbox = BBox(
                        cx=float(self.selection.start_x),
                        cy=float(self.selection.start_y),
                        w=float(config.DEFAULT_BBOX_WIDTH),
                        h=float(config.DEFAULT_BBOX_HEIGHT)
                    )
                else:
                    bbox = BBox.from_corners(
                        self.selection.start_x,
                        self.selection.start_y,
                        self.selection.end_x,
                        self.selection.end_y
                    )
            
            logger.info(f"Bbox selected: {bbox.to_rect()}")
            
            if self.on_bbox_selected:
                self.on_bbox_selected(bbox)
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to reset/cancel
            self.selection.active = False
            logger.debug("Selection cancelled")

    def _handle_button_click(self, button_name: str):
        """Handle button click events."""
        if button_name == 'light_low':
            self._set_light_mode(LightMode.LOW)
        elif button_name == 'light_mid':
            self._set_light_mode(LightMode.MID)
        elif button_name == 'light_high':
            self._set_light_mode(LightMode.HIGH)
        elif button_name == 'focus_minus':
            self.focus_position = max(0.0, self.focus_position - 0.5)
            if self.on_focus_change:
                self.on_focus_change(self.focus_position)
            logger.info(f"Focus: {self.focus_position}")
        elif button_name == 'focus_plus':
            self.focus_position += 0.5
            if self.on_focus_change:
                self.on_focus_change(self.focus_position)
            logger.info(f"Focus: {self.focus_position}")
        elif button_name == 'focus_auto':
            if self.on_autofocus:
                self.on_autofocus()
            logger.info("Autofocus triggered")

    def _set_light_mode(self, mode: LightMode):
        """Set light mode and update button states."""
        self.light_mode = mode
        
        # Update button active states
        self.buttons['light_low'].active = (mode == LightMode.LOW)
        self.buttons['light_mid'].active = (mode == LightMode.MID)
        self.buttons['light_high'].active = (mode == LightMode.HIGH)
        
        if self.on_light_mode_changed:
            self.on_light_mode_changed(mode)
        
        logger.info(f"Light mode: {mode.name}")

    def toggle_mode(self):
        """Toggle between selection modes."""
        if self.mode == SelectionMode.CLICK_CENTER:
            self.mode = SelectionMode.CLICK_DRAG
        else:
            self.mode = SelectionMode.CLICK_CENTER
        logger.info(f"Selection mode: {self.mode.name}")

    def draw_overlay(self, frame: np.ndarray, 
                    bbox: Optional[BBox],
                    state: TrackerState,
                    confidence: float,
                    features: Optional[np.ndarray] = None,
                    metrics: Optional[TrackingMetrics] = None) -> np.ndarray:
        """
        Draw tracking overlay on frame.
        """
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Initialize buttons if needed
        if not self._buttons_initialized:
            self._init_buttons(w, h)
        
        # Draw selection preview if active
        if self.selection.active and self.mode == SelectionMode.CLICK_DRAG:
            cv2.rectangle(
                display,
                (self.selection.start_x, self.selection.start_y),
                (self.selection.end_x, self.selection.end_y),
                (255, 255, 0),  # Cyan
                1
            )
        
        # Draw bbox if available
        if bbox is not None and state != TrackerState.IDLE:
            # Choose color based on state
            if state == TrackerState.TRACKING:
                color = config.BBOX_COLOR_TRACKING
            elif state == TrackerState.COASTING:
                color = config.BBOX_COLOR_COASTING
            else:
                color = config.BBOX_COLOR_LOST
            
            # Draw bbox rectangle
            corners = bbox.to_corners()
            cv2.rectangle(display, corners[0], corners[1], color, 2)
            
            # Draw center crosshair
            cx, cy = int(bbox.cx), int(bbox.cy)
            size = 10
            cv2.line(display, (cx - size, cy), (cx + size, cy), color, 1)
            cv2.line(display, (cx, cy - size), (cx, cy + size), color, 1)
            
            # Draw features if enabled
            if config.DRAW_FEATURES and features is not None and len(features) > 0:
                for pt in features:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(display, (x, y), 2, config.FEATURE_COLOR, -1)

        # Draw status text
        self._draw_status(display, state, confidence, metrics)
        
        # Draw buttons
        self._draw_buttons(display)
        
        # Draw mode indicator
        mode_text = f"Mode: {'Drag' if self.mode == SelectionMode.CLICK_DRAG else 'Click'} [M]"
        cv2.putText(display, mode_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        self.current_frame = display
        return display

    def _draw_buttons(self, frame: np.ndarray):
        """Draw all UI buttons."""
        for name, btn in self.buttons.items():
            if 'light' in name:
                # Light buttons: green when active
                color = (0, 180, 0) if btn.active else (60, 60, 60)
            elif 'focus' in name:
                # Focus buttons: blue tint
                color = (180, 100, 0)
            else:
                color = None
            btn.draw(frame, color)
        
        # Draw focus position indicator
        focus_text = f"Focus: {self.focus_position:.1f}"
        cv2.putText(frame, focus_text, (frame.shape[1] - 155, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_status(self, frame: np.ndarray, 
                    state: TrackerState,
                    confidence: float,
                    metrics: Optional[TrackingMetrics]):
        """Draw status information."""
        h, w = frame.shape[:2]
        
        # State text
        state_text = state.name
        if state == TrackerState.TRACKING:
            color = config.BBOX_COLOR_TRACKING
        elif state == TrackerState.COASTING:
            color = config.BBOX_COLOR_COASTING
        else:
            color = config.BBOX_COLOR_LOST
            
        cv2.putText(frame, f"State: {state_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence bar
        bar_x, bar_y = 10, 50
        bar_w, bar_h = 150, 15
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (bar_x + bar_w + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.TEXT_COLOR, 1)
        
        # Metrics
        if metrics is not None:
            y_offset = 80
            cv2.putText(frame, f"KLT: {metrics.klt_inliers}/{metrics.klt_total} "
                       f"({metrics.klt_quality:.2f})",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       config.TEXT_COLOR, 1)
            y_offset += 20
            cv2.putText(frame, f"NCC: {metrics.ncc_score:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       config.TEXT_COLOR, 1)
        
        # Instructions
        if state == TrackerState.IDLE:
            text = "Click to select target"
            cv2.putText(frame, text, (w // 2 - 100, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def show(self, frame: np.ndarray) -> int:
        """
        Display frame and handle key input.
        
        Returns:
            Key code pressed (or -1 if none)
        """
        if not self._window_created:
            self.create_window()
            
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def destroy(self):
        """Destroy the window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
