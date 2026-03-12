"""
Professional PyQt5 GUI for Click-to-Track system.
Modern dark theme with sliders, controls, and status display.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QFrame, QSizePolicy,
    QComboBox, QCheckBox, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

import config
from video_source import VideoSource
from tracker_cv import Tracker, BBox, TrackerState, TrackerType
from servo_controller import ServoController

# Style constants
DARK_BG = "#1a1a2e"
DARKER_BG = "#16162a"
ACCENT = "#0f9b8e"
ACCENT_HOVER = "#12b8a8"
TEXT_PRIMARY = "#e8e8e8"
TEXT_SECONDARY = "#888899"
BORDER_COLOR = "#2a2a4a"
SUCCESS = "#00d26a"
WARNING = "#ffc107"
DANGER = "#ff4757"

STYLESHEET = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}

QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
}}

QGroupBox {{
    background-color: {DARKER_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    margin-top: 12px;
    padding: 15px;
    padding-top: 25px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 8px;
    color: {ACCENT};
}}

QPushButton {{
    background-color: {DARKER_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 500;
    font-size: 12px;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {BORDER_COLOR};
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background-color: {ACCENT};
}}

QPushButton:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QPushButton#primaryButton {{
    background-color: {ACCENT};
    border: none;
    color: white;
    font-weight: 600;
}}

QPushButton#primaryButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton#dangerButton {{
    background-color: {DANGER};
    border: none;
    color: white;
}}

QSlider::groove:horizontal {{
    background: {BORDER_COLOR};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}

QComboBox {{
    background-color: {DARKER_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px 12px;
    min-width: 120px;
}}

QComboBox:hover {{
    border-color: {ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {DARKER_BG};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {ACCENT};
}}

QLabel#titleLabel {{
    font-size: 24px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.5px;
}}

QLabel#subtitleLabel {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
    letter-spacing: 0.5px;
}}

QLabel#valueLabel {{
    font-size: 13px;
    font-weight: 600;
    color: {ACCENT};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}}

QLabel#statusLabel {{
    font-size: 12px;
    padding: 8px 12px;
    border-radius: 4px;
}}

QFrame#videoFrame {{
    background-color: #000;
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
}}

QFrame#separator {{
    background-color: {BORDER_COLOR};
    max-height: 1px;
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {BORDER_COLOR};
    background-color: {DARKER_BG};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
"""


class VideoWidget(QLabel):
    """Widget to display video frames - scales with window."""
    
    clicked = pyqtSignal(int, int)  # x, y
    dragged = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2
    
    def __init__(self):
        super().__init__()
        self.setObjectName("videoFrame")
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: #000; border: 2px solid {BORDER_COLOR}; border-radius: 8px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self._drag_start = None
        self._current_frame = None
        self._frame_size = (640, 480)
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._offset_x = 0
        self._offset_y = 0
    
    def update_frame(self, frame: np.ndarray):
        """Update displayed frame."""
        if frame is None:
            return
        
        self._current_frame = frame
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        
        # Calculate display size maintaining aspect ratio
        label_w = self.width() - 4  # Account for border
        label_h = self.height() - 4
        
        if label_w <= 0 or label_h <= 0:
            return
        
        # Maintain aspect ratio
        scale = min(label_w / w, label_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Store for coordinate mapping
        self._scale_x = w / new_w if new_w > 0 else 1.0
        self._scale_y = h / new_h if new_h > 0 else 1.0
        self._offset_x = (label_w - new_w) // 2 + 2  # +2 for border
        self._offset_y = (label_h - new_h) // 2 + 2
        
        # Convert frame to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale and display
        pixmap = QPixmap.fromImage(q_img).scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)
    
    def resizeEvent(self, event):
        """Handle resize - redraw current frame at new size."""
        super().resizeEvent(event)
        if self._current_frame is not None:
            self.update_frame(self._current_frame)
    
    def _map_to_frame(self, x, y):
        """Map widget coordinates to frame coordinates."""
        frame_x = int((x - self._offset_x) * self._scale_x)
        frame_y = int((y - self._offset_y) * self._scale_y)
        # Clamp to frame bounds
        frame_x = max(0, min(self._frame_size[0] - 1, frame_x))
        frame_y = max(0, min(self._frame_size[1] - 1, frame_y))
        return frame_x, frame_y
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = (event.x(), event.y())
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start:
            x1, y1 = self._map_to_frame(*self._drag_start)
            x2, y2 = self._map_to_frame(event.x(), event.y())
            
            # Check if it's a click or drag
            if abs(x2 - x1) < 10 and abs(y2 - y1) < 10:
                self.clicked.emit(x1, y1)
            else:
                self.dragged.emit(x1, y1, x2, y2)
            
            self._drag_start = None


class ControlPanel(QWidget):
    """Side panel with all controls - scrollable and responsive."""
    
    # Signals
    light_mode_changed = pyqtSignal(str)  # "low", "mid", "high"
    focus_changed = pyqtSignal(float)
    autofocus_triggered = pyqtSignal()
    gain_changed = pyqtSignal(float)
    exposure_changed = pyqtSignal(int)
    reset_triggered = pyqtSignal()
    features_toggled = pyqtSignal(bool)
    flip_toggled = pyqtSignal(bool)
    tracker_type_changed = pyqtSignal(str)  # "CSRT", "KCF", "MOSSE"
    
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._setup_ui()
    
    def _setup_ui(self):
        # Main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area
        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {DARKER_BG};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ACCENT};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        
        # Content widget inside scroll area
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 12, 10, 12)
        
        # Title
        title = QLabel("CLICK-TO-TRACK")
        title.setObjectName("titleLabel")
        title.setStyleSheet("font-size: 18px;")
        layout.addWidget(title)
        
        subtitle = QLabel("Visual Object Tracker")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)
        
        layout.addSpacing(5)
        
        # Status section
        self.status_group = self._create_status_section()
        layout.addWidget(self.status_group)
        
        # Tracker settings
        tracker_group = self._create_tracker_section()
        layout.addWidget(tracker_group)
        
        # Camera controls
        camera_group = self._create_camera_section()
        layout.addWidget(camera_group)
        
        # Focus controls
        focus_group = self._create_focus_section()
        layout.addWidget(focus_group)
        
        # Servo controls
        servo_group = self._create_servo_section()
        layout.addWidget(servo_group)
        
        # Spacer
        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Reset button
        reset_btn = QPushButton("⟲  Reset")
        reset_btn.setObjectName("dangerButton")
        reset_btn.clicked.connect(self.reset_triggered.emit)
        layout.addWidget(reset_btn)
        
        # Keyboard shortcuts hint
        shortcuts = QLabel("Q=Quit  R=Reset  A=AF")
        shortcuts.setObjectName("subtitleLabel")
        shortcuts.setAlignment(Qt.AlignCenter)
        layout.addWidget(shortcuts)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _create_status_section(self):
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 20, 10, 10)
        
        # State indicator
        state_layout = QHBoxLayout()
        state_layout.addWidget(QLabel("State:"))
        self.state_label = QLabel("IDLE")
        self.state_label.setObjectName("valueLabel")
        state_layout.addWidget(self.state_label)
        state_layout.addStretch()
        layout.addLayout(state_layout)
        
        # Confidence
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Conf:"))
        self.conf_label = QLabel("0%")
        self.conf_label.setObjectName("valueLabel")
        conf_layout.addWidget(self.conf_label)
        conf_layout.addStretch()
        layout.addLayout(conf_layout)
        
        # Tracker type + FPS on same row
        stats_layout = QHBoxLayout()
        stats_layout.addWidget(QLabel("Type:"))
        self.type_label = QLabel("CSRT")
        self.type_label.setObjectName("valueLabel")
        stats_layout.addWidget(self.type_label)
        stats_layout.addSpacing(10)
        stats_layout.addWidget(QLabel("FPS:"))
        self.fps_label = QLabel("0")
        self.fps_label.setObjectName("valueLabel")
        stats_layout.addWidget(self.fps_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        return group

    def _create_tracker_section(self):
        group = QGroupBox("Tracker")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)
        
        # Tracker type selector
        layout.addWidget(QLabel("Algorithm:"))
        
        tracker_layout = QHBoxLayout()
        tracker_layout.setSpacing(4)
        
        self.tracker_buttons = {}
        for name, tooltip in [("CSRT", "Accurate"), ("KCF", "Fast"), ("MOSSE", "Fastest")]:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(name == "CSRT")
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, n=name: self._on_tracker_type(n))
            self.tracker_buttons[name] = btn
            tracker_layout.addWidget(btn)
        
        layout.addLayout(tracker_layout)
        
        # Info label
        self.tracker_info = QLabel("Best accuracy, handles scale")
        self.tracker_info.setObjectName("subtitleLabel")
        self.tracker_info.setWordWrap(True)
        layout.addWidget(self.tracker_info)
        
        return group

    def _on_tracker_type(self, tracker_type):
        # Update button states
        for name, btn in self.tracker_buttons.items():
            btn.setChecked(name == tracker_type)
        
        # Update info label
        info = {
            "CSRT": "Best accuracy, handles scale",
            "KCF": "Fast, good balance",
            "MOSSE": "Fastest, basic tracking"
        }
        self.tracker_info.setText(info.get(tracker_type, ""))
        
        self.tracker_type_changed.emit(tracker_type)

    def _create_servo_section(self):
        group = QGroupBox("Servo Pan")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)
        
        self.servo_status_label = QLabel("Connecting...")
        self.servo_status_label.setObjectName("subtitleLabel")
        layout.addWidget(self.servo_status_label)
        
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle:"))
        self.servo_angle_label = QLabel("0°")
        self.servo_angle_label.setObjectName("valueLabel")
        angle_layout.addWidget(self.servo_angle_label)
        angle_layout.addStretch()
        layout.addLayout(angle_layout)
        
        return group
    
    def update_servo_status(self, connected: bool):
        """Update servo connection status label."""
        if connected:
            self.servo_status_label.setText("● Connected")
            self.servo_status_label.setStyleSheet(f"color: {SUCCESS};")
        else:
            self.servo_status_label.setText("✗ Not connected")
            self.servo_status_label.setStyleSheet(f"color: {DANGER};")
    
    def update_servo_angle(self, angle: float):
        """Update displayed servo angle."""
        self.servo_angle_label.setText(f"{angle:.1f}°")
    
    def _create_camera_section(self):
        group = QGroupBox("Camera")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)
        
        # Light presets - vertical for narrow panel
        layout.addWidget(QLabel("Light Preset:"))
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        
        self.light_buttons = {}
        for name, label in [("low", "☀"), ("mid", "💡"), ("high", "🌙")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(name == "mid")
            btn.setMinimumWidth(40)
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, n=name: self._on_light_preset(n))
            self.light_buttons[name] = btn
            preset_layout.addWidget(btn)
        
        layout.addLayout(preset_layout)
        
        # Gain slider
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_value = QLabel("4.0")
        self.gain_value.setObjectName("valueLabel")
        gain_layout.addWidget(self.gain_value)
        layout.addLayout(gain_layout)
        
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(10, 160)
        self.gain_slider.setValue(40)
        self.gain_slider.valueChanged.connect(self._on_gain_change)
        layout.addWidget(self.gain_slider)
        
        # Exposure slider
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exp:"))
        self.exp_value = QLabel("33ms")
        self.exp_value.setObjectName("valueLabel")
        exp_layout.addWidget(self.exp_value)
        layout.addLayout(exp_layout)
        
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setRange(1, 200)
        self.exp_slider.setValue(33)
        self.exp_slider.valueChanged.connect(self._on_exposure_change)
        layout.addWidget(self.exp_slider)
        
        return group
    
    def _create_focus_section(self):
        group = QGroupBox("Focus")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)
        
        # Focus slider
        focus_layout = QHBoxLayout()
        focus_layout.addWidget(QLabel("Pos:"))
        self.focus_value = QLabel("1.0")
        self.focus_value.setObjectName("valueLabel")
        focus_layout.addWidget(self.focus_value)
        layout.addLayout(focus_layout)
        
        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setRange(0, 100)
        self.focus_slider.setValue(10)
        self.focus_slider.valueChanged.connect(self._on_focus_change)
        layout.addWidget(self.focus_slider)
        
        # Autofocus button
        af_btn = QPushButton("◎ Autofocus")
        af_btn.setObjectName("primaryButton")
        af_btn.clicked.connect(self.autofocus_triggered.emit)
        layout.addWidget(af_btn)
        
        return group
    
    def _create_display_section(self):
        group = QGroupBox("Display")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 20, 10, 10)
        
        self.features_check = QCheckBox("Show points")
        self.features_check.setChecked(True)
        self.features_check.toggled.connect(self.features_toggled.emit)
        layout.addWidget(self.features_check)
        
        self.flip_check = QCheckBox("Flip 180°")
        self.flip_check.setChecked(True)   # Camera mounted upside down
        self.flip_check.toggled.connect(self.flip_toggled.emit)
        layout.addWidget(self.flip_check)
        
        return group
    
    def _on_light_preset(self, preset):
        # Update button states
        for name, btn in self.light_buttons.items():
            btn.setChecked(name == preset)
        
        # Set slider values based on preset
        if preset == "low":
            self.gain_slider.setValue(10)
            self.exp_slider.setValue(20)
        elif preset == "mid":
            self.gain_slider.setValue(40)
            self.exp_slider.setValue(33)
        elif preset == "high":
            self.gain_slider.setValue(160)
            self.exp_slider.setValue(100)
        
        self.light_mode_changed.emit(preset)
    
    def _on_gain_change(self, value):
        gain = value / 10.0
        self.gain_value.setText(f"{gain:.1f}")
        self.gain_changed.emit(gain)
    
    def _on_exposure_change(self, value):
        self.exp_value.setText(f"{value}ms")
        self.exposure_changed.emit(value * 1000)  # Convert to microseconds
    
    def _on_focus_change(self, value):
        focus = value / 10.0
        self.focus_value.setText(f"{focus:.1f}")
        self.focus_changed.emit(focus)
    
    def update_status(self, state: str, confidence: float, fps: float):
        """Update status display."""
        self.state_label.setText(state)
        
        # Color code state
        if state == "TRACKING":
            self.state_label.setStyleSheet(f"color: {SUCCESS};")
        elif state == "LOST":
            self.state_label.setStyleSheet(f"color: {DANGER};")
        else:
            self.state_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        
        self.conf_label.setText(f"{confidence*100:.0f}%")
        self.fps_label.setText(f"{fps:.1f}")


class MainWindow(QMainWindow):
    """Main application window - fully responsive."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Click-to-Track")
        self.setMinimumSize(800, 500)
        self.resize(1100, 700)  # Default size
        
        # Apply dark theme
        self.setStyleSheet(STYLESHEET)
        
        # Initialize components
        self.video_source = None
        self.tracker = None
        self.servo = None
        self.flip_image = True   # Camera mounted upside down by default
        self.pending_bbox = None
        self.fps_counter = FPSCounter()
        
        self._setup_ui()
        self._setup_connections()
        self._init_tracker()
        
        # Start video timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.timer.start(40)  # ~25 FPS - avoids overwhelming the camera
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Video display (main area) - takes all available space
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 10, 0)
        
        self.video_widget = VideoWidget()
        video_layout.addWidget(self.video_widget)
        
        layout.addWidget(video_container, stretch=1)
        
        # Control panel (sidebar) - fixed width range
        self.control_panel = ControlPanel()
        layout.addWidget(self.control_panel, stretch=0)
    
    def _setup_connections(self):
        # Video widget signals
        self.video_widget.clicked.connect(self._on_click)
        self.video_widget.dragged.connect(self._on_drag)
        
        # Control panel signals
        self.control_panel.light_mode_changed.connect(self._on_light_mode)
        self.control_panel.focus_changed.connect(self._on_focus)
        self.control_panel.autofocus_triggered.connect(self._on_autofocus)
        self.control_panel.gain_changed.connect(self._on_gain)
        self.control_panel.exposure_changed.connect(self._on_exposure)
        self.control_panel.reset_triggered.connect(self._on_reset)
        self.control_panel.features_toggled.connect(self._on_features_toggle)
        self.control_panel.flip_toggled.connect(self._on_flip_toggle)
        self.control_panel.tracker_type_changed.connect(self._on_tracker_type)
    
    def _init_tracker(self):
        # Initialize video source
        self.video_source = VideoSource(config.VIDEO_SOURCE, config.TARGET_WIDTH)
        if not self.video_source.open():
            print("Failed to open video source!")
            return
        
        # Initialize tracker with default type
        self.tracker_type = TrackerType.CSRT
        self.tracker = Tracker(self.tracker_type)
        
        # Initialize servo controller and connect immediately
        simulate = getattr(config, 'SERVO_SIMULATE', True)
        invert   = getattr(config, 'SERVO_INVERT', False)
        self.servo = ServoController(simulate=simulate, invert=invert)
        ok = self.servo.connect()
        self.control_panel.update_servo_status(ok)
    
    def _process_frame(self):
        if self.video_source is None or self.tracker is None:
            return
        
        frame = self.video_source.read()
        if frame is None:
            return
        
        # Flip frame if enabled (camera mounted upside down)
        if self.flip_image:
            frame.image = cv2.rotate(frame.image, cv2.ROTATE_180)
            frame.gray  = cv2.rotate(frame.gray,  cv2.ROTATE_180)
        
        # Handle pending bbox selection
        if self.pending_bbox is not None:
            print(f"Tracker init: frame shape={frame.image.shape} bbox={self.pending_bbox}")
            self.tracker.initialize(frame.image, self.pending_bbox, frame.timestamp)
            self.pending_bbox = None
        
        # Update tracker
        bbox = None
        if self.tracker.get_state() != TrackerState.IDLE:
            bbox = self.tracker.update(frame.image, frame.timestamp)
        
        # Draw overlay
        display = self._draw_overlay(frame.image, bbox)
        
        # Update video widget
        self.video_widget.update_frame(display)
        
        # Update status
        fps = self.fps_counter.update()
        self.control_panel.update_status(
            self.tracker.get_state().name,
            self.tracker.get_confidence(),
            fps
        )
        
        # Drive servo to keep target centred
        if bbox is not None and self.servo is not None:
            self.servo.update(target_x=bbox.cx, frame_width=frame.image.shape[1])
            self.control_panel.update_servo_angle(self.servo.get_angle())
    
    def _draw_overlay(self, frame: np.ndarray, bbox) -> np.ndarray:
        """Draw tracking overlay on frame."""
        display = frame.copy()
        state = self.tracker.get_state()
        
        if bbox is not None and state != TrackerState.IDLE:
            # Choose color based on state
            if state == TrackerState.TRACKING:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red for lost
            
            # Draw bbox
            corners = bbox.to_corners()
            cv2.rectangle(display, corners[0], corners[1], color, 2)
            
            # Draw crosshair at center
            cx, cy = int(bbox.cx), int(bbox.cy)
            size = 15
            cv2.line(display, (cx - size, cy), (cx + size, cy), color, 2)
            cv2.line(display, (cx, cy - size), (cx, cy + size), color, 2)
            
            # Draw confidence bar above bbox
            conf = self.tracker.get_confidence()
            bar_w = int(bbox.w * conf)
            bar_y = corners[0][1] - 8
            if bar_y > 5:
                cv2.rectangle(display, (corners[0][0], bar_y - 4), 
                             (corners[0][0] + int(bbox.w), bar_y), (50, 50, 50), -1)
                cv2.rectangle(display, (corners[0][0], bar_y - 4), 
                             (corners[0][0] + bar_w, bar_y), color, -1)
        
        return display
    
    def _on_click(self, x, y):
        """Handle click to select target."""
        self.pending_bbox = BBox(
            cx=float(x),
            cy=float(y),
            w=float(config.DEFAULT_BBOX_WIDTH),
            h=float(config.DEFAULT_BBOX_HEIGHT)
        )
    
    def _on_drag(self, x1, y1, x2, y2):
        """Handle drag to select target."""
        self.pending_bbox = BBox.from_corners(x1, y1, x2, y2)
    
    def _on_light_mode(self, mode):
        if self.video_source is None:
            return
        
        if mode == "low":
            self.video_source.set_camera_settings(True, 20000, 1.0, 0.0)
        elif mode == "mid":
            self.video_source.set_camera_settings(True, 33000, 4.0, 0.1)
        elif mode == "high":
            self.video_source.set_camera_settings(False, 100000, 16.0, 0.3)
    
    def _on_focus(self, position):
        if self.video_source:
            self.video_source.set_manual_focus(position)
    
    def _on_autofocus(self):
        if self.video_source:
            self.video_source.trigger_autofocus()
    
    def _on_gain(self, gain):
        if self.video_source:
            self.video_source.set_camera_settings(
                analog_gain=gain
            )
    
    def _on_exposure(self, exposure_us):
        if self.video_source:
            self.video_source.set_camera_settings(
                auto_exposure=False,
                exposure_time=exposure_us
            )
    
    def _on_reset(self):
        if self.tracker:
            self.tracker.reset()
    
    def _on_features_toggle(self, show):
        config.DRAW_FEATURES = show
    
    def _on_flip_toggle(self, flipped):
        self.flip_image = flipped
    
    def _on_tracker_type(self, type_name):
        """Change tracker algorithm."""
        self.tracker_type = TrackerType[type_name]
        self.tracker = Tracker(self.tracker_type)
        self.control_panel.type_label.setText(type_name)
        print(f"Switched to {type_name} tracker")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_R:
            self._on_reset()
        elif event.key() == Qt.Key_A:
            self._on_autofocus()
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.timer.stop()
        if self.servo:
            self.servo.disconnect()
        if self.video_source:
            self.video_source.release()
        event.accept()


class FPSCounter:
    """Simple FPS counter."""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self):
        import time
        now = time.time()
        self.timestamps.append(now)
        
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
        
        if len(self.timestamps) < 2:
            return 0.0
        
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / duration


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Click-to-Track")
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
