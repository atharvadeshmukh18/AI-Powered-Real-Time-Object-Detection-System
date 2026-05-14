

from dataclasses import dataclass, field
from typing import Set, Dict, Tuple


@dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    model_path: str = "yolov8n.pt"
    confidence: float = 0.40          # detection confidence threshold
    iou_threshold: float = 0.45       # NMS IoU threshold

    # ── Capture ────────────────────────────────────────────────────────────
    source: int = 0                   # 0 = default webcam
    frame_width: int = 1280
    frame_height: int = 720

    # ── Focus classes (empty = detect all 80 COCO classes) ────────────────
    focus_classes: Set[str] = field(default_factory=lambda: {
        "person", "car", "bottle", "cell phone",
        "chair", "laptop", "dog", "cat",
        "bicycle", "bus", "truck", "cup", "book",
    })

    # ── Important objects → auto screenshot ───────────────────────────────
    important_objects: Set[str] = field(default_factory=lambda: {
        "person", "car", "laptop", "cell phone",
    })

    # ── Colours (BGR) ─────────────────────────────────────────────────────
    colours: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "person":     (0,   255,   0),
        "car":        (255, 165,   0),
        "bottle":     (0,   0,   255),
        "cell phone": (255, 0,   255),
        "chair":      (0,   255, 255),
        "laptop":     (255, 255,   0),
        "default":    (200, 200, 200),
    })

    # ── Directories ────────────────────────────────────────────────────────
    screenshot_dir: str = "screenshots"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # ── Screenshot ─────────────────────────────────────────────────────────
    auto_screenshot: bool = True
    screenshot_interval: int = 30    # seconds between auto-screenshots

    # ── Voice Assistant ────────────────────────────────────────────────────
    voice_enabled: bool = True
    tts_rate: int = 165              # words per minute
    speech_language: str = "en-US"
    mic_energy_threshold: int = 300
    mic_pause_threshold: float = 0.8

    # ── Display ────────────────────────────────────────────────────────────
    window_name: str = "AI Object Detection System"
    hud_sidebar_width: int = 220


# Singleton — import and use directly
CFG = Config()
