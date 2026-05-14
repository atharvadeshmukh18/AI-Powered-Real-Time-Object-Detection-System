
import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO


# ── Configuration ────────────────────────────────────────────────────────────

# Objects this system focuses on (subset of COCO 80-class dataset)
FOCUS_CLASSES = {
    "person", "car", "bottle", "cell phone",
    "chair", "laptop", "dog", "cat",
    "bicycle", "motorcycle", "bus", "truck",
    "cup", "book", "backpack",
}

# Auto-screenshot trigger: save frame when any of these appear
IMPORTANT_OBJECTS = {"person", "car", "laptop", "cell phone"}

# Colour palette for bounding boxes (BGR format for OpenCV)
COLOURS = {
    "person":      (0,   255,  0),    # green
    "car":         (255, 165,  0),    # orange
    "bottle":      (0,   0,   255),   # red
    "cell phone":  (255, 0,   255),   # magenta
    "chair":       (0,   255, 255),   # cyan
    "laptop":      (255, 255,  0),    # yellow
    "default":     (200, 200, 200),   # light grey
}


# ── ObjectDetector class ─────────────────────────────────────────────────────

class ObjectDetector:
    """
    Wraps YOLOv8 inference with drawing helpers, FPS tracking,
    object counting, and automatic screenshot saving.

    Usage:
        detector = ObjectDetector()
        annotated_frame, counts = detector.process_frame(frame)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        screenshot_dir: str = "screenshots",
        screenshot_interval: int = 30,   # seconds between auto-screenshots
    ):
        """
        Parameters
        ----------
        model_path           : Path to YOLOv8 weights (.pt file).
                               If not found locally, Ultralytics downloads it.
        confidence_threshold : Minimum confidence to keep a detection (0–1).
        screenshot_dir       : Directory where auto-screenshots are saved.
        screenshot_interval  : Minimum seconds between automatic screenshots.
        """
        self.confidence_threshold = confidence_threshold
        self.screenshot_dir = screenshot_dir
        self.screenshot_interval = screenshot_interval

        # FPS tracking state
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self.current_fps = 0.0

        # Screenshot cooldown
        self._last_screenshot_time = 0.0

        # Latest detection results (shared with voice assistant)
        self.current_counts: dict[str, int] = {}
        self.scene_objects: list[str] = []

        # ── Load YOLOv8 model ──────────────────────────────────────────────
        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)          # downloads automatically if absent
        self.class_names = self.model.names    # {0: 'person', 1: 'bicycle', ...}
        print(f"[Detector] Model loaded — {len(self.class_names)} classes available.")

        os.makedirs(screenshot_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """
        Run YOLOv8 detection on one frame.

        Returns
        -------
        annotated_frame : Frame with bounding boxes, labels, FPS overlay.
        object_counts   : {class_name: count} for all detected objects.
        """
        self._update_fps()

        # ── YOLOv8 inference ───────────────────────────────────────────────
        # verbose=False suppresses per-frame console output
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        # ── Parse detections ───────────────────────────────────────────────
        object_counts: dict[str, int] = {}
        detections = []  # list of (x1,y1,x2,y2,label,conf,colour)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id   = int(box.cls[0])
                label      = self.class_names[class_id]
                confidence = float(box.conf[0])

                # Only keep classes we care about OR all if focus list is empty
                if FOCUS_CLASSES and label not in FOCUS_CLASSES:
                    continue

                # Bounding-box pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                colour = COLOURS.get(label, COLOURS["default"])

                detections.append((x1, y1, x2, y2, label, confidence, colour))
                object_counts[label] = object_counts.get(label, 0) + 1

        # ── Update shared state (for voice assistant) ──────────────────────
        self.current_counts = object_counts
        self.scene_objects  = list(object_counts.keys())

        # ── Draw everything on a copy of the frame ─────────────────────────
        annotated = frame.copy()
        for det in detections:
            annotated = self._draw_detection(annotated, *det)

        annotated = self._draw_hud(annotated, object_counts)

        # ── Auto-screenshot trigger ────────────────────────────────────────
        if self._should_screenshot(object_counts):
            self._save_screenshot(annotated)

        return annotated, object_counts

    def get_scene_description(self) -> str:
        """
        Build a natural-language description of the current scene.
        Example: "I can see 2 persons, 1 laptop, and 1 bottle."
        """
        counts = self.current_counts
        if not counts:
            return "I don't see any recognizable objects right now."

        parts = []
        for label, count in sorted(counts.items()):
            noun = label if count == 1 else self._pluralize(label)
            parts.append(f"{count} {noun}")

        if len(parts) == 1:
            return f"I can see {parts[0]}."
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}."
        else:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}."

    def get_count(self, object_name: str) -> int:
        """Return the count of a specific object in the current frame."""
        # Handle common spoken variants
        name_map = {
            "phone":  "cell phone",
            "mobile": "cell phone",
            "people": "person",
            "human":  "person",
            "humans": "person",
        }
        key = name_map.get(object_name.lower(), object_name.lower())
        return self.current_counts.get(key, 0)

    def set_confidence(self, threshold: float) -> None:
        """Dynamically adjust the confidence threshold."""
        self.confidence_threshold = max(0.05, min(0.95, threshold))
        print(f"[Detector] Confidence threshold → {self.confidence_threshold:.2f}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _draw_detection(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        label: str, confidence: float,
        colour: tuple,
    ) -> np.ndarray:
        """Draw one bounding box with a filled label strip."""
        thickness = 2

        # Bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        # Label text: "person 0.92"
        text = f"{label} {confidence:.2f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thick = 1

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

        # Filled rectangle behind label for readability
        label_y = max(y1 - th - 8, 0)
        cv2.rectangle(
            frame,
            (x1, label_y),
            (x1 + tw + 6, label_y + th + 8),
            colour,
            cv2.FILLED,
        )

        # White text on coloured background
        cv2.putText(
            frame, text,
            (x1 + 3, label_y + th + 2),
            font, font_scale, (255, 255, 255), font_thick,
            cv2.LINE_AA,
        )
        return frame

    def _draw_hud(self, frame: np.ndarray, counts: dict[str, int]) -> np.ndarray:
        """Draw the heads-up display: FPS, object summary, controls hint."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent dark sidebar on the left
        cv2.rectangle(overlay, (0, 0), (220, h), (20, 20, 20), cv2.FILLED)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        small = 0.48
        mid   = 0.55
        white = (255, 255, 255)
        green = (80, 255, 80)
        gold  = (0, 215, 255)

        # Title
        cv2.putText(frame, "AI DETECTOR", (8, 28), font, 0.65, gold, 2, cv2.LINE_AA)
        cv2.line(frame, (8, 36), (212, 36), (80, 80, 80), 1)

        # FPS
        fps_colour = (0, 255, 0) if self.current_fps > 20 else (0, 165, 255)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (8, 58), font, mid, fps_colour, 1, cv2.LINE_AA)

        # Object counts
        cv2.putText(frame, "Objects:", (8, 82), font, small, gold, 1, cv2.LINE_AA)
        y = 102
        if counts:
            for obj, cnt in sorted(counts.items()):
                cv2.putText(frame, f"  {obj}: {cnt}", (8, y), font, small, white, 1, cv2.LINE_AA)
                y += 20
        else:
            cv2.putText(frame, "  none detected", (8, y), font, small, (150, 150, 150), 1, cv2.LINE_AA)
            y += 20

        # Keyboard shortcuts at the bottom
        hints = ["Q - quit", "S - screenshot", "+/- conf", "V - voice"]
        y_hint = h - len(hints) * 20 - 10
        cv2.line(frame, (8, y_hint - 8), (212, y_hint - 8), (80, 80, 80), 1)
        for hint in hints:
            cv2.putText(frame, hint, (8, y_hint), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
            y_hint += 20

        return frame

    def _update_fps(self) -> None:
        """Compute rolling FPS every 30 frames."""
        self._fps_frame_count += 1
        if self._fps_frame_count >= 30:
            elapsed = time.time() - self._fps_start_time
            self.current_fps = self._fps_frame_count / elapsed if elapsed > 0 else 0
            self._fps_frame_count = 0
            self._fps_start_time = time.time()

    def _should_screenshot(self, counts: dict[str, int]) -> bool:
        """Return True if an important object is detected and cooldown has passed."""
        now = time.time()
        has_important = any(obj in IMPORTANT_OBJECTS for obj in counts)
        cooldown_ok   = (now - self._last_screenshot_time) >= self.screenshot_interval
        return has_important and cooldown_ok

    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save a timestamped screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(self.screenshot_dir, f"detect_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self._last_screenshot_time = time.time()
        print(f"[Detector] Screenshot saved → {filename}")

    @staticmethod
    def _pluralize(word: str) -> str:
        """Very simple English pluraliser."""
        if word.endswith("s"):
            return word
        if word.endswith("son"):   # person → persons (not peopleson)
            return word + "s"
        return word + "s"
