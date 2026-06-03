"""
detection.py — YOLOv8 Object Detector (Person-Focused)
=======================================================
- Detects all 80 COCO classes
- Person gets EXTRA info: position in frame, size estimate, activity guess
- Thread-safe current_counts for voice assistant
- Rich HUD with person counter badge
"""

import cv2
import numpy as np
import time
import os
import threading
from datetime import datetime
from ultralytics import YOLO

LABEL_COLOURS = {
    "person":       (0,   230,   0),
    "car":          (0,   140, 255),
    "truck":        (0,   100, 200),
    "bus":          (0,    80, 180),
    "motorcycle":   (0,   200, 200),
    "bicycle":      (50,  200, 255),
    "bottle":       (0,     0, 255),
    "cup":          (60,    0, 220),
    "cell phone":   (255,   0, 230),
    "laptop":       (255, 220,   0),
    "tv":           (200, 180,   0),
    "chair":        (0,   255, 255),
    "couch":        (0,   200, 200),
    "dining table": (0,   180, 180),
    "dog":          (160, 100, 255),
    "cat":          (200, 130, 255),
    "book":         (200, 255, 100),
    "backpack":     (120, 255, 180),
    "handbag":      (100, 220, 160),
    "mouse":        (255, 200, 100),
    "keyboard":     (255, 180,  80),
    "remote":       (255, 160,  60),
    "clock":        (220, 220, 220),
    "default":      (200, 200, 200),
}

SPOKEN_SYNONYMS = {
    "people": "person",    "human": "person",     "humans": "person",
    "man": "person",       "woman": "person",      "men": "person",
    "women": "person",     "boy": "person",        "girl": "person",
    "phone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone",
    "iphone": "cell phone","android": "cell phone",
    "vehicle": "car",      "vehicles": "car",
    "computer": "laptop",  "notebook": "laptop",   "macbook": "laptop",
    "sofa": "couch",       "telly": "tv",          "television": "tv",
    "mug": "cup",          "glass": "cup",
    "bag": "backpack",     "purse": "handbag",
    "bike": "bicycle",     "motorbike": "motorcycle",
}

IMPORTANT = {"person", "car", "laptop", "cell phone", "truck", "bus"}


class PersonInfo:
    """Stores per-person analysis: position, size, activity hint and age estimate."""
    def __init__(self, x1, y1, x2, y2, frame_w, frame_h, nearby_objects):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        box_w  = x2 - x1
        box_h  = y2 - y1
        centre_x = (x1 + x2) / 2

        # Position in frame
        if centre_x < frame_w * 0.33:
            self.position = "on the left"
        elif centre_x > frame_w * 0.66:
            self.position = "on the right"
        else:
            self.position = "in the centre"

        # Size → proximity estimate
        area_ratio = (box_w * box_h) / (frame_w * frame_h)
        if area_ratio > 0.35:
            self.proximity = "very close"
        elif area_ratio > 0.15:
            self.proximity = "nearby"
        elif area_ratio > 0.05:
            self.proximity = "at mid distance"
        else:
            self.proximity = "far away"

        # Activity guess from nearby objects
        if "cell phone" in nearby_objects:
            self.activity = "using a phone"
        elif "laptop" in nearby_objects:
            self.activity = "working on a laptop"
        elif "book" in nearby_objects:
            self.activity = "reading"
        elif "cup" in nearby_objects:
            self.activity = "holding a cup"
        elif "tv" in nearby_objects:
            self.activity = "watching TV"
        elif "bicycle" in nearby_objects or "motorcycle" in nearby_objects:
            self.activity = "near a vehicle"
        else:
            self.activity = None

        # Age estimate from apparent height ratio
        height_ratio = box_h / frame_h
        if height_ratio >= 0.60:
            self.age_group = "an adult"
        elif height_ratio >= 0.45:
            self.age_group = "a teenager"
        elif height_ratio >= 0.28:
            self.age_group = "a child"
        else:
            self.age_group = "a young child"


class ObjectDetector:
    """
    Thread-safe YOLOv8 wrapper with person-focused analysis.
    """

    def __init__(
        self,
        model_path           = "yolov8n.pt",
        confidence_threshold = 0.35,
        screenshot_dir       = "screenshots",
        screenshot_interval  = 30,
    ):
        self.confidence_threshold = confidence_threshold
        self.screenshot_dir       = screenshot_dir
        self.screenshot_interval  = screenshot_interval

        self._fps_start  = time.time()
        self._fps_frames = 0
        self.current_fps = 0.0
        self._last_shot  = 0.0

        self._lock           = threading.Lock()
        self.current_counts  = {}
        self.scene_objects   = []
        self.person_details  = []   # list[PersonInfo]

        print(f"[Detector] Loading {model_path} …")
        self.model       = YOLO(model_path)
        self.class_names = self.model.names
        print(f"[Detector] Ready — {len(self.class_names)} COCO classes.")
        os.makedirs(screenshot_dir, exist_ok=True)

    # ── Main API ──────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        self._tick_fps()
        h, w = frame.shape[:2]

        results = self.model.predict(
            frame, conf=self.confidence_threshold,
            verbose=False, stream=False,
        )

        counts       = {}
        detections   = []
        person_boxes = []
        other_labels = set()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cid   = int(box.cls[0].item())
                label = self.class_names.get(cid, f"obj{cid}")
                conf  = float(box.conf[0].item())
                x1,y1,x2,y2 = (int(v) for v in box.xyxy[0].tolist())
                colour = LABEL_COLOURS.get(label, LABEL_COLOURS["default"])
                detections.append((x1, y1, x2, y2, label, conf, colour))
                counts[label] = counts.get(label, 0) + 1
                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))
                else:
                    other_labels.add(label)

        # Build person analysis
        persons = [
            PersonInfo(x1, y1, x2, y2, w, h, other_labels)
            for (x1, y1, x2, y2) in person_boxes
        ]

        with self._lock:
            self.current_counts = dict(counts)
            self.scene_objects  = list(counts.keys())
            self.person_details = persons

        # Draw
        out = frame.copy()
        for det in detections:
            if det[4] == "person":
                out = self._draw_person_box(out, *det)
            else:
                out = self._draw_box(out, *det)
        out = self._draw_hud(out, counts)

        if self._should_screenshot(counts):
            self._save_screenshot(out)

        return out, counts

    # ── Scene descriptions ────────────────────────────────────────────────────

    def get_scene_description(self) -> str:
        """
        Returns a natural spoken sentence.
        e.g. "I can see a person and a cell phone in the frame."
        """
        with self._lock:
            counts  = dict(self.current_counts)
            persons = list(self.person_details)

        if not counts:
            return "I don't see any objects in the frame right now."

        parts = []
        for label in sorted(counts):
            n = counts[label]
            if n == 1:
                art = "an" if label[0].lower() in "aeiou" else "a"
                parts.append(f"{art} {label}")
            else:
                parts.append(f"{n} {_pluralize(label)}")

        if len(parts) == 1:
            base = f"I can see {parts[0]} in the frame."
        elif len(parts) == 2:
            base = f"I can see {parts[0]} and {parts[1]} in the frame."
        else:
            base = f"I can see {', '.join(parts[:-1])}, and {parts[-1]} in the frame."

        return base

    def get_detailed_description(self) -> str:
        """
        Full description with person analysis.
        e.g. "I can see a person and a cell phone in the frame.
               The person is in the centre of the frame, nearby,
               and appears to be using a phone."
        """
        with self._lock:
            counts  = dict(self.current_counts)
            persons = list(self.person_details)

        if not counts:
            return "The frame is empty. I don't see any objects right now."

        base    = self.get_scene_description()
        context = _build_context(counts, persons)
        return (base + "  " + context).strip() if context else base

    def get_person_description(self) -> str:
        """
        Dedicated person report.
        e.g. "I can see 2 persons. Person 1 is on the left, nearby,
               and appears to be using a phone. Person 2 is on the right,
               at mid distance."
        """
        with self._lock:
            counts  = dict(self.current_counts)
            persons = list(self.person_details)

        n = counts.get("person", 0)
        if n == 0:
            return "I don't see any person in the frame right now."

        if n == 1:
            p   = persons[0] if persons else None
            msg = "I can see 1 person in the frame."
            if p:
                msg += f" The person is {p.position}, {p.proximity}, and appears to be {p.age_group}."
                if p.activity:
                    msg += f" They appear to be {p.activity}."
        else:
            msg = f"I can see {n} persons in the frame."
            details = []
            for i, p in enumerate(persons[:4], 1):   # cap at 4
                d = f"Person {i} is {p.position}, {p.proximity}, and appears to be {p.age_group}"
                if p.activity:
                    d += f", and appears to be {p.activity}"
                details.append(d + ".")
            if details:
                msg += " " + " ".join(details)

        return msg

    def get_count(self, spoken_name: str) -> int:
        key = spoken_name.strip().lower()
        key = SPOKEN_SYNONYMS.get(key, key)
        with self._lock:
            return self.current_counts.get(key, 0)

    def set_confidence(self, v: float):
        self.confidence_threshold = max(0.05, min(0.95, float(v)))
        print(f"[Detector] Confidence → {self.confidence_threshold:.0%}")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_person_box(self, frame, x1, y1, x2, y2, label, conf, colour):
        """Person gets a thicker box + activity and age labels if available."""
        with self._lock:
            persons = list(self.person_details)

        # Find matching PersonInfo by box coordinates
        activity = None
        age_group = None
        for p in persons:
            if abs(p.x1 - x1) < 10 and abs(p.y1 - y1) < 10:
                activity = p.activity
                age_group = p.age_group
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)   # thicker

        text  = f"person  {conf:.0%}"
        if age_group:
            age_label = age_group.replace("an ", "").replace("a ", "").strip()
            text += f"  [{age_label}]"
        if activity:
            text += f"  [{activity}]"

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.56
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        ty = max(y1 - 4, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 2),
                      colour, cv2.FILLED)
        cv2.putText(frame, text, (x1 + 4, ty - 2),
                    font, scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Small filled circle at the head area
        head_y = max(y1 - 14, 14)
        cx     = (x1 + x2) // 2
        cv2.circle(frame, (cx, head_y), 10, colour, cv2.FILLED)
        cv2.circle(frame, (cx, head_y), 10, (0, 0, 0), 1)

        return frame

    def _draw_box(self, frame, x1, y1, x2, y2, label, conf, colour):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        text  = f"{label}  {conf:.0%}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        ty = max(y1 - 4, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 2),
                      colour, cv2.FILLED)
        cv2.putText(frame, text, (x1 + 4, ty - 2),
                    font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_hud(self, frame, counts):
        h, w  = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        GOLD  = (0, 215, 255)
        GREEN = (0, 230, 0)
        WHITE = (255, 255, 255)

        # Title bar top
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 32), (10, 10, 10), cv2.FILLED)
        frame = cv2.addWeighted(ov, 0.75, frame, 0.25, 0)
        cv2.putText(frame, "AI OBJECT DETECTION  |  YOLOv8n",
                    (8, 22), font, 0.54, GOLD, 1, cv2.LINE_AA)

        # Person badge top-right (prominent)
        n_persons = counts.get("person", 0)
        badge_txt = f"PERSONS: {n_persons}"
        badge_col = (0, 200, 0) if n_persons > 0 else (60, 60, 60)
        (bw, bh), _ = cv2.getTextSize(badge_txt, font, 0.56, 2)
        bx = w - bw - 20
        by = 22
        cv2.rectangle(frame, (bx - 8, by - bh - 5),
                      (bx + bw + 8, by + 5), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, badge_txt, (bx, by),
                    font, 0.56, badge_col, 2, cv2.LINE_AA)

        # Bottom strip
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h - 38), (w, h), (10, 10, 10), cv2.FILLED)
        frame = cv2.addWeighted(ov2, 0.75, frame, 0.25, 0)

        fps_col = GREEN if self.current_fps > 20 else (0, 140, 255)
        cv2.putText(frame, f"FPS {self.current_fps:.0f}",
                    (10, h - 12), font, 0.52, fps_col, 1, cv2.LINE_AA)

        if counts:
            obj_str = "  |  ".join(
                f"{lbl}: {cnt}" for lbl, cnt in sorted(counts.items())
            )
        else:
            obj_str = "No objects detected"
        (ow, _), _ = cv2.getTextSize(obj_str, font, 0.47, 1)
        cv2.putText(frame, obj_str,
                    (max(90, (w - ow) // 2), h - 12),
                    font, 0.47, GOLD, 1, cv2.LINE_AA)

        tip = f"Conf {self.confidence_threshold:.0%}   Q=quit  S=snap  V=voice  SPACE=pause"
        (cw, _), _ = cv2.getTextSize(tip, font, 0.38, 1)
        cv2.putText(frame, tip, (w - cw - 8, h - 12),
                    font, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tick_fps(self):
        self._fps_frames += 1
        if self._fps_frames >= 20:
            elapsed = time.time() - self._fps_start
            self.current_fps = self._fps_frames / elapsed if elapsed else 0
            self._fps_frames = 0
            self._fps_start  = time.time()

    def _should_screenshot(self, counts):
        now = time.time()
        return (
            any(k in IMPORTANT for k in counts)
            and (now - self._last_shot) >= self.screenshot_interval
        )

    def _save_screenshot(self, frame):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.screenshot_dir, f"detect_{ts}.jpg")
        cv2.imwrite(path, frame)
        self._last_shot = time.time()
        print(f"[Detector] Auto-screenshot → {path}")


# ── Module helpers ────────────────────────────────────────────────────────────

def _pluralize(word: str) -> str:
    IRR = {"person": "persons", "mouse": "mice",
           "sheep": "sheep",    "fish":  "fish"}
    if word in IRR:
        return IRR[word]
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _build_context(counts: dict, persons: list) -> str:
    """
    Build a contextual sentence from detected objects + person analysis.
    Called by get_detailed_description().
    """
    has = counts.get

    if has("person") and has("cell phone"):
        p  = has("person")
        ph = has("cell phone")
        who = "a person" if p == 1 else f"{p} persons"
        phn = "a cell phone" if ph == 1 else f"{ph} cell phones"
        verb = "is" if p == 1 else "are"
        return f"It looks like {who} {verb} holding {phn}."

    if has("person") and has("laptop"):
        p    = has("person")
        who  = "someone" if p == 1 else f"{p} people"
        verb = "is" if p == 1 else "are"
        return f"It looks like {who} {verb} working on a laptop."

    if has("person") and has("tv"):
        p    = has("person")
        who  = "someone" if p == 1 else f"{p} people"
        verb = "is" if p == 1 else "are"
        return f"It looks like {who} {verb} watching TV."

    if has("person") and has("book"):
        p    = has("person")
        who  = "someone" if p == 1 else f"{p} people"
        verb = "is" if p == 1 else "are"
        return f"It looks like {who} {verb} reading a book."

    if has("person") and has("cup"):
        p    = has("person")
        who  = "someone" if p == 1 else f"{p} people"
        verb = "is" if p == 1 else "are"
        return f"It looks like {who} {verb} holding a cup."

    if has("person", 0) > 2:
        return f"There are {has('person')} persons visible in the frame."

    if has("person") and len(counts) == 1:
        p = has("person")
        return ("There is a person standing in the frame."
                if p == 1 else
                f"There are {p} persons standing in the frame.")

    cars = has("car", 0) + has("truck", 0) + has("bus", 0)
    if cars >= 3:
        return "There is significant vehicle traffic visible."
    if cars >= 1 and not has("person"):
        return "I can see vehicles but no people in the frame."

    if has("bottle") and not has("person"):
        return "I can see a bottle but no person is holding it."

    return ""
