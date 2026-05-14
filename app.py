import cv2
import time
import argparse
import sys
import os
from datetime import datetime

from detection import ObjectDetector
from assistant import VoiceAssistant


# ── Default configuration ─────────────────────────────────────────────────────
DEFAULT_MODEL      = "yolov8n.pt"    # nano model (fastest)
DEFAULT_CONFIDENCE = 0.40
DEFAULT_SOURCE     = 0               # 0 = default webcam
WINDOW_NAME        = "AI Object Detection System"
OUTPUT_DIR         = "outputs"


# ── CLI argument parser ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Real-Time Object Detection with Voice Assistant"
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help="Video source: webcam index (0,1,…) or path to video file"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to YOLOv8 weights (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE,
        help="Confidence threshold 0–1 (default: 0.40)"
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice assistant"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture height in pixels"
    )
    return parser.parse_args()


# ── Video source helper ───────────────────────────────────────────────────────
def open_source(source) -> cv2.VideoCapture:
    """
    Open a webcam or video file.
    Tries to convert the source to int (for webcam index) first.
    """
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = str(source)   # treat as file path

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[App] ERROR: Could not open video source: {source}")
        sys.exit(1)
    return cap


# ── Main application ──────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    print("=" * 60)
    print("  AI Real-Time Object Detection System")
    print(f"  Model      : {args.model}")
    print(f"  Source     : {args.source}")
    print(f"  Confidence : {args.conf}")
    print(f"  Voice      : {'disabled' if args.no_voice else 'enabled'}")
    print("=" * 60)

    # ── 1. Initialise detector ────────────────────────────────────────────────
    detector = ObjectDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        screenshot_dir="screenshots",
        screenshot_interval=30,
    )

    # ── 2. Initialise voice assistant ─────────────────────────────────────────
    detection_active = [True]    # mutable flag shared with callbacks

    def on_start():
        detection_active[0] = True
        print("[App] Detection STARTED via voice.")

    def on_stop():
        detection_active[0] = False
        print("[App] Detection PAUSED via voice.")

    def on_screenshot():
        # Force a screenshot by resetting the cooldown
        detector._last_screenshot_time = 0

    assistant = None
    if not args.no_voice:
        assistant = VoiceAssistant(
            detector=detector,
            on_start=on_start,
            on_stop=on_stop,
            on_screenshot=on_screenshot,
        )
        assistant.start()
    else:
        print("[App] Voice assistant disabled.")

    # ── 3. Open video source ──────────────────────────────────────────────────
    cap = open_source(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[App] Capture resolution: {actual_w} × {actual_h}")

    # ── 4. OpenCV window setup ─────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)

    # Confidence trackbar
    def on_conf_change(val):
        detector.set_confidence(val / 100.0)

    cv2.createTrackbar("Confidence %", WINDOW_NAME, int(args.conf * 100), 95, on_conf_change)

    print("[App] Window ready. Press Q to quit, SPACE to toggle detection.")

    # ── 5. Main loop ──────────────────────────────────────────────────────────
    voice_active = not args.no_voice
    frame_count  = 0
    paused_frame = None          # shown when detection is paused

    while True:
        ret, frame = cap.read()
        if not ret:
            # For video files: loop back; for webcam: error
            if isinstance(args.source, str) and os.path.isfile(str(args.source)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[App] Failed to grab frame. Exiting.")
            break

        frame_count += 1

        # ── Detect or show paused overlay ─────────────────────────────────
        if detection_active[0]:
            display_frame, counts = detector.process_frame(frame)
            paused_frame = display_frame.copy()   # cache for pause
        else:
            # Detection paused — show last frame with "PAUSED" banner
            display_frame = _draw_paused_overlay(
                paused_frame if paused_frame is not None else frame
            )

        # ── Show frame ─────────────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, display_frame)

        # ── Keyboard handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:        # Q or ESC → quit
            print("[App] Quitting...")
            break

        elif key == ord("s"):                    # S → manual screenshot
            _save_manual_screenshot(display_frame)

        elif key == ord(" "):                    # SPACE → toggle detection
            detection_active[0] = not detection_active[0]
            state = "STARTED" if detection_active[0] else "PAUSED"
            print(f"[App] Detection {state}.")

        elif key == ord("v") and assistant:      # V → toggle voice
            if voice_active:
                assistant.stop()
                voice_active = False
                print("[App] Voice assistant OFF.")
            else:
                assistant.start()
                voice_active = True
                print("[App] Voice assistant ON.")

        elif key in (ord("+"), ord("=")):        # + → raise confidence
            detector.set_confidence(detector.confidence_threshold + 0.05)
            cv2.setTrackbarPos("Confidence %", WINDOW_NAME,
                               int(detector.confidence_threshold * 100))

        elif key == ord("-"):                    # - → lower confidence
            detector.set_confidence(detector.confidence_threshold - 0.05)
            cv2.setTrackbarPos("Confidence %", WINDOW_NAME,
                               int(detector.confidence_threshold * 100))

        elif key == ord("0"):                    # 0 → reset confidence
            detector.set_confidence(DEFAULT_CONFIDENCE)
            cv2.setTrackbarPos("Confidence %", WINDOW_NAME,
                               int(DEFAULT_CONFIDENCE * 100))

        # Window closed by user (X button)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    if assistant:
        assistant.stop()
    print("[App] Resources released. Goodbye!")


# ── Helper functions ──────────────────────────────────────────────────────────

def _draw_paused_overlay(frame) -> None:
    """Dim the frame and print a PAUSED banner."""
    import numpy as np
    overlay = frame.copy()
    # Darken entire frame
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    h, w = frame.shape[:2]
    text = "DETECTION PAUSED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thick = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cx = (w - tw) // 2
    cy = (h + th) // 2

    # Shadow
    cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, text, (cx, cy), font, scale, (0, 200, 255), thick, cv2.LINE_AA)

    sub = "Press SPACE to resume"
    (sw, _), _ = cv2.getTextSize(sub, font, 0.6, 1)
    cv2.putText(frame, sub, ((w - sw) // 2, cy + 50), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def _save_manual_screenshot(frame) -> None:
    """Save a manually triggered screenshot."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("screenshots", f"manual_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[App] Manual screenshot → {path}")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
