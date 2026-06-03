"""
app.py — Main Entry Point (Person-Focused)
==========================================
Keyboard controls
-----------------
  Q / ESC   quit
  SPACE     pause / resume
  S         screenshot
  V         toggle voice
  + / =     confidence +5%
  -         confidence -5%
  0         reset confidence

Run
---
  python app.py
  python app.py --source 1
  python app.py --source clip.mp4
  python app.py --no-voice
  python app.py --conf 0.40
"""

import cv2
import os
import sys
import time
import argparse
from datetime import datetime

from detection import ObjectDetector
from assistant import VoiceAssistant

DEFAULT_CONF = 0.35
WIN          = "AI Object Detection  |  YOLOv8n  |  Person Tracking"


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",   default=0)
    p.add_argument("--model",    default="yolov8n.pt")
    p.add_argument("--conf",     type=float, default=DEFAULT_CONF)
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--width",    type=int, default=1280)
    p.add_argument("--height",   type=int, default=720)
    return p.parse_args()


# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_paused(frame):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0,0), (w, h), (0,0,0), cv2.FILLED)
    frame = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lbl   = "DETECTION PAUSED"
    (tw,th),_ = cv2.getTextSize(lbl, font, 1.4, 3)
    cx, cy    = (w-tw)//2, (h+th)//2
    cv2.putText(frame, lbl, (cx+2,cy+2), font, 1.4, (0,0,0),    4, cv2.LINE_AA)
    cv2.putText(frame, lbl, (cx,  cy),   font, 1.4, (0,200,255), 3, cv2.LINE_AA)
    hint = "SPACE to resume  |  V for voice"
    (sw,_),_ = cv2.getTextSize(hint, font, 0.55, 1)
    cv2.putText(frame, hint, ((w-sw)//2, cy+52), font, 0.55,
                (200,200,200), 1, cv2.LINE_AA)
    return frame


def draw_voice_badge(frame, on: bool):
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lbl   = "  MIC ON " if on else "  MIC OFF"
    col   = (0,160,0) if on else (0,0,180)
    (tw,th),_ = cv2.getTextSize(lbl, font, 0.50, 1)
    x = w - tw - 16
    y = 56
    cv2.rectangle(frame, (x-4, y-th-5), (x+tw+4, y+5), col, cv2.FILLED)
    cv2.putText(frame, lbl, (x, y), font, 0.50, (255,255,255), 1, cv2.LINE_AA)
    return frame


def draw_heard_strip(frame, heard: str, response: str):
    """Show last heard command and last response at bottom of frame."""
    if not heard and not response:
        return frame
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Response line (above)
    if response:
        resp_short = response[:110] + "…" if len(response) > 110 else response
        msg = f"  AI: {resp_short}"
        (tw,th),_ = cv2.getTextSize(msg, font, 0.46, 1)
        y = h - 60
        cv2.rectangle(frame, (0, y-th-6), (min(tw+12, w), y+4),
                      (10,60,10), cv2.FILLED)
        cv2.putText(frame, msg, (6, y), font, 0.46,
                    (180,255,180), 1, cv2.LINE_AA)

    # Heard line (bottom)
    if heard:
        msg = f"  You: {heard}"
        (tw,th),_ = cv2.getTextSize(msg, font, 0.46, 1)
        y = h - 42
        cv2.rectangle(frame, (0, y-th-6), (min(tw+12, w), y+4),
                      (10,10,60), cv2.FILLED)
        cv2.putText(frame, msg, (6, y), font, 0.46,
                    (180,200,255), 1, cv2.LINE_AA)

    return frame


def save_screenshot(frame, prefix="manual"):
    os.makedirs("screenshots", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("screenshots", f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[App] Screenshot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_args()
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)

    print()
    print("=" * 65)
    print("   AI Object Detection  +  Voice Assistant  (Person-Focused)")
    print("=" * 65)
    print(f"   Model   : {args.model}")
    print(f"   Source  : {args.source}")
    print(f"   Conf    : {args.conf:.0%}")
    print(f"   Voice   : {'OFF' if args.no_voice else 'ON'}")
    print()
    print("   Voice commands:")
    print("     'what do you see'              → all objects in frame")
    print("     'describe the scene'           → full context description")
    print("     'describe the person'          → person position + activity")
    print("     'what is the person doing'     → activity detection")
    print("     'where is the person'          → left / centre / right")
    print("     'how close is the person'      → proximity estimate")
    print("     'how old is the person'        → estimated age group")
    print("     'is there a person'            → yes/no + detail")
    print("     'how many persons'             → count")
    print("     'count cars'                   → count any object")
    print("     'are there any laptops'        → yes/no query")
    print("     'start / stop detection'")
    print("     'take a screenshot'")
    print("     'increase / decrease confidence'")
    print("     'help'                         → all commands")
    print("=" * 65)
    print()

    # ── Detector ──────────────────────────────────────────────────────────────
    detector = ObjectDetector(
        model_path           = args.model,
        confidence_threshold = args.conf,
        screenshot_dir       = "screenshots",
        screenshot_interval  = 30,
    )

    # ── Voice assistant ────────────────────────────────────────────────────────
    detection_active = [True]

    def on_start():
        detection_active[0] = True

    def on_stop():
        detection_active[0] = False

    def on_screenshot():
        detector._last_shot = 0   # reset cooldown

    assistant    = None
    voice_active = False

    if not args.no_voice:
        class TrackedAssistant(VoiceAssistant):
            def _handle(self, text):
                if text:
                    self.last_heard = text
                resp = super()._handle(text)
                if resp:
                    self.last_response = resp
                return resp

        assistant = TrackedAssistant(
            detector      = detector,
            on_start      = on_start,
            on_stop       = on_stop,
            on_screenshot = on_screenshot,
        )
        assistant.start()
        voice_active = True

    # ── Video source ──────────────────────────────────────────────────────────
    try:
        src = int(args.source)
    except (ValueError, TypeError):
        src = str(args.source)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[App] ERROR: Cannot open source '{args.source}'")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[App] Capture: {W}×{H}")

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    def _on_conf(val):
        detector.set_confidence(val / 100.0)

    cv2.createTrackbar("Confidence %", WIN,
                       int(args.conf * 100), 95, _on_conf)

    paused_frame = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            if not isinstance(src, int) and os.path.isfile(str(src)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[App] Stream ended.")
            break

        if detection_active[0]:
            display, _ = detector.process_frame(frame)
            paused_frame = display.copy()
        else:
            display = draw_paused(
                paused_frame if paused_frame is not None else frame
            )

        # Overlays
        display = draw_voice_badge(display, voice_active)
        if assistant:
            display = draw_heard_strip(
                display,
                assistant.last_heard,
                assistant.last_response,
            )

        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            save_screenshot(display)
        elif key == ord(" "):
            detection_active[0] = not detection_active[0]
            print(f"[App] Detection {'RUNNING' if detection_active[0] else 'PAUSED'}.")
        elif key == ord("v") and assistant:
            if voice_active:
                assistant.stop()
                voice_active = False
                print("[App] Voice OFF.")
            else:
                assistant.start()
                voice_active = True
                print("[App] Voice ON.")
        elif key in (ord("+"), ord("=")):
            detector.set_confidence(detector.confidence_threshold + 0.05)
            cv2.setTrackbarPos("Confidence %", WIN,
                               int(detector.confidence_threshold * 100))
        elif key == ord("-"):
            detector.set_confidence(detector.confidence_threshold - 0.05)
            cv2.setTrackbarPos("Confidence %", WIN,
                               int(detector.confidence_threshold * 100))
        elif key == ord("0"):
            detector.set_confidence(DEFAULT_CONF)
            cv2.setTrackbarPos("Confidence %", WIN,
                               int(DEFAULT_CONF * 100))

        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    if assistant:
        assistant.stop()
    print("[App] Done.")


if __name__ == "__main__":
    main()
