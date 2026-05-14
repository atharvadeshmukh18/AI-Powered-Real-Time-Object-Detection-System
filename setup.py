
import sys
import subprocess
import os


# ── Colour helpers (ANSI) ─────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")


# ── Steps ─────────────────────────────────────────────────────────────────────

def check_python():
    print(f"\n{BOLD}[1/6] Python Version{RESET}")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 9):
        ok(f"Python {major}.{minor} detected")
    else:
        err(f"Python {major}.{minor} — requires 3.9+. Please upgrade.")
        sys.exit(1)


def install_requirements():
    print(f"\n{BOLD}[2/6] Installing Requirements{RESET}")
    info("Running: pip install -r requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        capture_output=False,
    )
    if result.returncode == 0:
        ok("All packages installed")
    else:
        warn("Some packages may have failed — check output above")


def download_yolo():
    print(f"\n{BOLD}[3/6] YOLOv8 Model{RESET}")
    model_path = "yolov8n.pt"
    if os.path.exists(model_path):
        ok(f"Model already present: {model_path}")
        return
    info("Downloading yolov8n.pt (≈6 MB) ...")
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")   # triggers automatic download
        ok("yolov8n.pt downloaded successfully")
    except Exception as e:
        warn(f"Auto-download failed: {e}")
        warn("The model will be downloaded when you first run app.py")


def check_opencv():
    print(f"\n{BOLD}[4/6] OpenCV + Camera{RESET}")
    try:
        import cv2
        ok(f"OpenCV {cv2.__version__} installed")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok("Webcam (index 0) accessible")
            cap.release()
        else:
            warn("Webcam not found at index 0 — try --source 1 or a video file")
    except ImportError:
        err("OpenCV not installed — run: pip install opencv-python")


def check_tts():
    print(f"\n{BOLD}[5/6] Text-to-Speech{RESET}")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        ok(f"pyttsx3 ready — {len(voices)} voice(s) available")
    except ImportError:
        err("pyttsx3 not installed — run: pip install pyttsx3")
    except Exception as e:
        warn(f"pyttsx3 initialised with warning: {e}")


def check_speech_recognition():
    print(f"\n{BOLD}[6/6] Speech Recognition{RESET}")
    try:
        import speech_recognition as sr
        ok(f"SpeechRecognition {sr.__version__} installed")
        try:
            mic = sr.Microphone()
            ok("Microphone detected")
        except Exception:
            warn("Microphone not detected — voice INPUT may not work")
            warn("Voice OUTPUT (TTS) will still function")
    except ImportError:
        err("SpeechRecognition not installed — run: pip install SpeechRecognition")


def print_summary():
    print(f"\n{'='*55}")
    print(f"{BOLD}{CYAN}  Setup Complete!{RESET}")
    print(f"{'='*55}")
    print(f"\n  {BOLD}Run the application:{RESET}")
    print(f"  {GREEN}python app.py{RESET}                 # OpenCV window")
    print(f"  {GREEN}python app.py --no-voice{RESET}      # disable voice")
    print(f"  {GREEN}python app.py --source video.mp4{RESET} # use a video file")
    print(f"  {GREEN}streamlit run dashboard.py{RESET}    # web dashboard")
    print(f"\n  {BOLD}Keyboard shortcuts:{RESET}")
    print(f"    Q / ESC  → quit")
    print(f"    SPACE    → pause / resume")
    print(f"    S        → screenshot")
    print(f"    V        → toggle voice")
    print(f"    + / -    → confidence ±5%")
    print(f"\n  {BOLD}Voice commands:{RESET}")
    print(f"    'what do you see?'")
    print(f"    'how many persons?'")
    print(f"    'count cars'")
    print(f"    'describe the scene'")
    print(f"\n{'='*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}  AI Object Detection System — Setup{RESET}")
    print("=" * 55)
    check_python()
    install_requirements()
    download_yolo()
    check_opencv()
    check_tts()
    check_speech_recognition()
    print_summary()
