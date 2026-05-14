
import threading
import queue
import time
import re
from typing import Callable, Optional

# ── Try importing voice libraries gracefully ───────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[Assistant] WARNING: SpeechRecognition not installed. Voice input disabled.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[Assistant] WARNING: pyttsx3 not installed. Voice output disabled.")


# ── VoiceAssistant class ───────────────────────────────────────────────────

class VoiceAssistant:
    """
    Background-threaded voice assistant.

    It continuously listens for speech commands and delegates
    responses to the registered detector instance.

    Parameters
    ----------
    detector        : ObjectDetector instance (for live counts/descriptions).
    on_start        : Callback called when user says "start detection".
    on_stop         : Callback called when user says "stop detection".
    on_screenshot   : Callback called when user says "take a screenshot".
    speech_language : BCP-47 language code for Google STT (default: "en-US").
    tts_rate        : Words-per-minute for TTS engine.
    """

    WAKE_PHRASES = {"hey detector", "ok detector", "assistant"}

    def __init__(
        self,
        detector=None,
        on_start:      Optional[Callable] = None,
        on_stop:       Optional[Callable] = None,
        on_screenshot: Optional[Callable] = None,
        speech_language: str = "en-US",
        tts_rate: int = 165,
    ):
        self.detector        = detector
        self.on_start        = on_start
        self.on_stop         = on_stop
        self.on_screenshot   = on_screenshot
        self.speech_language = speech_language
        self.is_listening    = False
        self.is_active       = False      # set to True by .start()

        # Thread-safe queue for TTS (pyttsx3 must run on the same thread it was created on)
        self._tts_queue: queue.Queue[str] = queue.Queue()

        # ── Initialise TTS engine ──────────────────────────────────────────
        self._engine = None
        if TTS_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", tts_rate)
                # Try to pick a natural-sounding voice
                voices = self._engine.getProperty("voices")
                for voice in voices:
                    if "english" in voice.name.lower() or "en" in voice.id.lower():
                        self._engine.setProperty("voice", voice.id)
                        break
                print("[Assistant] TTS engine ready.")
            except Exception as e:
                print(f"[Assistant] TTS init error: {e}")
                self._engine = None

        # ── Initialise SpeechRecognition ───────────────────────────────────
        self._recognizer = None
        self._microphone = None
        if SR_AVAILABLE:
            try:
                self._recognizer = sr.Recognizer()
                self._recognizer.pause_threshold = 0.8   # seconds of silence = end of utterance
                self._recognizer.energy_threshold = 300  # background noise threshold
                self._microphone = sr.Microphone()
                print("[Assistant] Microphone ready.")
            except Exception as e:
                print(f"[Assistant] Microphone init error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background listener and TTS threads."""
        self.is_active = True
        threading.Thread(target=self._listen_loop, daemon=True, name="VoiceListener").start()
        threading.Thread(target=self._tts_loop,    daemon=True, name="TTSWorker").start()
        print("[Assistant] Voice assistant started. Say a command!")
        self.speak("Voice assistant is ready. Say 'what do you see' to start.")

    def stop(self) -> None:
        """Gracefully stop the assistant."""
        self.is_active = False
        print("[Assistant] Voice assistant stopped.")

    def speak(self, text: str) -> None:
        """Queue a text string for TTS playback (thread-safe)."""
        print(f"[Assistant] Speaking: {text}")
        self._tts_queue.put(text)

    def process_text_command(self, text: str) -> str:
        """
        Parse a text command and return the response string.
        Also executes side-effects (callbacks, threshold changes).
        Public so the Streamlit dashboard can call it directly.
        """
        return self._handle_command(text.lower().strip())

    # ── Background threads ────────────────────────────────────────────────────

    def _listen_loop(self) -> None:
        """
        Continuously captures microphone audio and converts to text.
        Runs in its own daemon thread.
        """
        if not SR_AVAILABLE or self._recognizer is None or self._microphone is None:
            print("[Assistant] Listening disabled (SpeechRecognition unavailable).")
            return

        print("[Assistant] Calibrating microphone for ambient noise...")
        try:
            with self._microphone as src:
                self._recognizer.adjust_for_ambient_noise(src, duration=2)
        except Exception as e:
            print(f"[Assistant] Calibration error: {e}")
            return

        print("[Assistant] Listening...")
        while self.is_active:
            try:
                with self._microphone as src:
                    # listen() blocks until speech then silence
                    audio = self._recognizer.listen(src, timeout=5, phrase_time_limit=8)

                # Send audio to Google for STT (free tier, no key required)
                text = self._recognizer.recognize_google(
                    audio, language=self.speech_language
                ).lower()
                print(f"[Assistant] Heard: '{text}'")

                response = self._handle_command(text)
                if response:
                    self.speak(response)

            except sr.WaitTimeoutError:
                pass   # no speech detected within timeout — normal
            except sr.UnknownValueError:
                pass   # could not understand audio — ignore
            except sr.RequestError as e:
                print(f"[Assistant] STT request error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"[Assistant] Listen error: {e}")
                time.sleep(1)

    def _tts_loop(self) -> None:
        """
        Drain the TTS queue and play each string aloud.
        Must run on the same thread as pyttsx3.init().
        """
        while self.is_active:
            try:
                text = self._tts_queue.get(timeout=1)
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    # Fallback: just print
                    print(f"[TTS Fallback] {text}")
                self._tts_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Assistant] TTS error: {e}")

    # ── Command handling ──────────────────────────────────────────────────────

    def _handle_command(self, text: str) -> str:
        """
        Match spoken text against known intents and return a response string.
        Order matters: more specific patterns checked first.
        """
        if not text:
            return ""

        d = self.detector  # shorthand

        # ── Scene / visibility ─────────────────────────────────────────────
        if self._match(text, ["what do you see", "what can you see",
                               "what objects", "what is in the frame"]):
            return d.get_scene_description() if d else "Detector not connected."

        if self._match(text, ["describe the scene", "describe what you see",
                               "scene description", "tell me what you see"]):
            return self._describe_scene()

        # ── Count specific objects ─────────────────────────────────────────
        count_match = self._extract_count_query(text)
        if count_match:
            obj   = count_match
            count = d.get_count(obj) if d else 0
            noun  = obj if count == 1 else f"{obj}s"
            if count == 0:
                return f"I don't see any {noun} right now."
            return f"I can see {count} {noun}."

        # ── Detection control ──────────────────────────────────────────────
        if self._match(text, ["start detection", "start detecting",
                               "begin detection", "turn on detection"]):
            if self.on_start:
                self.on_start()
            return "Detection started."

        if self._match(text, ["stop detection", "stop detecting",
                               "pause detection", "turn off detection"]):
            if self.on_stop:
                self.on_stop()
            return "Detection paused."

        # ── Screenshot ────────────────────────────────────────────────────
        if self._match(text, ["take a screenshot", "screenshot",
                               "capture the frame", "save the frame"]):
            if self.on_screenshot:
                self.on_screenshot()
            return "Screenshot saved."

        # ── Confidence threshold ───────────────────────────────────────────
        if self._match(text, ["increase confidence", "raise confidence",
                               "higher confidence", "more accurate"]):
            if d:
                d.set_confidence(d.confidence_threshold + 0.05)
                return f"Confidence raised to {d.confidence_threshold:.0%}."
            return "Detector not connected."

        if self._match(text, ["decrease confidence", "lower confidence",
                               "lower threshold", "more detections"]):
            if d:
                d.set_confidence(d.confidence_threshold - 0.05)
                return f"Confidence lowered to {d.confidence_threshold:.0%}."
            return "Detector not connected."

        # ── FPS / status ───────────────────────────────────────────────────
        if self._match(text, ["what is the fps", "current fps",
                               "how fast", "frame rate"]):
            fps = d.current_fps if d else 0
            return f"The detector is running at {fps:.0f} frames per second."

        # ── Help ──────────────────────────────────────────────────────────
        if self._match(text, ["help", "what can you do", "commands",
                               "list commands", "what commands"]):
            return (
                "You can ask me: what do you see, how many persons, "
                "count cars, describe the scene, start detection, "
                "stop detection, take a screenshot, or increase confidence."
            )

        # ── Greeting ──────────────────────────────────────────────────────
        if self._match(text, ["hello", "hi", "hey", "good morning",
                               "good afternoon", "how are you"]):
            return "Hello! I'm your AI detection assistant. Ask me what I see!"

        # ── Unrecognised ─────────────────────────────────────────────────
        return ""   # empty string = no response (suppress noise)

    def _describe_scene(self) -> str:
        """Return a richer scene description."""
        d = self.detector
        if not d or not d.current_counts:
            return "The scene appears empty. No recognizable objects detected."

        counts = d.current_counts
        desc   = d.get_scene_description()

        # Add contextual note
        if "person" in counts and "laptop" in counts:
            desc += " It looks like someone is working on a laptop."
        elif "person" in counts and "cell phone" in counts:
            desc += " Someone appears to be using a phone."
        elif counts.get("car", 0) > 2:
            desc += " There is significant vehicle traffic visible."
        elif "person" not in counts:
            desc += " No people are detected at the moment."

        return desc

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _match(text: str, phrases: list[str]) -> bool:
        """Return True if any phrase is a substring of text."""
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _extract_count_query(text: str) -> Optional[str]:
        """
        Detect "how many X" or "count X" patterns.
        Returns the object name string or None.
        """
        # Patterns: "how many persons", "count cars", "number of laptops"
        patterns = [
            r"how many (\w[\w\s]*?)(?:\s+(?:are|do|can|is))?\??$",
            r"count (?:the )?(\w[\w\s]*?)s?\??$",
            r"number of (\w[\w\s]*?)s?\??$",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                raw = match.group(1).strip().rstrip("s")  # naive de-plural
                # Map spoken words to COCO class names
                label_map = {
                    "person": "person",  "people": "person",
                    "human": "person",   "car": "car",
                    "vehicle": "car",    "bottle": "bottle",
                    "phone": "cell phone", "mobile": "cell phone",
                    "cell phone": "cell phone", "laptop": "laptop",
                    "computer": "laptop", "chair": "chair",
                }
                return label_map.get(raw, raw)
        return None
