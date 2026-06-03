"""
assistant.py — Voice Assistant (Person-Focused)
================================================
New commands added:
  "describe the person"        → position, proximity, activity of each person
  "what is the person doing"   → activity of detected person(s)
  "where is the person"        → position in frame (left/centre/right)
  "how close is the person"    → proximity estimate
  "is there a person"          → yes/no with detail
  "tell me about the people"   → full person report
"""

import threading
import queue
import time
import re
import subprocess
import platform
from typing import Optional, Callable

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[Voice] Install: pip install SpeechRecognition pyaudio")

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    PYTTSX3_OK = False
    print("[Voice] pyttsx3 not found — will use system TTS fallback.")


# ── System TTS fallback ───────────────────────────────────────────────────────

def _sys_speak(text: str) -> bool:
    OS   = platform.system()
    safe = text.replace("'", "").replace('"', "")
    try:
        if OS == "Darwin":
            subprocess.run(["say", "-r", "165", text], check=True, timeout=30)
            return True
        if OS == "Linux":
            for cmd in (["espeak-ng", "-s", "150", "-a", "200", text],
                        ["espeak",    "-s", "150", "-a", "200", text]):
                if subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0:
                    return True
        if OS == "Windows":
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate=1; "
                f"$s.Speak('{safe}');"
            )
            subprocess.run(["powershell", "-Command", ps],
                           check=True, capture_output=True, timeout=30)
            return True
    except Exception as e:
        print(f"[TTS] System error: {e}")
    return False


# ── VoiceAssistant ────────────────────────────────────────────────────────────

class VoiceAssistant:
    """
    Two daemon threads:
      VoiceListener  — mic → Google STT → command → TTS queue
      TTSWorker      — TTS queue → pyttsx3 (engine created ON this thread)
    """

    def __init__(
        self,
        detector           = None,
        on_start: Optional[Callable]      = None,
        on_stop:  Optional[Callable]      = None,
        on_screenshot: Optional[Callable] = None,
        language: str = "en-US",
        tts_rate: int = 155,
    ):
        self.detector      = detector
        self.on_start      = on_start
        self.on_stop       = on_stop
        self.on_screenshot = on_screenshot
        self.language      = language
        self.tts_rate      = tts_rate
        self.is_active     = False
        self.last_heard    = ""
        self.last_response = ""

        self._tts_q: queue.Queue[str] = queue.Queue()

        self._rec = None
        self._mic = None
        if SR_AVAILABLE:
            try:
                self._rec = sr.Recognizer()
                self._rec.pause_threshold       = 0.7
                self._rec.phrase_threshold      = 0.3
                self._rec.non_speaking_duration = 0.4
                self._rec.energy_threshold      = 300
                self._mic = sr.Microphone()
                print("[Voice] Microphone ready.")
            except Exception as e:
                print(f"[Voice] Mic error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self.is_active = True
        threading.Thread(target=self._tts_loop,    daemon=True, name="TTSWorker").start()
        time.sleep(0.4)
        threading.Thread(target=self._listen_loop, daemon=True, name="VoiceListener").start()
        print("[Voice] Assistant started.")
        self._tts_q.put(
            "Voice assistant ready. "
            "Say: what do you see, describe the person, what is the person doing, "
            "or how old is the person."
        )

    def stop(self):
        self.is_active = False

    def speak(self, text: str):
        if text:
            print(f"[Voice] Speaking → {text}")
            self._tts_q.put(text)

    def process_text(self, text: str) -> str:
        resp = self._handle(text.lower().strip())
        if resp:
            self.speak(resp)
        return resp

    # ── TTS thread (pyttsx3 lives HERE) ──────────────────────────────────────

    def _tts_loop(self):
        engine = None
        if PYTTSX3_OK:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate",   self.tts_rate)
                engine.setProperty("volume", 1.0)
                for v in engine.getProperty("voices"):
                    nm = (v.name or "").lower()
                    vi = (v.id   or "").lower()
                    if any(x in nm or x in vi for x in
                           ("english", "en_us", "en-us", "zira", "david", "hazel")):
                        engine.setProperty("voice", v.id)
                        break
                print("[TTS] pyttsx3 ready on TTS thread.")
            except Exception as e:
                print(f"[TTS] pyttsx3 init error: {e}")
                engine = None

        while self.is_active:
            try:
                text = self._tts_q.get(timeout=1.0)
            except queue.Empty:
                continue

            spoken = False

            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                    spoken = True
                except Exception as e:
                    print(f"[TTS] pyttsx3 error: {e}")
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty("rate", self.tts_rate)
                        engine.say(text)
                        engine.runAndWait()
                        spoken = True
                    except Exception:
                        engine = None

            if not spoken:
                spoken = _sys_speak(text)
            if not spoken:
                print(f"[TTS CONSOLE] {text}")

            self._tts_q.task_done()

    # ── Listener thread ───────────────────────────────────────────────────────

    def _listen_loop(self):
        if not SR_AVAILABLE or not self._rec or not self._mic:
            print("[Voice] Listening disabled.")
            return

        print("[Voice] Calibrating mic (2 s) …")
        try:
            with self._mic as src:
                self._rec.adjust_for_ambient_noise(src, duration=2)
            print("[Voice] Ready — try: 'what do you see' or 'describe the person'")
        except Exception as e:
            print(f"[Voice] Calibration error: {e}")
            return

        while self.is_active:
            try:
                with self._mic as src:
                    audio = self._rec.listen(src, timeout=5, phrase_time_limit=8)

                text = self._rec.recognize_google(
                    audio, language=self.language
                ).lower().strip()

                self.last_heard = text
                print(f"\n[Voice] Heard → '{text}'")

                resp = self._handle(text)
                if resp:
                    self.last_response = resp
                    self._tts_q.put(resp)
                else:
                    print("[Voice] No command matched.")

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"[Voice] STT error: {e}")
                time.sleep(3)
            except Exception as e:
                print(f"[Voice] Error: {type(e).__name__}: {e}")
                time.sleep(1)

    # ── Command handler ───────────────────────────────────────────────────────

    def _handle(self, text: str) -> str:
        if not text:
            return ""
        d = self.detector

        # ── "what do you see" — list everything visible ───────────────────
        if _any(text, [
            "what do you see",    "what can you see",
            "what objects",       "what is in the frame",
            "what's in the frame","what are you seeing",
            "what do you detect", "tell me what you see",
            "what are you looking at", "what's there",
            "what is there",      "what objects are there",
        ]):
            if not d:
                return "The detector is not connected."
            return d.get_scene_description()

        # ── "describe the scene" — everything + context sentence ─────────
        if _any(text, [
            "describe the scene",  "describe what you see",
            "full description",    "explain what you see",
            "narrate",             "tell me everything",
            "what's going on",     "what is going on",
            "give me a description",
        ]):
            if not d:
                return "Detector not connected."
            return d.get_detailed_description()

        # ── PERSON-SPECIFIC commands ──────────────────────────────────────

        # "describe the person" / "tell me about the person"
        if _any(text, [
            "describe the person",  "describe the people",
            "tell me about the person", "tell me about the people",
            "who is in the frame",  "who do you see",
            "person description",   "describe humans",
            "what does the person look like",
        ]):
            if not d:
                return "Detector not connected."
            return d.get_person_description()

        # "what is the person doing" / "what are they doing"
        if _any(text, [
            "what is the person doing", "what are they doing",
            "what is he doing",         "what is she doing",
            "what are the people doing","person activity",
            "what are they up to",      "what is happening with the person",
        ]):
            if not d:
                return "Detector not connected."
            with d._lock:
                persons = list(d.person_details)
                n       = d.current_counts.get("person", 0)

            if n == 0:
                return "I don't see any person in the frame right now."

            activities = [p.activity for p in persons if p.activity]
            if not activities:
                return (
                    "I can see a person in the frame, "
                    "but I cannot determine what they are doing."
                    if n == 1 else
                    f"I can see {n} persons but cannot determine their activity."
                )
            if n == 1:
                return f"The person appears to be {activities[0]}."

            # Multiple persons
            acts = list(set(activities))
            if len(acts) == 1:
                return f"The {n} persons appear to be {acts[0]}."
            return (
                f"I can see {n} persons. "
                + " ".join(
                    f"Person {i+1} appears to be {a}."
                    for i, a in enumerate(activities[:4])
                )
            )

        # "where is the person"
        if _any(text, [
            "where is the person", "where are the people",
            "where is he",         "where is she",
            "position of the person", "where in the frame",
        ]):
            if not d:
                return "Detector not connected."
            with d._lock:
                persons = list(d.person_details)
                n       = d.current_counts.get("person", 0)

            if n == 0:
                return "There is no person in the frame right now."
            if n == 1 and persons:
                return f"The person is {persons[0].position} of the frame."
            positions = [p.position for p in persons[:4]]
            return (
                f"There are {n} persons. "
                + "  ".join(
                    f"Person {i+1} is {pos}."
                    for i, pos in enumerate(positions)
                )
            )

        # "how close is the person"
        if _any(text, [
            "how close is the person", "how far is the person",
            "proximity",               "how near is the person",
            "distance of the person",  "how close are they",
        ]):
            if not d:
                return "Detector not connected."
            with d._lock:
                persons = list(d.person_details)
                n       = d.current_counts.get("person", 0)

            if n == 0:
                return "I don't see any person in the frame."
            if n == 1 and persons:
                return f"The person appears to be {persons[0].proximity}."
            proximities = [p.proximity for p in persons[:4]]
            return (
                f"I can see {n} persons. "
                + "  ".join(
                    f"Person {i+1} is {prx}."
                    for i, prx in enumerate(proximities)
                )
            )

        # "how old is the person" / "what is the person's age"
        if _any(text, [
            "how old is the person", "what is the person's age",
            "person age", "how old are they",
            "how old is he", "how old is she",
            "how old are the people",
        ]):
            if not d:
                return "Detector not connected."
            with d._lock:
                persons = list(d.person_details)
                n       = d.current_counts.get("person", 0)

            if n == 0:
                return "I don't see any person in the frame right now."
            if n == 1 and persons:
                return f"The person appears to be {persons[0].age_group}."
            if persons:
                return (
                    f"I can see {n} persons. "
                    + " ".join(
                        f"Person {i+1} appears to be {p.age_group}."
                        for i, p in enumerate(persons[:4])
                    )
                )
            return f"I can see {n} persons, but I cannot estimate their age."

        # "is there a person" / "is anyone there"
        if _any(text, [
            "is there a person", "is there anyone",
            "is anyone there",   "is someone there",
            "any person",        "is there a human",
            "do you see a person",
        ]):
            if not d:
                return "Detector not connected."
            n = d.get_count("person")
            if n == 0:
                return "No, I don't see any person in the frame right now."
            if n == 1:
                with d._lock:
                    persons = list(d.person_details)
                p   = persons[0] if persons else None
                msg = "Yes, I can see a person in the frame."
                if p:
                    msg += f" They are {p.position}, {p.proximity}."
                    if p.activity:
                        msg += f" They appear to be {p.activity}."
                return msg
            return f"Yes, I can see {n} persons in the frame."

        # ── Count a specific object ───────────────────────────────────────
        obj = _extract_object(text)
        if obj is not None:
            count = d.get_count(obj) if d else 0
            if count == 0:
                return f"I don't see any {obj} in the frame right now."
            noun = obj if count == 1 else _pluralize_word(obj)
            art  = "a" if count == 1 else str(count)
            return f"I can see {art} {noun} in the frame."

        # ── are there any X? ──────────────────────────────────────────────
        any_q = _extract_any_query(text)
        if any_q:
            count = d.get_count(any_q) if d else 0
            if count == 0:
                return f"No, I don't see any {any_q} right now."
            noun = any_q if count == 1 else _pluralize_word(any_q)
            art  = "a" if count == 1 else str(count)
            return f"Yes, I can see {art} {noun} in the frame."

        # ── start / stop ──────────────────────────────────────────────────
        if _any(text, ["start detection","start detecting",
                       "begin detection","resume","turn on"]):
            if self.on_start:
                self.on_start()
            return "Detection started."

        if _any(text, ["stop detection","stop detecting",
                       "pause","halt","turn off"]):
            if self.on_stop:
                self.on_stop()
            return "Detection paused."

        # ── screenshot ────────────────────────────────────────────────────
        if _any(text, ["screenshot","take screenshot","capture","save frame"]):
            if self.on_screenshot:
                self.on_screenshot()
            return "Screenshot saved."

        # ── confidence ────────────────────────────────────────────────────
        if _any(text, ["increase confidence","raise confidence","more accurate"]):
            if d:
                d.set_confidence(d.confidence_threshold + 0.05)
                return f"Confidence raised to {d.confidence_threshold:.0%}."

        if _any(text, ["decrease confidence","lower confidence","more detections"]):
            if d:
                d.set_confidence(d.confidence_threshold - 0.05)
                return f"Confidence lowered to {d.confidence_threshold:.0%}."

        # ── FPS ───────────────────────────────────────────────────────────
        if _any(text, ["fps","frame rate","how fast","frames per second"]):
            fps = d.current_fps if d else 0
            return f"Running at {fps:.0f} frames per second."

        # ── help ──────────────────────────────────────────────────────────
        if _any(text, ["help","what can you do","commands","list commands"]):
            return (
                "Say: what do you see, describe the scene, "
                "describe the person, what is the person doing, "
                "where is the person, how close is the person, "
                "how old is the person, is there a person, "
                "how many persons, count cars, start detection, "
                "stop detection, take screenshot, or frame rate."
            )

        # ── greetings ─────────────────────────────────────────────────────
        if _any(text, ["hello","hi","hey","good morning","good afternoon","how are you"]):
            return (
                "Hello! I am your AI detection assistant. "
                "Ask me what I see, or say describe the person."
            )

        return ""


# ── Module helpers ────────────────────────────────────────────────────────────

def _any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def _extract_object(text: str) -> Optional[str]:
    PATTERNS = [
        r"how many (\w[\w\s]*?)(?:\s+(?:are|do|can|is|were))?\s*\??$",
        r"count (?:the )?(\w[\w\s]*?)s?\s*\??$",
        r"number of (\w[\w\s]*?)s?\s*\??$",
    ]
    SYNONYMS = {
        "person":"person",      "people":"person",
        "human":"person",       "humans":"person",
        "man":"person",         "woman":"person",
        "car":"car",            "vehicle":"car",
        "bottle":"bottle",
        "phone":"cell phone",   "mobile":"cell phone",
        "cell phone":"cell phone","smartphone":"cell phone",
        "laptop":"laptop",      "computer":"laptop",
        "chair":"chair",        "dog":"dog",
        "cat":"cat",            "bicycle":"bicycle",
        "bike":"bicycle",       "bus":"bus",
        "truck":"truck",        "cup":"cup",
        "book":"book",          "backpack":"backpack",
        "keyboard":"keyboard",  "mouse":"mouse",
        "remote":"remote",      "clock":"clock",
        "tv":"tv",              "television":"tv",
    }
    for pat in PATTERNS:
        m = re.search(pat, text)
        if m:
            raw = m.group(1).strip().lower().rstrip("s")
            for k in (raw, raw + "s"):
                if k in SYNONYMS:
                    return SYNONYMS[k]
            return raw
    return None


def _extract_any_query(text: str) -> Optional[str]:
    PATTERNS = [
        r"are there any (\w[\w\s]*?)s?\s*\??$",
        r"is there (?:a|an) (\w[\w\s]*?)\s*\??$",
        r"do you see (?:a|an) (\w[\w\s]*?)\s*\??$",
        r"can you see (?:a|an) (\w[\w\s]*?)\s*\??$",
    ]
    SYNONYMS = {
        "person":"person",   "people":"person",  "human":"person",
        "car":"car",         "vehicle":"car",
        "phone":"cell phone","mobile":"cell phone","cell phone":"cell phone",
        "laptop":"laptop",   "computer":"laptop",
        "bottle":"bottle",   "chair":"chair",
        "dog":"dog",         "cat":"cat",
        "book":"book",       "cup":"cup",
        "bicycle":"bicycle", "bike":"bicycle",
        "bus":"bus",         "truck":"truck",
        "tv":"tv",           "television":"tv",
    }
    for pat in PATTERNS:
        m = re.search(pat, text)
        if m:
            raw = m.group(1).strip().lower().rstrip("s")
            for k in (raw, raw + "s"):
                if k in SYNONYMS:
                    return SYNONYMS[k]
            return raw
    return None


def _pluralize_word(word: str) -> str:
    IRR = {"person":"persons","mouse":"mice","sheep":"sheep","fish":"fish"}
    if word in IRR:
        return IRR[word]
    if word.endswith(("s","sh","ch","x","z")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"
