import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
from PIL import Image

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import project modules ────────────────────────────────────────────────────
try:
    from detection import ObjectDetector
    from assistant import VoiceAssistant
    MODULES_OK = True
except ImportError as e:
    MODULES_OK = False
    st.error(f"Import error: {e}. Run `pip install -r requirements.txt` first.")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size: 2.4rem; font-weight: 800; color: #00D4FF; margin-bottom: 0; }
  .sub-title  { font-size: 1rem;   color: #888;      margin-top: -6px; }
  .metric-box { background: #1E1E2E; border-radius: 10px; padding: 16px;
                border-left: 4px solid #00D4FF; margin-bottom: 8px; }
  .metric-val { font-size: 2rem; font-weight: 700; color: #00D4FF; }
  .metric-lbl { font-size: 0.8rem; color: #AAA; }
  .command-box{ background: #1a1a2e; border-radius: 8px;
                border: 1px solid #333; padding: 12px; font-style: italic; }
  .status-ok  { color: #00FF88; font-weight: 600; }
  .status-off { color: #FF4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session():
    defaults = {
        "detector":        None,
        "assistant":       None,
        "detection_on":    False,
        "conf":            0.40,
        "last_counts":     {},
        "last_frame":      None,
        "last_response":   "",
        "screenshot_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">🎯 AI Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">YOLOv8 + Voice Assistant</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ Settings")
    conf = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05,
                     help="Higher = fewer but more certain detections")
    st.session_state["conf"] = conf

    model_choice = st.selectbox("YOLOv8 Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="n=fastest, m=most accurate")

    st.divider()
    st.subheader("📷 Source")
    source_type = st.radio("Video Source", ["Webcam", "Upload Video"])
    if source_type == "Upload Video":
        uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    else:
        cam_idx = st.number_input("Camera Index", 0, 5, 0)

    st.divider()
    st.subheader("📁 Screenshots")
    screenshots = sorted(
        [f for f in os.listdir("screenshots") if f.endswith(".jpg")],
        reverse=True
    )[:6] if os.path.exists("screenshots") else []
    st.caption(f"{len(screenshots)} saved")
    if screenshots:
        for fn in screenshots[:3]:
            path = os.path.join("screenshots", fn)
            st.image(path, caption=fn, use_column_width=True)

    st.divider()
    st.caption("Built with YOLOv8 + Streamlit")


# ── Main content ───────────────────────────────────────────────────────────────
col_feed, col_info = st.columns([3, 1])

with col_feed:
    st.subheader("📺 Live Detection Feed")

    # Control buttons
    c1, c2, c3 = st.columns(3)
    start_btn = c1.button("▶ Start Detection", use_container_width=True)
    stop_btn  = c2.button("⏸ Stop Detection",  use_container_width=True)
    snap_btn  = c3.button("📸 Screenshot",      use_container_width=True)

    # Frame placeholder
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    # Detection info bar
    info_bar = st.empty()

with col_info:
    st.subheader("📊 Object Counts")
    counts_placeholder = st.empty()

    st.subheader("🔊 Voice Commands")
    voice_command = st.text_input(
        "Type a command:",
        placeholder="e.g. what do you see?",
        key="voice_input",
    )
    send_btn = st.button("Send Command", use_container_width=True)
    response_box = st.empty()

    st.subheader("💡 Commands")
    st.markdown("""
    - *what do you see?*
    - *how many persons?*
    - *count cars*
    - *describe the scene*
    - *increase confidence*
    - *help*
    """)

    st.subheader("ℹ️ System Info")
    info_placeholder = st.empty()


# ── Button handlers ───────────────────────────────────────────────────────────
if start_btn and MODULES_OK:
    if st.session_state["detector"] is None:
        with st.spinner("Loading YOLOv8 model..."):
            st.session_state["detector"] = ObjectDetector(
                model_path=model_choice,
                confidence_threshold=conf,
            )
            st.session_state["assistant"] = VoiceAssistant(
                detector=st.session_state["detector"]
            )
    st.session_state["detection_on"] = True
    status_placeholder.success("✅ Detection running")

if stop_btn:
    st.session_state["detection_on"] = False
    status_placeholder.warning("⏸ Detection paused")

if snap_btn and st.session_state.get("last_frame") is not None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("screenshots", f"dash_{ts}.jpg")
    cv2.imwrite(path, st.session_state["last_frame"])
    st.session_state["screenshot_count"] += 1
    st.toast(f"Screenshot saved: {path}")

# ── Voice command handler ─────────────────────────────────────────────────────
if send_btn and voice_command.strip():
    asst = st.session_state.get("assistant")
    if asst is None and st.session_state.get("detector"):
        asst = VoiceAssistant(detector=st.session_state["detector"])
        st.session_state["assistant"] = asst

    if asst:
        response = asst.process_text_command(voice_command)
        st.session_state["last_response"] = response or "Command not recognised."
    else:
        st.session_state["last_response"] = "Start detection first."

if st.session_state["last_response"]:
    response_box.markdown(
        f'<div class="command-box">🤖 {st.session_state["last_response"]}</div>',
        unsafe_allow_html=True,
    )

# ── Live detection loop (single frame per rerun) ──────────────────────────────
det = st.session_state.get("detector")

if st.session_state["detection_on"] and det is not None:
    # Update confidence if changed
    det.set_confidence(st.session_state["conf"])

    # Open webcam
    cap = cv2.VideoCapture(int(cam_idx) if source_type == "Webcam" else 0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Run detection
        annotated, counts = det.process_frame(frame)
        st.session_state["last_counts"] = counts
        st.session_state["last_frame"]  = annotated

        # Convert BGR → RGB for display
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

        # Object counts display
        if counts:
            count_md = "\n".join(
                f"| **{obj}** | {cnt} |"
                for obj, cnt in sorted(counts.items())
            )
            counts_placeholder.markdown(
                "| Object | Count |\n|--------|-------|\n" + count_md
            )
        else:
            counts_placeholder.info("No objects detected.")

        # System info
        info_placeholder.markdown(f"""
        - **FPS**: {det.current_fps:.1f}
        - **Confidence**: {det.confidence_threshold:.0%}
        - **Screenshots**: {st.session_state['screenshot_count']}
        - **Status**: <span class="status-ok">Running</span>
        """, unsafe_allow_html=True)
    else:
        frame_placeholder.error("Could not read from webcam.")

elif not st.session_state["detection_on"]:
    # Show placeholder image
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_img, "Press Start Detection", (140, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    frame_placeholder.image(placeholder_img, channels="BGR", use_column_width=True)

    info_placeholder.markdown("""
    - **Status**: <span class="status-off">Stopped</span>
    """, unsafe_allow_html=True)

# Auto-refresh every second while running
if st.session_state["detection_on"]:
    time.sleep(0.05)
    st.rerun()
