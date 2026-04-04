import streamlit as st
import numpy as np
import cv2
import sys, os, urllib.request
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.signal_engine import generate_2d_heatmap, generate_rf_scene, SIGNAL_MAX
from core.preprocessing import preprocess
from core.edge_detect   import process_heatmap
from model.rule_based   import RuleBasedDetector

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="WiSense — Live Sensing",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (same as other pages) ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,.stApp{background-color:#060b14!important;font-family:'Inter',sans-serif;color:#e2e8f0;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}
@keyframes slide-up{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
@keyframes detected-pulse{0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,0.4);}70%{box-shadow:0 0 0 12px rgba(52,211,153,0);}}
.panel-card{background:linear-gradient(135deg,#0a0f1e,#111827);border:1px solid #1e3a5f;border-radius:16px;padding:20px;margin-bottom:16px;position:relative;overflow:hidden;}
.panel-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.4),transparent);}
.panel-card:hover{border-color:#38bdf8;transition:all 0.3s;}
.panel-title{color:#38bdf8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;font-family:'JetBrains Mono',monospace;}
.detected-box{background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:10px;padding:16px;animation:detected-pulse 1.5s infinite;}
.empty-box{background:rgba(30,58,95,0.3);border:1px solid #1e3a5f;border-radius:10px;padding:16px;}
.stat-row{display:flex;gap:10px;margin:12px 0;}
.stat-box{flex:1;background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;padding:10px;text-align:center;}
.stat-val{font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#38bdf8;}
.stat-lbl{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-top:2px;}
.bridge-arrow{display:flex;align-items:center;justify-content:center;font-size:28px;color:#38bdf8;height:100%;min-height:200px;}
.section-label{display:flex;align-items:center;gap:10px;color:#38bdf8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;}
.section-label::before{content:'';width:20px;height:2px;background:#38bdf8;border-radius:2px;}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e3a5f,transparent);}
.how-it-works{background:linear-gradient(135deg,#0f172a,#1a1f4e);border:1px solid #1e3a5f;border-radius:12px;padding:16px 20px;margin-bottom:16px;font-size:13px;color:#94a3b8;line-height:1.8;}
.live-dot{width:7px;height:7px;background:#34d399;border-radius:50%;animation:blink 1s infinite;display:inline-block;margin-right:6px;}
[data-testid="stSidebar"]{background:#060b14!important;border-right:1px solid #1e3a5f!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;}
hr{border-color:#1e3a5f!important;}
</style>
""", unsafe_allow_html=True)


# ── Model paths ───────────────────────────────
MODELS_DIR       = os.path.join(ROOT, "data", "models")
HAND_MODEL_PATH  = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_PATH  = os.path.join(MODELS_DIR, "face_detector.task")

HAND_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task"


def ensure_model(path: str, url: str, name: str) -> bool:
    """Downloads model file if not already present."""
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with st.spinner(f"Downloading {name} model (~10MB)..."):
            urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        st.error(f"Could not download {name} model: {e}")
        return False


# ── MediaPipe detectors (new 0.10+ tasks API) ──
BoundingBox = Tuple[int, int, int, int, str]

def detect_hands_v2(img_rgb: np.ndarray) -> List[BoundingBox]:
    """Detects hands using MediaPipe HandLandmarker (tasks API)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions

        if not ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "Hand Landmarker"):
            return []

        base_opts = BaseOptions(model_asset_path=HAND_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        h, w = img_rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        boxes: List[BoundingBox] = []

        with mp_vision.HandLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(mp_img)

        if result.hand_landmarks:
            for i, lm_list in enumerate(result.hand_landmarks):
                xs = [lm.x for lm in lm_list]
                ys = [lm.y for lm in lm_list]
                x1 = max(0, int(min(xs) * w) - 20)
                y1 = max(0, int(min(ys) * h) - 20)
                x2 = min(w, int(max(xs) * w) + 20)
                y2 = min(h, int(max(ys) * h) + 20)

                label = "Hand"
                if result.handedness and i < len(result.handedness):
                    side = result.handedness[i][0].display_name
                    label = f"{side} Hand"

                boxes.append((x1, y1, x2 - x1, y2 - y1, label))

        return boxes

    except Exception as e:
        st.warning(f"Hand detection error: {e}")
        return []


def detect_face_v2(img_rgb: np.ndarray) -> List[BoundingBox]:
    """Detects faces using MediaPipe FaceDetector (tasks API)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions

        if not ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL, "Face Detector"):
            return []

        base_opts = BaseOptions(model_asset_path=FACE_MODEL_PATH)
        opts = mp_vision.FaceDetectorOptions(
            base_options=base_opts,
            min_detection_confidence=0.5,
        )

        h, w = img_rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        boxes: List[BoundingBox] = []

        with mp_vision.FaceDetector.create_from_options(opts) as detector:
            result = detector.detect(mp_img)

        if result.detections:
            for det in result.detections:
                bb = det.bounding_box
                boxes.append((
                    max(0, bb.origin_x),
                    max(0, bb.origin_y),
                    bb.width,
                    bb.height,
                    "Face"
                ))

        return boxes

    except Exception as e:
        st.warning(f"Face detection error: {e}")
        return []


def detect_any_object(img_bgr: np.ndarray) -> List[BoundingBox]:
    """Finds the largest foreground object via contour detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    valid = [c for c in contours if cv2.contourArea(c) > img_area * 0.02]
    if not valid:
        return []

    largest = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return [(x, y, w, h, "Object")]


# ── Image helpers ─────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def decode_uploaded(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def draw_bbox(img: np.ndarray, x: int, y: int, w: int, h: int, 
              label: str = "Object", color: tuple = (56, 189, 248)) -> np.ndarray:
    """Draws a glowing bounding box with corner accents."""
    out = img.copy()
    cv2.rectangle(out, (x-2, y-2), (x+w+2, y+h+2), color, 1)
    cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
    clen = max(8, min(w, h) // 5)
    for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(out, (cx, cy), (cx + dx*clen, cy), color, 3)
        cv2.line(out, (cx, cy), (cx, cy + dy*clen), color, 3)
    # Label pill
    lw = len(label) * 9 + 12
    cv2.rectangle(out, (x, y - 26), (x + lw, y), color, -1)
    cv2.putText(out, label, (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (6, 11, 20), 2)
    return out



def bbox_to_rf(bbox: BoundingBox, img_shape: tuple) -> Dict[str, float]:
    """Converts pixel bounding box → simulation parameters."""
    x, y, w, h, label = bbox
    ih, iw = img_shape[:2]
    cx = max(0.1, min(0.9, (x + w / 2) / iw))
    cy = max(0.1, min(0.9, (y + h / 2) / ih))
    area = (w * h) / (iw * ih)
    size = min(1.0, max(0.2, area * 8))
    return {
        'object_x': round(cx, 3), 
        'object_y': round(cy, 3),
        'object_size': round(size, 3), 
        'label': label,
        'bbox_area': round(area, 4)
    }

def run_rf(rf_params: Dict[str, float], noise: float = 0.02) -> Dict[str, Any]:
    """Runs WiSense pipeline from detected object parameters."""
    heatmap = generate_2d_heatmap(
        object_size=rf_params['object_size'],
        object_x=rf_params['object_x'],
        object_y=rf_params['object_y'],
        noise_level=noise
    )
    result = process_heatmap(heatmap)
    det_res = RuleBasedDetector(threshold=30).predict(heatmap)
    signal = generate_rf_scene(
        object_size=rf_params['object_size'],
        object_pos=int(rf_params['object_x'] * 100),
        noise_level=noise
    )
    return {
        'heatmap': heatmap,
        'edges': result['edges'],
        'edge_count': result['edge_count'],
        'signal': signal,
        'detected': det_res['label'] == 1,
        'confidence': det_res['confidence'],
    }



def heatmap_fig(heatmap: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(4,4), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear',
                   aspect='auto', vmin=0, vmax=SIGNAL_MAX)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors='#64748b', labelsize=7)
    ax.set_title(title, fontsize=9, color='#94a3b8', pad=6)
    ax.tick_params(colors='#475569', labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
    fig.tight_layout()
    return fig

def edge_fig(edges: np.ndarray, detected: bool = False):
    fig, ax = plt.subplots(figsize=(4,4), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    ax.imshow(edges, cmap='gray', aspect='auto')
    c = "#34d399" if detected else "#64748b"
    ax.set_title(f"{'DETECTED' if detected else 'EMPTY ROOM'}",
                 fontsize=10, color=c, pad=6, fontweight='600')
    ax.tick_params(colors='#475569', labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
    fig.tight_layout()
    return fig

def signal_fig(signal: np.ndarray):
    fig, ax = plt.subplots(figsize=(5,2.5), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    ax.plot(signal, color='#38bdf8', linewidth=1.2)
    ax.fill_between(range(len(signal)), signal, alpha=0.1, color='#38bdf8')
    ax.axhline(0, color='#1e3a5f', linewidth=0.8, linestyle='--')
    ax.set_title("Signal — 1D CSI", fontsize=9, color='#94a3b8', pad=6)
    ax.tick_params(colors='#475569', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_edgecolor('#1e3a5f')
    ax.set_xlim(0, len(signal)-1)
    fig.tight_layout()
    return fig



with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 4px;">
        <div style="font-size:18px;font-weight:700;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🖐️ Live Sensing
        </div>
        <div style="color:#475569;font-size:11px;font-family:'JetBrains Mono',monospace;">
            Real object → WIFI simulation
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                label="Home",         icon="🏠")
    st.page_link("pages/1_live_demo.py",   label="Live Demo",    icon="🎯")
    st.page_link("pages/4_live_sensing.py",label="Live Sensing", icon="🖐️")
    st.page_link("pages/2_research.py",    label="Research",     icon="📄")
    st.page_link("pages/3_about.py",       label="About",        icon="ℹ️")
    st.divider()

    mode = st.radio("Detection mode", [
        "🖐️  Hand (MediaPipe)",
        "😊  Face (MediaPipe)",
        "📦  Any object (contour)",
        "📁  Upload image",
    ])
    noise = st.slider("Noise Level", 0.01, 0.10, 0.02, 0.01)
    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#334155;line-height:1.7;">
        Object bounding box → normalized (x,y,size)
        → Heatmap blob position + amplitude.<br><br>
        <span style="color:#1e3a5f;">Models auto-download on first use (~10MB each).</span>
    </div>
    """, unsafe_allow_html=True)


# ── Page header ───────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px;">
    <div style="font-size:26px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">
        🖐️ Live Object → RF/Wifi Sensing
    </div>
    <div style="color:#64748b;font-size:13px;max-width:680px;line-height:1.7;">
        Capture a real object — the detected position and size feed directly
        into our simulation, showing exactly what the WiFi CSI signal
        would look like if sensed through our pipeline.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="how-it-works">
    <strong style="color:#38bdf8;">The bridge:</strong>
    &nbsp; Camera → MediaPipe/Contour →
    <code style="color:#818cf8;">bbox(x,y,w,h)</code> →
    <code style="color:#34d399;">generate_2d_heatmap(x, y, size)</code> →
    Edge detection → Detection output.<br>
    <span style="color:#475569; font-size:12px;">
    In production: bounding box comes from CSI triangulation.
    Here: computer vision acts as ground truth to validate the output
    for a real object at a real position.
    </span>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Input ─────────────────────────────────────
img_bgr: np.ndarray | None = None
img_rgb: np.ndarray | None = None

if "Upload" in mode:
    uploaded = st.file_uploader(
        "Upload any image — hand, face, object, anything",
        type=["jpg","jpeg","png","webp"]
    )
    if uploaded:
        img_bgr = decode_uploaded(uploaded)
        img_rgb = bgr_to_rgb(img_bgr)
else:
    st.markdown(
        f'<div style="margin-bottom:8px;font-size:13px;color:#64748b;">'
        f'<span class="live-dot"></span>'
        f'Point your camera and capture — '
        f'{"hold up your hand" if "Hand" in mode else "show your face" if "Face" in mode else "hold up any object"}'
        f'</div>',
        unsafe_allow_html=True
    )
    snapshot = st.camera_input("Capture", label_visibility="collapsed")
    if snapshot:
        file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = bgr_to_rgb(img_bgr)



if img_bgr is not None and img_rgb is not None:
    st.divider()
    st.markdown('<div class="section-label">Detection + Output</div>',
                unsafe_allow_html=True)

    
    boxes: List[BoundingBox] = []
    with st.spinner("Running detection..."):
        if "Hand" in mode:
            boxes = detect_hands_v2(img_rgb)
        elif "Face" in mode:
            boxes = detect_face_v2(img_rgb)
        elif "Any" in mode:
            boxes = detect_any_object(img_bgr)
        elif "Upload" in mode:
            # Try hand → face → contour (fallback cascade)
            boxes = detect_hands_v2(img_rgb)
            if not boxes:
                boxes = detect_face_v2(img_rgb)
            if not boxes:
                boxes = detect_any_object(img_bgr)

    if not boxes:
        st.warning(
            "No object detected. Tips: better lighting, move closer to camera, "
            "hold hand flat facing camera, or try 'Any object' mode."
        )
        st.image(img_rgb, caption="Input — no detection", use_container_width=True)

    else:
        # Use first box for main output
        rf_params = bbox_to_rf(boxes[0], img_bgr.shape)
        rf_out = run_rf(rf_params, noise)

        # Draw box on image
        color_map = {
            "Hand": (56,189,248), "Right Hand": (56,189,248),
            "Left Hand": (129,140,248), "Face": (52,211,153),
            "Object": (251,146,60),
        }
        color = color_map.get(rf_params['label'], (56,189,248))
        ann_img = draw_bbox(img_bgr, *boxes[0][:4],
                            label=rf_params['label'], color=color)
        ann_rgb = bgr_to_rgb(ann_img)

        
        col_img, col_arr, col_rf = st.columns([5, 1, 5])

        with col_img:
            st.markdown('<div class="panel-card">'
                        '<div class="panel-title">📷 Detected object</div>',
                        unsafe_allow_html=True)
            st.image(ann_rgb, use_container_width=True)
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                        color:#64748b;line-height:2.2;margin-top:10px;">
                label &nbsp;&nbsp;&nbsp;= <span style="color:#38bdf8;">{rf_params['label']}</span><br>
                object_x &nbsp;= <span style="color:#818cf8;">{rf_params['object_x']}</span><br>
                object_y &nbsp;= <span style="color:#818cf8;">{rf_params['object_y']}</span><br>
                obj_size &nbsp;= <span style="color:#34d399;">{rf_params['object_size']}</span><br>
                bbox_area = <span style="color:#fb923c;">{rf_params['bbox_area']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_arr:
            st.markdown("""
            <div class="bridge-arrow">
                <div style="text-align:center;">
                    <div>→</div>
                    <div style="font-size:9px;color:#334155;margin-top:6px;
                                font-family:'JetBrains Mono',monospace;">
                        Wifi<br>bridge
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_rf:
            st.markdown('<div class="panel-card">'
                        '<div class="panel-title">📡 Simulated output</div>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = heatmap_fig(rf_out['heatmap'], "Heatmap")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with c2:
                fig = edge_fig(rf_out['edges'], rf_out['detected'])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            fig = signal_fig(rf_out['signal'])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            det = rf_out['detected']
            conf = rf_out['confidence']
            col = "#34d399" if det else "#64748b"
            cls = "detected-box" if det else "empty-box"
            st.markdown(f"""
            <div class="{cls}" style="margin-top:10px;">
                <div style="color:{col};font-weight:700;font-size:15px;margin-bottom:4px;">
                    {'🟢 OBJECT DETECTED' if det else '⚫ EMPTY ROOM'}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{col}88;">
                    confidence: {conf:.0%} &nbsp;|&nbsp;
                    edges: {rf_out['edge_count']} &nbsp;|&nbsp;
                    pos: ({rf_params['object_x']}, {rf_params['object_y']})
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Stats ──
        st.divider()
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box"><div class="stat-val">{rf_params['object_x']}</div><div class="stat-lbl">Object X</div></div>
            <div class="stat-box"><div class="stat-val">{rf_params['object_y']}</div><div class="stat-lbl">Object Y</div></div>
            <div class="stat-box"><div class="stat-val">{rf_params['object_size']}</div><div class="stat-lbl">size</div></div>
            <div class="stat-box"><div class="stat-val">{rf_out['edge_count']}</div><div class="stat-lbl">Edge pixels</div></div>
            <div class="stat-box">
                <div class="stat-val" style="color:{'#34d399' if det else '#64748b'};">
                    {'YES' if det else 'NO'}
                </div>
                <div class="stat-lbl">Detected</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Multiple objects ──
        if len(boxes) > 1:
            st.divider()
            st.markdown(f'<div class="section-label">'
                        f'All {len(boxes)} objects detected</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(len(boxes), 4))
            for i, bbox in enumerate(boxes):
                rp = bbox_to_rf(bbox, img_bgr.shape)
                ro = run_rf(rp, noise)
                with cols[i % 4]:
                    fig = heatmap_fig(ro['heatmap'], rp['label'])
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    c = "#34d399" if ro['detected'] else "#64748b"
                    st.markdown(
                        f"<div style='text-align:center;font-size:12px;color:{c};'>"
                        f"{'DETECTED' if ro['detected'] else 'EMPTY'} "
                        f"({ro['confidence']:.0%})</div>",
                        unsafe_allow_html=True
                    )

else:
    # Landing state — no image yet
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:16px;
                padding:48px 32px;text-align:center;margin-top:8px;">
        <div style="font-size:48px;margin-bottom:16px;">🖐️</div>
        <div style="font-size:18px;font-weight:600;color:#e2e8f0;margin-bottom:8px;">
            Ready for your object
        </div>
        <div style="color:#475569;font-size:13px;max-width:480px;margin:0 auto;line-height:1.8;">
            Choose a mode from the sidebar.<br>
            Hold up your <strong style="color:#38bdf8;">hand</strong> or
            <strong style="color:#34d399;">face</strong> to the camera →
            WiSense maps it to an signal → shows the heatmap + edge detection.
        </div>
        <div style="margin-top:20px;font-size:12px;color:#334155;
                    font-family:'JetBrains Mono',monospace;">
            Models download automatically on first use (~10MB each)
        </div>
    </div>
    """, unsafe_allow_html=True)
