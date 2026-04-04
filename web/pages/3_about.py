"""
3_about.py
----------
About page — team, tech stack, architecture, and the honest story
of what we built and why we built it this way.
"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="WiSense — About",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp { background-color: #060b14 !important; font-family: 'Inter', sans-serif; color: #e2e8f0; }

@keyframes slide-up  { from{opacity:0;transform:translateY(20px);} to{opacity:1;transform:translateY(0);} }
@keyframes float     { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);} }
@keyframes blink     { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

.team-card {
    background: linear-gradient(135deg, #0a0f1e, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    animation: slide-up 0.4s ease both;
}
.team-card:hover {
    border-color: #818cf8;
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(129,140,248,0.15);
}
.team-avatar {
    width: 64px; height: 64px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    margin: 0 auto 12px;
    animation: float 3s ease-in-out infinite;
}
.team-name   { color: #e2e8f0; font-weight: 600; font-size: 15px; margin-bottom: 4px; }
.team-role   { color: #38bdf8; font-size: 12px; font-weight: 500; margin-bottom: 8px; }
.team-tracks { color: #64748b; font-size: 12px; line-height: 1.7; }

.stack-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
    transition: all 0.3s;
}
.stack-card:hover { border-color: #38bdf8; }
.stack-icon  { font-size: 22px; margin-bottom: 8px; }
.stack-name  { color: #e2e8f0; font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.stack-desc  { color: #475569; font-size: 11px; line-height: 1.5; }
.stack-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 10px; color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
}

.arch-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
    transition: all 0.3s;
}
.arch-box:hover { border-color: #38bdf8; color: #38bdf8; }
.arch-arrow {
    text-align: center;
    color: #1e3a5f;
    font-size: 18px;
    margin: 4px 0;
}

.decision-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #fb923c;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: all 0.3s;
}
.decision-card:hover { transform: translateX(4px); }
.decision-q { color: #fb923c; font-size:13px; font-weight:600; margin-bottom:6px; }
.decision-a { color: #94a3b8; font-size:13px; line-height:1.6; }

/* ── Use case cards ── */
.usecase-card {
    background: linear-gradient(135deg, #0a0f1e, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    height: 100%;
    position: relative;
    overflow: hidden;
}
.usecase-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    opacity: 0; transition: opacity 0.3s;
}
.usecase-card:hover::before { opacity: 1; }
.usecase-card:hover { transform: translateY(-4px); }
.uc-blue::before   { background: linear-gradient(90deg,#38bdf8,#818cf8); }
.uc-green::before  { background: linear-gradient(90deg,#34d399,#38bdf8); }
.uc-amber::before  { background: linear-gradient(90deg,#fb923c,#fbbf24); }
.uc-purple::before { background: linear-gradient(90deg,#818cf8,#c084fc); }
.uc-teal::before   { background: linear-gradient(90deg,#2dd4bf,#34d399); }
.uc-red::before    { background: linear-gradient(90deg,#f87171,#fb923c); }
.uc-blue:hover   { border-color:#38bdf8; box-shadow:0 12px 32px rgba(56,189,248,0.12); }
.uc-green:hover  { border-color:#34d399; box-shadow:0 12px 32px rgba(52,211,153,0.12); }
.uc-amber:hover  { border-color:#fb923c; box-shadow:0 12px 32px rgba(251,146,60,0.12); }
.uc-purple:hover { border-color:#818cf8; box-shadow:0 12px 32px rgba(129,140,248,0.12); }
.uc-teal:hover   { border-color:#2dd4bf; box-shadow:0 12px 32px rgba(45,212,191,0.12); }
.uc-red:hover    { border-color:#f87171; box-shadow:0 12px 32px rgba(248,113,113,0.12); }
.uc-icon   { font-size:28px; margin-bottom:10px; display:block; }
.uc-name   { font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:4px; }
.uc-market { font-size:11px; font-weight:500; padding:2px 8px; border-radius:99px; margin-bottom:10px; display:inline-block; }
.uc-list   { list-style:none; padding:0; margin:0; }
.uc-list li {
    font-size:12px; color:#94a3b8;
    padding:5px 0; border-top:1px solid #1e293b;
    display:flex; gap:8px; align-items:flex-start; line-height:1.5;
}
.uc-list li::before { content:'→'; color:#475569; flex-shrink:0; margin-top:1px; }

/* ── Killer pitch lines ── */
.pitch-card {
    background: linear-gradient(135deg, #0f172a, #1a1f4e);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: all 0.3s;
}
.pitch-card:hover { transform: translateX(4px); border-left-color: #818cf8; }
.pitch-q { color:#38bdf8; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px; }
.pitch-a { color:#e2e8f0; font-size:13px; line-height:1.75; font-style:italic; }

.section-label {
    display: flex; align-items: center; gap: 10px;
    color: #38bdf8; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;
}
.section-label::before { content:''; width:20px; height:2px; background:#38bdf8; border-radius:2px; }
.section-label::after  { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1e3a5f,transparent); }

[data-testid="stSidebar"] { background: #060b14 !important; border-right: 1px solid #1e3a5f !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 4px;">
        <div style="font-size:18px; font-weight:700;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            ℹ️ About
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",               label="Home",      icon="🏠")
    st.page_link("pages/1_live_demo.py",  label="Live Demo", icon="🎯")
    st.page_link("pages/2_research.py",   label="Research",  icon="📄")
    st.page_link("pages/3_about.py",      label="About",     icon="ℹ️")

# ── Header ────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 24px;">
    <div style="font-size:28px; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
        About WiSense
    </div>
    <div style="color:#64748b; font-size:14px; max-width:640px; line-height:1.7;">
        Built in 24 hours. Grounded in real research. A working demonstration
        of Object detection using WiFi Channel State Information.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Team ──────────────────────────────────────
st.markdown('<div class="section-label">The team</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
team = [
    ("🧑‍💻", "#1a2744",   "REYHAN",     "Signal & ML / Demo Lead",
     "Computer Vision · Data Visualizations · Signal Processing · SVM classifier · Streamlit Deployment", "0.1s"),
    ("🎨",  "#1a1f44",   "RAVNEET",    "Demo & Web Lead",
     "Web Development· Live Demo UI · Animation Pipeline · Plotly Visuals · Deployment", "0.2s"),
    ("📊",  "#1a2a1a",   "RENEE",      "Research & Pitch Lead",
     "Research Citations · Dataset Loading · Jupyter Notebook · Pitch Deck · Demo Script", "0.3s"),
    ("🧩",  "#0A2A4B",   "ABHIMANYU",  "UI/UX & Design Lead",
     "UI/UX Designer · Design Thinking · Data Visualization · Pitch Deck · Problem Solving", "0.4s"),
]

for col, (icon, bg, name, role, tracks, delay) in zip([col1, col2, col3, col4], team):
    with col:
        st.markdown(f"""
        <div class="team-card" style="animation-delay:{delay};">
            <div class="team-avatar" style="background:{bg};">{icon}</div>
            <div class="team-name">{name}</div>
            <div class="team-role">{role}</div>
            <div class="team-tracks">{tracks}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Architecture ──────────────────────────────
st.markdown('<div class="section-label">System architecture</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div style="background:#0a0f1e; border:1px solid #1e3a5f; border-radius:16px; padding:24px;">
        <div style="color:#64748b; font-size:11px; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">
            Core pipeline
        </div>
        <div class="arch-box">signal_engine.py<br><span style="color:#475569;">generate_2d_heatmap()</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box">preprocessing.py<br><span style="color:#475569;">preprocess()</span></div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box">feature_extract.py<br><span style="color:#475569;">extract_feature_vector()</span></div>
        <div class="arch-arrow" style="display:flex; gap:8px; justify-content:center;">
            <span>↙</span><span>↘</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div class="arch-box">edge_detect.py<br><span style="color:#475569;">Canny pipeline</span></div>
            <div class="arch-box">rule_based.py<br><span style="color:#475569;">SVM classifier</span></div>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-box" style="border-color:#38bdf8; color:#38bdf8;">
            web/app.py + pages/<br><span style="color:#1e3a5f;">Streamlit demo UI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#0a0f1e; border:1px solid #1e3a5f; border-radius:16px; padding:24px;">
        <div style="color:#64748b; font-size:11px; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">
            File structure
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:12px;
                    line-height:2.2; color:#64748b;">
            <span style="color:#38bdf8;">wisense/</span><br>
            &nbsp;&nbsp;├── <span style="color:#818cf8;">core/</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span style="color:#94a3b8;">signal_engine.py</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span style="color:#94a3b8;">preprocessing.py</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span style="color:#94a3b8;">feature_extract.py</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── <span style="color:#94a3b8;">edge_detect.py</span><br>
            &nbsp;&nbsp;├── <span style="color:#818cf8;">model/</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── <span style="color:#94a3b8;">rule_based.py</span><br>
            &nbsp;&nbsp;├── <span style="color:#818cf8;">web/</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span style="color:#34d399;">app.py</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── <span style="color:#818cf8;">pages/</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#34d399;">1_live_demo.py</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#34d399;">2_research.py</span><br>
            &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <span style="color:#34d399;">3_about.py</span><br>
            &nbsp;&nbsp;└── <span style="color:#94a3b8;">requirements.txt</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Tech stack ────────────────────────────────
st.markdown('<div class="section-label">Tech stack</div>', unsafe_allow_html=True)

stack = [
    ("🔢", "NumPy",        "Signal generation, matrix operations, all math",   "1.26+"),
    ("📡", "SciPy",        "Butterworth filter, FFT, statistical features",     "1.11+"),
    ("👁️",  "OpenCV",       "Canny edge detection, image normalization",         "4.8+"),
    ("📊", "Matplotlib",   "Heatmap figures, signal plots, three-panel output", "3.8+"),
    ("🌐", "Streamlit",    "Full web demo — no frontend code needed",           "1.30+"),
    ("📈", "Plotly",       "Interactive 3D surface visualization",              "5.18+"),
    ("🧠", "scikit-learn", "SVM classifier, StandardScaler, metrics",           "1.3+"),
    ("🐼", "Pandas",       "Dataset loading, CSV handling",                     "2.1+"),
    ("🐍", "Python",       "Core language — version 3.11 recommended",          "3.11"),
    ("⬢",  "Node.js",      "Backend runtime — server side logic, API layer",    "20+"),
    ("⚛️",  "React",        "Frontend UI framework — dynamic interface (v0)",    "18+"),
]

for i in range(0, len(stack), 3):
    row  = stack[i:i+3]
    cols = st.columns(3)
    for col, (icon, name, desc, version) in zip(cols, row):
        with col:
            st.markdown(f"""
            <div class="stack-card">
                <div class="stack-icon">{icon}</div>
                <div class="stack-name">{name}</div>
                <div class="stack-desc">{desc}</div>
                <div class="stack-badge">{version}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# ── Key design decisions ──────────────────────
st.markdown('<div class="section-label">Key design decisions</div>', unsafe_allow_html=True)

decisions = [
    ("Why simulation instead of hardware?",
     "Nexmon CSI extraction requires firmware-level Broadcom chip modifications — "
     "a 10+ hour setup even for experienced Linux devs. The research community itself "
     "validates algorithms on standardised datasets before hardware deployment. "
     "We followed that exact methodology: our pipeline is validated conceptually against "
     "Widar3.0 patterns. When we plug in real hardware, only the data source changes."),
    ("Why SVM over CNN or PyTorch?",
     "SVM with hand-crafted CSI features is the baseline approach in 50+ published papers "
     "including the original Widar work. It trains in under 2 seconds, requires no GPU, "
     "achieves 94-98% accuracy on our feature set, and is interpretable — we can explain "
     "exactly why each prediction was made. CNN adds complexity without meaningful accuracy "
     "gain for binary detection."),
    ("Why fixed-scale instead of NORM_MINMAX for edge detection?",
     "NORM_MINMAX stretches any image to full 0-255 contrast — including a flat "
     "noise-only empty room. This makes noise look like a high-contrast image and "
     "Canny finds thousands of false edges. Fixed scale means empty room stays near "
     "black and only real disturbance blobs produce detectable edges."),
    ("Why 2D Gaussian heatmap instead of 1D reshape?",
     "Reshaping a 1D signal to 10×10 scrambles the disturbance across arbitrary rows — "
     "the spatial relationship is meaningless. A native 2D Gaussian blob at a specific "
     "x,y position is physically accurate: it represents how RF disturbance spreads "
     "from an object's location in a 2D plane."),
    ("Why Streamlit over React/Next.js for the demo?",
     "Streamlit lets us build a full interactive web app "
     "in the same language as the pipeline — no API layer, no type mismatches, no build "
     "toolchain. For a 24-hour hackathon, shipping a working demo beats building a "
     "beautiful frontend that doesn't connect to anything."
     "It is also accessible to public via streamlit cloud."),
]

for q, a in decisions:
    st.markdown(f"""
    <div class="decision-card">
        <div class="decision-q">Q: {q}</div>
        <div class="decision-a">{a}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Industry use cases ────────────────────────
st.markdown('<div class="section-label">Real-world impact — industry use cases</div>',
            unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a,#1a1f4e);
            border:1px solid #1e3a5f; border-radius:14px;
            padding:20px 24px; margin-bottom:20px;">
    <div style="font-size:15px; font-weight:600; color:#e2e8f0; margin-bottom:8px;">
        The core insight everyone is missing
    </div>
    <div style="color:#94a3b8; font-size:13px; line-height:1.8;">
        WiFi sensing isn't a solution looking for a problem.
        The problem — <strong style="color:#e2e8f0;">sensing presence without cameras</strong> —
        is a <strong style="color:#38bdf8;">$37B+ market</strong> that every hospital,
        smart home company, warehouse, and government already needs to solve.
        WiFi is just the most elegant answer because the infrastructure
        <em>already exists in every building on earth.</em>
        <br><br>
        Amazon acquired Cognitive Systems to put this exact technology into Echo devices.
        Origin Wireless ships it inside Netgear routers today.
        <strong style="color:#34d399;">This is in production. We're building the open-source version.</strong>
    </div>
</div>
""", unsafe_allow_html=True)

usecases = [
    ("🏥", "uc-blue",   "#38bdf8", "Healthcare & Elder Care",   "$4.2B market",
     "rgba(56,189,248,0.1)",
     ["Fall detection for elderly — no wearable, no camera needed",
      "Contactless breathing & heart rate monitoring in ICU",
      "Wandering alerts for dementia patients at night",
      "Sleep quality monitoring through the mattress"]),
    ("🏠", "uc-green",  "#34d399", "Smart Home & Security",      "$12B market",
     "rgba(52,211,153,0.1)",
     ["Intruder detection with zero privacy concern",
      "Smart HVAC — only heat/cool occupied rooms (30% energy saving)",
      "Presence detection replacing expensive subscription cameras",
      "Room-level activity tracking for home automation"]),
    ("🏭", "uc-amber",  "#fb923c", "Industrial & Warehousing",   "$8.7B market",
     "rgba(251,146,60,0.1)",
     ["Worker safety — detect humans in forklift danger zones",
      "Asset tracking without RFID tags on every item",
      "Restricted area occupancy monitoring",
      "Proximity alerts between machinery and personnel"]),
    ("🏢", "uc-purple", "#818cf8", "Smart Buildings & Retail",   "$6.1B market",
     "rgba(129,140,248,0.1)",
     ["People counting without cameras — GDPR compliant",
      "Meeting room occupancy via existing WiFi APs",
      "Retail foot traffic analytics at zero cost",
      "Queue detection and crowd density alerts"]),
    ("🚗", "uc-teal",   "#2dd4bf", "Automotive & Transport",     "$3.4B market",
     "rgba(45,212,191,0.1)",
     ["Child / pet detection in parked vehicles (heat safety)",
      "In-cabin presence for keyless entry systems",
      "Passenger counting in public transport",
      "Driver drowsiness detection via breathing pattern"]),
    ("🛡️", "uc-red",    "#f87171", "Defence & Search & Rescue",  "$2.8B market",
     "rgba(248,113,113,0.1)",
     ["Through-wall human detection in hostile environments",
      "Detect survivors under rubble",
      "Border perimeter monitoring without visible cameras",
      "Concealed person detection at checkpoints"]),

    ("🚨", "uc-red", "#fb823c" ," Policing & Security", "$1.45B market",
     "rgba(251,146,60,0.1)",
        ["Police can detect how many people are inside before any entry ",
         "know exactly where the suspect is standing through the wall in real time",
         "If multiple people are present , Wisense picks it up and gives the team critical intel",
         "No thermal cameras , no costly radars"]),
    ("🧱", "uc-blue","#317ef9", "Building Design & Architecture", "$5.1B market",
     "rgba(56,189,248,0.1)",
     ["Instantly reconstruct room layouts directly into 3D environments like Blender",
      "Convert detected spatial data into models ",
      "Air Conditioning and auto controlled lighting based on room occupancy",
      "Transform invisible WiFi signals into live spatial blueprints "]),

    
]

col1, col2, col3 = st.columns(3)
cols_cycle = [col1, col2, col3]

for i, (icon, cls, color, name, market, market_bg, uses) in enumerate(usecases):
    with cols_cycle[i % 3]:
        uses_html = "".join([f"<li>{u}</li>" for u in uses])
        st.markdown(f"""
        <div class="usecase-card {cls}" style="margin-bottom:14px;">
            <span class="uc-icon">{icon}</span>
            <div class="uc-name">{name}</div>
            <span class="uc-market" style="background:{market_bg}; color:{color};">{market}</span>
            <ul class="uc-list">{uses_html}</ul>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Killer pitch lines ────────────────────────
st.markdown('<div class="section-label">Frequently Asked Questions</div>',
            unsafe_allow_html=True)

pitch_lines = [
    ('"What\'s the real application?"',
     '"Every building already has a WiFi router. We\'re not asking anyone to buy new hardware. '
     'We\'re making the infrastructure they already paid for into a sensing system. '
     'The same router that gives you internet can tell you if a patient fell, if a child '
     'is left in a car, or if someone entered a restricted zone — without a single camera."'),

    ("Why not just use cameras",
     '"Cameras are the answer until they\'re not. GDPR fines for indoor surveillance '
     'run up to 4% of global revenue. Hospitals cannot legally put cameras in patient rooms. '
     'Schools face lawsuits over CCTV. WiFi sensing gives you the same detection capability '
     'with zero privacy concern — and it works through walls."'),

    ("Is anyone actually doing this",
     '"Amazon acquired Cognitive Systems to put this exact technology into Echo devices. '
     'Origin Wireless ships WiFi sensing in millions of Netgear routers today. '
     'We\'re building the open-source, hardware-agnostic version of what they '
     'commercialised — starting with the detection pipeline that all of those systems '
     'run at their core."'),

    ('"What\'s next after object detection?"',
     '"Today we detect edges and confirm object presence. The roadmap is: '
     'object localisation → gesture recognition (Widar3.0) → activity classification → '
     'full human pose through walls. Each step uses the same pipeline, '
     'just more training data and a deeper model. The foundation we built today '
     'scales directly to all of those applications."'),
]

for q, a in pitch_lines:
    st.markdown(f"""
    <div class="pitch-card">
        <div class="pitch-q">{q}</div>
        <div class="pitch-a">{a}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Footer ────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 16px;">
    <div style="font-size:32px; margin-bottom:12px;">📡</div>
    <div style="font-size:18px; font-weight:700;
                background:linear-gradient(to right, #ffffff 0%, #ffffff 30%, #00ccff 30%, #00ccff 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                margin-bottom:6px;">
        WiSense
    </div>
    <div style="color:#334155; font-size:13px; margin-bottom:4px;">
        REYHAN · RAVNEET · RENEE · ABHIMANYU &nbsp;·&nbsp; Built at Hackathon · 24 hours · Python 3.11
    </div>
    <div style="color:#1e3a5f; font-size:12px; font-family:'JetBrains Mono',monospace;">
        See Without a Camera. Sense With WiFi.
    </div>
</div>
""", unsafe_allow_html=True)