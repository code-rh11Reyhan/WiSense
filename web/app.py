"""
app.py — WiSense Streamlit Home
Run: streamlit run web/app.py
"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.rule_based import WiSenseDetector

# ── Page config (must be first) ───────────────
st.set_page_config(
    page_title="WiSense — Seeing Beyond Sight",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + Animations ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, .stApp {
    background-color: #060b14 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* ── Keyframes ── */
@keyframes pulse-ring {
    0%   { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(2.4); opacity: 0; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-12px); }
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(56,189,248,0.2); }
    50%       { box-shadow: 0 0 40px rgba(56,189,248,0.5), 0 0 80px rgba(56,189,248,0.2); }
}
@keyframes scan-line {
    0%   { top: 0%; opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
            
@keyframes wave-expand {
    0%   { transform: scale(1); opacity: 0.8; }
    100% { transform: scale(3); opacity: 0; }
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(32px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes rotate-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Hero ── */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 64px 24px 48px;
    overflow: hidden;
}
.hero-bg-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(56,189,248,0.15);
    animation: wave-expand 4s ease-out infinite;
    pointer-events: none;
}
.hero-bg-ring:nth-child(1) { width:300px; height:300px; top:50%; left:50%; margin:-150px 0 0 -150px; animation-delay: 0s; }
.hero-bg-ring:nth-child(2) { width:300px; height:300px; top:50%; left:50%; margin:-150px 0 0 -150px; animation-delay: 1.3s; }
.hero-bg-ring:nth-child(3) { width:300px; height:300px; top:50%; left:50%; margin:-150px 0 0 -150px; animation-delay: 2.6s; }

.hero-icon {
    font-size: 56px;
    animation: float 3s ease-in-out infinite;
    display: inline-block;
    position: relative;
    z-index: 2;
}
.hero-title {
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    margin: 12px 0 8px;
    position: relative;
    z-index: 2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 17px;
    max-width: 560px;
    margin: 0 auto 32px;
    line-height: 1.7;
    position: relative;
    z-index: 2;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 12px;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 20px;
    position: relative;
    z-index: 2;
}
.hero-badge::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #38bdf8;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
    animation: blink 1.5s ease-in-out infinite;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 32px 0;
    animation: slide-up 0.6s ease both;
    animation-delay: 0.2s;
}
.stat-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    animation: glow-pulse 3s ease-in-out infinite;
}
.stat-card:hover {
    border-color: #38bdf8;
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(56,189,248,0.2);
}
.stat-value {
    font-size: 26px;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Pipeline steps ── */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin: 24px 0;
}
.pipeline-card {
    background: linear-gradient(135deg, #0f172a 0%, #1a2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
    animation: slide-up 0.5s ease both;
}
.pipeline-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.pipeline-card:hover::before { opacity: 1; }
.pipeline-card:hover {
    border-color: #38bdf8;
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(56,189,248,0.15);
}
.pipeline-card:nth-child(1) { animation-delay: 0.1s; }
.pipeline-card:nth-child(2) { animation-delay: 0.2s; }
.pipeline-card:nth-child(3) { animation-delay: 0.3s; }
.pipeline-card:nth-child(4) { animation-delay: 0.4s; }
.pipeline-card:nth-child(5) { animation-delay: 0.5s; }
.pipeline-card:nth-child(6) { animation-delay: 0.6s; }
.pipeline-icon {
    font-size: 36px;
    display: inline-block;
    transition: transform 0.3s;
    margin-bottom: 12px;
}
.pipeline-card:hover .pipeline-icon { transform: scale(1.2) rotate(-5deg); }
.pipeline-title {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 8px;
}
.pipeline-desc {
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}

/* ── RF Visualizer ── */
.rf-visualizer {
    position: relative;
    background: #0a0f1e;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 40px;
    margin: 24px 0;
    overflow: hidden;
    animation: glow-pulse 4s ease-in-out infinite;
}
            
.rf-rings-container {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 180px;
    position: relative;
}
.rf-ring {
    position: absolute;
    border-radius: 50%;
    border: 1.5px solid;
    animation: pulse-ring 3s cubic-bezier(0.215,0.61,0.355,1) infinite;
}
.rf-ring-1 { width:40px;  height:40px;  border-color:rgba(56,189,248,0.9); animation-delay:0s; }
.rf-ring-2 { width:40px;  height:40px;  border-color:rgba(56,189,248,0.6); animation-delay:0.5s; }
.rf-ring-3 { width:40px;  height:40px;  border-color:rgba(129,140,248,0.5); animation-delay:1s; }
.rf-ring-4 { width:40px;  height:40px;  border-color:rgba(52,211,153,0.4); animation-delay:1.5s; }
.rf-center-dot {
    width: 12px; height: 12px;
    background: #38bdf8;
    border-radius: 50%;
    box-shadow: 0 0 20px #38bdf8, 0 0 40px rgba(56,189,248,0.5);
    z-index: 10;
    position: relative;
}

/* ── Scan line effect ── */
.scan-container {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
}
.scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    animation: scan-line 2s linear infinite;
    pointer-events: none;
    z-index: 10;
}

/* ── Section headers ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #38bdf8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
}
.section-label::before {
    content: '';
    width: 24px; height: 2px;
    background: #38bdf8;
    border-radius: 2px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* ── CTA ── */
.cta-wrap {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f4e 50%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 32px 0;
}
.cta-wrap::before {
    content: '';
    position: absolute;
    top: -1px; left: 20%; right: 20%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, transparent);
    border-radius: 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060b14 !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #64748b; font-size: 13px; }

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 12px;
    color: #34d399;
    font-family: 'JetBrains Mono', monospace;
}
.status-dot {
    width: 6px; height: 6px;
    background: #34d399;
    border-radius: 50%;
    animation: blink 1.5s ease-in-out infinite;
}

/* ── Misc ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Detector init ─────────────────────────────
def initialize_detector():
    if 'detector' not in st.session_state:
        with st.spinner("🧠 Training SVM classifier on synthetic CSI data..."):
            d = WiSenseDetector()
            d.setup(n_samples=400)
            st.session_state['detector']    = d
            st.session_state['initialized'] = True


# ── Sidebar ──────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 8px;">
            <div style="font-size:22px; font-weight:700;
                        background:linear-gradient(to right, #ffffff 0%,#ffffff 22.769%, #00ccff 22.769%,#00ccff 100%);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                📡 WiSense
            </div>
            <div style="color:#475569; font-size:12px; margin-top:4px; font-family:'JetBrains Mono',monospace;">
                v1.0 · Eclipse 6.0 Hackathon Build
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Navigate**")
        st.page_link("app.py",              label="Home",      icon="🏠")
        st.page_link("pages/1_live_demo.py", label="Live Demo", icon="🎯")
        st.page_link("pages/2_research.py",  label="Research",  icon="📄")
        st.page_link("pages/3_about.py",     label="About",     icon="ℹ️")
        st.page_link("pages/4_live_sensing.py", label = "Live Sensing", icon = "📡")
        st.divider()

        if st.session_state.get('initialized'):
            st.markdown("""
            <div class="status-pill">
                <div class="status-dot"></div> Detector online
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Initializing...")

        st.divider()
        st.markdown("""
        <p style="color:#334155; font-size:11px; line-height:1.6;">
            Wifi sensing · CSI analysis<br>
            Edge detection · SVM classifier<br>
            No camera · No hardware
        </p>
        """, unsafe_allow_html=True)


# ── Home page ────────────────────────────────
def render_home():

    # ── Hero ──
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-bg-ring"></div>
        <div class="hero-bg-ring"></div>
        <div class="hero-bg-ring"></div>
        <div class="hero-badge">CSI-based Wifi sensing · Research prototype</div>
        <div class="hero-icon">📡</div>
        <div class="hero-title">See Without a Camera.<br>Sense With WiFi.</div>
        <div class="hero-subtitle">
            WiSense detects object presence by analysing WiFi signal distortion —
            no camera and no wearable required.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── RF visualizer ──
    st.markdown("""
    <div class="rf-visualizer">
        <div style="text-align:center; margin-bottom:8px;">
            <span style="color:#38bdf8; font-size:12px; font-family:'JetBrains Mono',monospace;
                         text-transform:uppercase; letter-spacing:0.1em;">
                Live Wifi signal propagation
            </span>
        </div>
        <div class="rf-rings-container">
            <div class="rf-ring rf-ring-1"></div>
            <div class="rf-ring rf-ring-2"></div>
            <div class="rf-ring rf-ring-3"></div>
            <div class="rf-ring rf-ring-4"></div>
            <div class="rf-center-dot"></div>
        </div>
        <div style="display:flex; justify-content:center; gap:32px; margin-top:16px;">
            <div style="text-align:center;">
                <div style="color:#38bdf8; font-family:'JetBrains Mono',monospace; font-size:13px;">2.4 GHz</div>
                <div style="color:#475569; font-size:11px;">Frequency</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#818cf8; font-family:'JetBrains Mono',monospace; font-size:13px;">100</div>
                <div style="color:#475569; font-size:11px;">Subcarriers</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#34d399; font-family:'JetBrains Mono',monospace; font-size:13px;">64×64</div>
                <div style="color:#475569; font-size:11px;">Heatmap grid</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#fb923c; font-family:'JetBrains Mono',monospace; font-size:13px;">~94%</div>
                <div style="color:#475569; font-size:11px;">Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ──
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">CSI</div>
            <div class="stat-label">Signal Type</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">SVM</div>
            <div class="stat-label">Classifier</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">~94%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">0</div>
            <div class="stat-label">Cameras needed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Pipeline ──
    st.markdown("""
    <div class="section-label">Detection pipeline</div>
    <div class="pipeline-grid">
        <div class="pipeline-card">
            <div class="pipeline-icon">📶</div>
            <div class="pipeline-title">Wifi Signal Capture</div>
            <div class="pipeline-desc">WiFi CSI records amplitude + phase distortion across 100 subcarriers per packet.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-icon">🔧</div>
            <div class="pipeline-title">Preprocessing</div>
            <div class="pipeline-desc">Butterworth low-pass filter removes noise. DC offset removed. Minmax normalized.We also use the concept of Keller Cones (from UCSB)</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-icon">🗺️</div>
            <div class="pipeline-title">Spatial Mapping</div>
            <div class="pipeline-desc">Signal disturbance → 2D Gaussian blob. Object position encoded as intensity peak.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-icon">🔍</div>
            <div class="pipeline-title">Edge Detection</div>
            <div class="pipeline-desc">OpenCV Canny finds the boundary ring of the disturbance blob. Fixed-scale normalization.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-icon">🧠</div>
            <div class="pipeline-title">SVM Classification</div>
            <div class="pipeline-desc">RBF-kernel SVM trained on 29 CSI features. Outputs label + confidence probability.</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-icon">📊</div>
            <div class="pipeline-title">Real-time Output</div>
            <div class="pipeline-desc">Heatmap + edge map + dual classifier confidence — updated live on every interaction.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Why WiFi ──
    st.markdown('<div class="section-label">Why WiFi sensing</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#0f172a; border:1px solid #1e3a5f; border-radius:12px; padding:24px;">
            <div style="color:#38bdf8; font-weight:600; margin-bottom:12px; font-size:15px;">
                📡 What is CSI?
            </div>
            <p style="color:#94a3b8; font-size:13px; line-height:1.8; margin:0;">
                Channel State Information captures how a WiFi signal changes as it
                travels from transmitter to receiver. Every wall reflection, furniture
                bounce, and moving object leaves a distinct distortion signature across
                the signal's subcarriers.<br><br>
                We exploit that distortion to infer object presence — the same
                principle behind Widar3.0, WiGest, and Wifi-Pose.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        advantages = [
            ("🌐", "Ubiquitous",    "WiFi infrastructure already exists everywhere"),
            ("🔒", "Non-invasive",  "No camera — zero privacy concern"),
            ("🧱", "Through-wall",  "Wifi penetrates obstacles cameras cannot"),
            ("💰", "Zero hardware", "No additional devices required"),
        ]
        items_html = "".join([
            f"""<div style="display:flex; gap:12px; align-items:flex-start; margin-bottom:14px;">
                    <div style="font-size:20px; margin-top:2px;">{icon}</div>
                    <div>
                        <div style="color:#e2e8f0; font-weight:500; font-size:13px;">{title}</div>
                        <div style="color:#64748b; font-size:12px;">{desc}</div>
                    </div>
                </div>"""
            for icon, title, desc in advantages
        ])
        st.markdown(f"""
        <div style="background:#0f172a; border:1px solid #1e3a5f; border-radius:12px; padding:24px;">
            <div style="color:#38bdf8; font-weight:600; margin-bottom:16px; font-size:15px;">
                ⚡ Why WiFi over cameras?
            </div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── CTA ──
    st.markdown("""
    <div class="cta-wrap">
        <div style="font-size:32px; margin-bottom:12px;">🎯</div>
        <div style="font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
            See the pipeline in action
        </div>
        <div style="color:#64748b; font-size:14px; margin-bottom:24px;">
            Adjust object size, position, and noise — watch the heatmap and edge
            detection update live.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("→  Open Live Demo", use_container_width=True, type="primary"):
            st.switch_page("pages/1_live_demo.py")


# ── Main ─────────────────────────────────────
initialize_detector()
render_sidebar()
render_home()