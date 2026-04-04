"""
1_live_demo.py — WiSense Live Demo
All existing panels preserved. Plotly 3D section added below.
"""

import streamlit as st
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.signal_engine import (generate_2d_heatmap, generate_rf_scene,
                                 generate_motion_sequence)
from core.preprocessing import preprocess
from core.edge_detect   import process_heatmap
from model.rule_based   import RuleBasedDetector

st.set_page_config(page_title="WiSense — Live Demo", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,.stApp{background-color:#060b14!important;font-family:'Inter',sans-serif;color:#e2e8f0;}
@keyframes glow-pulse{0%,100%{box-shadow:0 0 20px rgba(56,189,248,0.15);}50%{box-shadow:0 0 40px rgba(56,189,248,0.4);}}
@keyframes slide-up{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}
@keyframes detected-pulse{0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,0.4);}70%{box-shadow:0 0 0 12px rgba(52,211,153,0);}}
.panel-card{background:linear-gradient(135deg,#0a0f1e,#111827);border:1px solid #1e3a5f;border-radius:16px;padding:20px;margin-bottom:16px;animation:slide-up 0.4s ease both;position:relative;overflow:hidden;}
.panel-card:hover{border-color:#38bdf8;box-shadow:0 8px 32px rgba(56,189,248,0.1);transition:all 0.3s;}
.panel-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.4),transparent);}
.panel-title{color:#38bdf8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;font-family:'JetBrains Mono',monospace;}
.badge-detected{background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.4);border-radius:10px;padding:16px 20px;animation:detected-pulse 1.5s infinite;}
.badge-empty{background:rgba(30,58,95,0.3);border:1px solid #1e3a5f;border-radius:10px;padding:16px 20px;}
.badge-label{font-size:18px;font-weight:700;margin-bottom:4px;}
.badge-conf{font-size:12px;font-family:'JetBrains Mono',monospace;}
.conf-bar-wrap{margin:8px 0;}
.conf-bar-label{display:flex;justify-content:space-between;font-size:12px;color:#64748b;margin-bottom:4px;}
.conf-bar-bg{background:#1e293b;border-radius:99px;height:6px;overflow:hidden;}
.conf-bar-fill{height:100%;border-radius:99px;transition:width 0.5s ease;}
.metric-row{display:flex;gap:12px;margin:16px 0;}
.metric-box{flex:1;background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:12px 16px;text-align:center;}
.metric-val{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#38bdf8;}
.metric-lbl{font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-top:2px;}
.section-label{display:flex;align-items:center;gap:10px;color:#38bdf8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;}
.section-label::before{content:'';width:20px;height:2px;background:#38bdf8;border-radius:2px;}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e3a5f,transparent);}
.live-dot{display:inline-block;width:8px;height:8px;background:#34d399;border-radius:50%;animation:blink 1s ease-in-out infinite;margin-right:6px;}
.frame-counter{font-family:'JetBrains Mono',monospace;font-size:13px;color:#64748b;background:#0f172a;border:1px solid #1e3a5f;border-radius:6px;padding:4px 12px;display:inline-block;}
.viz3d-wrap{background:linear-gradient(135deg,#0a0f1e,#0f172a);border:1px solid #1e3a5f;border-radius:20px;padding:24px;margin-bottom:16px;position:relative;overflow:hidden;}
.viz3d-wrap::before{content:'';position:absolute;top:0;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,#38bdf8,#818cf8,transparent);}
[data-testid="stSidebar"]{background:#060b14!important;border-right:1px solid #1e3a5f!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;}
hr{border-color:#1e3a5f!important;}
</style>
""", unsafe_allow_html=True)


# ── Figure helpers ────────────────────────────

def heatmap_fig(heatmap, cmap='hot'):
    from core.signal_engine import SIGNAL_MAX
    fig, ax = plt.subplots(figsize=(4,4), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    im = ax.imshow(heatmap, cmap=cmap, interpolation='bilinear',
                   aspect='auto', vmin=0, vmax=SIGNAL_MAX)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors='#64748b', labelsize=8)
    cb.outline.set_edgecolor('#1e3a5f')
    ax.tick_params(colors='#475569', labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
    ax.set_xlabel("Spatial X", fontsize=8, color='#475569')
    ax.set_ylabel("Spatial Y", fontsize=8, color='#475569')
    fig.tight_layout(); return fig

def edge_fig(edges, detected=False):
    fig, ax = plt.subplots(figsize=(4,4), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    ax.imshow(edges, cmap='gray', aspect='auto')
    color = "#34d399" if detected else "#64748b"
    ax.set_title(f"Edge Detection\n{'OBJECT DETECTED' if detected else 'EMPTY ROOM'}",
                 fontsize=10, color=color, pad=8, fontweight='600')
    ax.tick_params(colors='#475569', labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
    fig.tight_layout(); return fig

def signal_fig(signal):
    fig, ax = plt.subplots(figsize=(5,3), facecolor='#0a0f1e')
    ax.set_facecolor('#0a0f1e')
    ax.plot(signal, color='#38bdf8', linewidth=1.4, alpha=0.9)
    ax.fill_between(range(len(signal)), signal, alpha=0.12, color='#38bdf8')
    ax.axhline(y=0, color='#1e3a5f', linewidth=0.8, linestyle='--')
    ax.set_title(" Signal (1D CSI)", fontsize=10, color='#94a3b8', pad=8)
    ax.set_xlabel("Subcarrier index", fontsize=8, color='#475569')
    ax.set_ylabel("Amplitude", fontsize=8, color='#475569')
    ax.tick_params(colors='#475569', labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlim(0, len(signal)-1)
    fig.tight_layout(); return fig

def conf_bar(label, val, color):
    p = int(val * 100)
    return f"""<div class="conf-bar-wrap">
        <div class="conf-bar-label"><span>{label}</span>
        <span style="color:{color};font-family:'JetBrains Mono',monospace;">{p}%</span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{p}%;background:linear-gradient(90deg,{color},{color}88);"></div>
        </div></div>"""

def det_badge(name, conf, detected):
    c  = "#34d399" if detected else "#64748b"
    cl = "badge-detected" if detected else "badge-empty"
    return f"""<div class="{cl}">
        <div class="badge-label" style="color:{c};">{'🟢' if detected else '⚫'} {name}</div>
        <div class="badge-conf" style="color:{c}88;">confidence: {conf:.0%}</div></div>"""

def run_det(obj_size, obj_x, obj_y, noise):
    hm   = generate_2d_heatmap(object_size=obj_size, object_x=obj_x,
                                object_y=obj_y, noise_level=noise)
    res  = process_heatmap(hm)
    rule = RuleBasedDetector(threshold=30).predict(hm)
    sig  = generate_rf_scene(object_size=obj_size,
                              object_pos=int(obj_x*100), noise_level=noise)
    svm  = {'label_name':'Not trained','confidence':0.0,'label':0}
    if st.session_state.get('initialized'):
        try: svm = st.session_state['detector'].svm_detector.predict(preprocess(sig))
        except: pass
    return {'heatmap':hm,'edges':res['edges'],'edge_count':res['edge_count'],
            'signal':sig,'rule':rule,'svm':svm}


# ── Sidebar ───────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("""<div style="padding:12px 0 4px;">
            <div style="font-size:18px;font-weight:700;
                background:linear-gradient(135deg,#38bdf8,#818cf8);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🎯 WiSense Demo</div>
            <div style="color:#475569;font-size:11px;font-family:'JetBrains Mono',monospace;">
                Live detection controls</div></div>""", unsafe_allow_html=True)
        st.divider()
        st.page_link("app.py",                  label="Home",         icon="🏠")
        st.page_link("pages/1_live_demo.py",     label="Live Demo",    icon="🎯")
        st.page_link("pages/4_live_sensing.py",  label="Live Sensing", icon="🖐️")
        st.page_link("pages/2_research.py",      label="Research",     icon="📄")
        st.page_link("pages/3_about.py",         label="About",        icon="ℹ️")
        st.divider()

        st.markdown("**Scene controls**")
        mode = st.radio("Mode", ["Static","Animation"], horizontal=True)
        st.markdown("---")
        obj_size = st.select_slider("Object size",
            options=[0.0,0.2,0.4,0.6,0.8,1.0], value=0.6,
            format_func=lambda x:{0.0:"None",0.2:"Tiny",0.4:"Small",
                                  0.6:"Medium",0.8:"Large",1.0:"Very large"}[x])
        obj_x = st.slider("Object X", 0.1, 0.9, 0.5, 0.05)
        obj_y = st.slider("Object Y", 0.1, 0.9, 0.5, 0.05)
        noise = st.slider("Noise",    0.01, 0.15, 0.03, 0.01)

        st.divider()
        st.markdown("**3D surface options**")
        cs      = st.selectbox("Colorscale",
                               ["Hot","Plasma","Viridis","Inferno","Turbo"], index=0)
        cmp     = st.checkbox("Show empty vs object comparison", value=False)
        mot     = st.checkbox("Show 3D motion animation",         value=False)

        anim_steps = anim_delay = anim_start = anim_end = None
        if mode == "Animation":
            st.divider()
            st.markdown("**Animation**")
            anim_steps = st.slider("Frames",     10,40,20)
            anim_delay = st.slider("Speed (ms)", 50,500,150,50)
            anim_start = st.slider("Start X",    0.05,0.4,0.1,0.05)
            anim_end   = st.slider("End X",      0.6,0.95,0.9,0.05)

        return mode,obj_size,obj_x,obj_y,noise,cs,cmp,mot,anim_steps,anim_delay,anim_start,anim_end


# ── 3D section (shared) ───────────────────────

def render_3d(heatmap, colorscale, detected, confidence,
              show_comparison, show_motion_3d, obj_x, obj_y, noise,
              obj_size, title="Spatial Heatmap — 3D Surface"):
    # Lazy import so the page still works if plotly isn't installed
    try:
        from web.viz_3d import (make_3d_surface, make_3d_comparison,
                                 make_3d_motion_frames)
    except ImportError:
        st.warning("Install plotly to enable 3D visuals: `pip install plotly`")
        return

    st.markdown("""<div class="viz3d-wrap">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="font-size:18px;">🌐</span>
            <div style="font-size:14px;font-weight:600;color:#e2e8f0;">3D Spatial Surface</div>
            <div style="margin-left:auto;font-size:11px;color:#475569;
                        font-family:'JetBrains Mono',monospace;">
                Drag to rotate · Scroll to zoom · Click peak for details</div>
        </div>
        <div style="font-size:12px;color:#475569;margin-bottom:0;">
            Object presence creates a mountain peak. Peak height = disturbance amplitude.
            Peak position = object location in the sensing space.
        </div></div>""", unsafe_allow_html=True)

    st.plotly_chart(make_3d_surface(heatmap, title=title, colorscale=colorscale,
                                    detected=detected, confidence=confidence),
                    use_container_width=True)

    if show_comparison:
        st.markdown('<div class="section-label">3D comparison — empty vs object</div>',
                    unsafe_allow_html=True)
        hm_empty = generate_2d_heatmap(object_size=0.0,object_x=obj_x,object_y=obj_y,noise_level=noise)
        st.plotly_chart(make_3d_comparison(hm_empty, heatmap),
                        use_container_width=True)
        st.markdown("""<div style="font-size:12px;color:#475569;text-align:center;margin-top:-8px;">
            Left: empty room — flat wifi landscape. &nbsp;
            Right: object present — clear peak at object position.
            </div>""", unsafe_allow_html=True)

    if show_motion_3d:
        st.markdown('<div class="section-label">3D motion — object crossing the space</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating frames..."):
            frames = generate_motion_sequence(
                object_size=max(obj_size, 0.4),
                start_x=0.1, end_x=0.9, steps=24, noise_level=noise)
        st.plotly_chart(make_3d_motion_frames(frames), use_container_width=True)
        st.markdown("""<div style="font-size:12px;color:#475569;text-align:center;margin-top:-8px;">
            Press ▶ Play. Watch the signal peak travel left → right —
            this is through-wall motion detection visualised.
            </div>""", unsafe_allow_html=True)


# ── Static page ───────────────────────────────

def static(obj_size, obj_x, obj_y, noise, cs, cmp, mot):
    st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <span style="font-size:24px;">🎯</span>
        <div><div style="font-size:20px;font-weight:700;color:#e2e8f0;">Live Detection</div>
        <div style="color:#475569;font-size:13px;">
            <span class="live-dot"></span>Adjust sliders — output updates instantly
        </div></div></div>""", unsafe_allow_html=True)

    d       = run_det(obj_size, obj_x, obj_y, noise)
    dr      = d['rule']['label'] == 1
    ds      = d['svm']['label']  == 1
    cr, cs2 = d['rule']['confidence'], d['svm']['confidence']

    st.markdown(f"""<div class="metric-row">
        <div class="metric-box"><div class="metric-val">{d['edge_count']}</div><div class="metric-lbl">Edge pixels</div></div>
        <div class="metric-box"><div class="metric-val" style="color:{'#34d399' if dr else '#64748b'};">
            {'DETECTED' if dr else 'EMPTY'}</div><div class="metric-lbl">Rule-based</div></div>
        <div class="metric-box"><div class="metric-val" style="color:{'#34d399' if ds else '#64748b'};">
            {'DETECTED' if ds else 'EMPTY'}</div><div class="metric-lbl">SVM</div></div>
        <div class="metric-box"><div class="metric-val">{max(cr,cs2):.0%}</div><div class="metric-lbl">Max confidence</div></div>
    </div>""", unsafe_allow_html=True)

    # Three panels
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel-card"><div class="panel-title">📡Heatmap</div>',
                    unsafe_allow_html=True)
        fig = heatmap_fig(d['heatmap']); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel-card"><div class="panel-title">🔍 Edge Detection</div>',
                    unsafe_allow_html=True)
        fig = edge_fig(d['edges'], dr); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown("""<div style="font-size:10px;color:#334155;text-align:center;
                        font-family:'JetBrains Mono',monospace;margin-top:6px;">
            disturbance boundary — signal response shape, not object outline</div>""",
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel-card"><div class="panel-title">📶 Signal 1D</div>',
                    unsafe_allow_html=True)
        fig = signal_fig(d['signal']); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Dual classifiers
    st.markdown('<div class="section-label">Dual classifier output</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="panel-card"><div class="panel-title">📐 Rule-based</div>
            {det_badge(d['rule']['label_name'], cr, dr)}
            {conf_bar('Detection confidence', cr, '#38bdf8')}
            <div style="margin-top:12px;font-size:12px;color:#475569;
                        font-family:'JetBrains Mono',monospace;">
                edge_count = {d['edge_count']} &nbsp;|&nbsp; threshold = 30
            </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="panel-card"><div class="panel-title">🧠 SVM (RBF kernel)</div>
            {det_badge(d['svm']['label_name'], cs2, ds)}
            {conf_bar('Class probability', cs2, '#818cf8')}
            <div style="margin-top:12px;font-size:12px;color:#475569;
                        font-family:'JetBrains Mono',monospace;">
                29 features &nbsp;|&nbsp; kernel=RBF &nbsp;|&nbsp; C=10
            </div></div>""", unsafe_allow_html=True)

    st.divider()

    # ── 3D surface ──
    render_3d(d['heatmap'], cs, dr, cr, cmp, mot,
              obj_x, obj_y, noise, obj_size)

    st.divider()

    # Alt colormaps
    st.markdown('<div class="section-label">Alternative colormaps</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-card"><div class="panel-title">🌈 Viridis</div>',
                    unsafe_allow_html=True)
        fig = heatmap_fig(d['heatmap'], cmap='viridis')
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel-card"><div class="panel-title">🌡️ Plasma</div>',
                    unsafe_allow_html=True)
        fig = heatmap_fig(d['heatmap'], cmap='plasma')
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)


# ── Animation page ────────────────────────────

def animation(obj_size, obj_y, noise, steps, delay_ms,
              start_x, end_x, cs):
    st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <span style="font-size:24px;">🎬</span>
        <div><div style="font-size:20px;font-weight:700;color:#e2e8f0;">Motion Animation</div>
        <div style="color:#475569;font-size:13px;">Watch object move — 3D surface updates every 4 frames</div>
        </div></div>""", unsafe_allow_html=True)

    cb1, cb2, _ = st.columns([1,1,4])
    with cb1: play = st.button("▶  Play", type="primary", use_container_width=True)
    with cb2: st.button("⏹  Stop", use_container_width=True)

    fi   = st.empty()
    c1, c2, c3 = st.columns(3)
    ph1  = c1.empty(); ph2 = c2.empty(); ph3 = c3.empty()
    rph  = st.empty(); tph = st.empty()  # result + 3D placeholder

    det  = RuleBasedDetector(threshold=30)

    if play:
        positions = np.linspace(start_x, end_x, steps)
        for i, xp in enumerate(positions):
            hm = generate_2d_heatmap(object_size=obj_size,object_x=xp,object_y=obj_y,noise_level=noise)
            res = process_heatmap(hm)
            rr  = det.predict(hm)
            sig = generate_rf_scene(obj_size, int(xp*100), noise)
            d   = rr['label'] == 1; cf = rr['confidence']

            fi.markdown(f"""<div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
                <span class="frame-counter">Frame {i+1}/{steps}</span>
                <span class="frame-counter">X={xp:.2f}</span>
                <span class="frame-counter">Edges:{res['edge_count']}</span>
                <span style="color:{'#34d399' if d else '#64748b'};
                             font-family:'JetBrains Mono',monospace;font-size:13px;">
                    {'● DETECTED' if d else '○ EMPTY'}</span>
            </div>""", unsafe_allow_html=True)

            with ph1.container():
                st.markdown('<div class="panel-card"><div class="panel-title">📡 Heatmap</div>',
                            unsafe_allow_html=True)
                fig = heatmap_fig(hm); st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            with ph2.container():
                st.markdown('<div class="panel-card"><div class="panel-title">🔍 Edges</div>',
                            unsafe_allow_html=True)
                fig = edge_fig(res['edges'], d); st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            with ph3.container():
                st.markdown('<div class="panel-card"><div class="panel-title">📶 Signal</div>',
                            unsafe_allow_html=True)
                fig = signal_fig(sig); st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3D update every 4 frames
            if i % 4 == 0:
                try:
                    from web.viz_3d import make_3d_surface
                    with tph.container():
                        st.markdown('<div class="section-label">3D surface (live)</div>',
                                    unsafe_allow_html=True)
                        st.plotly_chart(
                            make_3d_surface(hm, colorscale=cs, detected=d, confidence=cf,
                                            title=f"Frame {i+1} — X={xp:.2f}"),
                            use_container_width=True)
                except ImportError:
                    pass

            rph.markdown(f"""<div style="background:{'rgba(52,211,153,0.08)' if d else '#0f172a'};
                border:1px solid {'rgba(52,211,153,0.3)' if d else '#1e3a5f'};
                border-radius:10px;padding:14px 20px;
                display:flex;align-items:center;justify-content:space-between;">
                <div style="color:{'#34d399' if d else '#64748b'};font-weight:600;font-size:15px;">
                    {'🟢 DETECTED' if d else '⚫ EMPTY ROOM'}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#475569;">
                    confidence:{cf:.0%} &nbsp;|&nbsp; edges:{res['edge_count']} &nbsp;|&nbsp; frame:{i+1}/{steps}
                </div></div>""", unsafe_allow_html=True)

            time.sleep(delay_ms / 1000.0)
        st.success(f"Done — {steps} frames.")

    else:
        # Preview
        hm = generate_2d_heatmap(object_size=obj_size,object_x=start_x,object_y=obj_y,noise_level=noise)
        res = process_heatmap(hm); rr = det.predict(hm)
        sig = generate_rf_scene(obj_size, int(start_x*100), noise)
        d   = rr['label'] == 1

        fi.markdown(f"""<span class="frame-counter">Frame 1/{steps} — press Play</span>""",
                    unsafe_allow_html=True)
        with ph1.container():
            st.markdown('<div class="panel-card"><div class="panel-title">📡 Heatmap (preview)</div>',
                        unsafe_allow_html=True)
            fig = heatmap_fig(hm); st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with ph2.container():
            st.markdown('<div class="panel-card"><div class="panel-title">🔍 Edges (preview)</div>',
                        unsafe_allow_html=True)
            fig = edge_fig(res['edges'], d); st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with ph3.container():
            st.markdown('<div class="panel-card"><div class="panel-title">📶 Signal (preview)</div>',
                        unsafe_allow_html=True)
            fig = signal_fig(sig); st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        try:
            from web.viz_3d import make_3d_surface
            with tph.container():
                st.markdown('<div class="section-label">3D surface</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(
                    make_3d_surface(hm, colorscale=cs, detected=d,
                                    confidence=rr['confidence'],
                                    title="3D Preview — press Play to animate"),
                    use_container_width=True)
        except ImportError:
            pass


# ── Main ─────────────────────────────────────

(mode, obj_size, obj_x, obj_y, noise,
 cs, cmp, mot,
 asteps, adelay, astart, aend) = sidebar()

if mode == "Static":
    static(obj_size, obj_x, obj_y, noise, cs, cmp, mot)
else:
    animation(obj_size, obj_y, noise, asteps, adelay, astart, aend, cs)