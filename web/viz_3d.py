"""
viz_3d.py — WiSense Plotly 3D visualizations
Place in: wisense/web/viz_3d.py
Import:   from web.viz_3d import make_3d_surface, make_3d_comparison, make_3d_motion_frames
"""

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom


def make_3d_surface(heatmap, title="RF Spatial Heatmap — 3D",
                    colorscale="Hot", detected=False,
                    confidence=0.0, upsample=4):
    """
    Interactive 3D surface from a 2D RF heatmap.
    Empty room = flat. Object present = mountain peak.
    Peak position = object location. Peak height = disturbance amplitude.
    """
    data = np.clip(zoom(heatmap.astype(np.float64), upsample, order=3), 0, None)
    r, c = data.shape
    x    = np.linspace(0, 1, c)
    y    = np.linspace(0, 1, r)

    surface = go.Surface(
        z=data, x=x, y=y,
        colorscale=colorscale, showscale=True, opacity=0.92,
        colorbar=dict(thickness=12, len=0.6, x=0.92,
                      tickfont=dict(color='#94a3b8', size=10),
                      title=dict(text="Amplitude", font=dict(color='#94a3b8', size=10))),
        contours=dict(z=dict(show=True, usecolormap=True,
                             highlightcolor="#38bdf8", project=dict(z=True), width=1)),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4,
                      roughness=0.5, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=300),
    )

    # Peak marker
    peak_idx = np.unravel_index(np.argmax(data), data.shape)
    px = x[peak_idx[1]]; py = y[peak_idx[0]]; pz = data[peak_idx]
    sc = "#34d399" if detected else "#64748b"

    peak = go.Scatter3d(
        x=[px], y=[py], z=[pz * 1.08],
        mode='markers+text',
        marker=dict(size=8, color=sc, symbol='diamond',
                    line=dict(color='white', width=1)),
        text=["Object" if detected else ""],
        textfont=dict(color=sc, size=11),
        textposition="top center", showlegend=False,
        hovertemplate=(f"Position: ({px:.2f}, {py:.2f})<br>"
                       f"Amplitude: {pz:.3f}<br>"
                       f"Status: {'DETECTED' if detected else 'EMPTY'}"
                       "<extra></extra>"),
    )

    status = f"{'🟢 DETECTED' if detected else '⚫ EMPTY'} · {confidence:.0%}" if confidence > 0 else ""

    axis = dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f',
                tickfont=dict(color='#475569', size=9),
                showbackground=True, backgroundcolor='#0a0f1e')

    fig = go.Figure(data=[surface, peak], layout=go.Layout(
        title=dict(text=title, font=dict(color='#94a3b8', size=13),
                   x=0.5, xanchor='center', y=0.97),
        paper_bgcolor='#0a0f1e', plot_bgcolor='#0a0f1e',
        margin=dict(l=0, r=0, t=40, b=0), height=420,
        scene=dict(
            bgcolor='#060b14',
            xaxis=dict(title=dict(text='Spatial X', font=dict(color='#475569', size=10)), **axis),
            yaxis=dict(title=dict(text='Spatial Y', font=dict(color='#475569', size=10)), **axis),
            zaxis=dict(title=dict(text='RF Amplitude', font=dict(color='#475569', size=10)), **axis),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.0), up=dict(x=0, y=0, z=1)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5),
        ),
        annotations=[dict(text=status, xref='paper', yref='paper',
                          x=0.5, y=0.02, showarrow=False,
                          font=dict(color=sc, size=12), align='center')] if status else [],
    ))
    return fig


def make_3d_comparison(heatmap_empty, heatmap_object):
    """Two 3D surfaces side by side — empty room (blue) vs object (hot)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'surface'},{'type':'surface'}]],
                        subplot_titles=["Empty room", "Object present"])

    def _surf(hm, cs):
        d = np.clip(zoom(hm.astype(np.float64), 3, order=2), 0, None)
        r, c = d.shape
        return go.Surface(z=d, x=np.linspace(0,1,c), y=np.linspace(0,1,r),
                          colorscale=cs, showscale=False, opacity=0.9,
                          contours=dict(z=dict(show=True, usecolormap=True, width=1)))

    fig.add_trace(_surf(heatmap_empty,  "Blues"), row=1, col=1)
    fig.add_trace(_surf(heatmap_object, "Hot"),   row=1, col=2)

    sc = dict(bgcolor='#060b14',
              xaxis=dict(gridcolor='#1e3a5f', showbackground=True,
                         backgroundcolor='#0a0f1e', tickfont=dict(color='#475569', size=8)),
              yaxis=dict(gridcolor='#1e3a5f', showbackground=True,
                         backgroundcolor='#0a0f1e', tickfont=dict(color='#475569', size=8)),
              zaxis=dict(gridcolor='#1e3a5f', showbackground=True,
                         backgroundcolor='#0a0f1e', tickfont=dict(color='#475569', size=8)),
              camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9)),
              aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.45))

    fig.update_layout(
        paper_bgcolor='#0a0f1e', height=380,
        margin=dict(l=0, r=0, t=36, b=0),
        scene=sc, scene2=sc,
        title=dict(text="RF Landscape — Empty vs Object",
                   font=dict(color='#94a3b8', size=13), x=0.5, xanchor='center'),
        annotations=[
            dict(text="Flat → no disturbance", xref='paper', yref='paper',
                 x=0.18, y=-0.02, showarrow=False,
                 font=dict(color='#64748b', size=11)),
            dict(text="Peak → object detected", xref='paper', yref='paper',
                 x=0.78, y=-0.02, showarrow=False,
                 font=dict(color='#34d399', size=11)),
        ],
    )
    for ann in fig.layout.annotations[:2]:
        ann.font.color = '#94a3b8'; ann.font.size = 12
    return fig


def make_3d_motion_frames(frames):
    """Animated 3D surface — object moving left to right with play/pause."""
    if not frames:
        return go.Figure()

    def _surf(hm):
        d = np.clip(zoom(hm.astype(np.float64), 2, order=2), 0, None)
        r, c = d.shape
        return go.Surface(z=d, x=np.linspace(0,1,c), y=np.linspace(0,1,r),
                          colorscale="Hot", showscale=False, opacity=0.9)

    fig = go.Figure(data=[_surf(frames[0])])
    fig.frames = [go.Frame(data=[_surf(f)], name=str(i))
                  for i, f in enumerate(frames)]

    axis = dict(gridcolor='#1e3a5f', showbackground=True,
                backgroundcolor='#0a0f1e', tickfont=dict(color='#475569', size=8))

    fig.update_layout(
        paper_bgcolor='#0a0f1e', height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="RF Motion — Object Moving Through Space",
                   font=dict(color='#94a3b8', size=13), x=0.5, xanchor='center'),
        scene=dict(
            bgcolor='#060b14',
            xaxis=dict(title=dict(text='X', font=dict(color='#475569', size=9)), **axis),
            yaxis=dict(title=dict(text='Y', font=dict(color='#475569', size=9)), **axis),
            zaxis=dict(title=dict(text='Amplitude', font=dict(color='#475569', size=9)),
                       range=[0, 1.2], **axis),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.45),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0.02, x=0.5, xanchor='center',
            bgcolor='#0f172a', bordercolor='#1e3a5f',
            font=dict(color='#38bdf8', size=12),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=60))]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(frame=dict(duration=120, redraw=True),
                                              mode='immediate',
                                              transition=dict(duration=60))],
                        method='animate', label=str(i))
                   for i, f in enumerate(fig.frames)],
            active=0,
            currentvalue=dict(prefix="Frame: ",
                              font=dict(color='#64748b', size=11), visible=True),
            bgcolor='#0f172a', bordercolor='#1e3a5f', tickcolor='#1e3a5f',
            font=dict(color='#475569', size=9),
            pad=dict(t=10, b=0), len=0.85, x=0.08,
        )],
    )
    return fig