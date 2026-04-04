# 📡 WiSense — RF Object Detection

> See Without a Camera. Sense With WiFi.

WiSense detects object presence by analysing WiFi Channel State Information (CSI) distortion — no camera, no wearable, no additional hardware required.

## Live Demo
🔗 **[wisense.streamlit.app](https://wisense-b4dfjicsgagn3v5coabpaq.streamlit.app/)**

## What it does
- Simulates WiFi CSI signal distortion caused by object presence
- Reconstructs a 2D spatial heatmap from signal data
- Runs OpenCV Canny edge detection to find object boundaries
- Classifies presence/absence using an SVM trained on 29 CSI features
- Validated against real Widar3.0 hardware-captured CSI data

## Pipeline
```
RF Signal (CSI) → Preprocessing → 2D Heatmap → Edge Detection → SVM → Output
```

## Tech Stack
- **Python 3.11** — NumPy, SciPy, OpenCV, scikit-learn
- **Streamlit** — interactive web demo
- **Widar3.0** — real CSI dataset (IEEE Dataport)

## Run locally
```bash
git clone https://github.com/code-rh11Reyhan/WiSense
cd wisense
pip install -r requirements.txt
streamlit run web/app.py
```

## Research basis
- Widar3.0 (Zheng et al., IEEE TPAMI 2021)
- RF-Pose (Zhao et al., CVPR 2018)
- WiGest (Abdelnasser et al., INFOCOM 2015)

## Team
REYHAN - ML Lead And Deployment Lead
    Github Link: code-rh11Reyhan
RAVNEET - Web Dev Lead
RENEE - Research Citations And Pitch Lead
ABHIMANYU - UI/UX And Design Lead

---
*Built at Eclipse Hackathon 6.0 · 24hours · Python 3.11*