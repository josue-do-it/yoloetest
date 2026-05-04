# YOLOE · Block 1 — Mask & Crop

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Required system deps
```bash
# CLIP (for text prompt / set_classes)
pip install "git+https://github.com/ultralytics/CLIP.git"
# MobileCLIP (for free-prompt -pf models)
pip install "git+https://github.com/ultralytics/mobileclip.git"
```

## Modes
| Mode | UX | What it does |
|------|----|--------------|
| Text | Multi-line textarea → clickable tags | Detects objects matching selected class names |
| Visual | Draw box on anchor image → real coords | Detects object matching the drawn reference |
| Free | One click → full object list | Detects everything, list all with ✂ Crop button |
