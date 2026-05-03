# YOLOE · Block 1 — Mask & Crop

## Folder structure
```
yoloetest/
├── app.py          ← Streamlit app
├── helpers.py      ← YOLOE helper functions (self-contained)
├── requirements.txt
└── *.pt            ← model weights (auto-downloaded on first run)
```

## Quick start
```bash
cd yoloetest
pip install -r requirements.txt
streamlit run app.py
```

## Model weights
Weights are downloaded automatically from the Ultralytics hub on first use.
To pre-download manually:
```python
from ultralytics import YOLOE
YOLOE("yoloe-26l-seg.pt")      # seg model
YOLOE("yoloe-26l-seg-pf.pt")   # free-prompt model
```
Or place your own `.pt` files directly in this folder.

## Modes
| Mode | Input | What it does |
|------|-------|--------------|
| Text | Scene + class names | Detects objects matching the text labels |
| Visual | Anchor image + bbox + scene | Detects the object shown in the anchor |
| Free | Scene only | Detects all objects, you pick one to crop |

## Pipeline — Block 1 output
```
Scene → YOLOE mask → Crop (RGBA + square) + meta.json → Any6D (Block 2)
```
