# YOLOE · Block 1 — Mask & Crop

## Folder structure
```
yoloe_app/
├── app.py              ← Streamlit app
├── requirements.txt
├── helpers.py          ← copy from your notebooks/ folder
└── *.pt                ← model weights (auto-downloaded by ultralytics)
```

## Setup & launch
```bash
cp /home/josue_aims_ac_za/open-vocabulary-6d-pose-yoloe/notebooks/helpers.py ./
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (Block 1)
```
Scene image → Prompt → YOLOE mask → Crop → [Any6D - next block]
```

## Downloadable outputs
| File | Content |
|------|---------|
| mask.png | Binary segmentation mask |
| crop_rgba.png | Tight RGBA crop (transparent bg) |
| crop_square.png | Square-padded RGBA crop |
| meta.json | bbox, conf, crop coords, coverage |

## Sidebar controls
- **Confidence** — detection threshold
- **Largest component only** — removes mask noise blobs
- **Erode / Dilate** — refine mask boundary
- **Padding** — extra px around tight crop
