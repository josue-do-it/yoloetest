"""
helpers.py — YOLOE Block 1  (ultralytics 8.4.x)

Correct API (confirmed from source):
  TEXT   : model.set_classes(names)  →  model.predict(source)
           requires: pip install git+https://github.com/ultralytics/CLIP.git
  VISUAL : model.predict(source, visual_prompts={"bboxes":..,"cls":..},
                          refer_image=.., predictor=YOLOEVPSegPredictor)
  FREE   : model_pf.predict(source)
           requires: pip install git+https://github.com/ultralytics/mobileclip.git
"""

import cv2
import numpy as np


def _hw(scene_path: str):
    bgr = cv2.imread(scene_path)
    if bgr is None:
        raise ValueError(f"Cannot read: {scene_path}")
    return bgr.shape[:2]          # H, W


def _parse_best(results, H: int, W: int):
    """Return (bbox|None, mask bool HxW, conf float, results)."""
    r = results[0] if isinstance(results, list) else results
    if r.boxes is None or len(r.boxes) == 0:
        return None, np.zeros((H, W), dtype=bool), 0.0, results

    best = int(r.boxes.conf.argmax().item())
    bbox = r.boxes.xyxy[best].cpu().numpy().astype(int)
    conf = float(r.boxes.conf[best].item())

    mask = np.zeros((H, W), dtype=bool)
    if r.masks is not None:
        raw  = r.masks.data[best].cpu().numpy().astype(np.float32)
        mask = (cv2.resize(raw, (W, H)) > 0.5)
    else:
        x1, y1, x2, y2 = bbox
        mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = True

    return bbox, mask, conf, results


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PROMPT
# Correct API: model.set_classes(list[str])  then  model.predict(source)
# set_classes() internally calls get_text_pe() which uses CLIP to embed the text.
# ─────────────────────────────────────────────────────────────────────────────

def yoloe_text_prompt(model, scene_path: str, text_prompts: list,
                      conf: float = 0.25, **kwargs):
    H, W = _hw(scene_path)
    # set_classes generates CLIP text embeddings and stores them on the model
    model.set_classes(text_prompts)
    results = model.predict(source=scene_path, conf=conf, verbose=False, **kwargs)
    return _parse_best(results, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL PROMPT
# API: model.predict(source, visual_prompts={bboxes, cls}, refer_image, predictor)
# bboxes must be list[list[float]]; cls must be list[int]
# ─────────────────────────────────────────────────────────────────────────────

def yoloe_visual_prompt(model, scene_path: str, anchor_path: str,
                        anchor_bbox, conf: float = 0.25, **kwargs):
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    H, W = _hw(scene_path)

    bbox_arr = np.array(anchor_bbox, dtype=np.float32)
    if bbox_arr.ndim == 1:
        bbox_arr = bbox_arr[np.newaxis, :]          # → (N,4)

    bbox_list = bbox_arr.tolist()                   # list[list[float]]
    cls_list  = list(range(len(bbox_list)))         # [0] or [0,1,..] one int per box

    results = model.predict(
        source=scene_path,
        visual_prompts={"bboxes": bbox_list, "cls": cls_list},
        refer_image=anchor_path,
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_best(results, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# FREE PROMPT   (prompt-free *-pf.pt model — no setup needed)
# ─────────────────────────────────────────────────────────────────────────────

def yoloe_free_prompt(model, scene_path: str, conf: float = 0.25, **kwargs):
    H, W = _hw(scene_path)
    results = model.predict(source=scene_path, conf=conf, verbose=False, **kwargs)
    return _parse_best(results, H, W)
