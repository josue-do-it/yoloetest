"""
helpers.py — YOLOE Block 1
Correct API usage confirmed from ultralytics 8.4.x source:

  TEXT   : model.set_classes(names) → model.predict(source)
  VISUAL : model.predict(source, visual_prompts={bboxes,cls}, refer_image=..., predictor=YOLOEVPSegPredictor)
  FREE   : model_pf.predict(source)   [prompt-free -pf model, no setup needed]
"""

import cv2
import numpy as np


# ── shared result parser ──────────────────────────────────────────────────────

def _parse_best(results, H: int, W: int):
    """
    Extract highest-confidence detection from raw ultralytics Results.
    Returns (bbox int[4] | None, mask bool HxW, conf float, results).
    """
    r = results[0] if isinstance(results, list) else results

    if r.boxes is None or len(r.boxes) == 0:
        return None, np.zeros((H, W), dtype=bool), 0.0, results

    best = int(r.boxes.conf.argmax().item())
    bbox = r.boxes.xyxy[best].cpu().numpy().astype(int)
    conf = float(r.boxes.conf[best].item())

    mask = np.zeros((H, W), dtype=bool)
    if r.masks is not None:
        raw  = r.masks.data[best].cpu().numpy().astype(np.float32)
        raw  = cv2.resize(raw, (W, H))
        mask = raw > 0.5
    else:
        x1, y1, x2, y2 = bbox
        mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True

    return bbox, mask, conf, results


def _hw(scene_path: str):
    bgr = cv2.imread(scene_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {scene_path}")
    return bgr.shape[:2]   # H, W


# ── TEXT PROMPT ───────────────────────────────────────────────────────────────

def yoloe_text_prompt(model, scene_path: str, text_prompts: list,
                      conf: float = 0.25, **kwargs):
    """
    Detect objects by class name using YOLOE text embeddings.

    Correct API (ultralytics ≥ 8.2):
        model.set_classes(names)   ← generates text embeddings internally
        model.predict(source)      ← standard predict, no extra kwargs needed

    Args:
        model        : YOLOE instance (NOT the -pf variant)
        scene_path   : absolute path to the scene image
        text_prompts : list[str], e.g. ["ball", "tennis ball"]
        conf         : confidence threshold

    Returns:
        (bbox int[4] | None, mask bool HxW, conf float, raw_results)
    """
    H, W = _hw(scene_path)

    # Set classes — this calls get_text_pe() internally and stores embeddings
    model.set_classes(text_prompts)

    results = model.predict(
        source=scene_path,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_best(results, H, W)


# ── VISUAL PROMPT ─────────────────────────────────────────────────────────────

def yoloe_visual_prompt(model, scene_path: str, anchor_path: str,
                        anchor_bbox, conf: float = 0.25, **kwargs):
    """
    Detect by reference image + bounding box (visual prompt).

    Correct API:
        model.predict(
            source,
            visual_prompts={"bboxes": [[x1,y1,x2,y2]], "cls": [0]},
            refer_image=anchor_path,
            predictor=YOLOEVPSegPredictor,
        )

    Args:
        model        : YOLOE seg instance (NOT -pf)
        scene_path   : absolute path to scene/query image
        anchor_path  : absolute path to anchor/reference image
        anchor_bbox  : array-like (1,4) or (4,) — [x1,y1,x2,y2] in anchor pixels
        conf         : confidence threshold

    Returns:
        (bbox int[4] | None, mask bool HxW, conf float, raw_results)
    """
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    H, W = _hw(scene_path)

    bbox_arr = np.array(anchor_bbox, dtype=np.float32)
    if bbox_arr.ndim == 1:
        bbox_arr = bbox_arr[np.newaxis, :]   # → (1,4)

    # cls must be a list of ints, one per bbox
    cls_list = list(range(len(bbox_arr)))   # [0] for a single bbox

    results = model.predict(
        source=scene_path,
        visual_prompts={"bboxes": bbox_arr.tolist(), "cls": cls_list},
        refer_image=anchor_path,
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_best(results, H, W)


# ── FREE PROMPT ───────────────────────────────────────────────────────────────

def yoloe_free_prompt(model, scene_path: str, conf: float = 0.25, **kwargs):
    """
    Detect all objects with no prompt (prompt-free -pf model).
    No set_classes / set_vocab needed — the model already has built-in vocabulary.

    Args:
        model      : YOLOE instance loaded from a *-pf.pt weight file
        scene_path : absolute path to the scene image
        conf       : confidence threshold

    Returns:
        (bbox int[4] | None, mask bool HxW, conf float, raw_results)
        Pass raw_results to collect_all_detections() in app.py for all objects.
    """
    H, W = _hw(scene_path)

    results = model.predict(
        source=scene_path,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_best(results, H, W)


# ── visualise (notebook compat) ───────────────────────────────────────────────

def visualize_detection(scene_path: str, anchor_path: str,
                        bbox, mask, conf: float):
    """Simple matplotlib visualisation for notebook use."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        scene_rgb  = cv2.cvtColor(cv2.imread(scene_path),  cv2.COLOR_BGR2RGB)
        anchor_rgb = cv2.cvtColor(cv2.imread(anchor_path), cv2.COLOR_BGR2RGB)
        H, W = scene_rgb.shape[:2]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(anchor_rgb); axes[0].set_title("Anchor"); axes[0].axis("off")

        axes[1].imshow(scene_rgb)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            axes[1].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="cyan", facecolor="none"))
        axes[1].set_title(f"Detection  conf={conf:.3f}"); axes[1].axis("off")

        overlay = scene_rgb.copy().astype(np.float32)
        if mask is not None and np.asarray(mask).any():
            m = cv2.resize(np.asarray(mask, np.float32), (W, H)).astype(bool)
            overlay[m] = overlay[m] * 0.5 + np.array([0, 180, 255], np.float32) * 0.5
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title("Mask overlay"); axes[2].axis("off")

        plt.tight_layout(); plt.show()
    except ImportError:
        print("matplotlib not available — skipping visualisation.")
