"""
helpers.py — YOLOE Block 1
Self-contained helper functions for text, visual, and free prompts.
Drop this file next to app.py.  No other dependencies beyond ultralytics + cv2.
"""

import cv2
import numpy as np


# ── Internal: extract best detection from raw ultralytics result ──────────────

def _parse_result(results, scene_path: str):
    """
    Returns (bbox int[4], mask bool H×W, conf float, results)
    Works whether results is a list or a single Results object.
    """
    bgr = cv2.imread(scene_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {scene_path}")
    H, W = bgr.shape[:2]

    r = results[0] if isinstance(results, list) else results

    if r.boxes is None or len(r.boxes) == 0:
        return None, np.zeros((H, W), dtype=bool), 0.0, results

    best = int(r.boxes.conf.argmax().item())
    bbox = r.boxes.xyxy[best].cpu().numpy().astype(int)
    conf = float(r.boxes.conf[best].item())

    mask = np.zeros((H, W), dtype=bool)
    if r.masks is not None:
        raw  = r.masks.data[best].cpu().numpy()          # float32 [0,1]
        raw  = cv2.resize(raw.astype(np.float32), (W, H))
        mask = raw > 0.5
    else:
        x1, y1, x2, y2 = bbox
        mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True

    return bbox, mask, conf, results


# ── Public API ────────────────────────────────────────────────────────────────

def yoloe_text_prompt(model, scene_path: str, text_prompts: list,
                      conf: float = 0.25, **kwargs):
    """
    Run YOLOE with a text prompt.

    Args:
        model        : loaded YOLOE model instance
        scene_path   : absolute path to the scene image
        text_prompts : list of class name strings, e.g. ["ball", "tennis ball"]
        conf         : confidence threshold

    Returns:
        (bbox int[4], mask bool H x W, conf float, raw_results)
    """
    results = model.predict(
        source=scene_path,
        texts=text_prompts,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_result(results, scene_path)


def yoloe_visual_prompt(model, scene_path: str, anchor_path: str,
                        anchor_bbox: np.ndarray, conf: float = 0.25, **kwargs):
    """
    Run YOLOE with a visual prompt (reference image + bounding box).

    Args:
        model        : loaded YOLOE model instance
        scene_path   : absolute path to the query/scene image
        anchor_path  : absolute path to the anchor/reference image
        anchor_bbox  : np.array of shape (1,4) or (4,) -- [x1,y1,x2,y2] in anchor coords
        conf         : confidence threshold

    Returns:
        (bbox int[4], mask bool H x W, conf float, raw_results)
    """
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    anchor_bbox = np.array(anchor_bbox, dtype=np.float32)
    if anchor_bbox.ndim == 1:
        anchor_bbox = anchor_bbox[np.newaxis, :]   # (1,4)

    results = model.predict(
        source=scene_path,
        refer_image=anchor_path,
        visual_prompts={"bboxes": anchor_bbox, "cls": [0] * len(anchor_bbox)},
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_result(results, scene_path)


def yoloe_free_prompt(model, scene_path: str, conf: float = 0.25, **kwargs):
    """
    Run YOLOE in prompt-free mode (detects all objects, returns the best one).
    Use the raw results to iterate over all detections.

    Args:
        model      : loaded YOLOE-PF model instance
        scene_path : absolute path to the scene image
        conf       : confidence threshold

    Returns:
        (bbox int[4], mask bool H x W, conf float, raw_results)
        raw_results can be passed to collect_all_detections() in app.py
    """
    results = model.predict(
        source=scene_path,
        conf=conf,
        verbose=False,
        **kwargs,
    )
    return _parse_result(results, scene_path)


def visualize_detection(scene_path: str, anchor_path: str,
                        bbox, mask, conf: float):
    """
    Simple matplotlib visualisation (kept for notebook compatibility).
    Only used when running outside Streamlit.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        scene_bgr  = cv2.imread(scene_path)
        anchor_bgr = cv2.imread(anchor_path)
        scene_rgb  = cv2.cvtColor(scene_bgr,  cv2.COLOR_BGR2RGB)
        anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(anchor_rgb)
        axes[0].set_title("Anchor")
        axes[0].axis("off")

        axes[1].imshow(scene_rgb)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="cyan", facecolor="none"
            )
            axes[1].add_patch(rect)
        axes[1].set_title(f"Detection  conf={conf:.3f}")
        axes[1].axis("off")

        overlay = scene_rgb.copy().astype(np.float32)
        if mask is not None and mask.any():
            H, W = scene_rgb.shape[:2]
            m = cv2.resize(mask.astype(np.float32), (W, H)).astype(bool)
            overlay[m] = overlay[m] * 0.5 + np.array([0, 180, 255], np.float32) * 0.5
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title("Mask overlay")
        axes[2].axis("off")

        plt.suptitle("YOLOE Detection", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib not available -- skipping visualisation.")
