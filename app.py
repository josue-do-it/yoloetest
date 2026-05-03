"""
YOLOE · Block 1 — Prompt → Mask → Crop
Fixes applied:
  - show_results() defined BEFORE it is called (was defined after → NameError)
  - __file__ replaced by a robust APP_DIR computed at startup (fixes path crash in frozen/Streamlit env)
  - helpers import uses importlib + sys.path so stale cache never shadows a reload
  - _parse_best() no longer reads scene_path from disk (file may be deleted already) → uses H,W passed in
  - collect_all_detections() same fix — receives scene_rgb directly
  - os.unlink() wrapped in try/except so a missing temp file never crashes the app
  - anchor_up / scene_up NameError in run_v guard — variables guaranteed to exist
  - free-prompt: all_dets built before os.unlink so cv2.imread still works
  - mask dtype forced to bool after every resize (cv2.resize returns float32)
  - overlay_mask_rgb: added guard for empty mask so no crash on zero-pixel mask
  - crop_object: added guard when reg_rgb is empty (zero-area bbox after padding clamp)
  - download buttons always rendered (no conditional skip that left column blank)
  - helpers fallback prints the actual exception to stderr for easier debugging
"""

import sys
import os
import io
import json
import tempfile
import importlib
import traceback
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ── Resolve app directory once, robustly ──────────────────────────────────────
# __file__ is unreliable in some Streamlit/PyInstaller setups.
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path(os.getcwd())

# Add app dir to sys.path so helpers.py is always importable
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="YOLOE · Block 1",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, html, body { font-family: 'IBM Plex Sans', sans-serif; }
.stApp       { background: #080a0f; color: #d4d8e8; }
.stSidebar   { background: #0d1017 !important; border-right: 1px solid #1e2130 !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

.pipe-banner {
    background: #0f1520; border: 1px solid #1e2a40; border-radius: 10px;
    padding: 14px 20px; margin-bottom: 18px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.pipe-step {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    font-weight: 600; letter-spacing: 0.06em; padding: 4px 10px;
    border-radius: 4px; text-transform: uppercase;
}
.pipe-step.active  { background:#1a2a45; color:#5b9cf6; border:1px solid #2a4070; }
.pipe-step.pending { background:#1a1a2a; color:#3a3f5a; border:1px solid #252535; }
.pipe-arrow { color: #2a3050; font-size: 0.85rem; }

.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a5070; margin-bottom: 7px; padding-bottom: 5px;
    border-bottom: 1px solid #1a1e2e;
}
.mode-pill {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem;
    font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 14px;
}
.pill-text   { background:#0d2040; color:#5b9cf6; border:1px solid #1a3a6a; }
.pill-visual { background:#1e0d3a; color:#a87af5; border:1px solid #3a1a6a; }
.pill-free   { background:#0d2a1e; color:#3dd6a0; border:1px solid #1a4a30; }

.metric-card { background:#0d1220; border:1px solid #1a2035; border-radius:8px; padding:12px 16px; }
.metric-card .val { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#5b9cf6; line-height:1.1; }
.metric-card .lbl { font-size:0.63rem; color:#4a5070; text-transform:uppercase; letter-spacing:0.08em; margin-top:3px; }

.panel-title { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:#4a5070; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }

.crop-info {
    background:#0f2a1e; border:1px solid #1a4a30; border-radius:6px;
    padding:8px 12px; font-family:'IBM Plex Mono',monospace;
    font-size:0.68rem; color:#3dd6a0; margin-top:8px; line-height:1.9;
}
.stButton > button {
    background: #0d1a30 !important; color: #5b9cf6 !important;
    border: 1px solid #2a4a7a !important; border-radius: 7px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.74rem !important;
    font-weight: 600 !important; letter-spacing: 0.05em !important;
}
.stButton > button:hover { background: #1a2a50 !important; }
div[data-testid="stFileUploader"] {
    background: #0d1220 !important; border: 1px dashed #1e2a40 !important;
    border-radius: 8px !important;
}
.stTextArea textarea {
    background: #0d1220 !important; border: 1px solid #1e2a40 !important;
    color: #d4d8e8 !important; font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
}
label { color: #8090b0 !important; font-size: 0.78rem !important; }
hr    { border-color: #1a1e2e !important; }
.stDownloadButton > button {
    background: #0f2a1e !important; color: #3dd6a0 !important;
    border: 1px solid #1a4a30 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_upload(uploaded) -> str:
    """Save a Streamlit UploadedFile to a temp file; returns the path."""
    suffix = Path(uploaded.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded.seek(0)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def safe_unlink(*paths):
    """Delete temp files, ignoring errors if already gone."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def read_image_rgb(path: str) -> np.ndarray:
    """Read image from disk and return H×W×3 uint8 RGB array. Raises on failure."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cv2.imread failed for: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    uint8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(uint8, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest)

def refine_mask(mask: np.ndarray, dilate: int = 0, erode: int = 0) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if erode  > 0: m = cv2.erode (m, k, iterations=erode)
    if dilate > 0: m = cv2.dilate(m, k, iterations=dilate)
    return m.astype(bool)

def tight_bbox_from_mask(mask: np.ndarray):
    """Return (x1,y1,x2,y2) tight around mask's True pixels, or None."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2), int(y2)

def crop_object(img_rgb: np.ndarray, mask: np.ndarray, padding: int = 8):
    """
    Returns (crop_rgba, crop_square, (x1p,y1p,x2p,y2p)) or (None,None,None).
    crop_rgba   : tight RGBA PIL image (transparent bg)
    crop_square : square-padded RGBA PIL image
    """
    bbox = tight_bbox_from_mask(mask)
    if bbox is None:
        return None, None, None

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1p = max(0, x1 - padding); y1p = max(0, y1 - padding)
    x2p = min(W, x2 + padding); y2p = min(H, y2 + padding)

    reg_rgb  = img_rgb[y1p:y2p, x1p:x2p]
    reg_mask = mask   [y1p:y2p, x1p:x2p]

    # Guard: zero-area crop (can happen when mask touches the image border)
    if reg_rgb.size == 0 or reg_mask.size == 0:
        return None, None, None

    alpha     = (reg_mask * 255).astype(np.uint8)
    rgba_arr  = np.dstack([reg_rgb.copy(), alpha])
    crop_rgba = Image.fromarray(rgba_arr, "RGBA")

    # Square-padded version
    rh, rw = reg_rgb.shape[:2]
    side   = max(rh, rw)
    sq_arr = np.zeros((side, side, 4), dtype=np.uint8)
    oy = (side - rh) // 2
    ox = (side - rw) // 2
    sq_arr[oy:oy+rh, ox:ox+rw] = rgba_arr
    crop_square = Image.fromarray(sq_arr, "RGBA")

    return crop_rgba, crop_square, (x1p, y1p, x2p, y2p)

def overlay_mask_rgb(img_rgb: np.ndarray, mask: np.ndarray,
                     color=(60, 180, 255), alpha=0.45) -> np.ndarray:
    if mask is None or not mask.any():
        return img_rgb.copy()
    out = img_rgb.copy().astype(np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def draw_bbox_on(img_rgb: np.ndarray, bbox, color=(60, 180, 255), thickness=2) -> np.ndarray:
    out = img_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out

def ensure_bool_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize mask to (H,W) if needed and cast to bool."""
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.float32), (W, H))
    return mask.astype(bool)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading YOLOE model…")
def load_model(name: str):
    from ultralytics import YOLOE
    import torch
    m = YOLOE(name)
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    return m

@st.cache_resource(show_spinner="Loading YOLOE-PF model…")
def load_model_pf(name: str):
    from ultralytics import YOLOE
    import torch
    m = YOLOE(name)
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS IMPORTER  (safe, with clear error message)
# ═══════════════════════════════════════════════════════════════════════════════

def _import_helper(fn_name: str):
    """
    Try to import fn_name from helpers.py next to this app.
    Returns the function, or None if not found (caller uses raw ultralytics).
    """
    try:
        if "helpers" in sys.modules:
            importlib.reload(sys.modules["helpers"])
        import helpers
        fn = getattr(helpers, fn_name, None)
        if fn is None:
            print(f"[YOLOE app] helpers.py found but '{fn_name}' not defined — using fallback",
                  file=sys.stderr)
        return fn
    except ModuleNotFoundError:
        print(f"[YOLOE app] helpers.py not found in {APP_DIR} — using raw ultralytics fallback",
              file=sys.stderr)
        return None
    except Exception as e:
        print(f"[YOLOE app] helpers import error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT PARSER  (shared by all three modes)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_best_result(results, H: int, W: int):
    """
    Extract the highest-confidence detection from raw ultralytics results.
    Returns (bbox int[4], mask bool H×W, conf float, results)
    """
    r = results[0] if isinstance(results, list) else results
    if r.boxes is None or len(r.boxes) == 0:
        return None, np.zeros((H, W), dtype=bool), 0.0, results

    best  = int(r.boxes.conf.argmax().item())
    bbox  = r.boxes.xyxy[best].cpu().numpy().astype(int)
    conf  = float(r.boxes.conf[best].item())

    mask = np.zeros((H, W), dtype=bool)
    if r.masks is not None:
        raw  = r.masks.data[best].cpu().numpy()  # float32 [0,1]
        mask = ensure_bool_mask(raw, H, W)
    else:
        x1, y1, x2, y2 = bbox
        mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = True

    return bbox, mask, conf, results

def collect_all_detections(results, H: int, W: int):
    """
    Return list of (bbox int[4], mask bool H×W, conf float, label str).
    """
    r = results[0] if isinstance(results, list) else results
    dets = []
    if r.boxes is None:
        return dets

    for i in range(len(r.boxes)):
        bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(r.boxes.conf[i].item())
        cls  = int(r.boxes.cls[i].item())
        lbl  = r.names.get(cls, str(cls)) if hasattr(r, "names") and r.names else str(cls)
        mask = np.zeros((H, W), dtype=bool)
        if r.masks is not None:
            raw  = r.masks.data[i].cpu().numpy()
            mask = ensure_bool_mask(raw, H, W)
        else:
            x1, y1, x2, y2 = bbox
            mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = True
        dets.append((bbox, mask, conf, lbl))
    return dets


# ═══════════════════════════════════════════════════════════════════════════════
# CORE RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_text_prompt(model, scene_path: str, scene_rgb: np.ndarray,
                    text_prompts: list, conf: float):
    H, W = scene_rgb.shape[:2]
    fn = _import_helper("yoloe_text_prompt")
    if fn is not None:
        try:
            bbox, mask, c, res = fn(model, scene_path=scene_path,
                                    text_prompts=text_prompts, conf=conf)
            if mask is not None:
                mask = ensure_bool_mask(np.array(mask, dtype=np.float32), H, W)
            return bbox, mask, c, res
        except Exception as e:
            st.warning(f"helpers.yoloe_text_prompt failed ({e}), using fallback.")
            traceback.print_exc(file=sys.stderr)

    # Raw fallback
    res = model.predict(scene_path, texts=text_prompts, conf=conf)
    return parse_best_result(res, H, W)


def run_visual_prompt(model, scene_path: str, scene_rgb: np.ndarray,
                      anchor_path: str, anchor_bbox: np.ndarray, conf: float):
    H, W = scene_rgb.shape[:2]
    fn = _import_helper("yoloe_visual_prompt")
    if fn is not None:
        try:
            bbox, mask, c, res = fn(model, scene_path=scene_path,
                                    anchor_path=anchor_path,
                                    anchor_bbox=anchor_bbox, conf=conf)
            if mask is not None:
                mask = ensure_bool_mask(np.array(mask, dtype=np.float32), H, W)
            return bbox, mask, c, res
        except Exception as e:
            st.warning(f"helpers.yoloe_visual_prompt failed ({e}), using fallback.")
            traceback.print_exc(file=sys.stderr)

    # Raw fallback
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    res = model.predict(
        scene_path,
        refer_image=anchor_path,
        visual_prompts={"cls": [0], "bboxes": anchor_bbox},
        predictor=YOLOEVPSegPredictor,
        conf=conf,
    )
    return parse_best_result(res, H, W)


def run_free_prompt(model_pf, scene_path: str, scene_rgb: np.ndarray, conf: float):
    H, W = scene_rgb.shape[:2]
    fn = _import_helper("yoloe_free_prompt")
    if fn is not None:
        try:
            bbox, mask, c, res = fn(model_pf, scene_path=scene_path, conf=conf)
            if mask is not None:
                mask = ensure_bool_mask(np.array(mask, dtype=np.float32), H, W)
            return bbox, mask, c, res
        except Exception as e:
            st.warning(f"helpers.yoloe_free_prompt failed ({e}), using fallback.")
            traceback.print_exc(file=sys.stderr)

    res = model_pf.predict(scene_path, conf=conf)
    return parse_best_result(res, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT RENDERER  — defined BEFORE the mode blocks that call it
# ═══════════════════════════════════════════════════════════════════════════════

def show_results(scene_rgb: np.ndarray, mask, bbox, conf_val: float,
                 crop_padding: int, use_largest: bool,
                 erode_iters: int, dilate_iters: int,
                 mode_label: str = ""):

    H, W = scene_rgb.shape[:2]

    # ── Guard: nothing detected ──────────────────────────────────
    if mask is None or not np.asarray(mask).any():
        st.warning("No object detected — try lowering the confidence threshold.")
        return

    mask = ensure_bool_mask(np.asarray(mask, dtype=np.float32), H, W)

    # ── Mask refinement ──────────────────────────────────────────
    if use_largest:
        mask = largest_component(mask)
    if erode_iters > 0 or dilate_iters > 0:
        mask = refine_mask(mask, dilate=dilate_iters, erode=erode_iters)

    if not mask.any():
        st.warning("Mask is empty after refinement — try reducing erode or increasing dilate.")
        return

    # ── Visualisations ───────────────────────────────────────────
    overlay  = overlay_mask_rgb(scene_rgb, mask, color=(60, 180, 255))
    if bbox is not None:
        overlay = draw_bbox_on(overlay, bbox, color=(60, 180, 255))

    mask_vis = np.stack([(mask * 255).astype(np.uint8)] * 3, axis=-1)

    crop_rgba, crop_square, crop_bbox = crop_object(scene_rgb, mask, padding=crop_padding)

    # Composite crop onto dark background for preview
    if crop_square is not None:
        bg = Image.new("RGB", crop_square.size, (13, 18, 32))
        bg.paste(crop_square, mask=crop_square.split()[3])
        crop_preview = np.array(bg)
    else:
        crop_preview = None

    # ── Metrics ──────────────────────────────────────────────────
    mask_px  = int(mask.sum())
    mask_pct = mask_px / (H * W) * 100

    mc = st.columns(4)
    mc[0].markdown(f'<div class="metric-card"><div class="val">{conf_val:.3f}</div><div class="lbl">Confidence</div></div>', unsafe_allow_html=True)
    mc[1].markdown(f'<div class="metric-card"><div class="val">{mask_px:,}</div><div class="lbl">Mask pixels</div></div>', unsafe_allow_html=True)
    mc[2].markdown(f'<div class="metric-card"><div class="val">{mask_pct:.1f}%</div><div class="lbl">Coverage</div></div>', unsafe_allow_html=True)
    if bbox is not None:
        bw = int(bbox[2] - bbox[0]); bh = int(bbox[3] - bbox[1])
        mc[3].markdown(f'<div class="metric-card"><div class="val">{bw}×{bh}</div><div class="lbl">BBox size</div></div>', unsafe_allow_html=True)
    else:
        mc[3].markdown('<div class="metric-card"><div class="val">—</div><div class="lbl">BBox size</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Three result panels ───────────────────────────────────────
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown('<div class="panel-title">① Mask overlay</div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)

    with p2:
        st.markdown('<div class="panel-title">② Binary mask</div>', unsafe_allow_html=True)
        st.image(mask_vis, use_container_width=True)

    with p3:
        st.markdown('<div class="panel-title">③ Cropped object → Any6D</div>', unsafe_allow_html=True)
        if crop_preview is not None:
            st.image(crop_preview, use_container_width=True)
            if crop_bbox:
                cx1, cy1, cx2, cy2 = crop_bbox
                st.markdown(
                    f'<div class="crop-info">'
                    f'origin&nbsp;&nbsp; ({cx1}, {cy1})<br>'
                    f'size&nbsp;&nbsp;&nbsp;&nbsp; {cx2-cx1} × {cy2-cy1} px<br>'
                    f'padding&nbsp; {crop_padding} px'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Crop failed — mask may be empty or at image edge.")

    # ── Download row ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Export for Any6D</div>', unsafe_allow_html=True)

    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), "L")
    meta = {
        "mode":             mode_label,
        "conf":             round(float(conf_val), 4),
        "bbox_xyxy":        [int(v) for v in bbox] if bbox is not None else None,
        "crop_bbox_xyxy":   list(crop_bbox) if crop_bbox else None,
        "mask_pixels":      mask_px,
        "mask_coverage_pct": round(mask_pct, 3),
        "image_hw":         [H, W],
        "crop_padding":     crop_padding,
    }

    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button("⬇  mask.png",
        pil_to_bytes(mask_pil), "mask.png", "image/png",
        key=f"dl_mask_{mode_label}")
    dl2.download_button("⬇  crop_rgba.png",
        pil_to_bytes(crop_rgba) if crop_rgba else b"", "crop_rgba.png", "image/png",
        disabled=crop_rgba is None, key=f"dl_rgba_{mode_label}")
    dl3.download_button("⬇  crop_square.png",
        pil_to_bytes(crop_square) if crop_square else b"", "crop_square.png", "image/png",
        disabled=crop_square is None, key=f"dl_sq_{mode_label}")
    dl4.download_button("⬇  meta.json",
        json.dumps(meta, indent=2).encode(), "detection_meta.json", "application/json",
        key=f"dl_meta_{mode_label}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (must appear before any st.columns / mode blocks)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;
                font-weight:600;color:#5b9cf6;">YOLOE Studio</div>
    <div style="font-size:0.67rem;color:#4a5070;margin:2px 0 18px;
                font-family:'IBM Plex Mono',monospace;letter-spacing:0.1em;">
        BLOCK 1 · MASK &amp; CROP</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Prompt mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["📝  Text", "🖼️  Visual", "🔮  Free"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Seg model", [
        "yoloe-26l-seg.pt", "yoloe-11l-seg.pt",
        "yoloe-11m-seg.pt", "yoloe-11s-seg.pt",
    ])
    pf_choice = st.selectbox("Free-prompt model", [
        "yoloe-26l-seg-pf.pt", "yoloe-11l-seg-pf.pt",
    ])

    st.markdown("---")
    st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

    st.markdown('<div class="section-label">Mask refinement</div>', unsafe_allow_html=True)
    use_largest  = st.checkbox("Largest component only", value=True)
    erode_iters  = st.slider("Erode",  0, 5, 0)
    dilate_iters = st.slider("Dilate", 0, 5, 0)

    st.markdown('<div class="section-label">Crop settings</div>', unsafe_allow_html=True)
    crop_padding = st.slider("Padding (px)", 0, 80, 12)

    st.markdown("---")
    st.caption(f"App dir: `{APP_DIR}`")
    helpers_ok = (APP_DIR / "helpers.py").exists()
    if helpers_ok:
        st.caption("✅ helpers.py found")
    else:
        st.caption("⚠️ helpers.py not found — using ultralytics fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE BANNER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="pipe-banner">
  <div class="pipe-step active">① Scene image</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step active">② Prompt</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step active">③ YOLOE mask</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step active">④ Crop</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step pending">⑤ Any6D pose estimation</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — TEXT PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

if "📝" in mode:
    st.markdown('<span class="mode-pill pill-text">Text Prompt</span>', unsafe_allow_html=True)

    col_img, col_cfg = st.columns([1.5, 1])

    with col_img:
        st.markdown('<div class="section-label">Scene image</div>', unsafe_allow_html=True)
        scene_up = st.file_uploader(
            "scene", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key="t_scene",
        )
        if scene_up:
            scene_up.seek(0)
            st.image(scene_up, use_container_width=True)

    with col_cfg:
        st.markdown('<div class="section-label">Class names — one per line</div>', unsafe_allow_html=True)
        raw = st.text_area("", value="ball\ntennis ball", height=120,
                           label_visibility="collapsed", key="t_prompts")
        prompts = [p.strip() for p in raw.strip().splitlines() if p.strip()]
        st.caption(f"{len(prompts)} class(es): {', '.join(prompts)}")
        st.markdown("<br>", unsafe_allow_html=True)
        run_t = st.button("▶  Run — Text Prompt", use_container_width=True, key="run_t")

    if run_t:
        if not scene_up:
            st.error("Upload a scene image first.")
            st.stop()
        if not prompts:
            st.error("Enter at least one class name.")
            st.stop()

        scene_path = None
        try:
            scene_path = save_upload(scene_up)
            scene_rgb  = read_image_rgb(scene_path)
            model      = load_model(model_choice)

            with st.spinner("Segmenting…"):
                bbox, mask, conf_val, _ = run_text_prompt(
                    model, scene_path, scene_rgb, prompts, conf_thresh)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            traceback.print_exc(file=sys.stderr)
            st.stop()
        finally:
            safe_unlink(scene_path)

        st.markdown("---")
        show_results(scene_rgb, mask, bbox, conf_val,
                     crop_padding, use_largest, erode_iters, dilate_iters,
                     mode_label="text")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — VISUAL PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

elif "🖼️" in mode:
    st.markdown('<span class="mode-pill pill-visual">Visual Prompt</span>', unsafe_allow_html=True)

    col_anc, col_scn = st.columns(2)

    # Declare with defaults so the run_v block never hits NameError
    anchor_up = None
    scene_up  = None

    with col_anc:
        st.markdown('<div class="section-label">Anchor — reference object image</div>', unsafe_allow_html=True)
        anchor_up = st.file_uploader(
            "anchor", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key="v_anchor",
        )
        if anchor_up:
            anchor_up.seek(0)
            st.image(anchor_up, use_container_width=True)

        st.markdown('<div class="section-label">Object bbox in anchor (pixels)</div>', unsafe_allow_html=True)
        bc = st.columns(4)
        ax1 = bc[0].number_input("x1", 0, 99999, 0,   key="vx1")
        ay1 = bc[1].number_input("y1", 0, 99999, 0,   key="vy1")
        ax2 = bc[2].number_input("x2", 0, 99999, 640, key="vx2")
        ay2 = bc[3].number_input("y2", 0, 99999, 480, key="vy2")
        st.caption(f"Bbox  [{ax1}, {ay1}, {ax2}, {ay2}]")

    with col_scn:
        st.markdown('<div class="section-label">Scene — query image</div>', unsafe_allow_html=True)
        scene_up = st.file_uploader(
            "scene", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key="v_scene",
        )
        if scene_up:
            scene_up.seek(0)
            st.image(scene_up, use_container_width=True)

    run_v = st.button("▶  Run — Visual Prompt", use_container_width=True, key="run_v")

    if run_v:
        if not anchor_up or not scene_up:
            st.error("Upload both anchor and scene images.")
            st.stop()

        anchor_path = scene_path = None
        try:
            anchor_path  = save_upload(anchor_up)
            scene_path   = save_upload(scene_up)
            scene_rgb    = read_image_rgb(scene_path)
            anchor_bbox  = np.array([[ax1, ay1, ax2, ay2]], dtype=np.float32)
            model        = load_model(model_choice)

            with st.spinner("Segmenting…"):
                bbox, mask, conf_val, _ = run_visual_prompt(
                    model, scene_path, scene_rgb, anchor_path, anchor_bbox, conf_thresh)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            traceback.print_exc(file=sys.stderr)
            st.stop()
        finally:
            safe_unlink(anchor_path, scene_path)

        st.markdown("---")
        show_results(scene_rgb, mask, bbox, conf_val,
                     crop_padding, use_largest, erode_iters, dilate_iters,
                     mode_label="visual")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — FREE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

else:
    st.markdown('<span class="mode-pill pill-free">Free Prompt</span>', unsafe_allow_html=True)

    col_img, col_info = st.columns([1.5, 1])

    with col_img:
        st.markdown('<div class="section-label">Scene image</div>', unsafe_allow_html=True)
        scene_up = st.file_uploader(
            "scene", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key="f_scene",
        )
        if scene_up:
            scene_up.seek(0)
            st.image(scene_up, use_container_width=True)

    with col_info:
        st.info(
            "**Free prompt** detects all objects automatically — "
            "no class names or reference image needed.\n\n"
            "After detection, pick the object to crop for **Any6D**."
        )
        run_f = st.button("▶  Run — Free Prompt", use_container_width=True, key="run_f")

    if run_f:
        if not scene_up:
            st.error("Upload a scene image first.")
            st.stop()

        scene_path = None
        try:
            scene_path  = save_upload(scene_up)
            scene_rgb   = read_image_rgb(scene_path)   # read BEFORE unlink
            H, W        = scene_rgb.shape[:2]
            m_pf        = load_model_pf(pf_choice)

            with st.spinner("Detecting all objects…"):
                _, _, _, raw_results = run_free_prompt(m_pf, scene_path, scene_rgb, conf_thresh)
                all_dets = collect_all_detections(raw_results, H, W)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            traceback.print_exc(file=sys.stderr)
            st.stop()
        finally:
            safe_unlink(scene_path)   # safe to unlink now — we have scene_rgb in memory

        if not all_dets:
            st.warning("No objects detected. Try lowering the confidence threshold.")
            st.stop()

        # ── Overview with all detections ──────────────────────────
        COLORS = [
            (91, 156, 246), (163, 122, 245), (61, 214, 160),
            (246, 180, 91), (246,  91,  91), (91, 246, 220),
        ]
        overview = scene_rgb.copy()
        for i, (bb, mk, cv_, lbl) in enumerate(all_dets):
            col = COLORS[i % len(COLORS)]
            overview = overlay_mask_rgb(overview, mk, color=col, alpha=0.35)
            x1, y1, x2, y2 = [int(v) for v in bb]
            cv2.rectangle(overview, (x1, y1), (x2, y2), col, 2)
            cv2.putText(overview, f"[{i}] {lbl} {cv_:.2f}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

        st.markdown("---")
        st.markdown('<div class="section-label">All detections</div>', unsafe_allow_html=True)
        st.image(overview, use_container_width=True)

        mc = st.columns(3)
        mc[0].markdown(f'<div class="metric-card"><div class="val">{len(all_dets)}</div><div class="lbl">Objects found</div></div>', unsafe_allow_html=True)
        unique_cls = len(set(lbl for _, _, _, lbl in all_dets))
        mc[1].markdown(f'<div class="metric-card"><div class="val">{unique_cls}</div><div class="lbl">Classes</div></div>', unsafe_allow_html=True)
        best_c = max(c for _, _, c, _ in all_dets)
        mc[2].markdown(f'<div class="metric-card"><div class="val">{best_c:.3f}</div><div class="lbl">Best conf</div></div>', unsafe_allow_html=True)

        # ── Object selector ───────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">Select object to crop for Any6D</div>', unsafe_allow_html=True)
        options = [f"[{i}]  {lbl}  —  conf {c:.2f}" for i, (bb, mk, c, lbl) in enumerate(all_dets)]
        pick    = st.selectbox("", options, label_visibility="collapsed")
        idx     = int(pick.split("]")[0].replace("[", "").strip())

        sel_bbox, sel_mask, sel_conf, sel_lbl = all_dets[idx]

        st.markdown("---")
        show_results(scene_rgb, sel_mask, sel_bbox, sel_conf,
                     crop_padding, use_largest, erode_iters, dilate_iters,
                     mode_label=f"free/{sel_lbl}")
