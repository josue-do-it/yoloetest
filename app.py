"""
YOLOE · Block 1 — Prompt → Mask → Crop
All bugs fixed + new UX:
  TEXT   : multi-line textarea → parsed into clickable tags, each selected → detected
  VISUAL : drawable canvas on anchor image → real pixel coords sent to model
  FREE   : full object list with individual crop button per detection
"""

import sys, os, io, json, tempfile, importlib, traceback
from pathlib import Path

import streamlit as st

# ── OpenCV import guard ───────────────────────────────────────────────────────
# On Streamlit Cloud the native bootstrap fails if system libs are missing.
# packages.txt installs libgl1/libglib2.0-0 etc., but guard here for clarity.
try:
    import cv2
except ImportError as _e:
    st.error(
        f"OpenCV failed to import: {_e}\n\n"
        "Make sure `packages.txt` contains `libgl1` and `libglib2.0-0`, "
        "and `requirements.txt` uses `opencv-python-headless` (not `opencv-python`)."
    )
    st.stop()

import numpy as np
from PIL import Image

# ── App directory (robust) ───────────────────────────────────────────────────
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path(os.getcwd())

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOE Studio · Block 1",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — full design system, light + dark mode ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── TOKENS ── */
:root {
  --bg:          #f8f9fc;
  --bg2:         #ffffff;
  --bg3:         #f0f2f8;
  --border:      #e2e6f0;
  --border2:     #d0d6e8;
  --text:        #111827;
  --text2:       #4b5563;
  --text3:       #9ca3af;
  --accent:      #4f6ef7;
  --accent-bg:   #eef1fe;
  --accent-b:    #c7d0fc;
  --purple:      #7c3aed;
  --purple-bg:   #f3effe;
  --purple-b:    #ddd0fa;
  --green:       #059669;
  --green-bg:    #ecfdf5;
  --green-b:     #a7f3d0;
  --red:         #dc2626;
  --red-bg:      #fef2f2;
  --shadow:      0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md:   0 4px 12px rgba(0,0,0,.08);
  --radius:      10px;
  --radius-sm:   6px;
  --font:        'Inter', system-ui, sans-serif;
  --mono:        'JetBrains Mono', 'Fira Code', monospace;
}

[data-theme="dark"] {
  --bg:          #0c0e14;
  --bg2:         #12151f;
  --bg3:         #181c28;
  --border:      #1e2235;
  --border2:     #252a3f;
  --text:        #e8ecf8;
  --text2:       #8892b0;
  --text3:       #4a5070;
  --accent:      #6281f8;
  --accent-bg:   #0d1630;
  --accent-b:    #1e2e5a;
  --purple:      #a78bfa;
  --purple-bg:   #1a0f35;
  --purple-b:    #3a1f6a;
  --green:       #34d399;
  --green-bg:    #052e1a;
  --green-b:     #0a4a2a;
  --red:         #f87171;
  --red-bg:      #2a0808;
  --shadow:      0 1px 3px rgba(0,0,0,.3);
  --shadow-md:   0 4px 16px rgba(0,0,0,.4);
}

/* ── BASE ── */
*, html, body { font-family: var(--font); }

.stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ── LOGO ── */
.yoloe-logo {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 0 20px;
}
.yoloe-logo-icon {
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, var(--accent), var(--purple));
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(79,110,247,.35);
}
.yoloe-logo-text { font-size: .95rem; font-weight: 700; color: var(--text); letter-spacing: -.01em; }
.yoloe-logo-sub  { font-size: .62rem; color: var(--text3); font-family: var(--mono);
  letter-spacing: .1em; text-transform: uppercase; margin-top: 1px; }

/* ── SECTION LABEL ── */
.section-label {
  font-family: var(--mono); font-size: .6rem; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 8px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* ── PIPELINE BANNER ── */
.pipe-banner {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px 18px; margin-bottom: 20px;
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  box-shadow: var(--shadow);
}
.pipe-step {
  font-family: var(--mono); font-size: .62rem; font-weight: 600;
  letter-spacing: .06em; padding: 4px 11px; border-radius: 20px;
  text-transform: uppercase; white-space: nowrap;
}
.pipe-step.active  { background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent-b); }
.pipe-step.pending { background: var(--bg3); color: var(--text3); border: 1px solid var(--border); }
.pipe-arrow { color: var(--border2); font-size: .8rem; }

/* ── MODE PILL ── */
.mode-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px; border-radius: 20px;
  font-family: var(--mono); font-size: .65rem; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase; margin-bottom: 18px;
}
.pill-text   { background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent-b); }
.pill-visual { background: var(--purple-bg); color: var(--purple); border: 1px solid var(--purple-b); }
.pill-free   { background: var(--green-bg);  color: var(--green);  border: 1px solid var(--green-b); }

/* ── PAGE TITLE ── */
.page-title {
  font-size: 1.55rem; font-weight: 700; color: var(--text);
  letter-spacing: -.03em; margin-bottom: 2px; line-height: 1.2;
}
.page-subtitle {
  font-size: .8rem; color: var(--text2); margin-bottom: 20px; font-weight: 400;
}

/* ── CARDS ── */
.card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px 20px;
  box-shadow: var(--shadow);
}

/* ── METRIC CARDS ── */
.metric-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 18px;
  box-shadow: var(--shadow); text-align: center;
}
.metric-card .val {
  font-family: var(--mono); font-size: 1.5rem; font-weight: 700;
  color: var(--accent); line-height: 1.1;
}
.metric-card .lbl {
  font-size: .62rem; color: var(--text3); text-transform: uppercase;
  letter-spacing: .09em; margin-top: 4px; font-weight: 500;
}

/* ── PANEL TITLE ── */
.panel-title {
  font-family: var(--mono); font-size: .62rem; font-weight: 600;
  color: var(--text3); text-transform: uppercase; letter-spacing: .1em;
  margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}

/* ── CROP INFO ── */
.crop-info {
  background: var(--green-bg); border: 1px solid var(--green-b);
  border-radius: var(--radius-sm); padding: 9px 13px;
  font-family: var(--mono); font-size: .68rem; color: var(--green);
  margin-top: 10px; line-height: 2;
}

/* ── DETECTION ROW ── */
.det-row {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); margin-bottom: 8px;
  box-shadow: var(--shadow); transition: border-color .15s;
}
.det-row:hover { border-color: var(--accent-b); }
.det-badge {
  font-family: var(--mono); font-size: .6rem; font-weight: 700;
  background: var(--accent-bg); color: var(--accent);
  border: 1px solid var(--accent-b); border-radius: 4px;
  padding: 2px 6px; min-width: 28px; text-align: center;
}
.det-label { font-size: .82rem; color: var(--text); font-weight: 500; flex: 1; }
.det-conf  { font-family: var(--mono); font-size: .72rem; color: var(--accent); min-width: 55px; text-align: right; }
.det-mask  { font-size: .68rem; color: var(--text3); min-width: 70px; text-align: right; }

/* ── STATUS BADGES ── */
.status-ok  { display:inline-flex; align-items:center; gap:5px; font-size:.72rem;
  color: var(--green); background: var(--green-bg); border: 1px solid var(--green-b);
  border-radius: 20px; padding: 2px 9px; }
.status-err { display:inline-flex; align-items:center; gap:5px; font-size:.72rem;
  color: var(--red); background: var(--red-bg); border: 1px solid #fca5a5;
  border-radius: 20px; padding: 2px 9px; }
.status-warn { display:inline-flex; align-items:center; gap:5px; font-size:.72rem;
  color: #d97706; background: #fffbeb; border: 1px solid #fde68a;
  border-radius: 20px; padding: 2px 9px; }

/* ── UPLOAD AREA ── */
div[data-testid="stFileUploader"] {
  background: var(--bg3) !important; border: 2px dashed var(--border2) !important;
  border-radius: var(--radius) !important; transition: border-color .2s !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {
  background: var(--bg2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; font-family: var(--mono) !important;
  border-radius: var(--radius-sm) !important; font-size: .82rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(79,110,247,.12) !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: var(--accent) !important; color: #fff !important;
  border: none !important; border-radius: var(--radius-sm) !important;
  font-family: var(--font) !important; font-size: .82rem !important;
  font-weight: 600 !important; padding: .5rem 1.2rem !important;
  box-shadow: 0 2px 6px rgba(79,110,247,.3) !important;
  transition: all .15s !important;
}
.stButton > button:hover {
  opacity: .88 !important;
  box-shadow: 0 4px 12px rgba(79,110,247,.4) !important;
}

/* ── DOWNLOAD BUTTONS ── */
.stDownloadButton > button {
  background: var(--green-bg) !important; color: var(--green) !important;
  border: 1px solid var(--green-b) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font) !important; font-size: .78rem !important;
  font-weight: 600 !important;
}

/* ── CHECKBOXES ── */
.tag-checkbox {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-radius: 20px; cursor: pointer;
  font-size: .78rem; font-family: var(--mono);
  background: var(--bg3); color: var(--text2); border: 1px solid var(--border2);
  margin: 3px 3px 3px 0; transition: all .15s;
}
.tag-checkbox.selected {
  background: var(--accent-bg); color: var(--accent);
  border-color: var(--accent-b); font-weight: 600;
}

/* ── SIDEBAR WIDGETS ── */
.stRadio > div > label { color: var(--text) !important; font-size: .82rem !important; }
.stSelectbox label, .stSlider label, .stCheckbox label {
  color: var(--text2) !important; font-size: .78rem !important;
}
hr { border-color: var(--border) !important; margin: 12px 0 !important; }
.stAlert { border-radius: var(--radius) !important; }

/* ── IMAGE PANELS ── */
.img-panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow);
}
.img-panel-header {
  padding: 8px 14px; border-bottom: 1px solid var(--border);
  background: var(--bg3);
  font-family: var(--mono); font-size: .62rem; font-weight: 600;
  color: var(--text3); text-transform: uppercase; letter-spacing: .1em;
  display: flex; align-items: center; gap: 6px;
}
.img-panel-body { padding: 10px; }

/* ── DIVIDER ── */
.section-divider {
  border: none; border-top: 1px solid var(--border); margin: 20px 0;
}

/* ── EXPORT ROW ── */
.export-label {
  font-family: var(--mono); font-size: .6rem; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: var(--text3);
  margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def save_upload(up) -> str:
    suffix = Path(up.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    up.seek(0); tmp.write(up.read()); tmp.flush(); tmp.close()
    return tmp.name

def safe_unlink(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p): os.unlink(p)
        except Exception: pass

def pil_to_bytes(img, fmt="PNG") -> bytes:
    buf = io.BytesIO(); img.save(buf, format=fmt); return buf.getvalue()

def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None: raise ValueError(f"Cannot read: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def ensure_bool_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.float32), (W, H))
    return mask.astype(bool)

def largest_component(mask: np.ndarray) -> np.ndarray:
    u = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    if n <= 1: return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))

def refine_mask(mask, dilate=0, erode=0):
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    if erode  > 0: m = cv2.erode (m, k, iterations=erode)
    if dilate > 0: m = cv2.dilate(m, k, iterations=dilate)
    return m.astype(bool)

def tight_bbox(mask):
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    if not rows.any(): return None
    y1,y2 = np.where(rows)[0][[0,-1]]; x1,x2 = np.where(cols)[0][[0,-1]]
    return int(x1),int(y1),int(x2),int(y2)

def crop_object(img_rgb, mask, padding=8):
    bb = tight_bbox(mask)
    if bb is None: return None, None, None
    H,W = img_rgb.shape[:2]
    x1,y1,x2,y2 = bb
    x1p=max(0,x1-padding); y1p=max(0,y1-padding)
    x2p=min(W,x2+padding); y2p=min(H,y2+padding)
    rr = img_rgb[y1p:y2p, x1p:x2p]; rm = mask[y1p:y2p, x1p:x2p]
    if rr.size == 0: return None, None, None
    alpha = (rm*255).astype(np.uint8)
    rgba  = np.dstack([rr.copy(), alpha])
    crop_rgba = Image.fromarray(rgba, "RGBA")
    rh,rw = rr.shape[:2]; side = max(rh,rw)
    sq = np.zeros((side,side,4), dtype=np.uint8)
    oy=(side-rh)//2; ox=(side-rw)//2
    sq[oy:oy+rh, ox:ox+rw] = rgba
    return crop_rgba, Image.fromarray(sq,"RGBA"), (x1p,y1p,x2p,y2p)

def overlay_mask(img, mask, color=(60,180,255), alpha=0.45):
    if mask is None or not mask.any(): return img.copy()
    out = img.copy().astype(np.float32)
    out[mask] = out[mask]*(1-alpha) + np.array(color,np.float32)*alpha
    return np.clip(out,0,255).astype(np.uint8)

def draw_bbox_img(img, bbox, color=(60,180,255), t=2):
    out = img.copy()
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(out,(x1,y1),(x2,y2),color,t)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS  — argv patch prevents CLI dispatch under Streamlit
# ════════════════════════════════════════════════════════════════════════════════

def _load_yoloe(name: str):
    import torch
    from ultralytics import YOLOE
    orig = sys.argv[:]
    sys.argv = ["yoloe_app"]
    try:
        m = YOLOE(name, verbose=False)
    finally:
        sys.argv = orig
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    return m

@st.cache_resource(show_spinner="Loading YOLOE model…")
def load_model(name): return _load_yoloe(name)

@st.cache_resource(show_spinner="Loading YOLOE-PF model…")
def load_model_pf(name): return _load_yoloe(name)


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS IMPORTER  — always fresh, shows real error
# ════════════════════════════════════════════════════════════════════════════════

def _import_helper(fn_name: str):
    try:
        if "helpers" in sys.modules:
            importlib.reload(sys.modules["helpers"])
        import helpers
        fn = getattr(helpers, fn_name, None)
        if fn is None:
            st.warning(f"helpers.py: '{fn_name}' not defined — using fallback")
        return fn
    except ModuleNotFoundError:
        return None
    except Exception as e:
        st.warning(f"helpers import error: {e}")
        traceback.print_exc(file=sys.stderr)
        return None


# ════════════════════════════════════════════════════════════════════════════════
# RESULT PARSERS
# ════════════════════════════════════════════════════════════════════════════════

def parse_best(results, H, W):
    r = results[0] if isinstance(results, list) else results
    if r.boxes is None or len(r.boxes) == 0:
        return None, np.zeros((H,W),dtype=bool), 0.0, results
    best = int(r.boxes.conf.argmax().item())
    bbox = r.boxes.xyxy[best].cpu().numpy().astype(int)
    conf = float(r.boxes.conf[best].item())
    mask = np.zeros((H,W), dtype=bool)
    if r.masks is not None:
        raw = r.masks.data[best].cpu().numpy().astype(np.float32)
        mask = ensure_bool_mask(raw, H, W)
    else:
        x1,y1,x2,y2=bbox; mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]=True
    return bbox, mask, conf, results

def parse_all(results, H, W):
    r = results[0] if isinstance(results, list) else results
    dets = []
    if r.boxes is None: return dets
    for i in range(len(r.boxes)):
        bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(r.boxes.conf[i].item())
        cls  = int(r.boxes.cls[i].item())
        lbl  = r.names.get(cls, str(cls)) if hasattr(r,"names") and r.names else str(cls)
        mask = np.zeros((H,W), dtype=bool)
        if r.masks is not None:
            raw = r.masks.data[i].cpu().numpy().astype(np.float32)
            mask = ensure_bool_mask(raw, H, W)
        else:
            x1,y1,x2,y2=bbox; mask[max(0,y1):min(H,y2),max(0,x1):min(W,x2)]=True
        dets.append({"bbox":bbox,"mask":mask,"conf":conf,"label":lbl})
    return dets


# ════════════════════════════════════════════════════════════════════════════════
# RUNNERS
# ════════════════════════════════════════════════════════════════════════════════

def run_text(model, scene_path, scene_rgb, prompts, conf):
    H,W = scene_rgb.shape[:2]
    fn = _import_helper("yoloe_text_prompt")
    if fn:
        try:
            bbox,mask,c,res = fn(model, scene_path=scene_path, text_prompts=prompts, conf=conf)
            if mask is not None: mask = ensure_bool_mask(np.array(mask,np.float32),H,W)
            return bbox,mask,c,res
        except Exception as e:
            st.warning(f"helpers.yoloe_text_prompt failed ({e}) — fallback")
            traceback.print_exc(file=sys.stderr)
    # Fallback: correct API
    model.set_classes(prompts)
    res = model.predict(source=scene_path, conf=conf, verbose=False)
    return parse_best(res, H, W)

def run_visual(model, scene_path, scene_rgb, anchor_path, anchor_bbox, conf):
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    H,W = scene_rgb.shape[:2]
    fn = _import_helper("yoloe_visual_prompt")
    if fn:
        try:
            bbox,mask,c,res = fn(model, scene_path=scene_path,
                                  anchor_path=anchor_path, anchor_bbox=anchor_bbox, conf=conf)
            if mask is not None: mask = ensure_bool_mask(np.array(mask,np.float32),H,W)
            return bbox,mask,c,res
        except Exception as e:
            st.warning(f"helpers.yoloe_visual_prompt failed ({e}) — fallback")
            traceback.print_exc(file=sys.stderr)
    # Fallback
    bbox_arr = np.array(anchor_bbox, np.float32)
    if bbox_arr.ndim == 1: bbox_arr = bbox_arr[np.newaxis,:]
    res = model.predict(
        source=scene_path,
        visual_prompts={"bboxes": bbox_arr.tolist(), "cls": list(range(len(bbox_arr)))},
        refer_image=anchor_path,
        predictor=YOLOEVPSegPredictor,
        conf=conf, verbose=False,
    )
    return parse_best(res, H, W)

def run_free(model_pf, scene_path, scene_rgb, conf):
    H,W = scene_rgb.shape[:2]
    fn = _import_helper("yoloe_free_prompt")
    if fn:
        try:
            bbox,mask,c,res = fn(model_pf, scene_path=scene_path, conf=conf)
            if mask is not None: mask = ensure_bool_mask(np.array(mask,np.float32),H,W)
            return bbox,mask,c,res
        except Exception as e:
            st.warning(f"helpers.yoloe_free_prompt failed ({e}) — fallback")
            traceback.print_exc(file=sys.stderr)
    res = model_pf.predict(source=scene_path, conf=conf, verbose=False)
    return parse_best(res, H, W)


# ════════════════════════════════════════════════════════════════════════════════
# RESULT RENDERER  — mask overlay + binary mask + crop
# ════════════════════════════════════════════════════════════════════════════════

def show_results(scene_rgb, mask, bbox, conf_val, crop_padding,
                 use_largest, erode_iters, dilate_iters, mode_label="",
                 key_suffix=""):
    H,W = scene_rgb.shape[:2]
    if mask is None or not np.asarray(mask).any():
        st.warning("No object detected — try lowering confidence or adjusting the prompt.")
        return

    mask = ensure_bool_mask(np.asarray(mask,np.float32), H, W)
    if use_largest: mask = largest_component(mask)
    if erode_iters > 0 or dilate_iters > 0:
        mask = refine_mask(mask, dilate=dilate_iters, erode=erode_iters)
    if not mask.any():
        st.warning("Mask empty after refinement."); return

    overlay   = overlay_mask(scene_rgb, mask)
    if bbox is not None: overlay = draw_bbox_img(overlay, bbox)
    mask_vis  = np.stack([(mask*255).astype(np.uint8)]*3, axis=-1)
    crop_rgba, crop_sq, crop_bbox = crop_object(scene_rgb, mask, crop_padding)
    if crop_sq:
        bg = Image.new("RGB", crop_sq.size, (13,18,32))
        bg.paste(crop_sq, mask=crop_sq.split()[3])
        crop_prev = np.array(bg)
    else:
        crop_prev = None

    mask_px  = int(mask.sum())
    mask_pct = mask_px/(H*W)*100
    _bw = f"{int(bbox[2]-bbox[0])}×{int(bbox[3]-bbox[1])}" if bbox is not None else "—"
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin:12px 0 4px;">
      <div class="metric-card" style="flex:1"><div class="val">{conf_val:.3f}</div><div class="lbl">Confidence</div></div>
      <div class="metric-card" style="flex:1"><div class="val">{mask_px:,}</div><div class="lbl">Mask pixels</div></div>
      <div class="metric-card" style="flex:1"><div class="val">{mask_pct:.1f}%</div><div class="lbl">Coverage</div></div>
      <div class="metric-card" style="flex:1"><div class="val">{_bw}</div><div class="lbl">BBox size</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="img-panel-header">🎭 Mask overlay</div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
    with p2:
        st.markdown('<div class="img-panel-header">⬛ Binary mask</div>', unsafe_allow_html=True)
        st.image(mask_vis, use_container_width=True)
    with p3:
        st.markdown('<div class="img-panel-header">✂️ Cropped object → Any6D</div>', unsafe_allow_html=True)
        if crop_prev is not None:
            st.image(crop_prev, use_container_width=True)
            if crop_bbox:
                cx1, cy1, cx2, cy2 = crop_bbox
                st.markdown(
                    f'<div class="crop-info">' +
                    f'<b>origin</b> ({cx1}, {cy1})<br>' +
                    f'<b>size</b>&nbsp;&nbsp; {cx2-cx1} × {cy2-cy1} px<br>' +
                    f'<b>pad</b>&nbsp;&nbsp;&nbsp; {crop_padding} px' +
                    f'</div>', unsafe_allow_html=True)
        else:
            st.warning("Crop failed — mask may be empty or at image edge.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="export-label">⬇ Export for Any6D</div>', unsafe_allow_html=True)
    mask_pil = Image.fromarray((mask*255).astype(np.uint8),"L")
    meta = {"mode":mode_label,"conf":round(float(conf_val),4),
            "bbox_xyxy":[int(v) for v in bbox] if bbox is not None else None,
            "crop_bbox_xyxy":list(crop_bbox) if crop_bbox else None,
            "mask_pixels":mask_px,"coverage_pct":round(mask_pct,3),
            "image_hw":[H,W],"crop_padding":crop_padding}
    dl1,dl2,dl3,dl4 = st.columns(4)
    dl1.download_button("⬇ mask.png",      pil_to_bytes(mask_pil),              "mask.png","image/png",          key=f"dlm_{key_suffix}")
    dl2.download_button("⬇ crop_rgba.png", pil_to_bytes(crop_rgba) if crop_rgba else b"", "crop_rgba.png","image/png", disabled=not crop_rgba, key=f"dlr_{key_suffix}")
    dl3.download_button("⬇ crop_sq.png",   pil_to_bytes(crop_sq)   if crop_sq   else b"", "crop_sq.png",  "image/png", disabled=not crop_sq,   key=f"dls_{key_suffix}")
    dl4.download_button("⬇ meta.json",     json.dumps(meta,indent=2).encode(),  "meta.json","application/json",   key=f"dlj_{key_suffix}")


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

# ── Dark/light mode ─────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

def _inject_theme():
    if st.session_state.get("dark_mode", True):
        st.markdown("""<style>
        .stApp{background:#0c0e14!important;color:#e8ecf8!important}
        section[data-testid="stSidebar"]{background:#12151f!important;border-right:1px solid #1e2235!important}
        div[data-testid="stFileUploader"]{background:#181c28!important;border-color:#252a3f!important}
        .stTextArea textarea{background:#12151f!important;border-color:#252a3f!important;color:#e8ecf8!important}
        label{color:#8892b0!important}
        hr{border-color:#1e2235!important}
        .metric-card{background:#12151f!important;border-color:#1e2235!important}
        .metric-card .val{color:#6281f8!important}
        .metric-card .lbl{color:#4a5070!important}
        .det-row{background:#12151f!important;border-color:#1e2235!important}
        .det-label{color:#e8ecf8!important}
        .det-conf{color:#6281f8!important}
        .pipe-banner{background:#12151f!important;border-color:#1e2235!important}
        .pipe-step.active{background:#0d1630!important;color:#6281f8!important;border-color:#1e2e5a!important}
        .pipe-step.pending{background:#181c28!important;color:#4a5070!important;border-color:#1e2235!important}
        .card{background:#12151f!important;border-color:#1e2235!important}
        .section-label{color:#4a5070!important;border-color:#1e2235!important}
        .stButton>button{background:#4f6ef7!important}
        .stDownloadButton>button{background:#052e1a!important;color:#34d399!important;border-color:#0a4a2a!important}
        .crop-info{background:#052e1a!important;border-color:#0a4a2a!important;color:#34d399!important}
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
        .stApp{background:#f8f9fc!important;color:#111827!important}
        section[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid #e2e6f0!important}
        div[data-testid="stFileUploader"]{background:#f0f2f8!important;border-color:#d0d6e8!important}
        .stTextArea textarea{background:#ffffff!important;border-color:#d0d6e8!important;color:#111827!important}
        label{color:#4b5563!important}
        hr{border-color:#e2e6f0!important}
        .metric-card{background:#ffffff!important;border-color:#e2e6f0!important}
        .metric-card .val{color:#4f6ef7!important}
        .metric-card .lbl{color:#9ca3af!important}
        .det-row{background:#ffffff!important;border-color:#e2e6f0!important}
        .det-label{color:#111827!important}
        .det-conf{color:#4f6ef7!important}
        .pipe-banner{background:#ffffff!important;border-color:#e2e6f0!important}
        .pipe-step.active{background:#eef1fe!important;color:#4f6ef7!important;border-color:#c7d0fc!important}
        .pipe-step.pending{background:#f0f2f8!important;color:#9ca3af!important;border-color:#e2e6f0!important}
        .card{background:#ffffff!important;border-color:#e2e6f0!important}
        .section-label{color:#9ca3af!important;border-color:#e2e6f0!important}
        .stButton>button{background:#4f6ef7!important}
        .stDownloadButton>button{background:#ecfdf5!important;color:#059669!important;border-color:#a7f3d0!important}
        .crop-info{background:#ecfdf5!important;border-color:#a7f3d0!important;color:#059669!important}
        </style>""", unsafe_allow_html=True)

_inject_theme()

with st.sidebar:
    st.markdown("""
    <div class="yoloe-logo">
      <div class="yoloe-logo-icon">✂️</div>
      <div>
        <div class="yoloe-logo-text">YOLOE Studio</div>
        <div class="yoloe-logo-sub">Block 1 · Mask &amp; Crop</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.toggle("🌙  Dark mode", value=st.session_state["dark_mode"], key="dark_mode")
    _inject_theme()

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prompt mode</div>', unsafe_allow_html=True)
    mode = st.radio("", [" Text Prompt", " Visual Prompt", " Free-Prompt"],
                    label_visibility="collapsed")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Seg model",
        ["yoloe-26l-seg.pt","yoloe-11l-seg.pt","yoloe-11m-seg.pt","yoloe-11s-seg.pt"])
    pf_choice = st.selectbox("Free-prompt model",
        ["yoloe-26l-seg-pf.pt","yoloe-11l-seg-pf.pt","yoloe-11s-seg-pf.pt"])

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
    conf_thresh  = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

    st.markdown('<div class="section-label">Mask refinement</div>', unsafe_allow_html=True)
    use_largest  = st.checkbox("Largest component only", value=True)
    _c1, _c2 = st.columns(2)
    erode_iters  = _c1.slider("Erode",  0, 5, 0)
    dilate_iters = _c2.slider("Dilate", 0, 5, 0)

    st.markdown('<div class="section-label">Crop</div>', unsafe_allow_html=True)
    crop_padding = st.slider("Padding (px)", 0, 80, 12)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Environment</div>', unsafe_allow_html=True)
    _h_ok = (APP_DIR/"helpers.py").exists()
    st.markdown(f'<span class="status-{"ok" if _h_ok else "err"}">{"✓ helpers.py" if _h_ok else "✗ helpers.py missing"}</span>', unsafe_allow_html=True)
    try:
        import clip
        st.markdown('<span class="status-ok">✓ CLIP</span>', unsafe_allow_html=True)
    except ImportError:
        st.markdown('<span class="status-err">✗ CLIP missing</span>', unsafe_allow_html=True)
    try:
        import mobileclip
        st.markdown('<span class="status-ok">✓ MobileCLIP</span>', unsafe_allow_html=True)
    except ImportError:
        st.markdown('<span class="status-warn">⚠ MobileCLIP</span>', unsafe_allow_html=True)
    st.caption(f"📁 {APP_DIR.name}/")
st.markdown("""
<div class="pipe-banner">
  <div class="pipe-step active">① Image</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step active">② Prompt</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step active">③ YOLOE Mask</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step active">④ Crop</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step pending">⑤ Any6D Pose ·  next</div>
</div>
""", unsafe_allow_html=True)

# Page heading changes per mode
if " Text Prompt" in mode:
    st.markdown('<div class="page-title">Text Prompt Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter class names → select tags → detect &amp; crop all matched objects</div>', unsafe_allow_html=True)
elif " Visual Prompt" in mode:
    st.markdown('<div class="page-title">Visual Prompt Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Draw a bounding box on your reference image → find &amp; crop the same object in the scene</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="page-title">Free Prompt Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">No prompt needed — detect all objects, pick which ones to crop</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MODE 1 — TEXT PROMPT
# Large textarea → words parsed into clickable tag pills
# Selected tags are sent as text_prompts to YOLOE
# ════════════════════════════════════════════════════════════════════════════════

if " Text Prompt" in mode:
    st.markdown('<span class="mode-pill pill-text">Text Prompt</span>', unsafe_allow_html=True)

    col_img, col_cfg = st.columns([1.5, 1])

    with col_img:
        st.markdown('<div class="section-label">Scene image</div>', unsafe_allow_html=True)
        scene_up = st.file_uploader("scene", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed", key="t_scene")
        if scene_up:
            scene_up.seek(0); st.image(scene_up, use_container_width=True)

    with col_cfg:
        st.markdown('<div class="section-label">Enter object descriptions</div>', unsafe_allow_html=True)
        st.caption("One per line — each line becomes a clickable tag")
        raw_text = st.text_area(
            "", height=150, label_visibility="collapsed", key="t_raw",
            placeholder="tennis ball\nyellow ball\nball\nmustard bottle",
        )

        # Parse lines → unique non-empty tokens
        lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
        unique_lines = list(dict.fromkeys(lines))   # preserve order, deduplicate

        if unique_lines:
            st.markdown('<div class="section-label" style="margin-top:10px">Select tags to detect</div>', unsafe_allow_html=True)

            # Per-tag checkboxes — key includes tag text so changing
            # the textarea never reuses a stale checkbox state
            selected = []
            cols = st.columns(min(len(unique_lines), 3))
            for i, tag in enumerate(unique_lines):
                safe_key = "tag_" + "".join(c if c.isalnum() else "_" for c in tag)[:40]
                with cols[i % len(cols)]:
                    if st.checkbox(tag, value=True, key=safe_key):
                        selected.append(tag)

            if selected:
                st.caption(f"→ sending {len(selected)} prompt(s): {', '.join(selected)}")
            else:
                st.caption("Select at least one tag")
        else:
            selected = []

        st.markdown("<br>", unsafe_allow_html=True)
        run_t = st.button("▶  Run — Text Prompt", use_container_width=True, key="run_t")

    if run_t:
        if not scene_up:
            st.error("Upload a scene image."); st.stop()
        if not selected:
            st.error("Select at least one tag."); st.stop()

        scene_path = None
        scene_path_reuse = None
        try:
            scene_path = save_upload(scene_up)
            scene_path_reuse = scene_path  # keep ref for per-class re-runs below
            scene_rgb  = read_rgb(scene_path)
            model      = load_model(model_choice)
            with st.spinner(f"Detecting {len(selected)} class(es)…"):
                # Initial run to validate — per-class runs happen in show block
                bbox, mask, conf_val, _ = run_text(model, scene_path, scene_rgb, selected, conf_thresh)
        except Exception as e:
            st.error(f"Detection failed: {e}"); traceback.print_exc(file=sys.stderr); st.stop()
        finally:
            pass  # unlink after per-class crops below
        try:
            pass  # file still needed for per-class runs
        finally:
            safe_unlink(scene_path)

        st.markdown("---")
        # Run detection for EACH selected class and show individual crop
        # This respects the requirement: every selected tag gets detected & cropped
        _all_text_results = {}
        for _cls in selected:
            try:
                _bx, _mk, _cv, _ = run_text(model, scene_path_reuse, scene_rgb, [_cls], conf_thresh)
                _all_text_results[_cls] = (_bx, _mk, _cv)
            except Exception:
                _all_text_results[_cls] = (None, None, 0.0)

        if len(selected) == 1:
            _cls = selected[0]
            _bx, _mk, _cv = _all_text_results[_cls]
            show_results(scene_rgb, _mk, _bx, _cv,
                         crop_padding, use_largest, erode_iters, dilate_iters,
                         mode_label=f"text:{_cls}", key_suffix=f"t0")
        else:
            _tabs = st.tabs([f"✂ {c}" for c in selected])
            for _ti, _cls in enumerate(selected):
                with _tabs[_ti]:
                    _bx, _mk, _cv = _all_text_results[_cls]
                    if _mk is None or not np.asarray(_mk).any():
                        st.warning(f"No detection for '{_cls}' — try a lower confidence threshold.")
                    else:
                        show_results(scene_rgb, _mk, _bx, _cv,
                                     crop_padding, use_largest, erode_iters, dilate_iters,
                                     mode_label=f"text:{_cls}", key_suffix=f"t{_ti}")


# ════════════════════════════════════════════════════════════════════════════════
# MODE 2 — VISUAL PROMPT
# Drawable canvas on anchor → real pixel coords → model
# ════════════════════════════════════════════════════════════════════════════════

elif " Visual Prompt" in mode:
    st.markdown('<span class="mode-pill pill-visual">Visual Prompt</span>', unsafe_allow_html=True)

    col_anc, col_scn = st.columns(2)
    anchor_up = scene_up = None

    with col_anc:
        st.markdown('<div class="section-label">Anchor — draw a box around the object</div>', unsafe_allow_html=True)
        anchor_up = st.file_uploader("anchor", type=["jpg","jpeg","png","webp"],
                                     label_visibility="collapsed", key="v_anchor")

        drawn_bbox = None

        if anchor_up:
            anchor_up.seek(0)
            anchor_pil = Image.open(anchor_up).convert("RGB")
            orig_W, orig_H = anchor_pil.size

            # ── Pure Streamlit bbox selector ──────────────────────────
            # No external component needed.
            # Four sliders set the bbox; a PIL-rendered preview updates
            # in real-time showing the rectangle on the anchor image.
            # This works on any deployment with zero dependencies.

            st.caption("Set the bounding box using the sliders below:")

            # Init session_state defaults (full image)
            if "bbox_x1" not in st.session_state: st.session_state["bbox_x1"] = 0
            if "bbox_y1" not in st.session_state: st.session_state["bbox_y1"] = 0
            if "bbox_x2" not in st.session_state: st.session_state["bbox_x2"] = orig_W
            if "bbox_y2" not in st.session_state: st.session_state["bbox_y2"] = orig_H

            # Sliders in two rows
            row1 = st.columns(2)
            row2 = st.columns(2)
            _bx1 = row1[0].slider("x1 (left)",   0, orig_W-1, st.session_state["bbox_x1"], key="vx1")
            _bx2 = row1[1].slider("x2 (right)",  1, orig_W,   st.session_state["bbox_x2"], key="vx2")
            _by1 = row2[0].slider("y1 (top)",    0, orig_H-1, st.session_state["bbox_y1"], key="vy1")
            _by2 = row2[1].slider("y2 (bottom)", 1, orig_H,   st.session_state["bbox_y2"], key="vy2")

            # Clamp so x1<x2, y1<y2
            _bx1, _bx2 = min(_bx1,_bx2-1), max(_bx1+1,_bx2)
            _by1, _by2 = min(_by1,_by2-1), max(_by1+1,_by2)

            # Save to session_state
            st.session_state.update({"bbox_x1":_bx1,"bbox_x2":_bx2,
                                     "bbox_y1":_by1,"bbox_y2":_by2})

            # Live preview: draw bbox rectangle on a copy of the anchor image
            import cv2 as _cv2
            preview = np.array(anchor_pil.copy())
            # Draw filled rect with alpha
            overlay = preview.copy()
            _cv2.rectangle(overlay, (_bx1,_by1), (_bx2,_by2), (91,156,246), -1)
            preview = _cv2.addWeighted(overlay, 0.05, preview, 0.75, 0)
            # Draw border + corner handles
            _cv2.rectangle(preview, (_bx1,_by1), (_bx2,_by2), (91,156,246), 2)
            for cx,cy in [(_bx1,_by1),(_bx2,_by1),(_bx1,_by2),(_bx2,_by2)]:
                _cv2.circle(preview, (cx,cy), 6, (91,156,246), -1)
            # Draw dimensions label
            lbl = f"{_bx2-_bx1} x {_by2-_by1} px"
            _cv2.putText(preview, lbl, (_bx1+4, max(_by1-6,14)),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (91,156,246), 2)
            st.image(preview, use_container_width=True,
                     caption=f"Preview — box [{_bx1},{_by1},{_bx2},{_by2}]")

            drawn_bbox = [_bx1, _by1, _bx2, _by2]

    with col_scn:
        st.markdown('<div class="section-label">Scene — query image</div>', unsafe_allow_html=True)
        scene_up = st.file_uploader("scene", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed", key="v_scene")
        if scene_up:
            scene_up.seek(0); st.image(scene_up, use_container_width=True)

        if drawn_bbox:
            st.info(f"Reference box: [{drawn_bbox[0]}, {drawn_bbox[1]}, {drawn_bbox[2]}, {drawn_bbox[3]}]")
        else:
            st.info("Draw a box on the anchor image to define the reference object.")

    run_v = st.button("▶  Run — Visual Prompt", use_container_width=True, key="run_v")

    if run_v:
        if not anchor_up:
            st.error("Upload an anchor image."); st.stop()
        if not scene_up:
            st.error("Upload a scene image."); st.stop()
        if not drawn_bbox:
            st.error("Draw a bounding box on the anchor image first."); st.stop()

        # Sanity-check bbox: ensure x1<x2 and y1<y2, swap if needed
        _db = drawn_bbox
        _db = [min(_db[0],_db[2]), min(_db[1],_db[3]),
               max(_db[0],_db[2]), max(_db[1],_db[3])]
        if _db[2]-_db[0] < 4 or _db[3]-_db[1] < 4:
            st.error(f"Bounding box too small: {_db}. Draw a larger rectangle."); st.stop()
        drawn_bbox = _db

        anchor_path = scene_path = None
        try:
            anchor_up.seek(0); scene_up.seek(0)
            anchor_path = save_upload(anchor_up)
            scene_path  = save_upload(scene_up)
            scene_rgb   = read_rgb(scene_path)
            anchor_bbox = np.array([drawn_bbox], dtype=np.float32)
            model       = load_model(model_choice)
            with st.spinner("Segmenting via visual prompt…"):
                bbox, mask, conf_val, _ = run_visual(
                    model, scene_path, scene_rgb, anchor_path, anchor_bbox, conf_thresh)
        except Exception as e:
            st.error(f"Detection failed: {e}"); traceback.print_exc(file=sys.stderr); st.stop()
        finally:
            safe_unlink(anchor_path, scene_path)

        st.markdown("---")
        show_results(scene_rgb, mask, bbox, conf_val,
                     crop_padding, use_largest, erode_iters, dilate_iters,
                     mode_label="visual", key_suffix="v")


# ════════════════════════════════════════════════════════════════════════════════
# MODE 3 — FREE PROMPT
# Lists ALL detected objects with conf + mask px + individual crop button
# ════════════════════════════════════════════════════════════════════════════════

else:
    st.markdown('<span class="mode-pill pill-free">Free Prompt</span>', unsafe_allow_html=True)

    col_img, col_info = st.columns([1.5, 1])
    with col_img:
        st.markdown('<div class="section-label">Scene image</div>', unsafe_allow_html=True)
        scene_up = st.file_uploader("scene", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed", key="f_scene")
        if scene_up:
            scene_up.seek(0); st.image(scene_up, use_container_width=True)
    with col_info:
        st.info("**Free prompt** detects all objects automatically.\n\nNo class names or reference image needed. All detected objects are listed — pick any to crop for Any6D.")
        run_f = st.button("▶  Run — Free Prompt", use_container_width=True, key="run_f")

    # Cache detection results in session_state so ✂ Crop buttons
    # don't retrigger model.predict() on every Streamlit rerun
    if run_f:
        if not scene_up:
            st.error("Upload a scene image."); st.stop()

        scene_path = None
        try:
            scene_up.seek(0)
            scene_path  = save_upload(scene_up)
            scene_rgb   = read_rgb(scene_path)
            H, W        = scene_rgb.shape[:2]
            m_pf        = load_model_pf(pf_choice)
            with st.spinner("Detecting all objects…"):
                _, _, _, raw_results = run_free(m_pf, scene_path, scene_rgb, conf_thresh)
                all_dets = parse_all(raw_results, H, W)
            all_dets.sort(key=lambda d: d["conf"], reverse=True)
            # Store in session_state — persists across reruns triggered by crop buttons
            st.session_state["free_dets"]       = all_dets
            st.session_state["free_scene_rgb"]  = scene_rgb
        except Exception as e:
            st.error(f"Detection failed: {e}"); traceback.print_exc(file=sys.stderr); st.stop()
        finally:
            safe_unlink(scene_path)

    # Retrieve cached results (populated either this run or a previous one)
    all_dets  = st.session_state.get("free_dets",      [])
    scene_rgb_free = st.session_state.get("free_scene_rgb", None)

    if not all_dets or scene_rgb_free is None:
        if run_f:   # run was just pressed but produced nothing
            st.warning("No objects detected. Try lowering the confidence threshold.")
        st.stop()

    all_dets = all_dets  # already sorted

    # Overview image with all detections
    COLORS = [(91,156,246),(163,122,245),(61,214,160),(246,180,91),(246,91,91),(91,246,220)]
    overview = scene_rgb_free.copy()
    for i, d in enumerate(all_dets):
        col = COLORS[i % len(COLORS)]
        overview = overlay_mask(overview, d["mask"], color=col, alpha=0.3)
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(overview,(x1,y1),(x2,y2),col,2)
        cv2.putText(overview,f"[{i}] {d['label']} {d['conf']:.2f}",
                    (x1,max(y1-6,14)),cv2.FONT_HERSHEY_SIMPLEX,0.52,col,2)

        st.markdown("---")
    st.markdown('<div class="section-label">Detection overview</div>', unsafe_allow_html=True)
    st.image(overview, use_container_width=True)

    # Summary metrics
    mc = st.columns(3)
    mc[0].markdown(f'<div class="metric-card"><div class="val">{len(all_dets)}</div><div class="lbl">Objects</div></div>',unsafe_allow_html=True)
    uniq_cls = len(set(d["label"] for d in all_dets))
    mc[1].markdown(f'<div class="metric-card"><div class="val">{uniq_cls}</div><div class="lbl">Classes</div></div>',unsafe_allow_html=True)
    mc[2].markdown(f'<div class="metric-card"><div class="val">{all_dets[0]["conf"]:.3f}</div><div class="lbl">Best conf</div></div>',unsafe_allow_html=True)

    # ── Full object list ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">All detected objects — click to crop</div>', unsafe_allow_html=True)

    for i, d in enumerate(all_dets):
        col_info2, col_btn = st.columns([4, 1])
        mask_px = int(d["mask"].sum())
        col_info2.markdown(
            f'<div class="det-row">' +
            f'<span class="det-badge">#{i}</span>' +
            f'<span class="det-label">{d["label"]}</span>' +
            f'<span class="det-conf">{d["conf"]:.3f}</span>' +
            f'<span class="det-mask">{mask_px:,} px</span>' +
            f'</div>',
            unsafe_allow_html=True
        )
        crop_clicked = col_btn.button("✂ Crop", key=f"crop_{i}")

        if crop_clicked:
            st.markdown(f"#### Object [{i}] — {d['label']}")
            show_results(
                scene_rgb_free, d["mask"], d["bbox"], d["conf"],
                crop_padding, use_largest, erode_iters, dilate_iters,
                mode_label=f"free/{d['label']}", key_suffix=f"f{i}",
            )
            st.markdown("---")
