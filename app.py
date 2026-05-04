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
    page_title="YOLOE · Block 1",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
*, html, body { font-family: 'IBM Plex Sans', sans-serif; }
.stApp       { background: #080a0f; color: #d4d8e8; }
.stSidebar   { background: #0d1017 !important; border-right: 1px solid #1e2130 !important; }

.pipe-banner {
    background: #0f1520; border: 1px solid #1e2a40; border-radius: 10px;
    padding: 12px 18px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.pipe-step { font-family:'IBM Plex Mono',monospace; font-size:.68rem;
    font-weight:600; letter-spacing:.06em; padding:4px 10px;
    border-radius:4px; text-transform:uppercase; }
.pipe-step.active  { background:#1a2a45; color:#5b9cf6; border:1px solid #2a4070; }
.pipe-step.pending { background:#1a1a2a; color:#3a3f5a; border:1px solid #252535; }
.pipe-arrow { color:#2a3050; font-size:.85rem; }

.section-label { font-family:'IBM Plex Mono',monospace; font-size:.64rem;
    letter-spacing:.12em; text-transform:uppercase; color:#4a5070;
    margin-bottom:7px; padding-bottom:5px; border-bottom:1px solid #1a1e2e; }

.mode-pill { display:inline-block; padding:3px 11px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:.63rem;
    font-weight:600; letter-spacing:.08em; text-transform:uppercase; margin-bottom:14px; }
.pill-text   { background:#0d2040; color:#5b9cf6; border:1px solid #1a3a6a; }
.pill-visual { background:#1e0d3a; color:#a87af5; border:1px solid #3a1a6a; }
.pill-free   { background:#0d2a1e; color:#3dd6a0; border:1px solid #1a4a30; }

.metric-card { background:#0d1220; border:1px solid #1a2035; border-radius:8px; padding:12px 16px; }
.metric-card .val { font-family:'IBM Plex Mono',monospace; font-size:1.4rem;
    font-weight:600; color:#5b9cf6; line-height:1.1; }
.metric-card .lbl { font-size:.63rem; color:#4a5070; text-transform:uppercase;
    letter-spacing:.08em; margin-top:3px; }

.panel-title { font-family:'IBM Plex Mono',monospace; font-size:.66rem;
    color:#4a5070; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; }

.crop-info { background:#0f2a1e; border:1px solid #1a4a30; border-radius:6px;
    padding:8px 12px; font-family:'IBM Plex Mono',monospace;
    font-size:.68rem; color:#3dd6a0; margin-top:8px; line-height:1.9; }

.tag-pill { display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:.75rem; font-family:'IBM Plex Mono',monospace;
    background:#0d2040; color:#5b9cf6; border:1px solid #1a3a6a;
    margin:3px 4px 3px 0; cursor:pointer; }
.tag-pill.selected { background:#1a3a6a; border-color:#5b9cf6; }

.det-row { display:flex; align-items:center; gap:10px; padding:8px 12px;
    background:#0d1220; border:1px solid #1a2035; border-radius:8px; margin-bottom:6px; }
.det-label { font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:#d4d8e8; flex:1; }
.det-conf  { font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:#5b9cf6; min-width:50px; }
.det-mask  { font-size:.7rem; color:#4a5070; min-width:70px; }

.stButton > button {
    background:#0d1a30!important; color:#5b9cf6!important;
    border:1px solid #2a4a7a!important; border-radius:7px!important;
    font-family:'IBM Plex Mono',monospace!important; font-size:.74rem!important;
    font-weight:600!important; letter-spacing:.05em!important; }
.stButton > button:hover { background:#1a2a50!important; }
div[data-testid="stFileUploader"] {
    background:#0d1220!important; border:1px dashed #1e2a40!important; border-radius:8px!important; }
.stTextArea textarea {
    background:#0d1220!important; border:1px solid #1e2a40!important;
    color:#d4d8e8!important; font-family:'IBM Plex Mono',monospace!important; border-radius:6px!important; }
label { color:#8090b0!important; font-size:.78rem!important; }
hr    { border-color:#1a1e2e!important; }
.stDownloadButton > button {
    background:#0f2a1e!important; color:#3dd6a0!important;
    border:1px solid #1a4a30!important;
    font-family:'IBM Plex Mono',monospace!important; font-size:.7rem!important; }
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
    mc = st.columns(4)
    mc[0].markdown(f'<div class="metric-card"><div class="val">{conf_val:.3f}</div><div class="lbl">Confidence</div></div>',unsafe_allow_html=True)
    mc[1].markdown(f'<div class="metric-card"><div class="val">{mask_px:,}</div><div class="lbl">Mask pixels</div></div>',unsafe_allow_html=True)
    mc[2].markdown(f'<div class="metric-card"><div class="val">{mask_pct:.1f}%</div><div class="lbl">Coverage</div></div>',unsafe_allow_html=True)
    if bbox is not None:
        bw=int(bbox[2]-bbox[0]); bh=int(bbox[3]-bbox[1])
        mc[3].markdown(f'<div class="metric-card"><div class="val">{bw}×{bh}</div><div class="lbl">BBox size</div></div>',unsafe_allow_html=True)
    else:
        mc[3].markdown('<div class="metric-card"><div class="val">—</div><div class="lbl">BBox size</div></div>',unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    p1,p2,p3 = st.columns(3)
    with p1:
        st.markdown('<div class="panel-title">① Mask overlay</div>',unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
    with p2:
        st.markdown('<div class="panel-title">② Binary mask</div>',unsafe_allow_html=True)
        st.image(mask_vis, use_container_width=True)
    with p3:
        st.markdown('<div class="panel-title">③ Crop → Any6D</div>',unsafe_allow_html=True)
        if crop_prev is not None:
            st.image(crop_prev, use_container_width=True)
            if crop_bbox:
                cx1,cy1,cx2,cy2 = crop_bbox
                st.markdown(f'<div class="crop-info">origin&nbsp; ({cx1},{cy1})<br>size&nbsp;&nbsp;&nbsp; {cx2-cx1}×{cy2-cy1} px<br>pad&nbsp;&nbsp;&nbsp;&nbsp; {crop_padding} px</div>',unsafe_allow_html=True)
        else:
            st.warning("Crop failed.")

    st.markdown("---")
    st.markdown('<div class="section-label">Export for Any6D</div>',unsafe_allow_html=True)
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

with st.sidebar:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.05rem;font-weight:600;color:#5b9cf6;">YOLOE Studio</div>',unsafe_allow_html=True)
    st.markdown('<div style="font-size:.67rem;color:#4a5070;margin:2px 0 18px;font-family:\'IBM Plex Mono\',monospace;letter-spacing:.1em;">BLOCK 1 · MASK &amp; CROP</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-label">Prompt mode</div>',unsafe_allow_html=True)
    mode = st.radio("", ["📝  Text", "🖼️  Visual", "🔮  Free"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Models</div>',unsafe_allow_html=True)
    model_choice = st.selectbox("Seg model", ["yoloe-26l-seg.pt","yoloe-11l-seg.pt","yoloe-11m-seg.pt","yoloe-11s-seg.pt"])
    pf_choice    = st.selectbox("Free-prompt model", ["yoloe-26l-seg-pf.pt","yoloe-11l-seg-pf.pt","yoloe-11s-seg-pf.pt"])

    st.markdown("---")
    st.markdown('<div class="section-label">Detection</div>',unsafe_allow_html=True)
    conf_thresh  = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

    st.markdown('<div class="section-label">Mask refinement</div>',unsafe_allow_html=True)
    use_largest  = st.checkbox("Largest component only", value=True)
    erode_iters  = st.slider("Erode",  0, 5, 0)
    dilate_iters = st.slider("Dilate", 0, 5, 0)

    st.markdown('<div class="section-label">Crop</div>',unsafe_allow_html=True)
    crop_padding = st.slider("Padding (px)", 0, 80, 12)

    st.markdown("---")
    # Status
    h_ok = (APP_DIR/"helpers.py").exists()
    st.caption(f"{'✅' if h_ok else '⚠️'} helpers.py {'found' if h_ok else 'missing'}")
    try:
        import clip; st.caption("✅ clip installed")
    except ImportError:
        st.caption("❌ clip missing — run: pip install git+https://github.com/ultralytics/CLIP.git")
    try:
        import mobileclip; st.caption("✅ mobileclip installed")
    except ImportError:
        st.caption("❌ mobileclip missing — run: pip install git+https://github.com/ultralytics/mobileclip.git")
    st.caption(f"📁 {APP_DIR}")


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE BANNER
# ════════════════════════════════════════════════════════════════════════════════

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
  <div class="pipe-step pending">⑤ Any6D pose</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MODE 1 — TEXT PROMPT
# Large textarea → words parsed into clickable tag pills
# Selected tags are sent as text_prompts to YOLOE
# ════════════════════════════════════════════════════════════════════════════════

if "-" in mode:
    st.markdown('<span class="mode-pill pill-text"> Text Prompt</span>', unsafe_allow_html=True)

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
        try:
            scene_path = save_upload(scene_up)
            scene_rgb  = read_rgb(scene_path)
            model      = load_model(model_choice)
            with st.spinner(f"Detecting: {', '.join(selected)} …"):
                bbox, mask, conf_val, _ = run_text(model, scene_path, scene_rgb, selected, conf_thresh)
        except Exception as e:
            st.error(f"Detection failed: {e}"); traceback.print_exc(file=sys.stderr); st.stop()
        finally:
            safe_unlink(scene_path)

        st.markdown("---")
        show_results(scene_rgb, mask, bbox, conf_val,
                     crop_padding, use_largest, erode_iters, dilate_iters,
                     mode_label=f"text:{','.join(selected)}", key_suffix="t")


# ════════════════════════════════════════════════════════════════════════════════
# MODE 2 — VISUAL PROMPT
# Drawable canvas on anchor → real pixel coords → model
# ════════════════════════════════════════════════════════════════════════════════

elif "-" in mode:
    st.markdown('<span class="mode-pill pill-visual"> Visual Prompt</span>', unsafe_allow_html=True)

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
            preview = _cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
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
            f'<div class="det-row">'
            f'<div class="det-label">[{i}] {d["label"]}</div>'
            f'<div class="det-conf">conf {d["conf"]:.3f}</div>'
            f'<div class="det-mask">{mask_px:,} px</div>'
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
