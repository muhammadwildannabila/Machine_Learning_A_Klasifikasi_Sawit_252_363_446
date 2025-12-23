import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional heavy deps (biar UI tetap hidup walau env belum lengkap)
try:
    import joblib
except Exception:
    joblib = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import torch
    import timm
    import torchvision.transforms as T
except Exception:
    torch = None
    timm = None
    T = None


# =========================
# PAGE CONFIG + THEME
# =========================
st.set_page_config(page_title="SawitAI • Classifier", page_icon="🌴", layout="wide")

# Cyberpunk palette
BG = "#070712"          # near-black
CARD = "rgba(255,255,255,0.06)"
STROKE = "rgba(255,120,60,0.25)"     # orange stroke
NEON_ORANGE = "#FF7A3C"
NEON_PURPLE = "#B06CFF"
NEON_RED = "#FF3D6E"
TEXT = "rgba(255,255,255,0.92)"
MUTED = "rgba(255,255,255,0.70)"

CSS = f"""
<style>
:root {{
  --bg: {BG};
  --card: {CARD};
  --stroke: {STROKE};
  --orange: {NEON_ORANGE};
  --purple: {NEON_PURPLE};
  --red: {NEON_RED};
  --text: {TEXT};
  --muted: {MUTED};
}}

html, body, [class*="css"] {{
  background: var(--bg) !important;
  color: var(--text) !important;
}}

.block-container {{
  padding-top: 1.1rem;
  padding-bottom: 2.0rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(176,108,255,0.08), rgba(255,122,60,0.06)) !important;
  border-right: 1px solid rgba(255,122,60,0.22);
}}

.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,122,60,0.22);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}}

.hero {{
  border-radius: 22px;
  padding: 16px 18px;
  border: 1px solid rgba(176,108,255,0.25);
  background:
    radial-gradient(800px 180px at 10% 0%, rgba(176,108,255,0.18), transparent 55%),
    radial-gradient(900px 220px at 90% 10%, rgba(255,122,60,0.16), transparent 55%),
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  box-shadow: 0 12px 40px rgba(0,0,0,0.55);
}}

.h1 {{
  font-size: 34px;
  font-weight: 900;
  margin: 0;
  letter-spacing: 0.2px;
}}
.sub {{
  margin-top: 6px;
  color: var(--muted);
}}

.badge {{
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,122,60,0.28);
  background: rgba(255,122,60,0.10);
  font-size: 12px;
  margin-bottom: 8px;
}}

.tag {{
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.05);
  font-size: 12px;
}}

.tag-ok {{
  border-color: rgba(66,245,179,0.28);
  background: rgba(66,245,179,0.10);
}}
.tag-warn {{
  border-color: rgba(255,61,110,0.34);
  background: rgba(255,61,110,0.12);
}}

hr {{
  border: none;
  border-top: 1px solid rgba(176,108,255,0.18);
  margin: 1.0rem 0;
}}

.small {{
  font-size: 13px;
  color: var(--muted);
}}

.kpi {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 10px;
}}
.kpi .k {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,122,60,0.20);
  border-radius: 14px;
  padding: 10px 12px;
}}
.kpi .k .t {{
  color: var(--muted);
  font-size: 12px;
}}
.kpi .k .v {{
  font-size: 18px;
  font-weight: 900;
}}

a {{
  color: var(--purple) !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

APP_TITLE = "SawitAI • Ripeness Classifier"
APP_DESC = "Prediksi kematangan tandan sawit dari citra. Pilih model, upload gambar/ZIP, dapatkan label + confidence + saran foto ulang bila ragu."

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "sawit_models"

FILES = {
    "class_names": MODELS_DIR / "class_names.json",
    "xgb_model": MODELS_DIR / "xgb_hsv.joblib",
    "xgb_meta": MODELS_DIR / "xgb_meta.joblib",
    "effnet": MODELS_DIR / "model_effnetb0_lora_merged.keras",
    "maxvit": MODELS_DIR / "maxvit_merged.pt",
}

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}
IMG_160 = (160, 160)


# =========================
# UTIL
# =========================
def validate_assets() -> List[str]:
    missing = []
    for k, p in FILES.items():
        if not p.exists():
            missing.append(f"{k} -> {p}")
    return missing

@st.cache_data
def load_class_names(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        try:
            keys = list(data.keys())
            if all(str(k).isdigit() for k in keys):
                return [data[str(i)] for i in range(len(keys))]
        except Exception:
            pass
        return list(data.values())
    return data

def topk(prob: np.ndarray, class_names: List[str], k: int = 3) -> List[Tuple[str, float]]:
    idx = np.argsort(prob)[::-1][:k]
    return [(class_names[i], float(prob[i])) for i in idx]

def margin_top1_top2(prob: np.ndarray) -> float:
    s = np.sort(prob)[::-1]
    return float(s[0] - s[1]) if len(s) > 1 else 0.0

def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, Image.Image]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_IMG_EXT:
                continue
            with z.open(info) as f:
                try:
                    img = Image.open(f).convert("RGB")
                    out.append((Path(info.filename).name, img))
                except Exception:
                    pass
    return out

def safe_label(lbl: str) -> str:
    # tampilkan label yang lebih user-friendly
    mapping = {
        "unripe": "Unripe (Mentah)",
        "ripe": "Ripe (Matang)",
        "rotten": "Rotten (Busuk)",
        "decayed": "Decayed",
        "immature": "Immature",
        "over_ripe": "Over-ripe",
        "partially_ripe": "Partially-ripe",
        "fully_ripe": "Fully-ripe",
    }
    return mapping.get(lbl, lbl)

def insight_tips(top1: str, top2: str) -> List[str]:
    tips = [
        "Foto ulang lebih dekat (objek memenuhi frame).",
        "Gunakan cahaya merata (hindari bayangan tajam / backlight).",
        "Pastikan gambar tidak blur dan fokus pada tandan.",
        "Jika background ramai, coba background lebih bersih/kontras."
    ]
    pair = {top1, top2}
    if {"ripe", "unripe"} <= pair:
        tips.insert(0, "Kelas transisi **ripe ↔ unripe** sering mirip. Coba foto di cahaya natural dan jarak lebih dekat.")
    if {"rotten", "unripe"} <= pair:
        tips.insert(0, "Jika bagian busuk kecil, model bisa ragu. Pastikan area gelap/busuk terlihat jelas.")
    return tips


# =========================
# MODEL LOADERS
# =========================
@st.cache_resource
def load_xgb():
    if joblib is None:
        raise RuntimeError("joblib tidak tersedia.")
    xgb = joblib.load(str(FILES["xgb_model"]))
    meta = joblib.load(str(FILES["xgb_meta"]))
    return xgb, meta

def color_features_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-6)
    s_hist = s_hist / (s_hist.sum() + 1e-6)
    v_hist = v_hist / (v_hist.sum() + 1e-6)
    mean = hsv.mean(axis=(0, 1))
    std = hsv.std(axis=(0, 1))
    return np.concatenate([h_hist, s_hist, v_hist, mean, std]).astype(np.float32)

def xgb_predict(pil_img: Image.Image, xgb, meta) -> Tuple[str, float, np.ndarray, List[str]]:
    if cv2 is None:
        raise RuntimeError("opencv belum tersedia.")
    classes = meta.get("classes", [])
    img_size = tuple(meta.get("img_size", [160, 160]))
    rgb = np.array(pil_img.convert("RGB"))
    rgb = cv2.resize(rgb, img_size)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    feat = color_features_hsv(bgr).reshape(1, -1)
    prob = xgb.predict_proba(feat)[0]
    idx = int(np.argmax(prob))
    return classes[idx], float(prob[idx]), prob, classes

@st.cache_resource
def load_effnet():
    if tf is None:
        raise RuntimeError("TensorFlow tidak tersedia.")
    return tf.keras.models.load_model(str(FILES["effnet"]))

def effnet_predict(pil_img: Image.Image, model, class_names: List[str]) -> Tuple[str, float, np.ndarray]:
    img = pil_img.convert("RGB").resize(IMG_160)
    x = (np.array(img).astype("float32") / 255.0)[None, ...]
    prob = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(prob))
    return class_names[idx], float(prob[idx]), prob

@st.cache_resource
def load_maxvit():
    if torch is None or timm is None or T is None:
        raise RuntimeError("Torch/timm/torchvision tidak tersedia.")
    ckpt = torch.load(str(FILES["maxvit"]), map_location="cpu")
    arch = ckpt["arch"]
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 224))
    model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, classes, img_size

def maxvit_predict(pil_img: Image.Image, model, class_names: List[str], img_size: int) -> Tuple[str, float, np.ndarray]:
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(prob))
    return class_names[idx], float(prob[idx]), prob


# =========================
# HERO
# =========================
st.markdown(
    f"""
<div class="hero">
  <div class="badge">Willy UI • Dark Mode • Interaktif</div>
  <div class="h1">🌴 {APP_TITLE}</div>
  <div class="sub">{APP_DESC}</div>
</div>
""",
    unsafe_allow_html=True,
)

missing = validate_assets()
if missing:
    st.error("File model belum lengkap. Pastikan folder `sawit_models/` berisi file berikut:")
    for m in missing:
        st.write(f"- {m}")
    st.info("Run dari root project: `pdm run python -m streamlit run src/app.py`")
    st.stop()

class_names_global = load_class_names(FILES["class_names"])


# =========================
# SIDEBAR - GUIDED FLOW
# =========================
with st.sidebar:
    st.header("🚀 Panduan Singkat (4 Step)")
    st.caption("Ikuti alur: pilih model → pilih mode input → upload gambar/ZIP → lihat hasil & insight. Selesai ✅")

    model_choice = st.selectbox(
        "1) Pilih Model",
        [
            "MaxViT + LoRA",
            "EfficientNetB0 + LoRA",
            "XGBoost + HSV Color",
        ],
        index=1,
        help="Best = akurasi tertinggi. Balanced = cepat & stabil. Classic = sangat cepat namun sensitif lighting."
    )

    mode = st.radio(
        "2) Pilih Mode Input",
        ["Single / Multi Image", "ZIP Batch"],
        index=0,
        help="ZIP batch cocok untuk banyak gambar sekaligus."
    )

    st.divider()
    conf_th = st.slider("3) Confidence threshold", 0.0, 1.0, 0.60, 0.01,
                        help="Di bawah ini akan diberi warning (model ragu).")
    margin_th = st.slider("4) Ambiguity margin (Top1-Top2)", 0.0, 1.0, 0.15, 0.01,
                          help="Margin kecil = prediksi ambigu karena Top-1 dan Top-2 mirip.")
    show_top3 = st.checkbox("Tampilkan Top-3", value=True)
    show_insight = st.checkbox("Tampilkan Insight", value=True)


# =========================
# LOAD MODEL
# =========================
err_model = None
xgb_obj = meta_obj = None
eff_obj = None
maxvit_obj = maxvit_classes = None
maxvit_img = 224

try:
    if model_choice.startswith("XGBoost"):
        xgb_obj, meta_obj = load_xgb()
    elif model_choice.startswith("EfficientNet"):
        eff_obj = load_effnet()
    else:
        maxvit_obj, maxvit_classes, maxvit_img = load_maxvit()
except Exception as e:
    err_model = str(e)

if err_model:
    st.error("Gagal load model / dependency.")
    st.code(err_model)
    st.stop()


def predict_one(pil_img: Image.Image) -> Dict[str, Any]:
    if model_choice.startswith("XGBoost"):
        label, conf, prob, cn = xgb_predict(pil_img, xgb_obj, meta_obj)
        class_names = cn
    elif model_choice.startswith("EfficientNet"):
        label, conf, prob = effnet_predict(pil_img, eff_obj, class_names_global)
        class_names = class_names_global
    else:
        label, conf, prob = maxvit_predict(pil_img, maxvit_obj, maxvit_classes, maxvit_img)
        class_names = maxvit_classes

    prob = np.array(prob, dtype=float)
    idx_sorted = np.argsort(prob)[::-1]
    top1 = (class_names[int(idx_sorted[0])], float(prob[int(idx_sorted[0])]))
    top2 = (class_names[int(idx_sorted[1])], float(prob[int(idx_sorted[1])])) if len(prob) > 1 else ("-", 0.0)
    m = float(top1[1] - top2[1])

    low_conf = float(conf) < conf_th
    ambiguous = m < margin_th

    return {
        "pred_label": label,
        "confidence": float(conf),
        "margin": m,
        "top1": top1,
        "top2": top2,
        "top3": topk(prob, class_names, k=3),
        "low_conf": low_conf,
        "ambiguous": ambiguous,
    }


# =========================
# MAIN TABS
# =========================
tab_pred, tab_help = st.tabs(["🔮 Prediksi", "🧭 Panduan untuk User Awam"])

with tab_pred:
    left, right = st.columns([1.05, 1.0], gap="large")

    # ---- INPUT
    with left:
        st.markdown('<div class="card"><b>📥 Input</b><div class="small">Upload gambar/ZIP → sistem akan memprediksi + menilai risiko error (low confidence / ambiguous).</div></div>', unsafe_allow_html=True)
        st.write("")

        items: List[Tuple[str, Image.Image]] = []

        if mode == "Single / Multi Image":
            files = st.file_uploader("Upload gambar (bisa lebih dari 1)", type=["jpg","jpeg","png"], accept_multiple_files=True)
            if files:
                for f in files:
                    try:
                        items.append((f.name, Image.open(f).convert("RGB")))
                    except Exception:
                        pass
        else:
            zf = st.file_uploader("Upload ZIP berisi gambar", type=["zip"])
            if zf is not None:
                items = extract_images_from_zip(zf.read())

        if not items:
            st.info("👉 Pilih model & mode input di sidebar, lalu upload gambar/ZIP untuk mulai.")
            st.stop()

        st.markdown("#### 🖼️ Preview (maks 9 gambar)")
        cols = st.columns(3)
        for i, (name, img) in enumerate(items[:9]):
            with cols[i % 3]:
                st.image(img, caption=name, width="stretch")

    # ---- OUTPUT
    with right:
        st.markdown('<div class="card"><b>🧠 Output</b><div class="small">Hasil diurutkan berdasarkan confidence. Flag ⚠️ menandai prediksi berisiko.</div></div>', unsafe_allow_html=True)
        st.write("")

        rows = []
        risky_examples = []

        for name, img in items:
            out = predict_one(img)
            t1, t2 = out["top1"][0], out["top2"][0]
            rows.append({
                "filename": name,
                "pred_label": safe_label(out["pred_label"]),
                "confidence": round(out["confidence"], 4),
                "margin_top1_top2": round(out["margin"], 4),
                "low_conf": out["low_conf"],
                "ambiguous": out["ambiguous"],
                "top3": ", ".join([f"{safe_label(lbl)}:{p:.3f}" for lbl,p in out["top3"]]) if show_top3 else ""
            })
            if out["low_conf"] or out["ambiguous"]:
                risky_examples.append((name, img, out))

        df = pd.DataFrame(rows).sort_values(["confidence", "margin_top1_top2"], ascending=False)

        low_cnt = int(df["low_conf"].sum())
        amb_cnt = int(df["ambiguous"].sum())

        # KPIs
        st.markdown(
            f"""
<div class="kpi">
  <div class="k"><div class="t">Model Aktif</div><div class="v">{model_choice}</div></div>
  <div class="k"><div class="t">Total Gambar</div><div class="v">{len(df)}</div></div>
  <div class="k"><div class="t">Low confidence</div><div class="v">{low_cnt}</div></div>
  <div class="k"><div class="t">Ambiguous</div><div class="v">{amb_cnt}</div></div>
</div>
""",
            unsafe_allow_html=True
        )

        st.write("")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Distribusi Prediksi")
        st.bar_chart(df["pred_label"].value_counts())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download hasil (CSV)", data=csv, file_name="prediksi_sawit_cyberpunk.csv",
                           mime="text/csv", use_container_width=True)

        # Insight panel
        if show_insight:
            st.write("")
            st.markdown("#### 🔥 Insight & Analisis Risiko (tanpa ground-truth)")
            if not risky_examples:
                st.markdown('<span class="tag tag-ok">✅ Prediksi terlihat stabil (confidence & margin aman)</span>', unsafe_allow_html=True)
                st.write("- Jika hasil real-world turun, perhatikan pencahayaan & jarak foto.")
            else:
                # pilih yang paling berisiko: margin kecil + confidence kecil
                risky_examples.sort(key=lambda t: (t[2]["margin"], t[2]["confidence"]))
                name0, img0, out0 = risky_examples[0]
                top1, top2 = out0["top1"][0], out0["top2"][0]

                st.markdown('<span class="tag tag-warn">⚠️ Ada prediksi berisiko</span>', unsafe_allow_html=True)
                st.image(img0, caption=f"Contoh paling risk: {name0}", width="stretch")

                st.write(f"**Prediksi:** {safe_label(out0['pred_label'])}")
                st.write(f"**Confidence:** {out0['confidence']:.3f} | **Margin (Top1–Top2):** {out0['margin']:.3f}")
                if show_top3:
                    st.write("**Top-3:**")
                    st.table(pd.DataFrame([(safe_label(a), b) for a,b in out0["top3"]], columns=["kelas", "prob"]))

                st.write("**Kenapa bisa ragu?** (indikasi umum)")
                st.write("- Kelas transisi (warna mirip), blur, cahaya ekstrem, background ramai, objek terlalu kecil.")

                st.write("**Saran cepat untuk user:**")
                for tip in insight_tips(top1, top2):
                    st.write(f"- {tip}")

with tab_help:
    st.markdown('<div class="card"><b>🧭 Panduan Singkat (User Awam)</b><div class="small">Agar pengguna tidak bingung, ikuti langkah ini.</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        """
**Cara pakai (1 menit):**
1. **Pilih model** di sidebar  
   - *Balanced* (EffNet) → paling enak untuk pemakaian umum  
   - *Best* (MaxViT) → akurasi tinggi, tapi lebih lambat di CPU  
   - *Classic* (XGBoost) → sangat cepat, sensitif lighting
2. **Pilih mode input**: upload banyak gambar atau ZIP
3. **Upload** gambar/ZIP
4. Lihat **Label + Confidence**  
   - Kalau muncul **⚠️ low confidence / ambiguous**, coba foto ulang dengan saran yang ditampilkan.

**Tips foto terbaik:**
- Cahaya merata (natural), hindari bayangan
- Objek tandan memenuhi frame dan tidak blur
- Background sederhana/kontras
"""
    )
    st.info("Catatan: Error analysis full (Confusion Matrix/Classification Report) tetap dilakukan di notebook saat evaluasi test set.", icon="💡")
