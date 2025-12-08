# 🌴 Oil Palm Ripeness Classification — Classical ML vs Transfer Learning vs Transformers

**Klasifikasi kematangan buah kelapa sawit berbasis citra** untuk 3 kelas: **Mentah, Matang, Busuk**.  
Repositori ini menyajikan **pipeline eksperimen end-to-end** yang terstruktur, reproducible, dan berorientasi evaluasi:  
**dataset → split → EDA → preprocessing → classical ML → pretrained TL → transformers → evaluation & error analysis**.

---

## 📌 Highlights
- ✅ *Scientific workflow:* preprocessing terdefinisi, split train/val/test, evaluasi terstandar.
- ✅ *Multi-family comparison:* **Classical ML**, **Transfer Learning**, dan **Transformer-based**.
- ✅ Output paper-ready: **Accuracy, Macro-F1, Confusion Matrix, Learning Curves, Error Analysis**.
- ✅ Struktur folder eksperimen rapi (expA/expB/expC).

---

## 🎯 Problem Statement
Penentuan kematangan sawit di lapangan sering dilakukan secara manual sehingga berpotensi menimbulkan **variabilitas kualitas** dan **inefisiensi**.  
Studi ini mengevaluasi efektivitas pendekatan **computer vision** untuk klasifikasi kematangan sawit.

---

## 🧪 Research Objectives
1. Mengklasifikasikan kematangan buah sawit menjadi **Mentah / Matang / Busuk**.
2. Membandingkan performa **metode klasik**, **transfer learning**, dan **transformer**.
3. Menyediakan pipeline eksperimen yang **reproducible** untuk kebutuhan riset/paper/poster.

---

## 📦 Dataset
- Jumlah kelas: **3** (Mentah, Matang, Busuk)
- Format: citra (JPG/PNG)
- Split: **train / val / test**

---

## 🧹 Preprocessing
- Koreksi orientasi **EXIF**
- Resize & standardisasi ukuran (**224×224** untuk deep models)
- Normalisasi warna (opsional: Gray World)
- Simpan hasil preprocessing (JPEG quality 90) + verifikasi before/after

---

## 🏆 Key Results (Summary)
- **Best Classical:** **XGBoost + Color**  — **Acc 97%**
- **Best Transfer:** **EfficientNet-B0 + LoRA** — **Acc 97%** *(paling efisien untuk otomasi)*
- **Best Transformer:** **MaxViT-T + LoRA** — **Acc 98%** *(prediksi stabil antar kelas)*

---

## 👥 Kontributor
| Anggota Kelompok | NIM |
|---|---|
| Muhammad Wildan Nabila | 202210370311252 |
| Diemas Andung Prayoga | 202210370311363 |
| Irawana Juwita | 202210370311446 |

---

**Struktur yang direkomendasikan:**
```bash
dataset_sawit_split/
  train/
    mentah/
    matang/
    busuk/
  val/
    mentah/
    matang/
    busuk/
  test/
    mentah/
    matang/
    busuk/
