# 🌴 Sawit Ripeness Classification (Classical ML vs Transfer Learning vs Transformers)

> Sistem klasifikasi kematangan tandan sawit berbasis citra untuk **3 kelas: Mentah, Matang, Busuk**.  
> Repo ini berisi pipeline eksperimen end-to-end: **dataset → split → preprocessing → modeling → evaluation → error analysis**.

---

## 📌 Table of Contents
1. [Project](#project)  
   - [Latar Belakang](#latar-belakang)  
   - [Tujuan](#tujuan)  
2. [Dataset](#dataset)
3. [Eksperimen & Metodologi](#eksperimen--metodologi)  
   - [Preprocessing](#preprocessing)  
   - [Splitting Data](#splitting-data)  
   - [Pemodelan](#pemodelan)  
   - [Pemilihan Best Model](#pemilihan-best-model)
4. [Hasil Evaluasi & Analisis](#hasil-evaluasi--analisis)  
   - [Ringkasan Performa](#ringkasan-performa)  
   - [Confusion Matrix & Error Analysis](#confusion-matrix--error-analysis)  
5. [Cara Menjalankan (Lokal)](#cara-menjalankan-lokal)  
6. [Link Live Demo](#link-live-demo)  
7. [Keterbatasan](#keterbatasan)
8. [Struktur Folder](#struktur-folder) 
9. [Kontributor](#kontributor)  

---

## 🧾 Project <a id="project"></a>

### 🔍 Latar Belakang 
Penentuan kematangan tandan sawit di lapangan sering dilakukan secara manual sehingga berpotensi menimbulkan **subjektivitas**, dipengaruhi **pencahayaan**, **sudut pengambilan**, dan **variasi warna**.  
Penelitian ini mengevaluasi efektivitas pendekatan computer vision untuk membantu klasifikasi kematangan secara lebih konsisten.

### 🎯 Tujuan
1. Mengklasifikasikan kematangan sawit menjadi **Mentah / Matang / Busuk**.  
2. Membandingkan performa **metode klasik**, **transfer learning**, dan **transformer-based**.  
3. Menyediakan pipeline eksperimen yang rapi dan reproducible untuk kebutuhan riset/paper/poster.

---

## 📦 Dataset <a id="dataset"></a>
- Jumlah kelas: **3** (Mentah, Matang, Busuk)  
- Format: citra (JPG/PNG)  
- Sumber dataset: https://drive.google.com/drive/folders/1bntbpLT_nFVjbDB1XLJcmQl9zQtdeBBI?usp=sharing
- Jumlah data per kelas: 1000 data

---

## 🧪 Eksperimen & Metodologi

Bagian ini menjelaskan alur eksperimen dari penyiapan data hingga pemilihan model terbaik.  

---

### 🧼 Preprocessing
Preprocessing dilakukan untuk menyamakan format input dan meningkatkan konsistensi citra.
Langkah yang digunakan:
- Koreksi orientasi **EXIF**
- Resize & standardisasi ukuran (**224×224** untuk deep models)
- Normalisasi warna (opsional: Gray World)
- Simpan hasil preprocessing (JPEG quality 90) + verifikasi before/after

---

### ✂️ Splitting Data
Dataset dibagi menjadi **train / validation / test** agar evaluasi adil dan tidak bias, sebagai berikut:
- Train: 70%
- Validation: 15%
- Test: 15%

---

### 🧠 Pemodelan
Eksperimen dilakukan pada beberapa “keluarga” model untuk perbandingan menyeluruh.

#### A) Classical ML
- Model: XGBoost, SVM, dan ExtraTrees.
- Strategi: Extraction feature (color and texture (GLCM & LBP)).
- Kelebihan: cepat, ringan, relatif mudah dianalisis.

#### B) Transfer Learning
- Menggunakan backbone pretrained.
- Model: ResNet50 dan EfficientNet-B0.
- Strategi: **freeze backbone → train head, fine-tuning layer atas, dan LoRA**.
- Kelebihan: performa tinggi pada data terbatas.

#### C) Transformer-based 
- Menggunakan arsitektur transformer vision (contoh: ViT / MaxViT).
- Model: MaxVit-T dan ViT-B16
- Strategi: **freeze backbone → train head, fine-tuning layer atas, dan LoRA**.
- Kelebihan: kuat untuk pola visual kompleks.
  
📦 Output evaluasi yang disimpan per model:
- Confusion Matrix (CM)
- Classification Report (CR)
- Kurva accuracy/loss

---

### 🏆 Pemilihan Best Model
Best model dipilih berdasarkan gabungan beberapa indikator:
Kriteria pemilihan:
- Performa evaluasi tertinggi pada data test (Accuracy dan Macro-F1).
- Stabilitas training: gap train–validation kecil (indikasi overfitting lebih rendah).
- Confusion Matrix lebih baik: error antar kelas lebih sedikit dan lebih merata.

Best Model Pretrained
Best model: 
- XGBoost + Color
- EfficientNet-B0 + LoRA
- MaxVit-T + LoRA

---

## 📊 Hasil Evaluasi & Analisis

Bagian ini merangkum performa model dan menganalisis pola kesalahan prediksi menggunakan metrik evaluasi serta confusion matrix.  
Evaluasi dilakukan pada **test set** dan output disimpan pada folder `reports/` (CM, CR, kurva loss/acc, dll).

---

### 🏆 Hasil Evaluasi & Analisis

Berikut ringkasan model terbaik per “keluarga” berdasarkan evaluasi pada test set:
- **Best Classical:** **XGBoost + Color** 
- **Best Transfer:** **EfficientNet-B0 + LoRA**
- **Best Transformer:** **MaxViT-T + LoRA** 

Tabel ringkas:
| Kategori | Model Terbaik | Akurasi (Test) | Catatan |
|---|---|---:|---|
| Classical ML | XGBoost + Color | 97% | Cepat & ringan, kuat pada fitur warna |
| Transfer Learning | EfficientNet-B0 + LoRA | 97% | Efisien untuk otomasi, performa stabil |
| Transformer | MaxViT-T + LoRA | 98% | Prediksi paling stabil antar kelas |

📌 **Alasan pemilihan model terbaik:**  
Model dipilih berdasarkan **akurasi tinggi**, **stabilitas prediksi antar kelas**, dan (untuk TL/Transformer) **fine-tuning yang efisien** menggunakan LoRA sehingga lebih ringan dibanding full fine-tuning.












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
