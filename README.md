# 🌴 Sawit Ripeness Classification (Classical ML vs Transfer Learning vs Transformers)

> Sistem klasifikasi kematangan tandan sawit berbasis citra untuk **3 kelas: Mentah, Matang, Busuk**.  
> Repo ini berisi pipeline eksperimen end-to-end: **dataset → split → preprocessing → modeling → evaluation → error analysis**.

---

## 📌 Table of Contents
1. [Project](#project)  
   - [Latar Belakang](#latar-belakang)  
   - [Tujuan](#tujuan)  
2. [Dataset](#dataset)
3. [Eksperimen & Metodologi](#eksperimen-metodologi)  
   - [Preprocessing](#preprocessing)  
   - [Splitting Data](#splitting-data)  
   - [Pemodelan](#pemodelan)  
   - [Pemilihan Best Model](#pemilihan-best-model)
4. [Hasil Evaluasi & Analisis](#hasil-evaluasi-analisis)  
5. [Cara Menjalankan (Lokal)](#cara-menjalankan-lokal)  
6. [Link Demo Dashboard](#link-live-demo)  
7. [Keterbatasan](#keterbatasan)
8. [Struktur Folder](#struktur-folder) 
9. [Kontributor](#kontributor)  

---

## 🧾 Project <a id="project"></a>

### 🔍 Latar Belakang <a id="latar-belakang"></a>
Penentuan kematangan tandan sawit di lapangan sering dilakukan secara manual sehingga berpotensi menimbulkan **subjektivitas**, dipengaruhi **pencahayaan**, **sudut pengambilan**, dan **variasi warna**.  
Penelitian ini mengevaluasi efektivitas pendekatan computer vision untuk membantu klasifikasi kematangan secara lebih konsisten.

### 🎯 Tujuan <a id="tujuan"></a>
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

## 🧪 Eksperimen & Metodologi <a id="eksperimen-metodologi"></a>

Bagian ini menjelaskan alur eksperimen dari penyiapan data hingga pemilihan model terbaik.  

---

### 🧼 Preprocessing <a id="preprocessing"></a>
Preprocessing dilakukan untuk menyamakan format input dan meningkatkan konsistensi citra.
Langkah yang digunakan:
- Koreksi orientasi **EXIF**
- Resize & standardisasi ukuran (**224×224** untuk deep models)
- Normalisasi warna (opsional: Gray World)
- Simpan hasil preprocessing (JPEG quality 90) + verifikasi before/after

---

### ✂️ Splitting Data <a id="splitting-data"></a>
Dataset dibagi menjadi **train / validation / test** agar evaluasi adil dan tidak bias, sebagai berikut:
- Train: 70%
- Validation: 15%
- Test: 15%

---

### 🧠 Pemodelan <a id="pemodelan"></a>
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

### 🏆 Pemilihan Best Model <a id="pemilihan-best-model"></a>
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

## 📊 Hasil Evaluasi & Analisis <a id="hasil-evaluasi-analisis"></a>

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

---
## 💻 Cara Menjalankan (Lokal) <a id="cara-menjalankan-lokal"></a>

Bagian ini menjelaskan cara menjalankan project secara lokal.
### 1) Clone repository & masuk folder project
`git clone https://github.com/USERNAME/NAMA_REPO.git
cd NAMA_REPO/`

### 2) Buat virtual environment
Windows (PowerShell):
`python -m venv .venv`
`.\.venv\Scripts\Activate.ps1`

### 3) Install dependency
Jika menggunakan requirements.txt:
`python -m pip install --upgrade pip`
`python -m pip install -r requirements.txt`

### 4) Jalankan Demo Aplikasi (Streamlit)
`python -m streamlit run src/app.py`

## 🔗 Link Demo Dashboard <a id="link-live-demo"></a>

Aplikasi/hasil demo dapat diakses oleh pengguna lain melalui link berikut:
- **Live Demo:** https://dashboardklasifikasisawit252363446.streamlit.app/

📌 Jika link sedang tidak bisa diakses, project tetap dapat dijalankan secara lokal pada bagian **Cara Menjalankan (Lokal)**.

---

## ⚠️ Keterbatasan <a id="keterbatasan"></a>

Beberapa keterbatasan pada project ini:
1. **Sensitif terhadap kondisi foto**  
   Performa dapat menurun pada pencahayaan ekstrem (terlalu gelap/terang), blur, atau background terlalu ramai.

2. **Kemiripan visual antar kelas**  
   Pada kelas transisi kematangan, perbedaan visual bersifat gradual sehingga berpotensi menimbulkan prediksi ambigu.

3. **Model klasik bergantung pada fitur**  
   Perubahan warna akibat lighting dapat memengaruhi hasil ekstraksi fitur (mis. HSV/tekstur), sehingga error bisa meningkat pada kondisi tertentu.

Rencana perbaikan:
- Menambah data dengan variasi pencahayaan, sudut, dan background.
- Augmentasi yang lebih robust namun tetap realistis.

---

## 👥 Kontributor <a id="kontributor"></a>

| Nama | NIM |
|------|-----|
| Muhammad Wildan Nabila | 202210370311252 |
| Irawana Juwita | 202210370311446 |
| Diemas Andung Prayoga | 202210370311363 |



