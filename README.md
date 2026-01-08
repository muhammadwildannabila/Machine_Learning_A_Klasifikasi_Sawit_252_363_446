# 🌴 Oil Palm Fruit Ripeness Classification  
### A Comparative Study of Classical Machine Learning, CNN, and Vision Transformer Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classical%20ML-F7931E?style=for-the-badge&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Final Project – Machine Learning**  
Universitas Muhammadiyah Malang  

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://dashboard-sawit-ml-252.streamlit.app/)  
[![Source Code](https://img.shields.io/badge/💻_Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/muhammadwildannabila/Machine_Learning_A_Klasifikasi_Sawit_252_363_446)

</div>

---

## 👤 Author & Academic Context

| Attribute | Description |
| :-- | :-- |
| **Author** | **Muhammad Wildan Nabila** |
| **Program** | Informatics |
| **Course** | Machine Learning |
| **Institution** | Universitas Muhammadiyah Malang |
| **Academic Year** | 2024 / 2025 |

---

## 👥 Contributors <a id="kontributor"></a>

| Name | Student ID |
|------|-----------|
| **Muhammad Wildan Nabila** | 202210370311252 |
| **Irawana Juwita** | 202210370311446 |
| **Diemas Andung Prayoga** | 202210370311363 |

| Attribute | Description |
| :-- | :-- |
| **Program** | Informatics |
| **Course** | Machine Learning |
| **Institution** | Universitas Muhammadiyah Malang |
| **Academic Year** | 2024 / 2025 |


---

## 📌 Project Overview

Oil palm fruit ripeness significantly affects **harvest timing, oil quality, and economic value**. Manual visual inspection is subjective and inconsistent, motivating the development of an **automated image-based classification system**.

This study conducts a structured comparison of **three modeling paradigms**:
1. **Classical Machine Learning** using handcrafted features  
2. **Deep Learning (CNN)** with transfer learning and LoRA  
3. **Vision Transformer** with parameter-efficient fine-tuning  

Only the **best-performing model from each paradigm** is selected for in-depth evaluation and deployment.

---

## 📊 Dataset Description

- **Data Type:** RGB images  
- **Number of Classes:** 3  
- **Class Labels:**  
  - 🟢 Unripe (Mentah)  
  - 🟡 Ripe (Matang)  
  - 🔴 Rotten (Busuk)  

### 📷 Sample Images per Class

<div align="center">
  <img src="gambar/citra perkelas.png" width="850">
  <p><em>Figure 1. Sample oil palm fruit images for each ripeness class</em></p>
</div>

### 📊 Class Distribution

<div align="center">
  <img src="gambar/distribusi gambar per kelas.png" width="600">
  <p><em>Figure 2. Dataset class distribution</em></p>
</div>

---

## 🧪 Experimental Scope

This research evaluates multiple models and strategies before selecting the best representative from each category.

### 🔹 Classical Machine Learning
- SVM (Color, Texture, Gabor)
- XGBoost (Color/HSV, Texture, Gabor)

### 🔹 Transfer Learning (CNN)
- ResNet-50 (Freeze, Fine-Tuning, LoRA)
- EfficientNet-B0 (Freeze, Fine-Tuning, LoRA)

### 🔹 Vision Transformer
- ViT-B/16 (Freeze, Fine-Tuning, LoRA)
- MaxViT-T (Freeze, Fine-Tuning, LoRA)

---

## 🏆 Best Models (Final Selection)

| Category | Best Model |
|--------|-----------|
| Classical Machine Learning | **XGBoost + Color Features (HSV)** |
| CNN (Transfer Learning) | **EfficientNet-B0 + LoRA** |
| Vision Transformer | **MaxViT-T + LoRA** |

---

## 📈 Training & Evaluation Results (Best of the Best)

### 🔹 1. XGBoost + Color Features (HSV)

<div align="center">
  <img src="gambar/ACC_XGBOOST + Color.png" width="800">
  <img src="gambar/LOSS_XGBOOST + Color.png" width="800">
  <p><em>Figure 3. Accuracy and loss curve of XGBoost + HSV</em></p>

  <img src="gambar/CM_XGBOOST + Color.png" width="450">
  <p><em>Figure 4. Confusion matrix of XGBoost + HSV</em></p>
</div>

**Analysis:**  
XGBoost with HSV color features shows stable performance across all classes. Misclassification mainly occurs between adjacent ripeness stages, indicating visually similar color characteristics.

---

### 🔹 2. EfficientNet-B0 + LoRA

<div align="center">
  <img src="gambar/efficientnet_curve.png" width="800">
  <p><em>Figure 5. Accuracy and loss curve of EfficientNet-B0 + LoRA</em></p>

  <img src="gambar/efficientnet_cm.png" width="450">
  <p><em>Figure 6. Confusion matrix of EfficientNet-B0 + LoRA</em></p>
</div>

**Analysis:**  
EfficientNet-B0 with LoRA achieves high accuracy with significantly fewer trainable parameters. The model demonstrates strong generalization and balanced performance across classes.

---

### 🔹 3. MaxViT-T + LoRA

<div align="center">
  <img src="gambar/maxvit_curve.png" width="800">
  <p><em>Figure 7. Accuracy and loss curve of MaxViT-T + LoRA</em></p>

  <img src="gambar/maxvit_cm.png" width="450">
  <p><em>Figure 8. Confusion matrix of MaxViT-T + LoRA</em></p>
</div>

**Analysis:**  
MaxViT-T with LoRA delivers the best overall performance by effectively capturing both local and global visual patterns, resulting in superior class separation.

---

## 📊 Best Model Performance Comparison

| Model | Paradigm | Key Strength |
|------|---------|-------------|
| **XGBoost + HSV** | Classical ML | Fast, interpretable, low computational cost |
| **EfficientNet-B0 + LoRA** | CNN | High accuracy with parameter efficiency |
| **MaxViT-T + LoRA** | Transformer | Best overall generalization performance |

---

## 🚀 Interactive Deployment

A **Streamlit-based dashboard** is developed for real-time inference.

🔗 **Live Demo:**  
https://dashboard-sawit-ml-252.streamlit.app/

### ✨ Features
- Image upload & prediction
- Probability visualization
- Best-model inference
- User-friendly interface

---

## ▶️ Run Locally

```bash
git clone https://github.com/muhammadwildannabila/Machine_Learning_A_Klasifikasi_Sawit_252_363_446.git
cd Machine_Learning_A_Klasifikasi_Sawit_252_363_446
pip install -r requirements.txt
streamlit run app.py
