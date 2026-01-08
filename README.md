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

## 📑 Table of Contents

1. [🌴 Oil Palm Fruit Ripeness Classification](#oil-palm-fruit-ripeness-classification)  
2. [👤 Academic Information & Contributors](#academic-information--contributors)  
   1. [📌 Academic Context](#academic-context)  
   2. [👥 Contributors](#contributors)  
3. [📌 Project Overview](#project-overview)  
4. [📊 Dataset Description](#dataset-description)  
   1. [📷 Sample Images per Class](#sample-images-per-class)  
   2. [📊 Class Distribution](#class-distribution)  
5. [🧪 Experimental Scope](#experimental-scope)  
   1. [🔹 Classical Machine Learning](#classical-machine-learning)  
   2. [🔹 Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)  
   3. [🔹 Vision Transformer](#vision-transformer)  
6. [🏆 Best Models (Final Selection)](#best-models-final-selection)  
7. [📈 Training & Evaluation Results](#training--evaluation-results)  
   1. [🔹 XGBoost + Color Features (HSV)](#xgboost-color-features-hsv)  
   2. [🔹 EfficientNet-B0 + LoRA](#efficientnet-b0-lora)  
   3. [🔹 MaxViT-T + LoRA](#maxvit-t-lora)  
8. [📊 Best Model Performance Comparison](#best-model-performance-comparison)  
9. [🧾 Conclusion](#conclusion)  
10. [🚀 Interactive Deployment](#interactive-deployment)  
    1. [✨ Dashboard Features](#dashboard-features)  
11. [▶️ Run the Dashboard Locally](#run-the-dashboard-locally)

---

<a id="academic-information--contributors"></a>
## 👤 Academic Information & Contributors

<a id="academic-context"></a>
### 📌 Academic Context

| Attribute | Description |
| :-- | :-- |
| **Lead Author** | **Muhammad Wildan Nabila** |
| **Program** | Informatics |
| **Course** | Machine Learning |
| **Institution** | Universitas Muhammadiyah Malang |
| **Academic Year** | 2024 / 2025 |

<a id="contributors"></a>
### 👥 Contributors

| Name | Student ID |
|------|-----------|
| **Muhammad Wildan Nabila** | 202210370311252 |
| **Irawana Juwita** | 202210370311446 |
| **Diemas Andung Prayoga** | 202210370311363 |

---

<a id="project-overview"></a>
## 📌 Project Overview

Oil palm fruit ripeness significantly affects **harvest timing, oil quality, and economic value**. Conventional manual inspection is subjective and prone to inconsistency, motivating the development of an **automated computer vision-based classification system**.

This project conducts a **systematic experimental comparison** across three major modeling paradigms:

1. **Classical Machine Learning** using handcrafted visual features  
2. **Convolutional Neural Networks (CNN)**, including scratch and transfer learning  
3. **Vision Transformer architectures** with parameter-efficient fine-tuning (LoRA)

Only the **best-performing model from each paradigm** is selected for detailed analysis and deployment.

---

<a id="dataset-description"></a>
## 📊 Dataset Description

- **Data Type:** RGB Images  
- **Number of Classes:** 3  
- **Class Labels:**  
  - 🟢 Unripe (Mentah)  
  - 🟡 Ripe (Matang)  
  - 🔴 Rotten (Busuk)  

<a id="sample-images-per-class"></a>
### 📷 Sample Images per Class

<div align="center">
  <img src="gambar/citra perkelas.png" width="620">
  <p><em>Figure 1. Representative oil palm fruit images for each ripeness class</em></p>
</div>

<a id="class-distribution"></a>
### 📊 Class Distribution

<div align="center">
  <img src="gambar/distribusi gambar per kelas.png" width="460">
  <p><em>Figure 2. Dataset class distribution</em></p>
</div>

---

<a id="experimental-scope"></a>
## 🧪 Experimental Scope

Multiple models and configurations were evaluated before selecting the final candidates.

<a id="classical-machine-learning"></a>
### 🔹 Classical Machine Learning
- SVM (raw baseline, color, texture, gabor)
- XGBoost (raw baseline)
- XGBoost + Color (HSV)
- XGBoost + Texture
- XGBoost + Gabor

<a id="convolutional-neural-networks-cnn"></a>
### 🔹 Convolutional Neural Networks (CNN)
- CNN from Scratch
- ResNet-50 (Frozen, Fine-Tuning, LoRA)
- EfficientNet-B0 (Frozen, Fine-Tuning, LoRA)

<a id="vision-transformer"></a>
### 🔹 Vision Transformer
- ViT-B/16 (Frozen, Fine-Tuning, LoRA)
- MaxViT-T (Frozen, Fine-Tuning, LoRA)

> 📌 *Although numerous configurations were tested, this README focuses on the best-performing model from each paradigm to maintain clarity and academic rigor.*

---

<a id="best-models-final-selection"></a>
## 🏆 Best Models (Final Selection)

| Paradigm | Best Model |
|--------|------------|
| Classical ML | **XGBoost + Color Features (HSV)** |
| CNN (Transfer Learning) | **EfficientNet-B0 + LoRA** |
| Vision Transformer | **MaxViT-T + LoRA** |

---

<a id="training--evaluation-results"></a>
## 📈 Training & Evaluation Results  
### *Best of the Best Models*

<a id="xgboost-color-features-hsv"></a>
### 🔹 1. XGBoost + Color Features (HSV)

... *(gambar & analisis)*

<a id="efficientnet-b0-lora"></a>
### 🔹 2. EfficientNet-B0 + LoRA

... *(gambar & analisis)*

<a id="maxvit-t-lora"></a>
### 🔹 3. MaxViT-T + LoRA

... *(gambar & analisis)*

---

<a id="best-model-performance-comparison"></a>
## 📊 Best Model Performance Comparison

... *(tabel)*

---

<a id="conclusion"></a>
## 🧾 Conclusion

... *(konten)*

---

<a id="interactive-deployment"></a>
## 🚀 Interactive Deployment

🔗 **Live Demo:** https://dashboard-sawit-ml-252.streamlit.app/

<a id="dashboard-features"></a>
### ✨ Dashboard Features
- Image upload and ripeness prediction
- Class probability visualization
- Best-model inference
- Clean and responsive interface

---

<a id="run-the-dashboard-locally"></a>
## ▶️ Run the Dashboard Locally

```bash
git clone https://github.com/muhammadwildannabila/Machine_Learning_A_Klasifikasi_Sawit_252_363_446.git
cd Machine_Learning_A_Klasifikasi_Sawit_252_363_446
pip install -r requirements.txt
streamlit run app.py
```
