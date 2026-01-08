<a id="top"></a>
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

1. [Academic Information & Contributors](#academic)
2. [Project Overview](#overview)
3. [Dataset Description](#dataset)
4. [Experimental Scope](#experiment)
5. [Best Models (Final Selection)](#best-model)
6. [Training & Evaluation Results](#results)
7. [Best Model Performance Comparison](#comparison)
8. [Conclusion](#conclusion)
9. [Interactive Deployment](#deployment)
10. [Run the Dashboard Locally](#run-local)

---

<a id="academic"></a>
## 👤 Academic Information & Contributors

### 📌 Academic Context

| Attribute | Description |
| :-- | :-- |
| **Lead Author** | **Muhammad Wildan Nabila** |
| **Program** | Informatics |
| **Course** | Machine Learning |
| **Institution** | Universitas Muhammadiyah Malang |
| **Academic Year** | 2024 / 2025 |

### 👥 Contributors

| Name | Student ID |
|------|-----------|
| **Muhammad Wildan Nabila** | 202210370311252 |
| **Irawana Juwita** | 202210370311446 |
| **Diemas Andung Prayoga** | 202210370311363 |

---

<a id="overview"></a>
## 📌 Project Overview

Oil palm fruit ripeness significantly affects **harvest timing, oil quality, and economic value**. Conventional manual inspection is subjective and prone to inconsistency, motivating the development of an **automated computer vision-based classification system**.

This project conducts a **systematic experimental comparison** across three major modeling paradigms:

1. **Classical Machine Learning** using handcrafted visual features  
2. **Convolutional Neural Networks (CNN)**, including scratch and transfer learning  
3. **Vision Transformer architectures** with parameter-efficient fine-tuning (LoRA)

Only the **best-performing model from each paradigm** is selected for detailed analysis and deployment.

---

<a id="dataset"></a>
## 📊 Dataset Description

- **Data Type:** RGB Images  
- **Number of Classes:** 3  
- **Class Labels:**  
  - 🟢 Unripe (Mentah)  
  - 🟡 Ripe (Matang)  
  - 🔴 Rotten (Busuk)  

### 📷 Sample Images per Class

<div align="center">
  <img src="gambar/citra perkelas.png" width="600">
  <p><em>Figure 1. Representative oil palm fruit images for each ripeness class</em></p>
</div>

### 📊 Class Distribution

<div align="center">
  <img src="gambar/distribusi gambar per kelas.png" width="450">
  <p><em>Figure 2. Dataset class distribution</em></p>
</div>

---

<a id="experiment"></a>
## 🧪 Experimental Scope

### 🔹 Classical Machine Learning
- SVM (raw baseline, color, texture, gabor)
- XGBoost (raw baseline)
- XGBoost + Color (HSV)
- XGBoost + Texture
- XGBoost + Gabor

### 🔹 Convolutional Neural Networks (CNN)
- CNN from Scratch
- ResNet-50 (Frozen, Fine-Tuning, LoRA)
- EfficientNet-B0 (Frozen, Fine-Tuning, LoRA)

### 🔹 Vision Transformer
- ViT-B/16 (Frozen, Fine-Tuning, LoRA)
- MaxViT-T (Frozen, Fine-Tuning, LoRA)

> *This README highlights only the best-performing configuration from each paradigm to ensure clarity and academic rigor.*

---

<a id="best-model"></a>
## 🏆 Best Models (Final Selection)

| Paradigm | Best Model |
|--------|------------|
| Classical ML | **XGBoost + Color Features (HSV)** |
| CNN | **EfficientNet-B0 + LoRA** |
| Transformer | **MaxViT-T + LoRA** |

---

<a id="results"></a>
## 📈 Training & Evaluation Results

### 🔹 XGBoost + HSV  
**Accuracy:** **97.11%**  
Strong interpretable baseline; minor confusion occurs between adjacent ripeness stages.

### 🔹 EfficientNet-B0 + LoRA  
**Accuracy:** **97.78%**  
Excellent trade-off between accuracy and computational efficiency.

### 🔹 MaxViT-T + LoRA  
**Accuracy:** **98.67%**  
Best overall performance, capturing both local texture and global context.

---

<a id="comparison"></a>
## 📊 Best Model Performance Comparison

| Model | Paradigm | Accuracy | Strength |
|------|---------|----------|---------|
| XGBoost + HSV | Classical ML | 97.11% | Fast & interpretable |
| EfficientNet-B0 + LoRA | CNN | 97.78% | Efficient & accurate |
| MaxViT-T + LoRA | Transformer | **98.67%** | Best generalization |

---

<a id="conclusion"></a>
## 🧾 Conclusion

Vision Transformer models with **LoRA fine-tuning** achieve the highest accuracy, while CNN and classical approaches remain strong alternatives for resource-limited environments.

---

<a id="deployment"></a>
## 🚀 Interactive Deployment

A Streamlit dashboard enables real-time inference.

🔗 **Live Demo:**  
https://dashboard-sawit-ml-252.streamlit.app/

### ✨ Dashboard Features
- Image upload & prediction
- Probability visualization
- Best-model inference
- Responsive UI

---

<a id="run-local"></a>
## ▶️ Run the Dashboard Locally

```bash
git clone https://github.com/muhammadwildannabila/Machine_Learning_A_Klasifikasi_Sawit_252_363_446.git
cd Machine_Learning_A_Klasifikasi_Sawit_252_363_446
pip install -r requirements.txt
streamlit run app.py
