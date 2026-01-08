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

## 📌 Project Overview

Oil palm fruit ripeness directly affects **harvest quality, oil yield, and economic value**. Traditional visual inspection is subjective and inconsistent, motivating the need for an **automated computer vision-based solution**.

This project investigates and compares **three major modeling paradigms** for ripeness classification:

1. **Classical Machine Learning** (feature-based)
2. **Convolutional Neural Networks (CNN)** with Transfer Learning
3. **Vision Transformer (ViT)** architectures with LoRA fine-tuning

The best-performing model from each paradigm is evaluated and deployed in a **real-time Streamlit dashboard**.

---

## 📊 Dataset Description

- **Data Type:** RGB Images  
- **Number of Classes:** 3  
- **Class Labels:**  
  - 🟢 Unripe (Mentah)  
  - 🟡 Ripe (Matang)  
  - 🔴 Rotten (Busuk)  

### 📷 Sample Images per Class

<div align="center">
  <img src="gambar/citra perkelas.png" width="850">
  <p><em>Figure 1. Representative samples of oil palm fruit images for each ripeness class</em></p>
</div>

### 📊 Class Distribution

<div align="center">
  <img src="gambar/distribusi gambar per kelas.png" width="600">
  <p><em>Figure 2. Distribution of samples across classes</em></p>
</div>

---

## 🧠 Experimental Framework

To ensure **fair and interpretable comparison**, experiments are organized into three distinct categories.

### 🔹 Classical Machine Learning
- **XGBoost + Color Features (HSV)**
- Acts as an interpretable and fast baseline

### 🔹 Transfer Learning (CNN)
- **EfficientNet-B0 + LoRA**
- Lightweight CNN with parameter-efficient adaptation

### 🔹 Vision Transformer
- **MaxViT-T + LoRA**
- Transformer-based architecture capturing global visual context

---

## 🏆 Selected Best Models

Only the **best-performing model** from each paradigm is retained for final evaluation.

| Paradigm | Best Model |
|-------|------------|
| Classical ML | **XGBoost + HSV Color Features** |
| CNN (Transfer Learning) | **EfficientNet-B0 + LoRA** |
| Vision Transformer | **MaxViT-T + LoRA** |

---

## 📈 Training Dynamics

### 📉 Accuracy & Loss Curves (Best Models)

<div align="center">
  <img src="gambar/learning_curve_best_models.png" width="900">
  <p><em>Figure 3. Training and validation accuracy & loss for the selected best models</em></p>
</div>

---

## 🔍 Model Evaluation

### 🔹 Confusion Matrices

<div align="center">
  <img src="gambar/confusion_matrix_best_models.png" width="900">
  <p><em>Figure 4. Confusion matrices showing classification performance of the best models</em></p>
</div>

---

## 📊 Comparative Summary

| Model | Category | Strength |
|------|--------|---------|
| **XGBoost + HSV** | Classical ML | Interpretable & computationally efficient |
| **EfficientNet-B0 + LoRA** | CNN | Excellent accuracy-to-parameter ratio |
| **MaxViT-T + LoRA** | Transformer | Strong global feature modeling |

---

## 🚀 Interactive Deployment

A **Streamlit-based interactive dashboard** is developed to demonstrate real-time inference.

🔗 **Live Demo:**  
https://dashboard-sawit-ml-252.streamlit.app/

### ✨ Dashboard Features
- Image upload and prediction
- Probability distribution per class
- Best-model inference
- Clean and responsive interface

---

## ▶️ Run the Dashboard Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/muhammadwildannabila/Machine_Learning_A_Klasifikasi_Sawit_252_363_446.git
cd Machine_Learning_A_Klasifikasi_Sawit_252_363_446
```
### 2️⃣ Clone Repository
```bash
pip install -r requirements.txt
```
### 3️⃣ Launch Application
```bash
streamlit run app.py
```

## 👥 Kontributor <a id="kontributor"></a>

| Nama | NIM |
|------|-----|
| **Muhammad Wildan Nabila** | 202210370311252 |
| **Irawana Juwita** | 202210370311446 |
| **Diemas Andung Prayoga** | 202210370311363 |

