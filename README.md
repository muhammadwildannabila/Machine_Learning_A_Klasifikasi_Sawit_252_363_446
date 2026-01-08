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

## 👤 Academic Information & Contributors <a id="kontributor"></a>

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

## 📌 Project Overview

Oil palm fruit ripeness directly affects **harvest timing, oil quality, and economic value**. Manual inspection based on human vision is subjective and inconsistent, motivating the development of an **automated image-based classification system**.

This project performs a **systematic comparison of three modeling paradigms**:

1. **Classical Machine Learning** using handcrafted visual features  
2. **Convolutional Neural Networks (CNN)**, including scratch and transfer learning  
3. **Vision Transformer architectures** with parameter-efficient fine-tuning  

All models are evaluated under identical data splits, and only the **best-performing model from each paradigm** is selected for final analysis and deployment.

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
  <img src="gambar/citra perkelas.png" width="650">
  <p><em>Figure 1. Representative samples of oil palm fruit images for each ripeness class</em></p>
</div>

### 📊 Class Distribution

<div align="center">
  <img src="gambar/distribusi gambar per kelas.png" width="480">
  <p><em>Figure 2. Distribution of samples across ripeness classes</em></p>
</div>

---

## 🧪 Experimental Scope

This study explores **multiple models and learning strategies** before selecting the best representative from each paradigm.

### 🔹 Classical Machine Learning
- SVM (without features / raw pixel baseline)
- SVM + Color
- SVM + Texture
- SVM + Gabor
- XGBoost (without handcrafted features)
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

> 📌 *Although many configurations are evaluated, visual analysis is focused on the best-performing model from each category to ensure clarity and academic rigor.*

---

## 🏆 Best Models (Final Selection)

| Paradigm | Best Model |
|--------|------------|
| Classical ML | **XGBoost + Color Features (HSV)** |
| CNN (Transfer Learning) | **EfficientNet-B0 + LoRA** |
| Vision Transformer | **MaxViT-T + LoRA** |

---

## 📈 Training & Evaluation Results  
### *Best of the Best Models*

---

### 🔹 1. XGBoost + Color Features (HSV)

<div align="center">
  <img src="gambar/ACC_XGBOOST + Color.png" width="520">
  <img src="gambar/LOSS_XGBOOST + Color.png" width="520">
  <p><em>Figure 3. Training accuracy and loss of XGBoost + HSV</em></p>

  <img src="gambar/CM_XGBOOST + Color.png" width="360">
  <p><em>Figure 4. Confusion matrix of XGBoost + HSV</em></p>
</div>

**Analysis:**  
This model demonstrates stable and interpretable performance. Most misclassifications occur between adjacent ripeness stages, reflecting natural visual similarity in color transitions.

---

### 🔹 2. EfficientNet-B0 + LoRA

<div align="center">
  <img src="gambar/ACC_EffecientNet-B0 + LoRA.png" width="520">
  <img src="gambar/LOSS_EffecientNet-B0 + LoRA.png" width="520">
  <p><em>Figure 5. Training accuracy and loss of EfficientNet-B0 + LoRA</em></p>

  <img src="gambar/CM_EffecientNet-B0 + LoRA.png" width="360">
  <p><em>Figure 6. Confusion matrix of EfficientNet-B0 + LoRA</em></p>
</div>

**Analysis:**  
EfficientNet-B0 with LoRA achieves strong generalization while maintaining parameter efficiency. Performance is well-balanced across all classes.

---

### 🔹 3. MaxViT-T + LoRA

<div align="center">
  <img src="gambar/ACC_MaxVit-T + LoRA.png" width="520">
  <img src="gambar/LOSS_MaxVit-T + LoRA.png" width="520">
  <p><em>Figure 7. Training accuracy and loss of MaxViT-T + LoRA</em></p>

  <img src="gambar/CM_MaxVit-T + LoRA.png" width="360">
  <p><em>Figure 8. Confusion matrix of MaxViT-T + LoRA</em></p>
</div>

**Analysis:**  
MaxViT-T with LoRA provides the best overall performance by effectively capturing both local texture patterns and global spatial relationships.

---

## 📊 Best Model Performance Summary

| Model | Paradigm | Key Advantage |
|------|---------|---------------|
| **XGBoost + HSV** | Classical ML | Fast, interpretable, low computational cost |
| **EfficientNet-B0 + LoRA** | CNN | Excellent accuracy–efficiency trade-off |
| **MaxViT-T + LoRA** | Transformer | Best overall generalization capability |

---

## 🚀 Interactive Deployment

A **Streamlit-based interactive dashboard** is developed for real-time ripeness prediction.

🔗 **Live Demo:**  
https://dashboard-sawit-ml-252.streamlit.app/

### ✨ Dashboard Features
- Upload image and classify ripeness
- Class probability visualization
- Best-model inference
- Clean and responsive UI

---

## ▶️ Run the Dashboard Locally

```bash
git clone https://github.com/muhammadwildannabila/Machine_Learning_A_Klasifikasi_Sawit_252_363_446.git
cd Machine_Learning_A_Klasifikasi_Sawit_252_363_446
pip install -r requirements.txt
streamlit run app.py
