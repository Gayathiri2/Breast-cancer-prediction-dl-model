# 🧠 Breast Cancer Detection Using Deep Learning (VGG16)

This project uses deep learning and transfer learning techniques to detect breast cancer from CT scan images. The model is built using the VGG16 architecture and implemented in a Google Colab notebook for accessibility and ease of use.

---

## 📌 Project Overview

- **Model**: VGG16 (Pretrained on ImageNet)
- **Technique**: Transfer Learning + Fine-tuning
- **Dataset**: Breast cancer CT scan image dataset
- **Accuracy Achieved**: **98%**
- **File Format**: Colab Notebook (`.ipynb`)

---

## 📂 Repository Contents


---

## 🔗 Google Colab

You can directly run this project in Colab:  
👉 [Open in Google Colab]()

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Google Colab

---

## 📊 Project Highlights

- Preprocessed the image dataset (resizing, normalization, augmentation)
- Fine-tuned the VGG16 model for binary classification
- Achieved 98% accuracy on validation set
- Visualized model performance (accuracy, loss, confusion matrix)
- Predicts whether a CT scan is **cancerous** or **non-cancerous**

---

## 🧪 How to Use

1. **Open the Notebook in Google Colab**  
   Click [here]()

2. **Upload the Dataset**  
   - Place your images in two folders: `cancerous/` and `non-cancerous/`
   - Upload the folders to Colab as instructed in the notebook

3. **Run All Cells**  
   The notebook will guide you through:
   - Importing libraries  
   - Loading and preprocessing data  
   - Building and training the model  
   - Evaluating and predicting

---

## 📁 Dataset

- **Source**: Medical breast cancer image dataset
> *Note: Dataset is included for educational/research purposes only.*

---

## ✨ Results

- **Model**: VGG16 + Custom Dense Layers
- **Training Accuracy**: 98%
- **Validation Accuracy**: High generalization with minimal overfitting
- **Performance Metrics**: Accuracy, Loss Curve, Confusion Matrix



