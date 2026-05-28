# 🌿 Plant Disease Detector using CNN

A deep learning web application that detects plant leaf diseases using a custom **Convolutional Neural Network (CNN)** model trained on the **PlantVillage Dataset**.

---

## 📌 Features

* 🌱 Detect plant diseases from leaf images
* 🧠 CNN-based deep learning model
* 📷 Upload JPG / PNG / JPEG images
* 📊 Displays prediction with confidence score
* 💊 Provides treatment recommendations
* ⚡ Fast and user-friendly interface

---

# 🌿 Disease Classes

| Class           | Risk Level | Description                          |
| --------------- | ---------- | ------------------------------------ |
| ✅ Healthy       | None       | No infection detected                |
| ⚠️ Early Blight | Medium     | Caused by *Alternaria solani* fungus |
| 🔴 Late Blight  | High       | Caused by *Phytophthora infestans*   |

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Flask
* HTML / CSS / JavaScript

---

## 🚀 How to Use

1. Open **Plant Disease Detector** in any modern browser
2. Upload a plant leaf image (`JPG / PNG / JPEG`, max 10MB)
3. Click **Detect Disease**
4. View:

   * Disease Prediction
   * Confidence Score
   * Treatment Plan

---

## 📦 Installation

### Clone the repository

```bash id="nxnljn"
git clone https://github.com/khushi-blip/PLANT-DISEASE-DETECTION-USING-CNN.git
cd PLANT-DISEASE-DETECTION-USING-CNN
```

### Install dependencies

```bash id="wj10iu"
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train the model

```bash id="r0z4ui"
python train.py
```

### Run prediction

```bash id="3s9lmz"
python predict.py
```

### Run the web application

```bash id="v1dbz2"
python app.py
```

---

## 🧠 Model Architecture

The CNN model consists of:

* 4 Convolutional Blocks
  `(32 → 64 → 128 → 256 filters)`
* Max Pooling Layers
* Dropout Layers
* Flatten Layer
* Dense Fully Connected Layers
* Softmax Activation (3-class output)

### CNN Workflow

```text id="1yj4f4"
Input Image → CNN Layers → Feature Extraction → Classification → Disease Prediction
```

---

## 📂 Project Structure

```bash id="3c6ojg"
PLANT-DISEASE-DETECTION-USING-CNN/
│
├── dataset/                 # Plant leaf image dataset
├── models/                  # Saved trained models
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
│   ├── train.py
│   ├── predict.py
│   └── preprocessing.py
│
├── Plant Disease Detector   # Frontend UI
├── app.py                   # Flask backend
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses plant leaf image datasets for training and testing.

### Dataset Sources

* PlantVillage Dataset
* Kaggle Plant Disease Datasets

---

## 📁 Files

```bash id="64rpwp"
├── Plant Disease Detector   # Main frontend interface
├── app.py                   # Flask backend
├── README.md
```

---
This project runs entirely in the browser using TensorFlow.js.
No Python backend or installation is required.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

Repository Link:

https://github.com/khushi-blip/PLANT-DISEASE-DETECTION-USING-CNN

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
