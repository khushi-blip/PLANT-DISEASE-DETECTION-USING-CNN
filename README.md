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

1. Open `index.html` in any modern browser
2. Upload a plant leaf image (`JPG / PNG / JPEG`, max 10MB)
3. Click **Detect Disease**
4. View:

   * Disease Prediction
   * Confidence Score
   * Treatment Plan

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/khushi-blip/PLANT-DISEASE-DETECTION-USING-CNN.git
cd PLANT-DISEASE-DETECTION-USING-CNN
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train the model

```bash
python train.py
```

### Run prediction

```bash
python predict.py
```

### Run the web application

```bash
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

### Workflow

```text
Input Image → CNN Layers → Feature Extraction → Classification → Disease Prediction
```

---

## 📂 Project Structure

```bash
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
├── index.html               # Frontend UI
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

```bash
├── index.html      # Frontend interface
├── app.py          # Flask application
├── README.md
```

---

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
