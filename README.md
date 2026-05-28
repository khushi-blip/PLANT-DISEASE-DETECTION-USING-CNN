# 🌿 Plant Disease Detector

A deep learning based web application that detects plant leaf diseases from leaf images using a custom Convolutional Neural Network (CNN).  
The system can identify diseased and healthy plant leaves using AI-powered image classification.

The project combines TensorFlow/Keras, Flask, and a modern frontend UI to provide real-time disease prediction directly from uploaded plant leaf images.

---

# 📌 Features

- 🌱 Detects plant leaf diseases using CNN
- 📷 Upload plant leaf images
- 🔍 Real-time disease prediction
- 📊 Confidence score display
- 💊 Treatment recommendations
- 🧠 Custom-trained deep learning model
- 🌐 Flask-based web application
- 🎨 Responsive modern UI
- ⚡ Fast prediction system

---

# 🧠 Disease Classes

| Disease | Risk Level | Description |
|---|---|---|
| Healthy | None | No disease detected |
| Early Blight | Medium | Fungal disease affecting plant leaves |
| Late Blight | High | Rapidly spreading plant leaf disease |

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- HTML5
- CSS3

---

# 📂 Project Structure

```bash
Plant-Disease-Detector/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── plant_model.h5
├── class_indices.json
├── training_history.png
├── confusion_matrix.png
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── dataset/
│
└── screenshots/
```

---

# 🧪 Model Architecture

The CNN model contains:

- Multiple Conv2D layers
- Batch Normalization
- MaxPooling layers
- Dropout regularization
- Dense fully connected layers
- Softmax output layer

Input Image Size:

```python
224 × 224 × 3
```

---

# 📊 Dataset

Dataset structure:

```bash
dataset/
├── Early_blight
├── Late_blight
└── Healthy
```

Dataset includes augmented plant leaf images for training and validation.

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Plant-Disease-Detector.git
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Application

```bash
python app.py
```

---

## 4️⃣ Open in Browser

```bash
http://127.0.0.1:5000
```

---

# 📸 Application Workflow

1. Upload plant leaf image  
2. CNN model processes image  
3. Disease prediction generated  
4. Confidence score displayed  
5. Treatment recommendation shown  

---

# 📈 Training Features

- Data Augmentation
- Class Weight Balancing
- Early Stopping
- ReduceLROnPlateau
- Model Checkpointing

---

# 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Validation Loss
- Confusion Matrix
- Classification Report

---

# 💡 Future Improvements

- Support more plant species
- Mobile application integration
- Real-time camera detection
- Cloud deployment
- Explainable AI visualizations
- Multi-disease classification

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

- PlantVillage Dataset
- TensorFlow
- Flask
- Open Source AI Community
