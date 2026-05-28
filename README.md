#  Plant Disease Detector

A deep learning web app that detects plant leaf diseases using a custom CNN model trained on the **PlantVillage Dataset**.

## 🌿 Disease Classes
| Class | Risk | Description |
|-------|------|-------------|
| ✅ Healthy | None | No infection detected |
| ⚠️ Early Blight | Medium | Caused by *Alternaria solani* fungus |
| 🔴 Late Blight | High | Caused by *Phytophthora infestans* |

🛠️ Technologies Used
Python
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Scikit-learn

## 🚀 How to Use
1. Open `index.html` in any modern browser
2. Upload a plant leaf photo (JPG / PNG / JPEG — max 10MB)
3. Click **"Detect Disease"**
4. View the prediction, confidence score, and treatment plan

📦 Installation

Clone the repository:

git clone https://github.com/khushi-blip/PLANT-DISEASE-DETECTION-USING-CNN
cd plant-disease-detection

Install dependencies:

pip install -r requirements.txt
▶️ Run the Project

Train the model:

python train.py

Run prediction:

python predict.py

If using a web app:

python app.py

## 🧠 Model Architecture
- **4 Convolutional Blocks** (32 → 64 → 128 → 256 filters)
- Flatten → Dense → Softmax (3-class output)
- Framework: TensorFlow + Flask
- Dataset: PlantVillage
- Plant-Disease-Detection
│
├── dataset/                # Dataset containing plant leaf images
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── train.py
│   ├── predict.py
│   └── preprocessing.py
│
├── requirements.txt
├── README.md
└── app.py

 CNN Architecture

The CNN model includes:

->Convolution Layers
->Max Pooling Layers
->Dropout Layers
->Fully Connected Dense Layers
->Softmax Activation for Classification

Example workflow:

Input Image → CNN Layers → Feature Extraction → Classification → Disease Prediction

📊 Dataset

This project uses plant leaf image datasets for training and testing.
You can use datasets from:

PlantVillage Dataset
Kaggle Plant Disease datasets

## 📁 Files
```
├── index.html      # Main frontend (all-in-one HTML/CSS/JS)
└── README.md


🤝 Contributing

Contributions are welcome!

Fork the repository  https://github.com/khushi-blip/PLANT-DISEASE-DETECTION-USING-CNN
Create a new branch
Commit your changes
Push to the branch
Open a Pull Request

📜 License
This project is licensed under the MIT License.

If you like this project, give it a ⭐ on GitHub!
If you like this project, give it a ⭐ on GitHub!
``
