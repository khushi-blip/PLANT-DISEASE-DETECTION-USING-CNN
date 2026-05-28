import os

DATASET_DIR = 'dataset'

# Dataset folder structure should be:
# dataset/
#   Potato___Early_blight/   (1000 images)
#   Potato___Late_blight/    (1000 images)
#   Potato___healthy/        ( 152 images)

classes = sorted(os.listdir(DATASET_DIR))
print(f"Total classes: {len(classes)}")

total = 0
for cls in classes:
    path  = os.path.join(DATASET_DIR, cls)
    count = len(os.listdir(path))
    total += count
    print(f"  {cls}: {count} images")

print(f"Total images: {total}")
# ── All settings in one place ──────────────────────
DATASET_DIR  = 'dataset'
IMG_SIZE     = (224, 224)   # CNN input size
BATCH_SIZE   = 32           # images per batch
EPOCHS       = 30           # max training rounds
NUM_CLASSES  = 3            # Healthy, Early, Late
LR           = 0.001        # Adam learning rate
MODEL_PATH   = 'plant_model.h5'
CLASS_JSON   = 'class_indices.json'
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = 'dataset'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# ── TRAINING: augmentation + normalize ─────────────
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,  # pixel 0-255 → 0-1
    rotation_range     = 20,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    zoom_range         = 0.15,
    horizontal_flip    = True,
    shear_range        = 0.1,
    fill_mode          = 'nearest',
    validation_split   = 0.20        # 80% train, 20% val
)

# ── VALIDATION: only normalize, NO augmentation ────
valid_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = 0.20
)

# ── Load from Kaggle folder structure ──────────────
# folder name = label automatically
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',  # one-hot labels
    shuffle     = True,
    subset      = 'training'
)

valid_gen = valid_datagen.flow_from_directory(
    DATASET_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    shuffle     = False,
    subset      = 'validation'
)

print("Train:", train_gen.samples, "| Valid:", valid_gen.samples)
print("Classes:", train_gen.class_indices)
# {'Potato___Early_blight':0,'Potato___Late_blight':1,'Potato___healthy':2}

# ── Save class index mapping for prediction ─────────
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
with open('class_indices.json', 'w') as f:
    json.dump(idx_to_class, f)

# ── Fix class imbalance (Healthy only 152 images) ──
labels = train_gen.classes
cw_arr = compute_class_weight(
    'balanced',
    classes = np.unique(labels),
    y       = labels
)
class_weights = dict(enumerate(cw_arr))
print("Class weights:", class_weights)
# {0: 0.72, 1: 0.72, 2: 4.73}  → Healthy gets ~6x weight
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(num_classes=3):
    model = models.Sequential(name='PotatoDisease_CNN')

    # ── BLOCK 1: Detect edges & basic colors ──────────
    # Input: 224×224×3  →  Output: 112×112×32
    model.add(layers.Conv2D(32, (3,3), padding='same',
                            input_shape=(224,224,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))   # 224→112
    model.add(layers.Dropout(0.25))

    # ── BLOCK 2: Detect textures & disease spots ──────
    # Input: 112×112×32  →  Output: 56×56×64
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))   # 112→56
    model.add(layers.Dropout(0.25))

    # ── BLOCK 3: Detect complex disease patterns ──────
    # Input: 56×56×64  →  Output: 28×28×128
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))   # 56→28
    model.add(layers.Dropout(0.30))

    # ── BLOCK 4: High-level disease signatures ────────
    # Input: 28×28×128  →  Output: 14×14×256
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))   # 28→14
    model.add(layers.Dropout(0.30))

    # ── CLASSIFICATION HEAD ───────────────────────────
    # 14×14×256 = 50,176 values → 1D vector
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.50))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.30))

    # Output: 3 probabilities that sum to 1.0
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = build_cnn(3)
model.summary()
print(f"Total params: {model.count_params():,}")
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# ── Compile ───────────────────────────────────────
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

# ── Callbacks ─────────────────────────────────────
cb_list = [
    # Stop if val_accuracy doesn't improve for 8 epochs
    callbacks.EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 8,
        restore_best_weights = True,
        verbose              = 1
    ),
    # Halve LR if val_loss doesn't improve for 4 epochs
    callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 4,
        min_lr   = 1e-7,
        verbose  = 1
    ),
    # Save only the best model automatically
    callbacks.ModelCheckpoint(
        'plant_model.h5',
        monitor        = 'val_accuracy',
        save_best_only = True,
        verbose        = 1
    )
]

# ── Train ─────────────────────────────────────────
# Backpropagation happens automatically inside fit()
history = model.fit(
    train_gen,
    epochs          = 30,
    validation_data = valid_gen,
    class_weight    = class_weights,   # fix imbalance
    callbacks       = cb_list,
    verbose         = 1
)

# ── Evaluate ──────────────────────────────────────
loss, acc = model.evaluate(valid_gen, verbose=0)
print(f"Validation Accuracy: {acc*100:.2f}%")

# ── Plot Training Graphs ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train', color='green')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='red')
axes[0].set_title('Accuracy'); axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train', color='green')
axes[1].plot(history.history['val_loss'], label='Val',   color='red')
axes[1].set_title('Loss'); axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# ── Confusion Matrix ──────────────────────────────
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

valid_gen.reset()
preds        = model.predict(valid_gen, verbose=1)
pred_classes = preds.argmax(axis=1)
true_classes = valid_gen.classes
labels       = ['Early Blight', 'Late Blight', 'Healthy']

cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print(classification_report(true_classes, pred_classes,
                             target_names=labels))
import numpy as np
import json
import tensorflow as tf
from PIL import Image

# ── Load saved model ──────────────────────────────
model = tf.keras.models.load_model('plant_model.h5')

# ── Load class name mapping ───────────────────────
with open('class_indices.json') as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

DISPLAY = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight' : 'Late Blight',
    'Potato___healthy'     : 'Healthy',
}

def preprocess(image_path):
    # Step 1: Open image
    img = Image.open(image_path).convert('RGB')  # force 3 channels
    # Step 2: Resize to match model input
    img = img.resize((224, 224))
    # Step 3: Convert to numpy array
    arr = np.array(img, dtype=np.float32)
    # Step 4: Normalize pixels 0-255 → 0-1
    arr = arr / 255.0
    # Step 5: Add batch dimension (224,224,3) → (1,224,224,3)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_disease(image_path):
    arr        = preprocess(image_path)
    # CNN forward pass → shape: (1, 3)
    preds      = model.predict(arr, verbose=0)[0]
    # Get predicted class index
    pred_idx   = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    class_name = idx_to_class[pred_idx]
    display    = DISPLAY[class_name]

    # All 3 class probabilities
    all_probs = {
        DISPLAY[idx_to_class[i]]: round(float(preds[i]) * 100, 2)
        for i in range(len(preds))
    }

    return {
        'class'     : class_name,
        'name'      : display,
        'confidence': round(confidence, 2),
        'all_probs' : all_probs,
        'healthy'   : 'healthy' in class_name.lower()
    }

# ── Usage ─────────────────────────────────────────
result = predict_disease('test_leaf.jpg')
print(f"Prediction : {result['name']}")
print(f"Confidence : {result['confidence']}%")
print(f"All probs  : {result['all_probs']}")
import os, json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
os.makedirs('static/uploads', exist_ok=True)

ALLOWED = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
CLASS_NAMES = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]
DISPLAY = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight' : 'Late Blight',
    'Potato___healthy'     : 'Healthy',
}
INFO = {
    'Potato___Early_blight': {
        'desc'    : 'Alternaria solani fungus. Dark brown rings on leaves.',
        'treat'   : 'Spray Chlorothalonil every 7 days. Remove infected leaves.',
        'severity': 'Medium'
    },
    'Potato___Late_blight': {
        'desc'    : 'Phytophthora infestans — spreads very fast.',
        'treat'   : 'Apply Mancozeb IMMEDIATELY. Destroy infected plants.',
        'severity': 'High'
    },
    'Potato___healthy': {
        'desc'    : 'No disease detected. Plant is healthy.',
        'treat'   : 'Continue regular watering and fertilization.',
        'severity': 'None'
    }
}

# Load model once at startup
print("Loading CNN model...")
model = tf.keras.models.load_model('plant_model.h5')
print("Model ready!")

def allowed(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED

def predict(path):
    img   = Image.open(path).convert('RGB')
    img   = img.resize((224, 224))
    arr   = np.array(img, dtype=np.float32) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(np.max(preds)) * 100
    name  = CLASS_NAMES[idx]
    info  = INFO[name]
    probs = {DISPLAY[CLASS_NAMES[i]]: round(float(preds[i])*100,2)
             for i in range(3)}
    return {
        'display_name': DISPLAY[name],
        'confidence'  : round(conf, 2),
        'desc'        : info['desc'],
        'treat'       : info['treat'],
        'severity'    : info['severity'],
        'all_probs'   : probs
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return render_template('index.html', error='No file!')
    file = request.files['file']
    if not allowed(file.filename):
        return render_template('index.html', error='JPG/PNG only!')
    fname     = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(save_path)
    result    = predict(save_path)
    image_url = f'/static/uploads/{fname}'
    return render_template('index.html',
                            result=result, image_url=image_url)

if __name__ == '__main__':
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True)
    <!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Potato Disease Detector</title>
  <link rel="stylesheet"
        href="{{ url_for('static', filename='style.css') }}"/>
</head>
<body>

<div class="header">
  <h1>🥔 Potato Disease Detector</h1>
  <p>CNN Deep Learning — 3 Classes</p>
</div>

<div class="main">
  <!-- Error message -->
  {% if error %}
  <div class="alert">⚠ {{ error }}</div>
  {% endif %}

  <!-- Upload Form -->
  <form action="/predict" method="POST"
        enctype="multipart/form-data">
    <div class="drop-zone"
         onclick="document.getElementById('fi').click()">
      <span>📷 Click or drag &amp; drop a leaf photo</span>
      <input type="file" id="fi" name="file"
             accept=".jpg,.jpeg,.png"
             onchange="previewImg(event)"/>
    </div>
    <img id="preview" style="display:none;max-height:200px"/>
    <button type="submit" class="btn">
      🔍 Detect Disease
    </button>
  </form>

  <!-- Result (shown after prediction) -->
  {% if result %}
  <div class="result-card">
    <h2>{{ result.display_name }}</h2>
    <p>Confidence: {{ result.confidence }}%</p>
    <p>Severity: {{ result.severity }}</p>
    <p>{{ result.desc }}</p>
    <p><b>Treatment:</b> {{ result.treat }}</p>
    {% if image_url %}
    <img src="{{ image_url }}" width="200"/>
    {% endif %}
    <a href="/">↩ Try Another</a>
  </div>
  {% endif %}
</div>

<script>
function previewImg(e) {
  const file = e.target.files[0];
  const preview = document.getElementById('preview');
  const reader = new FileReader();
  reader.onload = ev => {
    preview.src = ev.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
}
</script>
</body>
</html>
/* ── Reset ───────────────────────── */
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: Arial, sans-serif;
  background: #f5f0e8;
  color: #1a1208;
  min-height: 100vh;
}

/* ── Header ──────────────────────── */
.header {
  background: #1a1208;
  color: #f5f0e8;
  padding: 1.5rem;
  text-align: center;
}
.header h1 { font-size: 1.6rem; font-weight: 700; }
.header p  { font-size: .85rem; color: rgba(255,255,255,.5); margin-top: 4px; }

/* ── Main container ──────────────── */
.main {
  max-width: 560px;
  margin: 2rem auto;
  padding: 0 1rem 3rem;
}

/* ── Drop zone ───────────────────── */
.drop-zone {
  border: 2px dashed #c8bfa0;
  border-radius: 12px;
  padding: 2.5rem 1rem;
  text-align: center;
  cursor: pointer;
  background: #faf7f0;
  min-height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .2s;
}
.drop-zone:hover {
  border-color: #4a7c3f;
  background: #f0f7ee;
}
input[type=file] { display: none; }

/* ── Predict button ──────────────── */
.btn {
  width: 100%;
  padding: 13px;
  background: #2d5a1b;
  color: #fff;
  border: none;
  border-radius: 50px;
  font-size: .95rem;
  font-weight: 700;
  cursor: pointer;
  margin-top: 1rem;
  transition: background .2s;
}
.btn:hover { background: #3d7a28; }

/* ── Alert box ───────────────────── */
.alert {
  background: #fcebeb;
  border: 1px solid #f09595;
  color: #791f1f;
  border-radius: 8px;
  padding: 9px 13px;
  font-size: .83rem;
  margin-bottom: 1rem;
}

/* ── Result card ─────────────────── */
.result-card {
  background: #fff;
  border: 1px solid #ddd5bb;
  border-radius: 14px;
  padding: 1.4rem;
  margin-top: 1.2rem;
}
.result-card h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: .6rem;
}
.result-card p { font-size: .88rem; margin-bottom: .4rem; line-height: 1.6; }
.result-card a {
  display: block;
  margin-top: 1rem;
  text-align: center;
  color: #2d5a1b;
  font-weight: 600;
  text-decoration: none;
}
.result-card a:hover { text-decoration: underline; }

/* ── Responsive ──────────────────── */
@media (max-width: 480px) {
  .main { padding: 0 .75rem 2rem; }
  .header h1 { font-size: 1.3rem; }
} this is code for the project