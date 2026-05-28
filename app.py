import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs('static/uploads', exist_ok=True)

ALLOWED = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

CLASS_NAMES = [
    'Early_blight',
    'Late_blight',
    'Healthy'
]

DISPLAY = {
    'Early_blight': 'Early Blight',
    'Late_blight': 'Late Blight',
    'Healthy': 'Healthy',
}

INFO = {
    'Early_blight': {
        'desc': 'Fungal disease causing dark brown spots on leaves.',
        'treat': 'Apply fungicide and remove infected leaves.',
        'severity': 'Medium'
    },

    'Late_blight': {
        'desc': 'Rapidly spreading disease affecting plant leaves.',
        'treat': 'Apply fungicide immediately and isolate infected plants.',
        'severity': 'High'
    },

    'Healthy': {
        'desc': 'No disease detected. Plant looks healthy.',
        'treat': 'Continue proper watering and care.',
        'severity': 'None'
    }
}

print("Loading CNN model...")

print("Model ready!")

def allowed(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED

def predict(path):

    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100

    name = CLASS_NAMES[idx]
    info = INFO[name]

    probs = {
        DISPLAY[CLASS_NAMES[i]]: round(float(preds[i]) * 100, 2)
        for i in range(3)
    }

    return {
        'display_name': DISPLAY[name],
        'confidence': round(conf, 2),
        'desc': info['desc'],
        'treat': info['treat'],
        'severity': info['severity'],
        'all_probs': probs
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():

    if 'file' not in request.files:
        return render_template('index.html',
                               error='No file uploaded!')

    file = request.files['file']

    if not allowed(file.filename):
        return render_template('index.html',
                               error='Only JPG/PNG images allowed!')

    fname = secure_filename(file.filename)

    save_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        fname
    )

    file.save(save_path)

    result = predict(save_path)

    image_url = f'/static/uploads/{fname}'

    return render_template(
        'index.html',
        result=result,
        image_url=image_url
    )

if __name__ == '__main__':
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True)
