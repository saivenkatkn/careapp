from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
import tensorflow as tf
import logging
from flask_cors import CORS

# Get the absolute path of the directory where this script is located
base_dir = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder relative to the script's location
UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- THIS IS THE CORRECTED PATH ---
# Load pre-trained model relative to the script's location
MODEL_PATH = os.path.join(base_dir, 'precare.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match training order!)
classes = ['COPD', 'Bronchiolitis', 'Pneumonia', 'URTI', 'Healthy']

precautions_data = {
    "COPD": "Avoid smoking, avoid pollution, continue prescribed inhalers.",
    "Bronchiolitis": "Rest, hydration, use humidifier, seek care if worsening.",
    "Pneumonia": "Medical evaluation needed. Rest and stay hydrated.",
    "URTI": "Rest, steam inhalation, warm fluids, avoid exposure to cold.",
    "Healthy": "Maintain a healthy lifestyle and regular checkups."
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path, n_mfcc=64):
    """Extract MFCC features in same shape as training."""
    # [cite_start]Your model's config.json shows it expects (None, 64, 1) [cite: 1988, 1990]
    # This function correctly returns that shape.
    audio, sr = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc.reshape(1, n_mfcc, 1)  # (1, 64, 1)

@app.route('/')
def index():
    """Serve the HTML file for the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Please upload a .wav file only.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    try:
        file.save(file_path)

        features = extract_features(file_path)
        prediction = model.predict(features)[0]

        class_index = np.argmax(prediction)
        predicted_class = classes[class_index]
        confidence = float(prediction[class_index]) * 100

        precautions = precautions_data.get(predicted_class, "No precautions available.")

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'precautions': precautions,
            'all_probabilities': {
                classes[i]: f"{float(prediction[i]) * 100:.2f}%" 
                for i in range(len(classes))
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error.', 'details': str(e)}), 500
    
    finally:
        # --- ADDED THIS FIX AGAIN ---
        # This ensures the uploaded file is deleted to prevent disk quota errors.
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Successfully removed uploaded file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {str(e)}")
