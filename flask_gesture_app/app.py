from flask import Flask, render_template, Response, jsonify, request
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import base64
import threading
import json

app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
model = None
le = None
gesture_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

def load_gesture_model():
    """Load the gesture recognition model"""
    global model, le
    try:
        if os.path.exists('hand_gesture_model.keras'):
            model = load_model('hand_gesture_model.keras', compile=False)
            le = LabelEncoder()
            le.classes_ = np.array(gesture_classes)
            print("Model loaded successfully")
            return True
        else:
            print("Model file not found")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_hand_connections(image):
    """Extract hand connections from an input image using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        connections = []
        for connection in mp_hands.HAND_CONNECTIONS:
            start = results.multi_hand_landmarks[0].landmark[connection[0]]
            end = results.multi_hand_landmarks[0].landmark[connection[1]]
            vector = [end.x - start.x, end.y - start.y, end.z - start.z]
            connections.extend(vector)
        return np.array(connections), results
    return None, None

def predict_gesture(frame, model, le):
    """Predict the hand gesture from the given frame using the trained model."""
    if model is None:
        return None, None
    
    connections, results = extract_hand_connections(frame)
    if connections is not None:
        connections = connections.reshape(1, -1)
        prediction = model.predict(connections)
        gesture_index = np.argmax(prediction)
        gesture_label = le.inverse_transform([gesture_index])[0]
        return gesture_label, results
    return None, None

@app.route('/')
def index():
    """Main page"""
    model_status = "loaded" if model is not None else "not found"
    return render_template('index.html', model_status=model_status)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload for gesture recognition"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Predict gesture
        gesture, results = predict_gesture(image, model, le)
        
        if gesture:
            return jsonify({'gesture': gesture, 'status': 'success'})
        else:
            return jsonify({'gesture': None, 'status': 'no_hand_detected'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mediapipe_ready': hands is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_gesture_model()
    
    # Get port from environment variable (required for Render)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)