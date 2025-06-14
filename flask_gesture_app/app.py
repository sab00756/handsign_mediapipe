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
import io
from PIL import Image

app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model (with error handling)
model = None
try:
    model = load_model('hand_gesture_model.keras', compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define gesture classes
gesture_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
le = LabelEncoder()
le.classes_ = np.array(gesture_classes)

# Global variables for gesture tracking
current_gesture = None
gesture_start_time = 0
gesture_duration = 1.5  # seconds
last_processed_frame = None

def extract_hand_connections(image):
    """
    Extract hand connections (landmark vectors) from an input image using MediaPipe.
    """
    if image is None:
        return None, None
        
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
    """
    Predict the hand gesture from the given frame using the trained model.
    """
    if model is None:
        return None, None
        
    connections, results = extract_hand_connections(frame)
    if connections is not None:
        connections = connections.reshape(1, -1)  # Reshape for model input
        prediction = model.predict(connections)  # Predict probabilities
        gesture_index = np.argmax(prediction)  # Get index of highest probability
        gesture_label = le.inverse_transform([gesture_index])[0]  # Convert to label
        return gesture_label, results
    return None, None

def process_frame_with_landmarks(frame):
    """
    Process frame and return image with landmarks drawn
    """
    global current_gesture, gesture_start_time
    
    # Predict the gesture
    gesture, results = predict_gesture(frame, model, le)
    
    # Check for gesture change
    if gesture is not None:
        if current_gesture != gesture:
            if time.time() - gesture_start_time >= gesture_duration:
                current_gesture = gesture
                gesture_start_time = time.time()
        else:
            gesture_start_time = time.time()
    
    # Draw the hand landmarks
    if results and results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
    
    # Display the gesture
    if current_gesture:
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame sent from the browser"""
    global last_processed_frame
    
    try:
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process the frame
        processed_frame = process_frame_with_landmarks(opencv_image)
        
        # Convert back to base64 for sending to browser
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Store for the processed video feed
        last_processed_frame = processed_base64
        
        return jsonify({
            'processed_image': f"data:image/jpeg;base64,{processed_base64}",
            'gesture': current_gesture,
            'success': True
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/current_gesture')
def get_current_gesture():
    """API endpoint to get current gesture"""
    return jsonify({
        'gesture': current_gesture,
        'model_loaded': model is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK', 
        'message': 'Flask app is running',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    try:
        app.run(debug=False, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        print("Server stopped")