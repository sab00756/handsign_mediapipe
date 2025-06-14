from flask import Flask, render_template, Response, jsonify
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

# Load the trained model
model = load_model('hand_gesture_model.keras' ,compile=False)

# Define gesture classes
gesture_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
le = LabelEncoder()
le.classes_ = np.array(gesture_classes)

# Global variables for gesture tracking
current_gesture = None
gesture_start_time = 0
gesture_duration = 1.5  # seconds
camera = None

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.current_gesture = None
        self.gesture_start_time = 0
        
    def __del__(self):
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Predict the gesture
        gesture, results = predict_gesture(frame, model, le)
        
        # Check for gesture change
        if gesture is not None:
            if self.current_gesture != gesture:
                if time.time() - self.gesture_start_time >= gesture_duration:
                    self.current_gesture = gesture
                    self.gesture_start_time = time.time()
            else:
                self.gesture_start_time = time.time()
        
        # Display the gesture
        if self.current_gesture:
            cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        return frame_bytes, self.current_gesture

def extract_hand_connections(image):
    """
    Extract hand connections (landmark vectors) from an input image using MediaPipe.
    """
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
    connections, results = extract_hand_connections(frame)
    if connections is not None:
        connections = connections.reshape(1, -1)  # Reshape for model input
        prediction = model.predict(connections)  # Predict probabilities
        gesture_index = np.argmax(prediction)  # Get index of highest probability
        gesture_label = le.inverse_transform([gesture_index])[0]  # Convert to label
        return gesture_label, results
    return None, None

def generate_frames():
    """Generator function for video streaming"""
    global camera
    if camera is None:
        camera = Camera()
    
    while True:
        try:
            frame_data = camera.get_frame()
            if frame_data is None:
                break
                
            frame_bytes, gesture = frame_data
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_gesture')
def get_current_gesture():
    """API endpoint to get current gesture"""
    global camera
    if camera and camera.current_gesture:
        return jsonify({'gesture': camera.current_gesture})
    return jsonify({'gesture': None})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera"""
    global camera
    if camera:
        del camera
        camera = None
    return jsonify({'status': 'Camera stopped'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 1000))  # Use PORT env from Render
    try:
        app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if camera:
            del camera