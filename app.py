from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64

app = Flask(__name__)

# Cargar modelo, escalador, codificador y PCA
model = joblib.load('Entrega3/random_forest_model.pkl')
scaler = joblib.load('Entrega3/scaler.pkl')
pca = joblib.load('Entrega3/pca.pkl')
label_encoder = joblib.load('Entrega3/label_encoder.pkl')

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_angles(landmarks):
    angles = {}
    angles['right_elbow'] = calculate_angle(
        [landmarks[16].x, landmarks[16].y],
        [landmarks[14].x, landmarks[14].y],
        [landmarks[12].x, landmarks[12].y]
    )
    angles['right_knee'] = calculate_angle(
        [landmarks[24].x, landmarks[24].y],
        [landmarks[26].x, landmarks[26].y],
        [landmarks[28].x, landmarks[28].y]
    )
    angles['right_hip'] = calculate_angle(
        [landmarks[12].x, landmarks[12].y],
        [landmarks[24].x, landmarks[24].y],
        [landmarks[26].x, landmarks[26].y]
    )
    angles['left_knee'] = calculate_angle(
        [landmarks[23].x, landmarks[23].y],
        [landmarks[25].x, landmarks[25].y],
        [landmarks[27].x, landmarks[27].y]
    )
    angles['left_hip'] = calculate_angle(
        [landmarks[11].x, landmarks[11].y],
        [landmarks[23].x, landmarks[23].y],
        [landmarks[25].x, landmarks[25].y]
    )
    return angles

def extract_features(landmarks):
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    angles = extract_angles(landmarks)
    features.extend(list(angles.values()))
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Obtener el frame enviado desde el cliente
        data = request.json['frame']
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Procesar el frame con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = extract_features(landmarks)
            if len(features) == scaler.n_features_in_:
                scaled_features = scaler.transform([features])
                reduced_features = pca.transform(scaled_features)
                predicted_label = model.predict(reduced_features)[0]
                activity = label_encoder.inverse_transform([predicted_label])[0]
                return jsonify({'activity': activity})

        return jsonify({'activity': 'No se detectó actividad'})

    except Exception as e:
        print(f"Error procesando frame: {e}")
        return jsonify({'error': 'Error procesando el frame'})

if __name__ == "__main__":
    app.run(debug=True)
