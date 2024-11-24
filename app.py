from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib

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

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            features = extract_features(landmarks)
            if len(features) == scaler.n_features_in_:
                scaled_features = scaler.transform([features])
                reduced_features = pca.transform(scaled_features)
                predicted_label = model.predict(reduced_features)[0]
                activity = label_encoder.inverse_transform([predicted_label])[0]
                cv2.putText(frame, f'Actividad: {activity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/activity', methods=['GET'])
def get_activity():
    return jsonify({"message": "Endpoint not yet implemented."})

if __name__ == "__main__":
    app.run(debug=True)
