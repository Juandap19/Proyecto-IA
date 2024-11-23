import cv2
import mediapipe as mp
import numpy as np
import joblib

# Cargar modelo, escalador, codificador y PCA
model = joblib.load('Entrega3/random_forest_model.pkl')
scaler = joblib.load('Entrega3/scaler.pkl')
pca = joblib.load('Entrega3/pca.pkl')
label_encoder = joblib.load('Entrega3/label_encoder.pkl')

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Función para extraer características de los landmarks
def extract_features(landmarks):
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features

# Configurar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Dibujar landmarks en el frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extraer características de los landmarks
        features = extract_features(landmarks)

        # Validar tamaño de las características
        if len(features) != scaler.n_features_in_:
            print(f"Error: Número de características ({len(features)}) no coincide con el esperado ({scaler.n_features_in_}).")
            continue

        # Escalar características
        scaled_features = scaler.transform([features])

        # Reducir dimensiones con PCA
        reduced_features = pca.transform(scaled_features)

        # Predecir actividad
        predicted_label = model.predict(reduced_features)[0]
        activity = label_encoder.inverse_transform([predicted_label])[0]

        # Mostrar actividad en el frame
        cv2.putText(frame, f'Actividad: {activity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame procesado
    cv2.imshow('Clasificación en Tiempo Real', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()