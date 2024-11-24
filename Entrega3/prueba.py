import cv2
import mediapipe as mp
import numpy as np
import joblib

# Cargar modelo, escalador, codificador y PCA (si aplica)
model = joblib.load('Entrega3/random_forest_model.pkl')
scaler = joblib.load('Entrega3/scaler.pkl')
pca = joblib.load('Entrega3/pca.pkl')  # Si usaste PCA
label_encoder = joblib.load('Entrega3/label_encoder.pkl')

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Función para calcular ángulos entre tres puntos
def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Limitar valores para evitar errores numéricos
    return np.degrees(angle)

# Función para extraer ángulos clave
def extract_angles(landmarks):
    angles = {}
    # Ángulo del codo derecho
    angles['right_elbow'] = calculate_angle(
        [landmarks[16].x, landmarks[16].y],  # Muñeca derecha
        [landmarks[14].x, landmarks[14].y],  # Codo derecho
        [landmarks[12].x, landmarks[12].y]   # Hombro derecho
    )

    # Ángulo de la rodilla derecha
    angles['right_knee'] = calculate_angle(
        [landmarks[24].x, landmarks[24].y],  # Cadera derecha
        [landmarks[26].x, landmarks[26].y],  # Rodilla derecha
        [landmarks[28].x, landmarks[28].y]   # Tobillo derecho
    )

    # Ángulo de la cadera derecha
    angles['right_hip'] = calculate_angle(
        [landmarks[12].x, landmarks[12].y],  # Hombro derecho
        [landmarks[24].x, landmarks[24].y],  # Cadera derecha
        [landmarks[26].x, landmarks[26].y]   # Rodilla derecha
    )

    # Ángulo de la rodilla izquierda
    angles['left_knee'] = calculate_angle(
        [landmarks[23].x, landmarks[23].y],  # Cadera izquierda
        [landmarks[25].x, landmarks[25].y],  # Rodilla izquierda
        [landmarks[27].x, landmarks[27].y]   # Tobillo izquierdo
    )

    # Ángulo de la cadera izquierda
    angles['left_hip'] = calculate_angle(
        [landmarks[11].x, landmarks[11].y],  # Hombro izquierdo
        [landmarks[23].x, landmarks[23].y],  # Cadera izquierda
        [landmarks[25].x, landmarks[25].y]   # Rodilla izquierda
    )

    return angles

# Función para extraer características de los landmarks
def extract_features(landmarks):
    features = []
    # Extraer coordenadas x, y, z
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])

    # Extraer ángulos clave
    angles = extract_angles(landmarks)
    features.extend(list(angles.values()))  # Agregar los ángulos al vector de características

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

        # Verificar si el número de características coincide con el escalador
        if len(features) != scaler.n_features_in_:
            print(f"Error: Número de características ({len(features)}) no coincide con el esperado ({scaler.n_features_in_}).")
            continue

        # Escalar características
        scaled_features = scaler.transform([features])

        # Reducir dimensiones con PCA (si aplica)
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