import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model

# 1. CARGAR EL MODELO Y LAS CLASES
model = load_model('modelo_senas_estatico.h5')

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# 2. CONFIGURAR MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 3. SELECCIONAR CÁMARA
def listar_camaras():
    print("Buscando cámaras disponibles...")
    camaras = []
    # Verificar las primeras 10 cámaras
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camaras.append(i)
            cap.release()
    return camaras

camaras_disponibles = listar_camaras()

if not camaras_disponibles:
    print("No se encontraron cámaras.")
    exit()

print("\nCámaras disponibles:")
for i, cam_idx in enumerate(camaras_disponibles):
    print(f"{i}: Cámara {cam_idx}")

seleccion = input("\nSelecciona el número de la cámara (0, 1, ...): ")
try:
    indice_seleccionado = int(seleccion)
    if indice_seleccionado < 0 or indice_seleccionado >= len(camaras_disponibles):
        print("Selección inválida. Usando la primera cámara encontrada.")
        camera_id = camaras_disponibles[0]
    else:
        camera_id = camaras_disponibles[indice_seleccionado]
except ValueError:
    print("Entrada no válida. Usando la primera cámara encontrada.")
    camera_id = camaras_disponibles[0]

print(f"Iniciando cámara {camera_id}...")
cap = cv2.VideoCapture(camera_id)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Espejo para que sea más natural
    frame = cv2.flip(frame, 1)
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar el esqueleto de la mano
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # --- PREPROCESAMIENTO (Igual que en el entrenamiento) ---
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            min_x, min_y = min(x_), min(y_)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)
            # -------------------------------------------------------

            # PREDICCIÓN
            prediction = model.predict(np.array([data_aux]))
            predicted_index = np.argmax(prediction)
            predicted_character = class_names[predicted_index]
            confidence = np.max(prediction)

            # Mostrar resultado si la confianza es alta (> 70%)
            if confidence > 0.7:
                cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
                cv2.putText(frame, f"Letra: {predicted_character} ({int(confidence*100)}%)", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Reconocimiento de Señas', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()