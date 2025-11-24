import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# ------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------
DATA_DIR = './asl_alphabet_train'  # La carpeta donde tienes tus carpetas de fotos (A, B, C...)
OUTPUT_FILE = 'data.pickle'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

# Obtener lista de clases (nombres de las carpetas)
classes = os.listdir(DATA_DIR)
print(f"Clases detectadas: {classes}")

# ------------------------------------------
# PROCESAMIENTO DE IMÁGENES
# ------------------------------------------
for idx, label in enumerate(classes):
    path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(path): continue
    
    print(f"Procesando clase: {label}...")
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detección de mano con MediaPipe
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer coordenadas (x, y) de los 21 puntos
                data_aux = []
                
                # Normalización simple (opcional, pero recomendada)
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                
                min_x, min_y = min(x_), min(y_)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Guardamos la coordenada relativa al punto más bajo para centrar la mano
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)
                
                data.append(data_aux)
                labels.append(idx) # Guardamos el índice (0, 1, 2...) en lugar del nombre

# Guardar todo en un archivo pickle
f = open(OUTPUT_FILE, 'wb')
pickle.dump({'data': data, 'labels': labels, 'class_names': classes}, f)
f.close()

print(f"¡Proceso terminado! Datos guardados en {OUTPUT_FILE}")