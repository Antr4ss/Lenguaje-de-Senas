import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. CARGAR DATOS
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
class_names = data_dict['class_names']

num_classes = len(class_names)

print(f"Total de muestras: {len(data)}")
print(f"Tamaño de entrada (puntos por mano): {len(data[0])}") # Debería ser 42 (21 puntos * 2 coords)

# 2. PREPARAR DATASET
# Convertir etiquetas a one-hot encoding
y_categorical = to_categorical(labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    data, y_categorical, test_size=0.2, shuffle=True, stratify=labels
)

# 3. CREAR MODELO
# Input shape es 42 (coord x e y para 21 puntos)
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(len(data[0]),)))
model.add(Dropout(0.2)) # Ayuda a evitar overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) # Salida = número de letras

# 4. COMPILAR Y ENTRENAR
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Iniciando entrenamiento...")
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

# 5. EVALUAR Y GUARDAR
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en test: {accuracy*100:.2f}%")

model.save('my_model.keras')
print("Modelo guardado como 'my_model.keras'")

# Guardar también los nombres de las clases para usarlos en la webcam
import json
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)