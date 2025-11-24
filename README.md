# Reconocimiento de Lenguaje de Señas

Este proyecto implementa un sistema de reconocimiento de lenguaje de señas estático utilizando **MediaPipe** para la detección de manos y **TensorFlow/Keras** para la clasificación de gestos.

## Descripción

El sistema funciona en tres etapas principales:
1.  **Creación de Datos**: Procesa imágenes de un dataset (como ASL Alphabet) para extraer los puntos clave (landmarks) de las manos.
2.  **Entrenamiento**: Entrena una red neuronal simple para clasificar los gestos basándose en los puntos clave.
3.  **Prueba en Tiempo Real**: Utiliza la cámara web para detectar manos y predecir la letra en tiempo real.

## Requisitos

Asegúrate de tener instalado Python y las siguientes librerías:

```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn
```

## Estructura del Proyecto

- `1_crear_datos.py`: Script para procesar las imágenes y crear el dataset de entrenamiento (`data.pickle`).
- `2_entrenar_datos.py`: Script para entrenar el modelo y guardarlo (`my_model.keras` / `modelo_senas_estatico.h5`).
- `3_probar_camara.py`: Script principal para probar el reconocimiento con la cámara web.
- `asl_alphabet_train/`: Carpeta que debe contener las subcarpetas con las imágenes de entrenamiento (A, B, C...).
- `data.pickle`: Archivo generado que contiene los datos procesados.
- `class_names.json`: Archivo que guarda los nombres de las clases (letras).

## Uso

### 1. Crear el Dataset
Asegúrate de tener tus imágenes en la carpeta `asl_alphabet_train`. Ejecuta:

```bash
python 1_crear_datos.py
```
Esto generará el archivo `data.pickle`.

### 2. Entrenar el Modelo
Entrena la red neuronal con los datos procesados:

```bash
python 2_entrenar_datos.py
```
Esto generará el modelo entrenado y el archivo `class_names.json`.
> **Nota:** El script de entrenamiento guarda el modelo como `my_model.keras`. El script de prueba busca `modelo_senas_estatico.h5`. Asegúrate de renombrar el archivo o actualizar el código según corresponda.

### 3. Probar con la Cámara
Ejecuta el script de prueba:

```bash
python 3_probar_camara.py
```
- El sistema buscará cámaras disponibles y te pedirá seleccionar una.
- Se abrirá una ventana mostrando la detección en tiempo real.
- Presiona `q` para salir.

## Créditos
Basado en el uso de MediaPipe Hands y TensorFlow.
