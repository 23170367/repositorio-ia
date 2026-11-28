# Proyecto de Clasificación Facial con CNN

Sistema de reconocimiento facial en tiempo real utilizando una Red Neuronal Convolucional (CNN), desarrollado con Python, TensorFlow/Keras, OpenCV y MediaPipe.

## Estructura del Proyecto

- `captura_rostros.py`: Script para captura, alineación estricta en proporción 1:1 y filtrado de rostros en tiempo real.
- `dataset/`: Carpeta local donde se almacenan las imágenes por clase (ignorada en Git).
- `entrenamiento.ipynb`: Notebook de preprocesamiento, data augmentation y transfer learning de la CNN.
- `reconocimiento.py`: Script de clasificación facial e inferencia en tiempo real desde la cámara web con optimización de FPS y seguimiento espacial.

### Requisitos de Instalación

```bash
pip install opencv-python mediapipe numpy tensorflow matplotlib
```

---

## 1. Captura de Rostros (`captura_rostros.py`)

El script de captura recolecta imágenes faciales alineadas horizontalmente según los ojos, en formato cuadrado estricto (1:1) sin incluir hombros ni bordes negros. Cuenta con un HUD en vivo y validación en tiempo real de:
- **Iluminación:** Brillo promedio, exceso de sombras (píxeles negros) y luz quemada.
- **Biometría y Geometría:** Distancia al rostro, parpadeo (EAR - Eye Aspect Ratio) y detección de rostros borrosos (Varianza del Laplaciano).
- **Pose (SolvePnP):** Validación de ángulos Yaw y Pitch.

### Instrucciones de Uso

1. Ejecutar el script desde la terminal:
```bash
python captura_rostros.py
```
2. Ingresar el nombre o ID del integrante. El script creará automáticamente la carpeta `dataset/<nombre_integrante>/` y descargará el modelo base de MediaPipe `face_landmarker.task` si no está presente.
3. Seguir las indicaciones en pantalla (HUD) para completar las capturas correspondientes a cada bloque de pose (*Frente*, *Leve Izquierda*, *Leve Derecha*, *Leve Arriba*, *Leve Abajo*).

### Controles en Pantalla
- `n`: Saltar manualmente al siguiente bloque de pose.
- `q`: Salir del programa.

---

## 2. Preprocesamiento y Entrenamiento de la CNN (`entrenamiento.ipynb`)

El entrenamiento del modelo utiliza **Transfer Learning** basado en la arquitectura **MobileNetV2** (pre-entrenada con ImageNet), optimizada para dispositivos móviles y predicción en tiempo real. 

- **Carga de Datos:** División automática 80/20 (Entrenamiento/Validación) manteniendo la relación de aspecto. Las clases detectadas se exportan directamente a `clases.json`.
- **Data Augmentation:** Se aplica un pipeline de aumento de datos nativo de Keras que incluye volteo horizontal, rotación, zoom, ajustes de contraste/brillo y traslación.
- **Arquitectura y Entrenamiento:**
  - Base: MobileNetV2 (congelada pasando `training=False` para evitar la degradación de las capas *Batch Normalization*).
  - Cabeza de Clasificación: `GlobalAveragePooling2D` -> `BatchNormalization` -> `Dropout(0.2)` -> `Dense(softmax)`.
- El modelo se compila con el optimizador Adam y utiliza *EarlyStopping* y *ModelCheckpoint* para evitar sobreajuste, guardando automáticamente la mejor versión iteración como `modelo_facial.keras`.

---

## 3. Reconocimiento Facial en Tiempo Real (`reconocimiento.py`)

El script de inferencia en vivo carga el modelo entrenado y detecta rostros de forma fluida, integrando algoritmos de seguimiento y optimizaciones de hardware.

### Características y Optimizaciones
- **Centroid Tracking (Seguimiento Espacial):** Soluciona el problema de intercambio de nombres cuando hay múltiples personas en pantalla. Mide la distancia matemática entre los centros de los rostros frame a frame para asociar correctamente cada identidad mientras se mueven.
- **Optimización de Inferencia (Skip-Frames):** La clasificación pesada de la CNN solo se ejecuta 1 de cada 5 frames. En los frames intermedios, se utiliza el caché del tracking de MediaPipe, reduciendo el uso del procesador.
- **Ejecución Eager y Reducción de Resolución:** La cámara se inicializa a 640x480 para agilizar las operaciones de rotación de OpenCV, y la CNN se ejecuta de forma directa (sin batching) para eliminar el *lag* de TensorFlow.
- **Alineación Dinámica:** Recorta y alinea el rostro detectado en vivo de forma idéntica a la fase de captura, utilizando `BORDER_REPLICATE` para evitar esquinas negras falsas.
- **Umbral de Confianza:** Requiere al menos un 65% de certeza en la predicción. Valores menores se etiquetan como "Desconocido".

### Instrucciones de Uso
1. Asegurarse de tener generados en la raíz del proyecto los archivos `modelo_facial.keras`, `clases.json` y `face_landmarker.task`.
2. Ejecutar la inferencia:
```bash
python reconocimiento.py
```
3. Presionar `q` o `ESC` para salir.

---

## Archivos Ignorados en Git (`.gitignore`)

Reglas recomendadas para mantener el repositorio limpio de archivos binarios pesados y cachés:

```gitignore
dataset/
**/dataset/

face_landmarker.task
**/face_landmarker.task

modelo_facial.keras
**/modelo_facial.keras

clases.json
**/clases.json
```