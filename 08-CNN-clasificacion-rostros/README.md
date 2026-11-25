# Proyecto de Clasificación Facial con CNN

Sistema de reconocimiento facial en tiempo real utilizando una Red Neuronal Convolucional (CNN), desarrollado con Python, TensorFlow/Keras, OpenCV y MediaPipe.

## Estructura del Proyecto

* `captura_rostros.py`: Script para captura, alineación estricta en proporción 1:1 y filtrado de rostros en tiempo real.
* `dataset/`: Carpeta local donde se almacenan las imágenes por clase (ignorada en Git).
* `entrenamiento.ipynb`: Notebook de preprocesamiento, data augmentation, transfer learning y fine-tuning de la CNN.
* `reconocimiento.py`: Script de clasificación facial e inferencia en tiempo real desde la cámara web con optimización de FPS.

### Requisitos de Instalación

```bash
pip install opencv-python mediapipe numpy tensorflow matplotlib pillow
```

---

## 1. Captura de Rostros (`captura_rostros.py`)

El script de captura recolecta imágenes faciales alineadas horizontalmente según los ojos, en formato cuadrado estricto (1:1) sin incluir hombros ni bordes negros. Cuenta con un HUD en vivo y validación en tiempo real de:
* **Iluminación:** Brillo promedio, exceso de sombras (píxeles negros) y luz quemada.
* **Biometría y Geometría:** Distancia al rostro, parpadeo (EAR - Eye Aspect Ratio) y detección de rostros borrosos (Varianza del Laplaciano).
* **Pose (SolvePnP):** Validación de ángulos Yaw y Pitch.

### Instrucciones de Uso

1. Ejecutar el script desde la terminal:
```bash
python captura_rostros.py
```
2. Ingresar el nombre o ID del integrante. El script creará automáticamente la carpeta `dataset/<nombre_integrante>/` y descargará el modelo base de MediaPipe `face_landmarker.task` si no está presente.
3. Seguir las indicaciones en pantalla (HUD) para completar las capturas correspondientes a cada bloque de pose (*Frente*, *Leve Izquierda*, *Leve Derecha*, *Leve Arriba*, *Leve Abajo*). Se capturan x fotos válidas por bloque.

### Controles en Pantalla
* `n`: Saltar manualmente al siguiente bloque de pose.
* `q`: Salir del programa.

---

## 2. Preprocesamiento y Entrenamiento de la CNN (`entrenamiento.ipynb`)

El entrenamiento del modelo utiliza **Transfer Learning** basado en la arquitectura **MobileNetV2** (pre-entrenada con ImageNet), optimizada para dispositivos móviles y predicción en tiempo real. El dataset está compuesto por clases de famosos y un integrante, procesado de la siguiente manera:

* **Limpieza y Carga:** Verificación de integridad de las imágenes (eliminando archivos corruptos) y división 80/20 (Entrenamiento/Validación). Las clases detectadas se exportan a `clases.json`.
* **Data Augmentation:** Se aplica un pipeline de aumento de datos nativo de Keras que incluye volteo horizontal, rotación, zoom, ajustes de contraste y brillo, y traslación.
* **Arquitectura:**
  * Base: MobileNetV2 (congelada en la primera fase).
  * Cabeza de Clasificación: `GlobalAveragePooling2D` -> `BatchNormalization` -> `Dropout(0.2)` -> `Dense(6, softmax)`.
* **Fases de Entrenamiento:**
  1. **Fase 1 (Transfer Learning):** Entrenamiento de la cabeza de clasificación con optimizador Adam.
  2. **Fase 2 (Fine-Tuning):** Descongelamiento a partir del bloque 14 de MobileNetV2 (`block_14_expand`). Entrenamiento con una tasa de aprendizaje reducida (1e-5) y callbacks (`ReduceLROnPlateau`, `EarlyStopping`, `ModelCheckpoint`).
* El modelo final optimizado se guarda como `modelo_facial_finetuned.keras`.

---

## 3. Reconocimiento Facial en Tiempo Real (`reconocimiento.py`)

El script de inferencia en vivo carga el modelo pre-entrenado y detecta rostros de forma fluida.

### Características y Optimizaciones
* **Calentamiento del Modelo (Warm-up):** Se realiza una predicción inicial con tensores vacíos (Dummy Input) para cargar la gráfica computacional en memoria y evitar el lag en el primer frame detectado.
* **Optimización de Inferencia (Skip-Frames):** La clasificación CNN solo se ejecuta 1 de cada 5 frames (`PROCESS_EVERY_N_FRAMES = 5`). En los frames intermedios, se utiliza el caché del último resultado junto con el tracking ultra rápido de MediaPipe.
* **Alineación Dinámica:** Recorta y alinea el rostro detectado en vivo de forma idéntica a la fase de captura.
* **Umbral de Confianza:** Requiere al menos un 65% (`UMBRAL_CONFIANZA = 0.65`) de certeza en la predicción de la red neuronal. Predicciones por debajo de este umbral se etiquetan como "Desconocido".

### Instrucciones de Uso
1. Asegurarse de tener generados en la raíz del proyecto los archivos `modelo_facial_finetuned.keras`, `clases.json` y `face_landmarker.task`.
2. Ejecutar la inferencia:
```bash
python reconocimiento.py
```
3. Presionar `q` o `ESC` para salir.

---

## Archivos Ignorados en Git (`.gitignore`)

Reglas recomendadas para mantener el repositorio limpio de archivos binarios pesados y cachés de Python:

```gitignore
dataset/
**/dataset/

face_landmarker.task
**/face_landmarker.task

modelo_facial.keras
**/modelo_facial.keras

modelo_facial_finetuned.keras
**/modelo_facial_finetuned.keras

clases.json
**/clases.json
```