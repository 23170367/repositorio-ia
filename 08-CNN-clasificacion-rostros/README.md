# Proyecto de Clasificación Facial con CNN

Sistema de reconocimiento facial en tiempo real utilizando una Red Neuronal Convolucional (CNN), desarrollado con Python, OpenCV y MediaPipe.

## Estructura del Proyecto

* `captura_rostros.py`: Script para captura, alineación en proporción 1:1 y filtrado de rostros en tiempo real.
* `dataset/`: Carpeta local donde se almacenan las imágenes por clase (ignorada en Git).
* `entrenamiento.py`: Script de preprocesamiento, data augmentation y entrenamiento de la CNN *(en desarrollo)*.
* `reconocimiento.py`: Script de clasificación facial e inferencia en tiempo real desde la cámara web *(en desarrollo)*.

---

## 1. Captura de Rostros (`captura_rostros.py`)

El script de captura recolecta imágenes faciales alineadas horizontalmente según los ojos, en formato cuadrado (1:1) y con validación en tiempo real de brillo, nitidez, parpadeo y poses.

### Requisitos de Instalación

```bash
pip install opencv-python mediapipe numpy
```

### Instrucciones de Uso

1. Ejecutar el script desde la terminal:

```bash
python captura_rostros.py
```

2. Ingresar el nombre o ID del integrante cuando la consola lo solicite. El script creará automáticamente la carpeta `dataset/<nombre_integrante>/` y descargará `face_landmarker.task` si no está presente.

3. Seguir las indicaciones en pantalla para completar las capturas correspondientes a cada bloque de pose (*Frente*, *Leve Izquierda*, *Leve Derecha*, *Leve Arriba*, *Leve Abajo*).

### Controles en Pantalla

* `n`: Saltar manualmente al siguiente bloque de pose.
* `q`: Salir del programa.

---

## 2. Preprocesamiento y Entrenamiento de la CNN

*(Sección en desarrollo: documentará la carga de clases de alumnos y famosos, la división 80/20, el escalado/normalización, el data augmentation y los callbacks de entrenamiento).*

---

## 3. Reconocimiento Facial en Tiempo Real

*(Sección en desarrollo: documentará la ejecución de la prueba en vivo y las métricas de evaluación del modelo).*

---

## Archivos Ignorados en Git (`.gitignore`)

Reglas recomendadas para mantener el repositorio libre de binarios y datasets pesados:

```gitignore
dataset/
**/dataset/
face_landmarker.task
**/face_landmarker.task
```