"""
Script de Captura y Alineación de Rostros para Dataset CNN (1:1 Cuadrado - Solo Rostro)
Desarrollado con OpenCV y MediaPipe Tasks API (Face Landmarker).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

# ==============================================================================
# 🛠️ VARIABLES CONFIGURABLES GLOBALES
# ==============================================================================

# 📷 Configuración de Cámara
CAMERA_INDEX = 0                  # Índice de la cámara web
RESOLUTION_W = 1280               # Resolución objetivo Ancho
RESOLUTION_H = 720                # Resolución objetivo Alto

# 📁 Configuración de Almacenamiento
DATASET_BASE_DIR = "08-CNN-clasificacion-rostros/dataset"      # Carpeta raíz del dataset
MODEL_FILE = "08-CNN-clasificacion-rostros/face_landmarker.task" # Archivo del modelo de MediaPipe
CUOTA_POR_BLOQUE = 20             # Cuota de imágenes válidas a capturar por bloque de pose
MIN_INTERVAL_SEC = 0.3            # Intervalo mínimo en segundos entre capturas exitosas

# 📐 Filtros de Calidad y Geometría
MIN_FACE_WIDTH_PERCENT = 0.20     # El ancho del rostro debe ocupar mínimo el 20% del frame
# Padding reducido al 15% para encuadrar estrictamente la cara sin agarrar hombros
PADDING_PERCENT = 0.15            

# 💡 Filtros de Iluminación
MIN_AVG_BRIGHTNESS = 60           # Brillo promedio mínimo (escala de grises 0-255)
MAX_AVG_BRIGHTNESS = 200          # Brillo promedio máximo
MAX_PURE_BLACK_PERCENT = 5.0      # Máximo % permitido de píxeles <= 5 (negros)
MAX_PURE_WHITE_PERCENT = 5.0      # Máximo % permitido de píxeles >= 250 (quemados/saturados)

# 👁️ Filtros Biomédicos y Ópticos
EAR_THRESHOLD = 0.22              # Umbral de Eye Aspect Ratio (menor = cerrado/parpadeo)
LAPLACIAN_VAR_THRESH = 60.0       # Umbral de nitidez (varianza del Laplaciano). Ajustar según la cámara.

# 🔄 Bloques de Pose (Yaw y Pitch permitidos por etapa)
POSE_BLOCKS = [
    {"name": "Frente",         "yaw": (-15, 15), "pitch": (-15, 15), "count": 0, "target": CUOTA_POR_BLOQUE},
    {"name": "Leve Izquierda", "yaw": (-45, -20), "pitch": (-15, 15), "count": 0, "target": CUOTA_POR_BLOQUE},
    {"name": "Leve Derecha",   "yaw": (20, 45),  "pitch": (-15, 15), "count": 0, "target": CUOTA_POR_BLOQUE},
    {"name": "Leve Arriba",    "yaw": (-25, 25), "pitch": (15, 45),  "count": 0, "target": CUOTA_POR_BLOQUE},
    {"name": "Leve Abajo",     "yaw": (-25, 25), "pitch": (-45, -15), "count": 0, "target": CUOTA_POR_BLOQUE},
]

# Índices clave de MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Puntos para resolver PnP (3D Model genérico para OpenCV)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],             # Nariz
    [0.0, 150.0, -50.0],         # Mentón
    [-70.0, -60.0, -40.0],       # Esquina ojo izquierdo
    [70.0, -60.0, -40.0],        # Esquina ojo derecho
    [-45.0, 70.0, -30.0],        # Comisura boca izquierda
    [45.0, 70.0, -30.0]          # Comisura boca derecha
], dtype=np.float64)
PNP_2D_INDICES = [1, 152, 33, 263, 61, 291]

# ==============================================================================
# 🧠 FUNCIONES AUXILIARES
# ==============================================================================

def download_model_if_missing():
    """Descarga el modelo face_landmarker.task de los servidores de Google si no existe."""
    if not os.path.exists(MODEL_FILE):
        print(f"📥 Descargando el modelo de MediaPipe ({MODEL_FILE}) esto tomará unos segundos...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_FILE)
        print("✅ Descarga completada.")

def calculate_ear(eye_points):
    """Calcula el Eye Aspect Ratio (EAR) para detectar parpadeo."""
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
    return ear

def estimate_pose(image_w, image_h, landmarks_2d):
    """Estima Pitch, Yaw y Roll del rostro usando solvePnP."""
    focal_length = image_w
    cam_matrix = np.array([
        [focal_length, 0, image_w / 2],
        [0, focal_length, image_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, landmarks_2d, cam_matrix, dist_matrix)
    if not success:
        return 0, 0, 0
        
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    return pitch, yaw, roll

def get_brightness_stats(gray_img):
    """Retorna promedio de brillo, % píxeles negros y % píxeles blancos."""
    mean_val = np.mean(gray_img)
    total_pixels = gray_img.size
    black_pct = np.sum(gray_img <= 5) / total_pixels * 100.0
    white_pct = np.sum(gray_img >= 250) / total_pixels * 100.0
    return mean_val, black_pct, white_pct

def align_and_crop(image, landmarks_list, pad_pct):
    """Alinea el rostro y realiza un recorte CUADRADO 1:1 estricto al rostro, sin hombros ni bordes negros."""
    h, w, _ = image.shape
    
    left_eye_pts = np.array([landmarks_list[i] for i in LEFT_EYE])
    right_eye_pts = np.array([landmarks_list[i] for i in RIGHT_EYE])
    
    left_center = left_eye_pts.mean(axis=0)
    right_center = right_eye_pts.mean(axis=0)
    
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Casting a float nativo para evitar errores de tipo en NumPy/OpenCV
    face_center_raw = np.array(landmarks_list).mean(axis=0)
    face_center = (float(face_center_raw[0]), float(face_center_raw[1]))
    
    M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    ones = np.ones((len(landmarks_list), 1))
    pts3d = np.hstack([landmarks_list, ones])
    transformed_pts = M.dot(pts3d.T).T
    
    min_x = int(np.min(transformed_pts[:, 0]))
    max_x = int(np.max(transformed_pts[:, 0]))
    min_y = int(np.min(transformed_pts[:, 1]))
    max_y = int(np.max(transformed_pts[:, 1]))
    
    face_w = max_x - min_x
    face_h = max_y - min_y
    
    # 2. Calcular recorte cuadrado 1:1 tomando el lado mayor (generalmente el alto del rostro)
    base_side = max(face_w, face_h)
    cx = int((min_x + max_x) / 2)
    
    # Ajuste CLAVE: Desplazar el centro un 5% de la altura hacia arriba.
    # Esto asegura que el cuadro inferior se corte cerca del mentón y no llegue a los hombros/cuello.
    cy = int((min_y + max_y) / 2) - int(face_h * 0.05)
    
    # Aplicar padding simétrico (reducido) desde este nuevo centro
    half_side = int((base_side / 2.0) * (1.0 + pad_pct))
    
    start_x = cx - half_side
    end_x = cx + half_side
    start_y = cy - half_side
    end_y = cy + half_side
    
    # 3. Rechazar si el recorte sale de los límites (evita marcos negros virtualmente creados por el padding)
    if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
        return None  
        
    return rotated_img[start_y:end_y, start_x:end_x]

# ==============================================================================
# 🚀 FUNCIÓN PRINCIPAL DEL PIPELINE
# ==============================================================================

def main():
    download_model_if_missing()

    nombre = input("▶ Ingrese el nombre/ID del integrante: ").strip().replace(" ", "_")
    if not nombre:
        print("⚠️ Nombre inválido. Saliendo...")
        return
        
    save_dir = os.path.join(DATASET_BASE_DIR, nombre)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"✅ Guardando en: {save_dir}")
    print("▶ Presione 'q' para salir, 'n' para saltar al siguiente bloque de pose.")

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2, 
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)

    current_block_idx = 0
    last_save_time = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error capturando video.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)
        display_frame = frame.copy()
        
        # Variables por defecto para el HUD
        pitch, yaw = 0.0, 0.0 
        
        if current_block_idx >= len(POSE_BLOCKS):
            cv2.putText(display_frame, "DATASET COMPLETADO!", (w//2 - 200, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Captura CNN", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        block = POSE_BLOCKS[current_block_idx]
        reason = ""
        passed = False
        
        if not detection_result.face_landmarks:
            reason = "Sin rostro"
        elif len(detection_result.face_landmarks) > 1:
            reason = "Mas de un rostro"
        else:
            face_landmarks = detection_result.face_landmarks[0]
            landmarks_px = []
            for lm in face_landmarks:
                landmarks_px.append([int(lm.x * w), int(lm.y * h)])
            landmarks_px = np.array(landmarks_px)
            
            min_x, min_y = np.min(landmarks_px, axis=0)
            max_x, max_y = np.max(landmarks_px, axis=0)
            face_width = max_x - min_x
            
            pnp_2d = np.array([landmarks_px[idx] for idx in PNP_2D_INDICES], dtype=np.float64)
            
            left_eye_pts = np.array([landmarks_px[i] for i in LEFT_EYE])
            right_eye_pts = np.array([landmarks_px[i] for i in RIGHT_EYE])
            ear_left = calculate_ear(left_eye_pts)
            ear_right = calculate_ear(right_eye_pts)
            ear_max = max(ear_left, ear_right) 
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_b, black_p, white_p = get_brightness_stats(gray)
            
            pitch, yaw, _ = estimate_pose(w, h, pnp_2d)

            if min_x <= 0 or min_y <= 0 or max_x >= w-1 or max_y >= h-1:
                reason = "Rostro toca los bordes"
            elif (face_width / w) < MIN_FACE_WIDTH_PERCENT:
                reason = "Acercate mas"
            elif mean_b < MIN_AVG_BRIGHTNESS:
                reason = "Iluminacion muy oscura"
            elif mean_b > MAX_AVG_BRIGHTNESS:
                reason = "Iluminacion muy brillante"
            elif black_p > MAX_PURE_BLACK_PERCENT:
                reason = "Exceso de sombras"
            elif white_p > MAX_PURE_WHITE_PERCENT:
                reason = "Exceso de luz quemada"
            elif ear_max < EAR_THRESHOLD:
                reason = "Abre los ojos (parpadeo)"
            elif not (block["yaw"][0] <= yaw <= block["yaw"][1] and block["pitch"][0] <= pitch <= block["pitch"][1]):
                reason = f"Gira hacia: {block['name']}"
            elif time.time() - last_save_time < MIN_INTERVAL_SEC:
                reason = "Esperando intervalo..."
                passed = True
            else:
                cropped = align_and_crop(frame, landmarks_px, PADDING_PERCENT)
                if cropped is None:
                    reason = "Recorte fuera de marco (centrate)"
                else:
                    crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
                    
                    if laplacian_var < LAPLACIAN_VAR_THRESH:
                        reason = "Imagen borrosa (quedate quieto)"
                    else:
                        passed = True
                        reason = "CAPTURANDO..."
                        
                        filename = f"{block['name'].replace(' ', '_')}_{int(time.time()*1000)}.jpg"
                        filepath = os.path.join(save_dir, filename)
                        cv2.imwrite(filepath, cropped)
                        
                        last_save_time = time.time()
                        block["count"] += 1
                        saved_count += 1
                        
                        if block["count"] >= block["target"]:
                            current_block_idx += 1

            color = (0, 255, 0) if passed else (0, 0, 255)
            cv2.rectangle(display_frame, (min_x, min_y), (max_x, max_y), color, 2)
            cv2.circle(display_frame, tuple(landmarks_px[33]), 2, (255, 255, 0), -1)
            cv2.circle(display_frame, tuple(landmarks_px[263]), 2, (255, 255, 0), -1)
            cv2.circle(display_frame, tuple(landmarks_px[1]), 2, (255, 0, 0), -1)

        # -------------------------------------------------------------
        # 🎨 HUD EN VIVO
        # -------------------------------------------------------------
        hud_bg = display_frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, display_frame, 0.4, 0, display_frame)
        
        cv2.putText(display_frame, f"Bloque: {block['name']} ({block['count']}/{block['target']})", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Yaw: {yaw:.1f} | Pitch: {pitch:.1f}", 
                    (w - 350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        total_target = sum(b['target'] for b in POSE_BLOCKS)
        cv2.putText(display_frame, f"Total: {saved_count}/{total_target}", 
                    (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
        msg_color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.putText(display_frame, f"Estado: {reason}", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)

        cv2.imshow("Captura CNN", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            print(f"⏭️ Bloque '{block['name']}' omitido manualmente.")
            current_block_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Proceso finalizado. Total guardadas: {saved_count}")

if __name__ == "__main__":
    main()