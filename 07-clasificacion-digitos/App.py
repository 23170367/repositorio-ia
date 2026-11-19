from pathlib import Path
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2

# Obtiene la carpeta donde está guardado este archivo App.py (07-clasificacion-digitos)
BASE_DIR = Path(__file__).resolve().parent
# Construye la ruta completa hacia Modelo.keras dentro de esa misma carpeta
ruta_modelo = BASE_DIR / "Modelo.keras"

# 1. Cargar el modelo previamente entrenado
try:
    modelo = tf.keras.models.load_model(ruta_modelo)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

class AppClasificadorNumeros:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Dígitos MNIST")
        
        # Configuración del lienzo (Canvas)
        self.ancho_canvas = 280
        self.alto_canvas = 280
        # MNIST usa trazos gruesos; simulamos esto con un pincel ancho
        self.grosor_pincel = 20
        
        # Creamos el Canvas de Tkinter visible para el usuario (fondo negro)
        self.canvas = tk.Canvas(self.root, width=self.ancho_canvas, height=self.alto_canvas, bg='black', cursor='cross')
        self.canvas.pack(pady=10)
        
        # Enlazamos el evento de arrastrar el ratón con la función de dibujar
        self.canvas.bind("<B1-Motion>", self.dibujar)
        self.canvas.bind("<ButtonRelease-1>", self.predecir) # Predecir al soltar el clic
        
        # Creamos una imagen PIL "oculta" que sincronizaremos con el Canvas
        # Es más confiable procesar esta imagen que intentar tomar captura de pantalla al Canvas
        self.imagen_pil = Image.new("L", (self.ancho_canvas, self.alto_canvas), color=0) # "L" = escala de grises, 0 = negro
        self.draw = ImageDraw.Draw(self.imagen_pil)
        
        # Interfaz: Etiquetas de resultados y botones
        self.lbl_prediccion = tk.Label(self.root, text="Dibuja un número", font=("Helvetica", 18, "bold"))
        self.lbl_prediccion.pack()
        
        self.lbl_probabilidades = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.lbl_probabilidades.pack(pady=5)
        
        self.btn_limpiar = tk.Button(self.root, text="Limpiar", command=self.limpiar, font=("Helvetica", 12))
        self.btn_limpiar.pack(pady=10)

    def dibujar(self, event):
        # Coordenadas del ratón
        x, y = event.x, event.y
        r = self.grosor_pincel
        
        # Dibujamos un óvalo blanco en el Canvas de Tkinter visible
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        
        # Dibujamos exactamente lo mismo en nuestra imagen PIL oculta (color 255 = blanco)
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def limpiar(self):
        # Borra el Canvas de Tkinter
        self.canvas.delete("all")
        # Reinicia la imagen PIL oculta a negro
        self.draw.rectangle([0, 0, self.ancho_canvas, self.alto_canvas], fill=0)
        self.lbl_prediccion.config(text="Dibuja un número")
        self.lbl_probabilidades.config(text="")

    def predecir(self, event=None):
        # 1. Convertimos la imagen a arreglo numpy
        img_array = np.array(self.imagen_pil)
        
        # Buscamos todas las coordenadas donde el color sea mayor a 0 (donde hay dibujo)
        y_idxs, x_idxs = np.where(img_array > 0)
        
        if len(x_idxs) == 0:  # Si el lienzo está vacío, no hacemos nada
            return
            
        x_min, x_max = np.min(x_idxs), np.max(x_idxs)
        y_min, y_max = np.min(y_idxs), np.max(y_idxs)
        
        # B. Recortar la imagen al tamaño exacto de lo dibujado
        recorte = img_array[y_min:y_max+1, x_min:x_max+1]
        
        # C. Redimensionar para que el lado más largo sea de 20 píxeles
        alto, ancho = recorte.shape
        if alto > ancho:
            nuevo_alto = 20
            # Regla de 3 para mantener la proporción del ancho
            nuevo_ancho = int(ancho * (20.0 / alto)) 
        else:
            nuevo_ancho = 20
            nuevo_alto = int(alto * (20.0 / ancho))
            
        # Prevenir dimensiones de valor 0 si el trazo es una línea perfecta
        nuevo_ancho = max(1, nuevo_ancho)
        nuevo_alto = max(1, nuevo_alto)
        
        recorte_resized = cv2.resize(recorte, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
        
        # D. Crear un lienzo negro final de 28x28 y pegar el número en el centro exacto
        img_final_28x28 = np.zeros((28, 28), dtype=np.uint8)
        
        y_offset = (28 - nuevo_alto) // 2
        x_offset = (28 - nuevo_ancho) // 2
        
        img_final_28x28[y_offset:y_offset+nuevo_alto, x_offset:x_offset+nuevo_ancho] = recorte_resized
        
        # 3. Normalizamos los píxeles (0 a 1)
        img_normalizada = img_final_28x28 / 255.0
        
        # 4. Ajustamos la forma para el modelo: (1, 28, 28)
        img_final = np.expand_dims(img_normalizada, axis=0)
        
        # --- INFERENCIA ---
        predicciones = modelo.predict(img_final, verbose=0)[0]
        digito_predicho = np.argmax(predicciones)
        probabilidad = predicciones[digito_predicho] * 100
        
        # Actualizamos la interfaz
        self.lbl_prediccion.config(text=f"Predicción: {digito_predicho} ({probabilidad:.1f}%)")
        
        top_3_indices = np.argsort(predicciones)[-3:][::-1]
        texto_probs = " | ".join([f"{i}: {predicciones[i]*100:.1f}%" for i in top_3_indices])
        self.lbl_probabilidades.config(text=f"Top 3: {texto_probs}")

# Inicializamos la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = AppClasificadorNumeros(root)
    root.mainloop()