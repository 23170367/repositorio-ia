# Clasificación con k-NN: Adult Census Income

Adaptación de un cuaderno educativo de **k-Nearest Neighbors (k-NN)** —originalmente hecho para un ejemplo de 28 filas sobre el clima— para trabajar con el dataset real **[Adult Census Income](https://huggingface.co/datasets/scikit-learn/adult-census-income)** de Hugging Face (+32,000 registros). El objetivo cambia de predecir si se puede salir a jugar golf a predecir si una persona gana más o menos de **50,000 USD anuales**.

---

## Resumen rápido: Original vs. Adaptado

| Aspecto | Código original | Código adaptado |
| :--- | :--- | :--- |
| **Dataset** | 28 filas manuales (clima) | ~32,500 filas desde Hugging Face (censo) |
| **Variable objetivo** | `Play` (`Yes` / `No`) | `income` (`>50K` / `<=50K`) |
| **Limpieza** | Ninguna | Eliminación de `'?'` y columnas que estorban |
| **División (Train/Test)** | 50% / 50% sin mezclar | 80% / 20% mezclado y balanceado |
| **Vecinos (`k`)** | `5` (pruebas: `3, 5, 7`) | `15` (pruebas: `9, 15, 21, 31`) |
| **Distancia** | Euclidiana (`euclidean`) | Manhattan (`cityblock`) |
| **Procesamiento** | 1 núcleo | Todos los núcleos disponibles (`n_jobs=-1`) |

---

## Modificaciones celda por celda

### Celda 1: Carga de datos
* **Cambio:** Se reemplazó el diccionario escrito a mano por `pd.read_csv()` apuntando directo al archivo `adult.csv` en Hugging Face.

### Celda 2: Limpieza, preparación y división
* **Datos faltantes (`'?'`):** Se reemplazaron los signos `'?'` por `NaN` y se borraron esas filas con `dropna()` para evitar decisiones con datos incompletos.
* **Columnas eliminadas (`fnlwgt` y `education`):** `fnlwgt` es un peso estadístico del censo que distorsiona las distancias, y `education` repite en texto lo que ya contiene `education.num` en número.
* **Texto a números (*One-Hot Encoding*):** Se detectan automáticamente todas las columnas de texto y se convierten en columnas de `0` y `1` con `pd.get_dummies()`, conservando prefijos para evitar nombres duplicados.
* **Columna objetivo (`income`):** Se eliminaron posibles espacios extra (`.str.strip()`) y se convirtió a `1` (`>50K`) y `0` (`<=50K`).
* **Enteros a decimales (`float`):** Como los números del censo vienen como enteros y el código original usa `float64` para saber qué escalar con `StandardScaler`, convertirlos a `float` mantiene la lógica intacta sin escalar por error las columnas de `0` y `1`.
* **División mejorada (`train_size=0.8, shuffle=True, stratify=y, random_state=42`):** Da más datos para entrenar (80/20), revuelve el orden de las filas (`shuffle`), mantiene la misma proporción de ingresos en entrenamiento y prueba (`stratify`), y asegura resultados repetibles (`random_state`).

### Celda 3: Número de vecinos (`k`)
* **Cambio:** `k` pasó de `5` a `15`.
* **Por qué:** Con más de 24,000 filas de entrenamiento, consultar solo a 5 vecinos hace al modelo muy sensible a casos aislados; usar 15 da una votación más estable.

### Celda 4: Métrica de distancia e índices
* **Cambio:** Se cambió `'euclidean'` por `'cityblock'` (distancia Manhattan), se añadió `ord=1` a `np.linalg.norm`, y se sustituyó `.loc[0]`/`.loc[1]` por `.iloc[0]`/`.iloc[1]`.
* **Por qué:** Con casi 80 columnas tras el *One-Hot Encoding*, la distancia Manhattan (suma de diferencias por columna) suele rendir mejor y sufrir menos por valores extremos que la línea recta euclidiana. Además, `.iloc` toma las dos primeras filas disponibles y evita errores de índice tras borrar datos y barajar.

### Celda 5: Inicialización del clasificador
* **Cambio:** Se agregó `n_jobs=-1`.
* **Por qué:** Usa todos los núcleos del procesador en paralelo para acelerar el cálculo de distancias entre miles de filas.

### Celda 6: Distancias con SciPy
* **Cambio:** Se cambió `'euclidean'` fijo por `metric=distance_metric`.
* **Por qué:** Garantiza que esta celda use automáticamente la misma métrica Manhattan definida en la Celda 4.

### Celda 7: Vista de predicciones
* **Cambio:** Se agregó `[:20]` al imprimir `y_test` y `y_pred`.
* **Por qué:** Muestra solo los primeros 20 resultados en lugar de saturar la pantalla con más de 6,000 predicciones.

### Celda 8: Evaluación (*Accuracy*)
* **Sin cambios:** El cálculo del porcentaje de aciertos funciona igual sin importar el tamaño del dataset.

### Celda 9: Comparación de distintos `k`
* **Cambio:** Se cambió `k_list` de `[3, 5, 7]` a `[9, 15, 21, 31]` y se añadieron `metric=distance_metric` y `n_jobs=-1` al clasificador dentro del ciclo.
* **Por qué:** Evalúa valores de `k` acordes al volumen real del censo, mantiene consistencia con la métrica Manhattan y acelera la ejecución.