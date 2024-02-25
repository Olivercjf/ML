import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import ConfusionMatrix
import matplotlib.pyplot as plt
from yolov8 import YOLOv8

x_test = np.load("/home/oliver18/Documents/ML/ML/data/train/images")
y_test = np.load("/home/oliver18/Documents/ML/ML/data/train/labels")
numero_de_clases = 1

# Carga el modelo YOLOv8
modelo = YOLOv8(model_path="ruta/a/tu/modelo.h5")

# Obtener las predicciones del modelo para el conjunto de prueba
predicciones = modelo.predict(x_test)

# Convertir las predicciones en etiquetas
etiquetas_predichas = np.argmax(predicciones, axis=-1)

# Crear una matriz de confusión
matriz_confusion = ConfusionMatrix(num_classes=numero_de_clases)

# Actualizar la matriz de confusión con las etiquetas reales y predichas
matriz_confusion.update_state(y_true=y_test, y_pred=etiquetas_predichas)

# Obtener la matriz de confusión
matriz_confusion = matriz_confusion.result().numpy()

# Visualizar la matriz de confusión
plt.imshow(matriz_confusion, cmap="hot")
plt.xlabel("Etiqueta real")
plt.ylabel("Etiqueta predicha")
plt.show()