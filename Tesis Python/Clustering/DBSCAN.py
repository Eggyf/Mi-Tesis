import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Cargar la imagen segmentada
image_path = "C:\Study\Tesis\Mi Tesis\imagen_zoom.png"
# Cambia esta ruta a tu imagen
image = cv2.imread(image_path)

if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar la imagen si es necesario (opcional)
# image = cv2.resize(image, (width, height))

# Convertir la imagen a un espacio de color LAB
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# Obtener los píxeles de la imagen
pixels = image_lab.reshape(-1, 3)

# Aplicar DBSCAN para agrupar colores
dbscan = DBSCAN(eps=4, min_samples=1000)  # Ajusta eps y min_samples según sea necesario
labels = dbscan.fit_predict(pixels)

clustered_image = np.zeros((pixels.shape[0], 3), dtype=np.uint8)

# Visualizar los resultados
unique_labels = set(labels)
colors = [
    np.random.randint(0, 255, size=3) for _ in range(len(unique_labels) - 1)
]  # Excluir ruido

for idx, label in enumerate(labels):
    if label == -1:
        # Asignar un color negro para el ruido
        clustered_image[idx] = [0, 0, 0]
    else:
        clustered_image[idx] = colors[label]

# Reshape the clustered image back to the original image shape
clustered_image = clustered_image.reshape(image.shape).astype(np.uint8)

# Mostrar la imagen original y la imagen con clusters coloreados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Clusters Coloreados")
plt.imshow(clustered_image)
plt.axis("off")

plt.show()
