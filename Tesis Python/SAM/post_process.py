import cv2
import numpy as np
import os

# Definir la ruta a la imagen segmentada (máscara)
image_path = "imagen_con_solo_mascara.png"  # Cambia esto a la ruta correcta

# Verificar si la ruta existe
if not os.path.exists(image_path):
    print("La ruta al archivo no es válida.")
    exit()

# Cargar la imagen (máscara)
mask = cv2.imread(image_path)
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# Verificar si la imagen se cargó correctamente
if mask is None:
    print("Error al cargar la imagen.")
    exit()

# Encontrar los contornos en la máscara
contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Verificar si se encontraron contornos
if len(contours) == 0:
    print("No se encontraron contornos en la máscara.")
    exit()

# Encontrar el contorno más grande (puedes ajustar esto según tus necesidades)
largest_contour = max(contours, key=cv2.contourArea)

# Obtener el rectángulo delimitador del contorno
x, y, width, height = cv2.boundingRect(largest_contour)

# Recortar el área de interés (ROI) usando el rectángulo delimitador
roi = mask[y : y + height, x : x + width]

# Redimensionar el área recortada para hacer zoom (por ejemplo, duplicar el tamaño)
zoomed_image = cv2.resize(roi, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

# Guardar la imagen con zoom
output_path = "imagen_zoom.png"  # Cambia esto a donde quieras guardar la imagen
cv2.imwrite(output_path, zoomed_image)


# Imprimir las dimensiones de las imágenes para verificación
print("Dimensiones originales:", mask.shape)
print("Dimensiones del área de interés (ROI):", roi.shape)
print("Dimensiones de la imagen con zoom:", zoomed_image.shape)
