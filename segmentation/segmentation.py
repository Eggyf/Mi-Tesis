import cv2
import numpy as np


def segment_ulcer(image):
    # Preprocesamiento de la imagen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Selección de semilla
    seed_x, seed_y = 441, 208  # Coordenadas de la semilla (ajusta según tu imagen)

    # Crecimiento de la región
    mask = np.zeros_like(gray)
    queue = [(seed_x, seed_y)]
    mask[seed_y, seed_x] = 255

    # Obtener características de la semilla
    seed_intensity = int(blur[seed_y, seed_x])
    seed_texture = cv2.Laplacian(blur, cv2.CV_64F)[seed_y, seed_x]

    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                if mask[ny, nx] == 0:
                    # Calcular diferencia de intensidad y textura
                    intensity_diff = abs(int(blur[ny, nx]) - seed_intensity)
                    texture_diff = abs(
                        cv2.Laplacian(blur, cv2.CV_64F)[ny, nx] - seed_texture
                    )

                    # Criterios de similitud
                    if intensity_diff < 40 and texture_diff <= 11:
                        mask[ny, nx] = 255
                        queue.append((nx, ny))

    # Refinamiento de la segmentación
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Visualización y análisis
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Calcular métricas de interés
    ulcer_area = np.sum(mask == 255)
    print(f"Área de la úlcera: {ulcer_area} píxeles")

    return segmented_image


# Cargar la imagen del pie con la úlcera
image = cv2.imread("data/cimg-1205.png")

# Segmentar la úlcera
segmented_image = segment_ulcer(image)

# Mostrar la imagen segmentada
cv2.imshow("Ulcera", image)
cv2.imshow("Úlcera segmentada", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
