import numpy as np
import cv2
from ultralytics.models.sam import Predictor as SAMPredictor

overrides = dict(
    conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam_b.pt"
)
model1 = SAMPredictor(overrides=overrides)  # Cambia a 'mobile_sam.pt' si es necesario

# Cargar la imagen
image_path = "C:\Study\Tesis\Mi Tesis\data\cimg-500.png"
image = cv2.imread(image_path)

# Realizar la segmentación
results = model1(source=image_path, multimask_output=True)

# Acceder a las máscaras generadas
masks = results[0].masks
print(masks)
print("------------------------------")
print(results)

masks = results[0].masks  # Esto te dará acceso a las máscaras segmentadas
masks = masks.data

if masks is not None and len(masks) > 0:
    mask_index = (
        6  # Cambia este índice para seleccionar otra máscara (0 para la primera)
    )

    # Asegúrate de que la máscara sea un array NumPy
    selected_mask = (
        masks[mask_index].numpy()
        if hasattr(masks[mask_index], "numpy")
        else np.array(masks[mask_index])
    )

    # Convertir la máscara a formato binario (0 o 1)
    mask_image = (selected_mask > 0).astype(
        np.uint8
    )  # Convertir a formato binario (0 o 1)

    # Crear una imagen vacía del mismo tamaño que la original (negra)
    mask_only_image = np.zeros_like(
        image
    )  # Imagen negra del mismo tamaño que la original

    # Copiar los píxeles de la imagen original donde hay máscara
    mask_only_image[mask_image == 1] = image[mask_image == 1]

    # Guardar la imagen que contiene solo la máscara
    cv2.imwrite("imagen_con_solo_mascara1.png", mask_only_image)
    print("Imagen guardada como 'imagen_con_solo_mascara.png'")
else:
    print("No se encontraron máscaras.")
