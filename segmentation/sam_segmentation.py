import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt

# Inicializar el modelo SAM
sam_checkpoint = "weigths/sam_vit_b_01ec64.pth"  # Ruta al archivo del modelo
model_type = "vit_b"  # Tipo de modelo a usar (vit_h, vit_b, etc.)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

# Cargar el modelo SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# Cargar la imagen
image_path = "data/cimg-306.png"  # Ruta a tu imagen
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mostrar la imagen en una ventana
# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb)
# plt.axis("off")  # Ocultar los ejes
# plt.show()


mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

sam_results = mask_generator.generate(image_rgb)
print(len(sam_results))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack(img, m=0.35))


plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
show_anns(sam_results)
plt.axis("off")
plt.show()
