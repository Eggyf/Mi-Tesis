import torch
import cv2
import os
from pathlib import Path
import numpy as np
import json
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib import pyplot as plt


# Inicializar el modelo SAM
sam_checkpoint = "./weigths/sam_vit_b_01ec64.pth"  # Ruta al archivo del modelo
model_type = "vit_b"  # Tipo de modelo a usar (vit_h, vit_b, etc.)
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "imagen_zoom.png"


class SAM:
    def __init__(self, _checkpoint, _model_type, _device):
        self.checkpoint = _checkpoint
        self.model_type = _model_type
        self.device = _device
        self.predictor = None

    def prepare(self):
        print("Cargando el modelo")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        print("Modelo cargado")
        self.predictor = SamPredictor(sam)
        self.model = sam

    def set_embedding(self, img, format="RGB"):
        self.predictor.set_image(img, format)

    def save_embedding(self, path):
        if not self.predictor.is_image_set:
            raise RuntimeError("No hay embedding")
        res = {
            "original_size": self.predictor.original_size,
            "input_size": self.predictor.input_size,
            "features": self.predictor.features,
            "is_image_set": True,
        }
        torch.save(res, path)

    def load_embedding(self, path):
        self.predictor.reset_image()
        res = torch.load(path, self.predictor.device)
        for k, v in res.items():
            setattr(self.predictor, k, v)

    def segmentate(self, img):
        print("Segmentando con SAM")
        mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.96,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        self.results = mask_generator.generate(img)
        print("Segmentacion terminada")

    def segmentate_from_box(self, x1, y1, x2, y2):
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks

    def segmentate_from_points(self, fg_points, bg_points, multimask=True):
        points = fg_points + bg_points
        labels = [1] * len(fg_points) + [0] * len(bg_points)
        mask, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=multimask,
        )
        return mask

    def segmentate_from_box_and_middle_point(self, x1, y1, x2, y2):
        input_box = np.array([x1, y1, x2, y2])
        input_points = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
        label = np.array([1])
        masks, _, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=label,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks

    def show_anns(self, anns):
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
            ax.imshow(np.dstack((img, m * 0.35)))

    def show_segmentation(self, img):
        plt.figure(figsize=(10, 10))

        plt.imshow(img)
        self.show_anns(self.results)
        plt.axis("off")
        plt.show()


def save_embeddings(path_list, out_folder, model_type, model_path):
    # Crear el modelo
    model = SAM(model_path, model_type, device)
    model.prepare()

    result = []
    # crear carpeta para los resultados
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    data_path = os.path.join(out_folder, "data.json")
    for index, img_path in enumerate(path_list):
        print(f"{index + 1}/{len(path_list)}")
        # Cargar la imagen
        img = cv2.imread(str(img_path))
        # Crear el embedding
        model.set_embedding(img)
        # Crear el path del embedding
        path = Path(img_path)
        img_name = path.name.split(".")[0]
        emb_name = img_name + ".torch"
        emb_path = os.path.join(out_folder, emb_name)
        # Salvando el embedding
        model.save_embedding(emb_path)

        # Anotando los datos
        result.append(
            {
                "path": str(img_path),
                "emb_path": str(emb_path),
            }
        )
        with open(data_path, "w") as f:
            json.dump(result, f)
    return data_path


def load_img(img_path):
    # Ruta a tu imagen

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


img = load_img(image_path)
sam_model = SAM(sam_checkpoint, model_type, device)
sam_model.prepare()
sam_model.segmentate(img=img)
print(sam_model.results[0]["predicted_iou"])
sam_model.show_segmentation(img=img)
