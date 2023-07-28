"""
 this is a adapted code from 
 https://github.com/facebookresearch/segment-anything/
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.image
from urllib.request import urlopen, urlretrieve, Request
import matplotlib
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

matplotlib.use("agg")


class ExtractMasks:
    def __init__(self, img_input_dir: str = None, img_input_file=None) -> None:
        self.img_input_file = img_input_file
        self.model_dir = "models/sam_vit_b_01ec64.pth"
        self.model_type = "vit_b"

    def mark_areas(self, anns: np.ndarray, type_bg: str = "color"):
        if type_bg == "color":
            bg_color = [0.0, 0.0, 0.0, 0.0]
            ar_color = [0.5, 0.0, 0.0, 0.40]
        else:
            bg_color = [0.0, 0.0, 0.0, 1]
            ar_color = [1, 1, 1, 1]

        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for idx, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            if idx == 0:
                color = bg_color
            else:
                color = ar_color
            img[m] = color
        ax.imshow(img)
        return img

    def pad_images(
        self, img1: np.ndarray, y: int, x: int, color: List[int] = [255, 255, 255]
    ) -> np.ndarray:
        constant = cv2.copyMakeBorder(
            img1, y, y, x, x, cv2.BORDER_CONSTANT, value=color
        )
        return constant

    def extract_roi_images(
        self, img: np.ndarray, mask: np.ndarray, size_roi: Tuple = (128, 128)
    ) -> List[np.ndarray]:
        imgs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        th, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        list_masks = []
        selected_contours = [c for c in contours if cv2.contourArea(c) >= 100]
        img_with_draw_ctour = cv2.drawContours(
            imgs, selected_contours, -1, (255, 255, 255), 1
        )
        # list_masks.append(img_with_draw_ctour)

        for idx, area in enumerate(selected_contours):
            x, y, w, h = cv2.boundingRect(area)
            cropped_img = imgs[y - 1 : y + h + 1, x - 1 : x + w + 1]

            difference_y, difference_x = 0, 0
            h2, w2 = cropped_img.shape

            # Add padding to new images if neccesary
            if cropped_img.shape < size_roi:
                if cropped_img.shape[0] <= 3 or cropped_img.shape[1] <= 3:
                    continue
                if cropped_img.shape[0] < size_roi[0]:
                    difference_y = (size_roi[0] - h2) // 2 + 1
                if cropped_img.shape[1] < size_roi[1]:
                    difference_x = (size_roi[1] - w2) // 2 + 1

                cropped_img = self.pad_images(
                    cropped_img, y=difference_y, x=difference_x
                )
            list_masks.append(cropped_img)

        return img_with_draw_ctour, list_masks

    def image_to_numpy_array(self, transform_gray: bool = False) -> np.ndarray:
        file_bytes = np.fromstring(self.img_input_file, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if transform_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_masks(self) -> np.ndarray:
        print("Extrayendo RoI's...")
        image = self.image_to_numpy_array()
        sam = sam_model_registry[self.model_type](checkpoint=self.model_dir)
        mask_generator = SamAutomaticMaskGenerator(sam)
        layer_mask = mask_generator.generate(image)
        layer_mask = self.mark_areas(layer_mask, type_bg="gray")
        layer_mask = layer_mask.astype("uint8")

        mask, list_masks = self.extract_roi_images(image, layer_mask)
        return image, mask, list_masks
