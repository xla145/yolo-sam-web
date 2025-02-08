import cv2
import numpy as np

rng = np.random.default_rng(2)
colors = rng.uniform(0, 255, size=(100, 3))

def contours_to_coco_format(contours, label_masks, annotation_id):
    coco_annotations = []
    
    for contour, label in zip(contours, label_masks):
        # 将每个轮廓的点展平为一个平坦的列表
        segmentation = contour.flatten().tolist()  # 将每个轮廓的 (x, y) 点展平为 1D 列表
        
        # 创建 COCO 注解对象
        annotation = {
            "segmentation": [segmentation],  # 每个轮廓是一个多边形
            "area": cv2.contourArea(contour),  # 面积
            "iscrowd": 0,  # 如果是 crowd 则设为 1，默认是 0
            "category_id": label,  # 标签 ID
            "id": annotation_id  # 标注 ID
        }
        coco_annotations.append(annotation)
        annotation_id += 1  # 更新标注 ID
    
    return coco_annotations


def draw_masks(image: np.ndarray, masks: dict[int, np.ndarray], alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()

    mask_pionts = []

    for label_id, label_masks in masks.items():
        if label_masks is None:
            continue
        color = colors[label_id]
        mask_image, contours = draw_mask(mask_image, label_masks, (color[0], color[1], color[2]), alpha, draw_border)
        # 把contours的点转成list
        coco = contours_to_coco_format(contours, [label_id], 1)
        mask_pionts.append(coco)
    return mask_image, mask_pionts

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image, contours
