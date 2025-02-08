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


def draw_masks(image, masks):
    """
    在图像上绘制所有分割掩码
    
    Args:
        image: 原始图像
        masks: 包含所有掩码的字典，格式为 {mask_id_idx: mask}
    
    Returns:
        image: 绘制了掩码的图像
        mask_points: 所有掩码的轮廓点
    """
    mask_points = {}
    # 为每个掩码生成不同的随机颜色
    colors = {}
    
    # 首先获取基础类别（去掉索引号）
    for mask_id in masks.keys():
        base_id = mask_id.split('_')[0]  # 获取基础类别ID
        if base_id not in colors:
            colors[base_id] = tuple(np.random.randint(0, 255, 3).tolist())

    # 按掩码绘制
    for mask_id, mask in masks.items():
        base_id = mask_id.split('_')[0]  # 获取基础类别ID
        color = colors[base_id]
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color
        
        # 将掩码与原图像混合
        alpha = 0.5
        mask_area = (mask == 1)
        image[mask_area] = cv2.addWeighted(image[mask_area], 1-alpha, colored_mask[mask_area], alpha, 0)
        
        # 获取掩码轮廓
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # 存储轮廓点
        if contours:
            mask_points[mask_id] = contours[0].squeeze().tolist()
            # 绘制轮廓
            cv2.drawContours(image, contours, -1, color, 2)

    return image, mask_points

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image, contours
