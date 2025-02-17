import cv2
import numpy as np

rng = np.random.default_rng(2)
colors = rng.uniform(0, 255, size=(100, 3))

# 预定义颜色列表（可根据需要扩展）
PREDEFINED_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (0, 0, 0)         # Black
]

def get_color(index):
    """
    根据索引返回预定义的颜色，循环使用颜色列表
    """
    return PREDEFINED_COLORS[index % len(PREDEFINED_COLORS)]

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
    在图像上绘制所有分割掩码，并美化颜色

    Args:
        image: 原始图像（BGR格式）
        masks: 包含所有掩码的字典，格式为 {mask_id_idx: mask}

    Returns:
        image: 绘制了掩码的图像
        mask_points: 所有掩码的轮廓点
    """
    mask_points = {}
    colors = {}
    
    # 为每个类别分配一个固定颜色
    for mask_id in masks.keys():
        base_id = mask_id.split('_')[0]  # 获取基础类别ID
        if base_id not in colors:
            # 使用类别索引分配颜色，假设 base_id 是整数或可转换为整数
            try:
                class_index = int(base_id)
            except ValueError:
                class_index = hash(base_id)  # 否则使用哈希值
            colors[base_id] = get_color(class_index)
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建一个透明层用于绘制掩码
    overlay = image.copy()
    
    for mask_id, mask in masks.items():
        base_id = mask_id.split('_')[0]
        color = colors[base_id]
        
        # 确保掩码尺寸与图像匹配
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 标准化掩码值到 0-1
        mask = (mask > 0.5).astype(np.uint8)
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask == 1] = color
        
        # 将彩色掩码叠加到透明层上
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
        
        # 获取掩码轮廓
        contours, _ = cv2.findContours(mask, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # 存储轮廓点并绘制边界
        if contours:
            mask_points[mask_id] = contours[0].squeeze().tolist()
            cv2.drawContours(overlay, contours, -1, color, 2)
    
    # 合并透明层与原图像
    image = overlay
    
    return image, mask_points

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.8, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image, contours
