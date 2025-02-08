import cv2
import numpy as np
from sam2 import SAM2Image
from utils import draw_masks
import os
from yolov8_onnx import YOLOv8
import gradio as gr

def process_image(image):
    """
    处理图片的函数，适配 Gradio 接口
    """
    encoder_model_path = "./models/sam2_hiera_base_plus_encoder.onnx"
    decoder_model_path = "./models/decoder.onnx"
    yolo_model_path = "./models/yolov8x.onnx"
    sam2 = SAM2Image(encoder_model_path, decoder_model_path)

    # 转换 Gradio 图片格式为 OpenCV 格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    classes = ['person']
    detection = YOLOv8(yolo_model_path, 0.5, 0.5, classes)

    image_copy = image.copy()
    _results = detection.predict(image)

    if len(_results) > 0:
        # 设置SAM的输入图像
        sam2.set_image(image_copy)
        all_masks = {}
        for (box, score, class_id) in _results:
            print(box)
            x1, y1, w, h = map(int, box)
            sam2.set_box(((x1, y1), (x1 + w, y1 + h)), label_id=class_id)
            masks = sam2.get_masks()
            all_masks.update(masks)
        
        # 在图像上绘制所有分割结果
        if all_masks:
            image, mask_points = draw_masks(image_copy, all_masks)
        
        # 绘制所有边界框和标签
        for (box, score, class_id) in _results:
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h
            label = classes[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 转换回 RGB 格式用于 Gradio 显示
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        print("No objects detected in the image")
        # 转换回 RGB 格式用于 Gradio 显示
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def create_demo():
    """
    创建 Gradio 演示界面
    """
    demo = gr.Interface(
        fn=process_image,
        inputs=gr.Image(),
        outputs=gr.Image(),
        title="人物检测与分割演示",
        description="上传一张图片，系统将自动检测并分割出人物。",
        examples=["test.png"]
    )
    return demo

if __name__ == "__main__":
    # 创建并启动 Gradio 界面
    demo = create_demo()
    demo.launch(share=True) 