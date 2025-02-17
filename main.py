import cv2
import numpy as np
from sam2 import SAM2Image
from utils import draw_masks
import os
from yolov8_onnx import YOLOv8
import gradio as gr

def download_model(url):
    import requests
    model_path = "./models/" + url.split('/')[-1]
    if not os.path.exists(model_path):
        from io import BytesIO
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return model_path

def get_sam_model_paths(model_type):
    """
    根据选择的模型类型返回对应的模型路径
    """
    if model_type == "large":
        return {
            "encoder": "./models/sam2_hiera_large_encoder.onnx",
            "decoder": "./models/sam2_hiera_large_decoder.onnx"
        }
    else:  # base
        return {
            "encoder": "./models/sam2_hiera_base_plus_encoder.onnx",
            "decoder": "./models/sam2_hiera_base_plus_decoder.onnx"
        }

def process_image(image, yolo_model_path, class_names, sam_model_type):
    """
    处理图片的函数，适配 Gradio 接口
    """
    model_paths = get_sam_model_paths(sam_model_type)
    encoder_model_path = model_paths["encoder"]
    decoder_model_path = model_paths["decoder"]

    if image is None:
        return "请上传图片"

     # 验证YOLO模型文件是否存在
    if not yolo_model_path:
        return f"YOLO模型文件不存在: {yolo_model_path}"
    
    if yolo_model_path.startswith('http') or yolo_model_path.startswith('https'):
        yolo_model_path = download_model(yolo_model_path)
    
    # 处理类别名称
    classes = [name.strip() for name in class_names.split(',')]
    if not classes:
        return "请输入至少一个类别名称"
    
    sam2 = SAM2Image(encoder_model_path, decoder_model_path)

    # 转换 Gradio 图片格式为 OpenCV 格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    detection = YOLOv8(yolo_model_path, 0.5, 0.5, classes)

    image_copy = image.copy()
    _results = detection.predict(image)

    if len(_results) > 0:
        # 设置SAM的输入图像
        sam2.set_image(image_copy)
        all_masks = {}
        
        # 先获取所有边界框的 masks
        for idx, (box, score, class_id) in enumerate(_results):
            x1, y1, w, h = map(int, box)
            sam2.set_box(((x1, y1), (x1 + w, y1 + h)), label_id=class_id)
            masks = sam2.get_masks()
            
            print(f"Original mask shape for box {idx}:")
            for k, mask in masks.items():
                print(f"Mask {k}: shape={mask.shape}, type={type(mask)}, values={np.unique(mask)}")
            
            resized_masks = {}
            for k, mask in masks.items():
                if isinstance(mask, list):  # 检查是否是列表
                    mask = mask[0]  # 获取列表中的第一个掩码
                
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                resized_masks[k] = mask
                
                print(f"Resized mask {k}: shape={mask.shape}, type={type(mask)}, values={np.unique(mask)}")
            
            renamed_masks = {f"{k}_{idx}": v for k, v in resized_masks.items()}
            all_masks.update(renamed_masks)
        
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
    with gr.Blocks() as demo:
        gr.Markdown("# 目标检测与分割演示")
        gr.Markdown("上传图片并设置模型参数，系统将自动检测并分割目标。")
        
        with gr.Row():
            with gr.Column():
                # 输入部分
                input_image = gr.Image(label="输入图片")
                yolo_model_path = gr.Textbox(
                    label="YOLO模型路径",
                    value="./models/yolov8x.onnx",
                    placeholder="输入YOLO模型的路径"
                )
                class_names = gr.Textbox(
                    label="检测类别（用逗号分隔）",
                    value="person",
                    placeholder="例如: person,car,dog"
                )
                sam_model_type = gr.Radio(
                    choices=["base", "large"],
                    value="large",
                    label="SAM模型类型",
                    info="选择分割模型的类型：base 或 large"
                )
                process_button = gr.Button("开始处理")
            
            with gr.Column():
                # 输出部分
                output_image = gr.Image(label="处理结果")
        
        # 设置处理流程
        process_button.click(
            fn=process_image,
            inputs=[input_image, yolo_model_path, class_names, sam_model_type],
            outputs=output_image
        )
        
        # 添加示例
        gr.Examples(
            examples=[
                ["bus.jpg", "https://xulamodel.oss-cn-beijing.aliyuncs.com/xulamodel/xula/onnx/20250213/best.onnx", "person", "base"],
                ["bus.jpg", "https://xulamodel.oss-cn-beijing.aliyuncs.com/yolo/yolov8x.onnx", "person,car,dog", "base"]
            ],
            inputs=[input_image, yolo_model_path, class_names, sam_model_type]
        )
    
    return demo

if __name__ == "__main__":
    # 创建并启动 Gradio 界面
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860) 