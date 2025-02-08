## 使用yolo结合sam实现精细分割

### yolo介绍
YOLO (You Only Look Once) 是一个高效的目标检测算法，能够实时检测图像中的多个对象。它将图像分割成网格，每个网格预测边界框和类别概率。

### sam介绍
SAM (Segment Anything Model) 是Meta AI推出的图像分割模型，可以根据提示（如边界框）生成高质量的分割掩码。它能够适应各种分割任务，具有强大的泛化能力。

### 使用yolo进行检测，使用sam进行分割

本项目将YOLO的目标检测能力与SAM的精细分割能力相结合。YOLO首先检测出目标位置和类别，然后将检测框输入SAM生成精确的分割掩码。

#### 代码下载，安装

1. 下载代码

   ```git clone https://github.com/xla145/yolo-sam-web.git```

2. 下载模型

   - 下载yolo模型

   访问 https://github.com/ultralytics/yolov8.git 下载模型

   - 下载sam模型

   访问 https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/tree/main 下载模型


2. 安装依赖

   ```pip install -r requirements.txt```

3. 运行代码

   ```python main.py```

4. 打开网站

   运行后会自动打开本地网页界面，一般为 http://localhost:7860

#### 结果展示

1. 输入图片
   上传任意包含人物的图片到网页界面

2. 显示检测和分割结果
   系统会自动显示检测框和分割结果，不同颜色表示不同目标

### 总结
本项目展示了如何结合YOLO和SAM的优势，实现既快速又精确的目标检测与分割。该方案具有良好的实用性，可以应用于各种计算机视觉任务中。


