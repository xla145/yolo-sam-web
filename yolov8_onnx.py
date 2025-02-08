# Ultralytics YOLO 🚀, AGPL-3.0 license
import os

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres, classes):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = classes
        self.pad_height = 0
        self.pad_width = 0

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, _results):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        for box, score, class_id in _results:
            # Extract the coordinates of the bounding box
            x1, y1, w, h = box

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {score:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = input_image

        # Get the height and width of the input image
        self.img_height, self.img_width, chanel = self.img.shape[:3]

        # Check if the image has 3 channels (RGB) or 1 channel (grayscale)
        if chanel == 3:
            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        elif chanel == 1:
            # Convert the single-channel image to RGB by duplicating the channel
            img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        else:
            # Handle unsupported image formats (e.g., images with more than 3 channels)
            raise ValueError("Unsupported image format. Expected 3 or 1 channels.")

        # Calculate the aspect ratio of the input image
        aspect_ratio = self.img_width / self.img_height

        # Resize the image to match the input shape while preserving the aspect ratio
        if aspect_ratio > (self.input_width / self.input_height):
            new_height = int(self.input_width / aspect_ratio)
            img = cv2.resize(img, (self.input_width, new_height))
        else:
            new_width = int(self.input_height * aspect_ratio)
            img = cv2.resize(img, (new_width, self.input_height))

        # Pad the resized image to match the input shape
        pad_height = self.input_height - img.shape[0]
        pad_width = self.input_width - img.shape[1]

        self.pad_height = pad_height
        self.pad_width = pad_width

        img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        x_factor = self.img_width / self.input_width

        # Calculate the scaling factors for the bounding box coordinates
        if self.img_width < self.img_height:
            x_factor = self.img_height / self.input_width

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4: 4 + len(self.classes)]  # 因为前四个是坐标
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)  # 最大的分数
            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)  # 最大的分数的索引

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][:4]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * x_factor)
                width = int(w * x_factor)
                height = int(h * x_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(float(max_score))
                boxes.append([left, top, width, height])
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        # Iterate over the selected indices after non-maximum suppression
        _results = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            _results.append((box, score, class_id))
        return _results

    def predict(self, input_image):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        # Preprocess the image data
        img_data = self.preprocess(input_image)
        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})
        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(outputs)  # output image


if __name__ == "__main__":
    pass


