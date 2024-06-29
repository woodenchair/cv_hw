import os
import glob
import numpy as np
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
import time

# 激活函数
def activation_function(iou):
    threshold = 0.8
    return iou >= threshold

# 定义计算IoU的函数
def calculate_iou(box1, box2):
    # 计算两个边界框的交集
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

# 定义计算检测率和误报率的函数
def calculate_detection_metrics(true_positives, false_positives, total_images):
    detection_rate = true_positives / total_images if total_images > 0 else 0
    false_positive_rate = false_positives / (total_images - true_positives) if total_images > 0 else 0
    return detection_rate, false_positive_rate

# 定义计算FPS的函数
def calculate_fps(time_start, time_end):
    return 1 / (time_end - time_start)

# 处理图片和标签数据
def process_images_and_labels(image_dir, label_dir, output_dir):
    true_positives = 0
    false_positives = 0
    total_images = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_filename in sorted(glob.glob(os.path.join(image_dir, '*.jpg'))):
        # 使用face_recognition库加载图像
        image = face_recognition.load_image_file(image_filename)
        assert image is not None, f"Failed to load image: {image_filename}"

        # 获取图像的高度和宽度
        height, width = image.shape[:2]

        image_name = os.path.basename(image_filename)
        label_filename = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

        with open(label_filename, 'r') as f:
            line = f.readline()
            if not line:
                raise ValueError(f"Label file '{label_filename}' is empty or does not contain enough data.")
            labels = line.split()  # 分割行数据
            labels = [float(x) for i, x in enumerate(labels) if i != 0]

        # 计算真实边界框
        true_box = [labels[1] * width, labels[2] * height,
                    labels[3] * width, labels[4] * height]

        # 使用face_recognition进行人脸检测
        face_locations = face_recognition.face_locations(image)

        ious = []
        for face_location in face_locations:
            iou = calculate_iou(face_location, true_box)
            ious.append(iou)
            # 使用激活函数判断识别是否成功
            if activation_function(iou):
                true_positives += 1
            else:
                false_positives += 1

        total_images += 1

        output_filename = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))
        with open(output_filename, 'w') as f:
            for iou in ious:
                f.write(f"{image_name} IoU: {iou:.2f}, Success: {activation_function(iou)}\n")

        print(f"IoU values for {image_name} have been successfully saved to {output_filename}")

    return true_positives, false_positives, total_images

def plot_roc_curve(true_positive_rates, false_positive_rates):
    """
    绘制ROC曲线。

    参数:
    true_positive_rates -- 真阳性率的列表或数组
    false_positive_rates -- 假阳性率的列表或数组
    """
    # 检查输入是否为列表或数组
    if not (isinstance(true_positive_rates, (list, np.ndarray)) and isinstance(false_positive_rates,
                                                                               (list, np.ndarray))):
        raise ValueError("True positive rates and false positive rates must be lists or numpy arrays.")

    # 检查两个参数的长度是否相同
    if len(true_positive_rates) != len(false_positive_rates):
        raise ValueError("The lengths of true_positive_rates and false_positive_rates must be the same.")

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(false_positive_rates, true_positive_rates, label='ROC curve', color='blue', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linestyle='--', linewidth=2)  # 随机分类器的ROC曲线

    # 设置图表标题和轴标签
    plt.title('Receiver Operating Characteristic Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # 优化图表显示
    plt.grid(True)
    plt.axis('square')

    # 显示图表
    plt.show()

def main():
    image_dir = r'celeba\images\test'
    label_dir = r'celeba\lables\test'
    output_dir = r'face-location'

    time_start = time.time()
    true_positives, false_positives, total_images = process_images_and_labels(image_dir, label_dir, output_dir)
    time_end = time.time()

    fps = calculate_fps(time_start, time_end)
    detection_rate, false_positive_rate = calculate_detection_metrics(true_positives, false_positives, total_images)

    print(f"Detection Rate: {detection_rate:.2f}")
    print(f"False Positive Rate: {false_positive_rate:.2f}")
    print(f"FPS: {fps:.2f}")

    # 这里需要填充roc_data以绘制ROC曲线
    roc_data = []  # 这里需要填充数据
    if roc_data:
        plot_roc_curve(*zip(*roc_data))

if __name__ == "__main__":
    main()