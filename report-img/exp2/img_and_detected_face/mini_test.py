import os
import glob
import matplotlib.pyplot as plt
import face_recognition
import time
from PIL import Image

# 设置图片文件夹路径
image_folder = './test_img'

# 获取文件夹中所有.jpg图片的路径
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

# 遍历所有图片路径
for image_path in image_paths:
    a = time.time()  # 记录开始时间

    # 通过PIL加载图片
    image = face_recognition.load_image_file(image_path)

    # 基于cnn识别人脸，选择使用cpu
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    b = time.time()  # 记录结束时间

    # 计算并打印识别耗时
    time_elapsed = b - a
    print(f"识别图片 {image_path} 耗时: {time_elapsed}秒")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    # 如果检测到人脸，打印人脸信息并显示
    if face_locations:
        for face_location in face_locations:
            # 打印人脸信息
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # 提取人脸
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)

            # 显示人脸图像
            plt.imshow(pil_image)
            plt.axis('off')
            plt.title("Detected Face")
            plt.show()
    else:
        print("No faces found in this photograph.")