import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class DarkChannel:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.processed_image = None
        self.dark_image = None
        self.patch_size = 15
        self.omega = 0.95
        self.result = None  # 大气光值
        self.t_values = None
        self.lamda = 1e-4

    def read_image(self):
        """读取图片"""
        self.image = cv2.imread(self.image_path)

    def remove_haze(self):
        """去雾处理"""
        height, width, channels = self.image.shape

        self.dark_image = np.zeros((height, width), dtype=np.uint8)
        self.t_values = np.zeros((height, width))
        dehazed_image = np.zeros_like(self.image)

        for y in range(height):
            for x in range(width):
                window_size = self.patch_size // 2

                x_start = max(0, x - window_size)
                x_end = min(width - window_size, x + window_size)
                y_start = max(0, y - window_size)
                y_end = min(height - window_size, y + window_size)
                # 提取局部窗口
                local_window = self.image[y_start:y_end + 1, x_start:x_end + 1]

                # 计算得到暗通道
                pixel_min_rgbs = np.min(local_window, axis=(1, 2))
                global_min_rgb = np.min(pixel_min_rgbs)
                self.dark_image[y, x] = global_min_rgb

        num_brightest = int(0.001 * self.dark_image.size)

        # 图像进行分区排序
        brightest_indices = np.argpartition(self.dark_image.flatten(), -num_brightest)[-num_brightest:]
        brightest_coordinates = np.unravel_index(brightest_indices, self.dark_image.shape)
        corresponding_pixels = self.image[brightest_coordinates]

        # 计算原始图像中这些像素点的 RGB 通道均值
        avg_red = np.mean(corresponding_pixels[:, 0])
        avg_green = np.mean(corresponding_pixels[:, 1])
        avg_blue = np.mean(corresponding_pixels[:, 2])

        result = {'R': avg_red, 'G': avg_green, 'B': avg_blue}
        self.result = result
        # # 打印结果字典
        # print(result)

        for y in range(height):
            for x in range(width):
                window_size = self.patch_size // 2
                # 提取局部窗口
                x_start = max(0, x - window_size)
                x_end = min(width - window_size, x + window_size)
                y_start = max(0, y - window_size)
                y_end = min(height - window_size, y + window_size)

                local_window = self.image[y_start:y_end + 1, x_start:x_end + 1]
                r_ratio = local_window[:, :, 0] / result['R']
                g_ratio = local_window[:, :, 1] / result['G']
                b_ratio = local_window[:, :, 2] / result['B']

                min_ratios = np.min([r_ratio, g_ratio, b_ratio], axis=0)
                local_min_ratio = np.min(min_ratios)
                # 计算 t[k]
                self.t_values[y, x] = 1.0 - local_min_ratio * self.omega

        # 去除伪影
        self.guided_filter()

        for y in range(height):
            for x in range(width):
                # 获取原图像中当前像素在RGB三通道上的值
                r_val = self.image[y, x, 0]
                g_val = self.image[y, x, 1]
                b_val = self.image[y, x, 2]

                # 计算去雾后的RGB三通道值
                jr = (r_val - result['R']) / self.t_values[y, x] + result['R']
                jg = (g_val - result['G']) / self.t_values[y, x] + result['G']
                jb = (b_val - result['B']) / self.t_values[y, x] + result['B']

                # 将计算得到的值存储到去雾后的图片
                dehazed_image[y, x, 0] = np.clip(jr, 0, 255)  # 红通道
                dehazed_image[y, x, 1] = np.clip(jg, 0, 255)  # 绿通道
                dehazed_image[y, x, 2] = np.clip(jb, 0, 255)  # 蓝通道
        self.processed_image = dehazed_image

    def guided_filter(self):
        r = 20
        d = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype('float64') / 255
        # i = self.image
        t = self.t_values
        eps = 1e-4
        # 1
        mean_I = cv2.blur(d , (r, r))
        mean_p = cv2.blur(t , (r, r))
        corr_I = cv2.blur(d * d , (r, r))
        corr_Ip = cv2.blur(d * t , (r, r))
        # 2
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        # 3
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        # 4
        mean_a = cv2.blur(a , (r, r))
        mean_b = cv2.blur(b , (r, r))
        # 5
        q = mean_a * d + mean_b
        q = cv2.max(q, 0.25)
        self.t_values = q

    def show_image(self):
        """展示原始图片和处理后的图片"""
        if self.image is None:
            print("请先读取图片")
            return

        # 展示原始图片
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # 如果处理后的图片已经存在，则展示处理后的图片
        if self.processed_image is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            plt.title('Processed Image')
            plt.axis('off')
        plt.show()

    def save_image(self, image_name='processed_image.png'):
        """保存处理后的图片到指定路径"""
        if self.processed_image is None:
            print("处理后的图片不存在，请先处理图片。")
            return

        # 将处理后的图片保存到指定路径
        success = cv2.imwrite('./img/' + image_name, self.processed_image)

        if success:
            print(f"处理后的图片已保存到 {image_name}")
        else:
            print("保存图片时出现错误。")


def process_images_in_directory(directory_path, results_path):
    # 确保结果文件夹存在
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # 获取目录中的所有图片文件
    for image_name in os.listdir(directory_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory_path, image_name)
            # 创建DarkChannel实例并处理图片
            dark_channel = DarkChannel(file_path)
            dark_channel.read_image()
            dark_channel.remove_haze()
            dehazed_image_name = f'dehazed_{image_name}'
            dark_channel.save_image(os.path.join(results_path, dehazed_image_name))
            print(f"Processed and saved {dehazed_image_name}")


def main():
    directory_path = "./img/latex"
    results_path = "./results"

    process_images_in_directory(directory_path, results_path)
    print("Done!")


if __name__ == "__main__":
    main()