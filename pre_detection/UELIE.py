import cv2
import os
import LIME.demo as demo
import time
from concurrent.futures import ThreadPoolExecutor



# 减少图片局部过曝
def reduce_overexposure(img):
    # Convert image to YUV color space
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Split YUV channels
    y, u, v = cv2.split(yuv)

    # Apply adaptive histogram equalization to Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    y = clahe.apply(y)

    # Merge YUV channels
    yuv = cv2.merge((y, u, v))

    # Convert back to BGR color space
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return result

def process_image(file_path, filename, output_dir):
    img = cv2.imread(file_path)
    # mean = cv2.mean(img)[0]  # 仅使用亮度通道进行亮度计算
    # 求图片亮度
    mean = cv2.mean(img)
    # print("图片亮度：", sum(mean) / 3)

    if mean <= 120:
        print("亮度较低，使用LIME")
        demo.LIME_api_single(file_path, filename, output_dir)

    elif mean >= 180:
        print("过曝，减少过曝")
        img = reduce_overexposure(img)
        save_path = os.path.join(output_dir, filename)
        print("保存到: ", save_path)
        cv2.imwrite(save_path, img)
    else:
        print("正常图片")
        save_path = os.path.join(output_dir, filename)
        print("保存到: ", save_path)
        cv2.imwrite(save_path, img)

if __name__ == '__main__':
    threshold_value1 = 115
    threshold_value2 = 180

    T1 = time.time()

    input_dir = r"../cut_datasets/VOC2007/JPEGImages"
    output_dir = os.path.join("../datasets_UELIE/VOC2007/",
                              str(threshold_value1) + "_" + str(threshold_value2)
                              )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 控制线程数，例如设置为8，单图耗时最短 99.82252 ms
    max_workers = 8

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            print("文件路径: ", file_path)

            # 将图像处理任务提交给ThreadPoolExecutor
            executor.submit(process_image, file_path, filename, output_dir)



    T2 = time.time()
    print("每张图像的处理时间", (T2 - T1) / len(os.listdir(input_dir)) * 1000, "毫秒")
    print("FPS: ", 1 / ((T2 - T1) / len(os.listdir(input_dir))))