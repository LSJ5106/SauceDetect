import cv2
import os
import LIME.demo as demo

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

if __name__ == '__main__':
    input_dir = r"../datasets/VOC2007/JPEGImages_cut"
    output_dir = r"../datasets\VOC2007\JPEGImages_UELIE"


    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir,filename)
        print("file_path: ", file_path)
        img = cv2.imread(file_path)

        # 求图片亮度
        mean = cv2.mean(img)

        # 对于亮度不足的图片
        if sum(mean) / 3 <= 120:
            print("LIME")
            demo.LIME_api_single(file_path, filename, output_dir)

        # 对于存在过曝的图片
        elif sum(mean) / 3 >= 180:
            print("OverExposure")
            img = reduce_overexposure(img)
            save_path = os.path.join(output_dir, filename)
            print("Save to : ", save_path)
            cv2.imwrite(save_path, img)