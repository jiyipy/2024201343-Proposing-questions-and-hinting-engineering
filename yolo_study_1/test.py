from ultralytics import YOLO
import cv2
import numpy as np

# 加载训练好的 YOLO 模型
a1 = YOLO(r'runs\detect\train\weights\best.pt')

# 捕获视频（如果使用视频源，0 表示摄像头）
cap = cv2.VideoCapture(0)  # 或者替换成你的视频文件路径，例如 'your_video.mp4'

while True:
    ret, color_img = cap.read()  # 获取视频帧（彩色图像）

    if not ret:
        break  # 如果没有读取到图像，退出

    # 推理检测（包括获取掩码）
    results = a1(color_img)

    # 获取掩码数据
    masks = results[0].masks  # 获取推理结果的掩码数据

    # 如果掩码数据不为空
    if masks is not None:
        for mask in masks:
            mask_image = mask.cpu().numpy()  # 获取每个物体的掩码，转换为 NumPy 数组

            # 将掩码应用到原图像上（提取物体）
            masked_image = color_img * mask_image[:, :, np.newaxis]  # 扩展掩码形状，适配图像

            # 可视化掩码（将掩码显示为白色物体，背景为黑色）
            cv2.imshow('Masked Image', masked_image)  # 显示带掩码的图像

    # 可视化推理结果（掩码和边界框）
    annotated_frame = results[0].plot(show_labels=False)  # 显示掩码和检测结果，禁用标签显示
    cv2.imshow("YOLO Inference with Masks", annotated_frame)

    # 按键 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()
