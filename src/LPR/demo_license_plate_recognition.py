"""
车牌识别系统演示
使用优化后的字符边界检测算法
"""

import cv2
import os
import sys
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from license_plate_recognition import preprocess_image, detect_edges, apply_morphology, detect_contours, extract_plate_region, preprocess_plate_region, detect_character_boundaries

def demo_license_plate_recognition(image_path):
    """
    演示车牌识别系统
    :param image_path: 输入图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"处理图像: {image_path}")
    
    # 1. 预处理图像
    preprocessed = preprocess_image(image)
    
    # 2. 边缘检测
    edges = detect_edges(preprocessed)
    
    # 3. 形态学操作
    morph = apply_morphology(edges)
    
    # 4. 轮廓检测
    contours = detect_contours(morph)
    
    # 5. 提取车牌区域
    plate_img, plate_contour = extract_plate_region(image, contours)
    if plate_img is None:
        print("未检测到车牌区域")
        return
    
    print("成功检测到车牌区域")
    
    # 6. 预处理车牌图像
    preprocessed_plate = preprocess_plate_region(plate_img)
    print("完成车牌预处理")
    
    # 7. 检测字符边界
    boundaries = detect_character_boundaries(preprocessed_plate)
    print(f"检测到 {len(boundaries)} 个字符边界")
    
    # 4. 可视化结果
    result_img = plate_img.copy()
    
    # 在车牌上绘制字符边界
    height, width = result_img.shape[:2]
    for i, (start, end) in enumerate(boundaries):
        x1 = int(start * width)
        x2 = int(end * width)
        cv2.rectangle(result_img, (x1, 0), (x2, height), (0, 255, 0), 2)
        cv2.putText(result_img, str(i+1), (x1+5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 车牌区域
    plt.subplot(3, 2, 2)
    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    plt.title('车牌区域')
    plt.axis('off')
    
    # 预处理后的车牌
    plt.subplot(3, 2, 3)
    plt.imshow(preprocessed_plate, cmap='gray')
    plt.title('预处理后的车牌')
    plt.axis('off')
    
    # 字符边界检测结果
    plt.subplot(3, 2, 4)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'字符边界检测结果 ({len(boundaries)} 个字符)')
    plt.axis('off')
    
    # 垂直投影
    plt.subplot(3, 2, 5)
    # 计算垂直投影
    vertical_projection = np.sum(preprocessed_plate, axis=0)
    plt.plot(vertical_projection)
    plt.title('垂直投影')
    plt.xlabel('水平位置')
    plt.ylabel('像素强度和')
    
    # 字符边界位置
    plt.subplot(3, 2, 6)
    plt.imshow(preprocessed_plate, cmap='gray')
    for i, (start, end) in enumerate(boundaries):
        x1 = int(start * width)
        x2 = int(end * width)
        plt.axvline(x=x1, color='r', linestyle='--')
        plt.axvline(x=x2, color='r', linestyle='--')
    plt.title('字符边界位置')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = "../output/LPR/demo"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"demo_{os.path.basename(image_path)}")
    plt.savefig(output_path, dpi=300)
    print(f"演示结果已保存到: {output_path}")
    
    # 不显示图形界面，直接关闭
    plt.close()

if __name__ == "__main__":
    import numpy as np
    
    # 测试图像路径
    test_images = [
        "../assets/images/LPR/1.jpg",
        "../assets/images/LPR/2.jpg",
        "../assets/images/LPR/3.jpg",
        "../assets/images/LPR/4.jpg",
        "../assets/images/LPR/5.jpg"
    ]
    
    # 对每个测试图像进行演示
    for image_path in test_images:
        if os.path.exists(image_path):
            demo_license_plate_recognition(image_path)
        else:
            print(f"图像不存在: {image_path}")