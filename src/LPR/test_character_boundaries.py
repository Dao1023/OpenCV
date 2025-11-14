"""
测试字符边界检测函数
"""

import cv2
import numpy as np
import os

# 导入车牌识别模块中的函数
from license_plate_recognition import (
    preprocess_image,
    detect_edges,
    apply_morphology,
    detect_contours,
    extract_plate_region,
    extract_and_correct_plate_region,
    preprocess_plate_region,
    detect_character_boundaries
)

def test_character_boundaries():
    """测试字符边界检测函数"""
    # 测试所有图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    for test_file in test_files:
        print(f"\n测试图像: {test_file}")
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 先进行车牌定位和预处理
        processed_image = preprocess_image(image)
        edge_image = detect_edges(processed_image)
        morph_image = apply_morphology(edge_image)
        contours = detect_contours(morph_image)
        
        if not contours:
            print(f"图像 {test_file} 未检测到轮廓")
            continue
            
        # 提取最佳轮廓
        _, best_contour = extract_plate_region(image, contours)
        
        if best_contour is None:
            print(f"图像 {test_file} 未找到最佳轮廓")
            continue
            
        # 提取并矫正车牌区域
        plate_region = extract_and_correct_plate_region(image, best_contour)
        
        if plate_region is None:
            print(f"图像 {test_file} 车牌区域提取失败")
            continue
        
        # 预处理车牌区域
        preprocessed_plate = preprocess_plate_region(plate_region)
        
        if preprocessed_plate is None:
            print(f"图像 {test_file} 车牌区域预处理失败")
            continue
        
        # 测试字符边界检测
        print(f"开始检测图像 {test_file} 的字符边界...")
        boundaries = detect_character_boundaries(preprocessed_plate)
        
        if not boundaries:
            print(f"图像 {test_file} 字符边界检测失败")
            continue
        
        print(f"图像 {test_file} 最终检测到 {len(boundaries)} 个字符边界")
        
        # 创建一个图像来显示字符边界
        boundary_image = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR)
        for i, (start, end) in enumerate(boundaries):
            cv2.line(boundary_image, (start, 0), (start, preprocessed_plate.shape[0]), (0, 255, 0), 1)
            cv2.line(boundary_image, (end, 0), (end, preprocessed_plate.shape[0]), (0, 0, 255), 1)
        
        # 保存字符边界检测结果
        output_dir = "../output/LPR"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{test_file}_debug_boundaries.jpg"
        cv2.imwrite(output_path, boundary_image)
        print(f"字符边界检测结果已保存到 {output_path}")
    
    return True

if __name__ == "__main__":
    test_character_boundaries()