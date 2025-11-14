"""
测试车牌识别系统中的各个函数
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

def test_preprocess_image():
    """测试图像预处理函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        print(f"测试图像: {test_file}, 尺寸: {image.shape}")
        
        # 测试预处理函数
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            print(f"图像 {test_file} 预处理失败")
            continue
        
        print(f"图像 {test_file} 预处理成功，处理后尺寸: {processed_image.shape}")
        
        # 保存处理后的图像用于检查，使用新的命名规范
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_1_preprocessed.jpg"
        cv2.imwrite(output_path, processed_image)
        print(f"预处理后的图像已保存到 {output_path}")
        
        success_count += 1
    
    print(f"图像预处理测试完成，成功处理 {success_count}/{len(test_files)} 张图像")
    return success_count == len(test_files)

def test_detect_edges():
    """测试边缘检测函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 先进行预处理
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            print(f"图像 {test_file} 预处理失败，跳过边缘检测")
            continue
        
        # 测试边缘检测函数
        edge_image = detect_edges(processed_image)
        
        if edge_image is None:
            print(f"图像 {test_file} 边缘检测失败")
            continue
        
        # 保存边缘检测结果，使用新的命名规范
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_2_edge.jpg"
        cv2.imwrite(output_path, edge_image)
        print(f"边缘检测结果已保存到 {output_path}")
        
        success_count += 1
    
    print(f"边缘检测测试完成，成功处理 {success_count}/{len(test_files)} 张图像")
    return success_count == len(test_files)

def test_apply_morphology():
    """测试形态学操作函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 先进行预处理和边缘检测
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            print(f"图像 {test_file} 预处理失败，跳过形态学操作")
            continue
        
        edge_image = detect_edges(processed_image)
        
        if edge_image is None:
            print(f"图像 {test_file} 边缘检测失败，跳过形态学操作")
            continue
        
        # 测试形态学操作函数
        morph_image = apply_morphology(edge_image)
        
        if morph_image is None:
            print(f"图像 {test_file} 形态学操作失败")
            continue
        
        # 保存形态学操作结果，使用新的命名规范
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_3_morphology.jpg"
        cv2.imwrite(output_path, morph_image)
        print(f"形态学操作结果已保存到 {output_path}")
        
        success_count += 1
    
    print(f"形态学操作测试完成，成功处理 {success_count}/{len(test_files)} 张图像")
    return success_count == len(test_files)

def test_detect_contours():
    """测试轮廓检测函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_contours = 0
    
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 先进行预处理、边缘检测和形态学操作
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            print(f"图像 {test_file} 预处理失败，跳过轮廓检测")
            continue
        
        edge_image = detect_edges(processed_image)
        
        if edge_image is None:
            print(f"图像 {test_file} 边缘检测失败，跳过轮廓检测")
            continue
        
        morph_image = apply_morphology(edge_image)
        
        if morph_image is None:
            print(f"图像 {test_file} 形态学操作失败，跳过轮廓检测")
            continue
        
        # 测试轮廓检测函数
        contours = detect_contours(morph_image)
        
        if contours is None:
            print(f"图像 {test_file} 轮廓检测失败")
            continue
        
        contour_count = len(contours)
        total_contours += contour_count
        print(f"图像 {test_file} 检测到 {contour_count} 个可能是车牌的轮廓")
        
        # 在原图上绘制所有轮廓（绿色）
        result_image = image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        
        # 保存轮廓检测结果，使用新的命名规范
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_4_contours.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"轮廓检测结果已保存到 {output_path}")
        
        # 提取最佳轮廓用于后续处理
        _, best_contour = extract_plate_region(image, contours)
        
        # 如果有最佳轮廓，创建一个额外的图像显示最佳轮廓（红色）
        if best_contour is not None:
            best_contour_image = image.copy()
            cv2.drawContours(best_contour_image, [best_contour], -1, (0, 0, 255), 3)
            output_path = f"{output_dir}/{base_name}_5_best_contour.jpg"
            cv2.imwrite(output_path, best_contour_image)
            print(f"最佳轮廓结果已保存到 {output_path}")
            
            # 直接进行车牌区域提取和矫正
            plate_region = extract_and_correct_plate_region(image, best_contour)
            if plate_region is not None:
                output_path = f"{output_dir}/{base_name}_6_plate.jpg"
                cv2.imwrite(output_path, plate_region)
                print(f"矫正后的车牌区域已保存到 {output_path}")
        
        success_count += 1
    
    print(f"轮廓检测测试完成，成功处理 {success_count}/{len(test_files)} 张图像，共检测到 {total_contours} 个轮廓")
    return success_count == len(test_files)

def test_preprocess_plate_region():
    """测试车牌区域预处理函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 先进行车牌定位
        processed_image = preprocess_image(image)
        edge_image = detect_edges(processed_image)
        morph_image = apply_morphology(edge_image)
        contours = detect_contours(morph_image)
        
        if not contours:
            print(f"图像 {test_file} 未检测到轮廓，跳过车牌区域预处理测试")
            continue
            
        # 提取最佳轮廓
        _, best_contour = extract_plate_region(image, contours)
        
        if best_contour is None:
            print(f"图像 {test_file} 未找到最佳轮廓，跳过车牌区域预处理测试")
            continue
            
        # 提取并矫正车牌区域
        plate_region = extract_and_correct_plate_region(image, best_contour)
        
        if plate_region is None:
            print(f"图像 {test_file} 车牌区域提取失败，跳过车牌区域预处理测试")
            continue
        
        # 测试车牌区域预处理函数
        preprocessed_plate = preprocess_plate_region(plate_region)
        
        if preprocessed_plate is None:
            print(f"图像 {test_file} 车牌区域预处理失败")
            continue
        
        print(f"图像 {test_file} 车牌区域预处理成功")
        
        # 保存预处理后的车牌区域
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_7_preprocessed_plate.jpg"
        cv2.imwrite(output_path, preprocessed_plate)
        print(f"预处理后的车牌区域已保存到 {output_path}")
        
        success_count += 1
    
    print(f"车牌区域预处理测试完成，成功处理 {success_count}/{len(test_files)} 张图像")
    return success_count == len(test_files)

def test_character_segmentation():
    """测试字符分割函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for test_file in test_files:
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
            print(f"图像 {test_file} 未检测到轮廓，跳过字符分割测试")
            continue
            
        # 提取最佳轮廓
        _, best_contour = extract_plate_region(image, contours)
        
        if best_contour is None:
            print(f"图像 {test_file} 未找到最佳轮廓，跳过字符分割测试")
            continue
            
        # 提取并矫正车牌区域
        plate_region = extract_and_correct_plate_region(image, best_contour)
        
        if plate_region is None:
            print(f"图像 {test_file} 车牌区域提取失败，跳过字符分割测试")
            continue
        
        # 预处理车牌区域
        preprocessed_plate = preprocess_plate_region(plate_region)
        
        if preprocessed_plate is None:
            print(f"图像 {test_file} 车牌区域预处理失败，跳过字符分割测试")
            continue
        
        # 测试字符边界检测
        print(f"开始检测图像 {test_file} 的字符边界...")
        boundaries = detect_character_boundaries(preprocessed_plate)
        
        if not boundaries:
            print(f"图像 {test_file} 字符边界检测失败")
            continue
        
        print(f"图像 {test_file} 检测到 {len(boundaries)} 个字符边界")
        
        # 创建一个图像来显示字符边界
        boundary_image = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR)
        for i, (start, end) in enumerate(boundaries):
            cv2.line(boundary_image, (start, 0), (start, preprocessed_plate.shape[0]), (0, 255, 0), 1)
            cv2.line(boundary_image, (end, 0), (end, preprocessed_plate.shape[0]), (0, 0, 255), 1)
        
        # 保存字符边界检测结果
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_8_character_boundaries.jpg"
        cv2.imwrite(output_path, boundary_image)
        print(f"字符边界检测结果已保存到 {output_path}")
        
        success_count += 1
    
    print(f"字符分割测试完成，成功处理 {success_count}/{len(test_files)} 张图像")
    return success_count == len(test_files)

def test_extract_plate_region():
    """测试车牌区域提取函数"""
    # 测试所有LPR目录下的图像
    test_dir = "../assets/images/LPR"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_files.sort()  # 确保按顺序测试
    
    # 确保output/LPR目录存在
    output_dir = "../output/LPR"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 先进行预处理、边缘检测、形态学操作和轮廓检测
        processed_image = preprocess_image(image)
        edge_image = detect_edges(processed_image)
        morph_image = apply_morphology(edge_image)
        contours = detect_contours(morph_image)
        
        if not contours:
            print(f"图像 {test_file} 未检测到轮廓，跳过车牌区域提取测试")
            continue
            
        # 测试车牌区域提取函数
        plate_image, best_contour = extract_plate_region(image, contours)
        
        if plate_image is None or best_contour is None:
            print(f"图像 {test_file} 车牌区域提取失败")
            continue
            
        print(f"图像 {test_file} 车牌区域提取成功")
        
        # 保存最佳轮廓结果
        base_name = os.path.splitext(test_file)[0]
        output_path = f"{output_dir}/{base_name}_5_best_contour.jpg"
        cv2.imwrite(output_path, plate_image)
        print(f"最佳轮廓结果已保存到 {output_path}")
        
        # 测试车牌区域矫正函数
        plate_region = extract_and_correct_plate_region(image, best_contour)
        
        if plate_region is None:
            print(f"图像 {test_file} 车牌区域矫正失败")
            continue
            
        print(f"图像 {test_file} 车牌区域矫正成功")
        
        # 保存矫正后的车牌区域
        output_path = f"{output_dir}/{base_name}_6_plate.jpg"
        cv2.imwrite(output_path, plate_region)
        print(f"矫正后的车牌区域已保存到 {output_path}")
        
        success_count += 1
    
    print(f"车牌区域提取与矫正测试完成，成功处理 {success_count}/{len(test_files)} 张图像")
    return success_count == len(test_files)

if __name__ == "__main__":
    print("测试图像预处理函数...")
    test_preprocess_image()
    print("\n测试边缘检测函数...")
    test_detect_edges()
    print("\n测试形态学操作函数...")
    test_apply_morphology()
    print("\n测试轮廓检测函数...")
    test_detect_contours()
    
    print("\n步骤5: 测试车牌区域提取与矫正函数")
    success = test_extract_plate_region()
    if success:
        print("车牌区域提取与矫正测试通过")
    else:
        print("车牌区域提取与矫正测试失败")
    
    print("\n步骤6: 测试车牌区域预处理函数")
    success = test_preprocess_plate_region()
    if success:
        print("车牌区域预处理测试通过")
    else:
        print("车牌区域预处理测试失败")
    
    print("\n步骤7: 测试字符分割函数")
    success = test_character_segmentation()
    if success:
        print("字符分割测试通过")
    else:
        print("字符分割测试失败")
    
    print("\n测试完成！")
    print("处理步骤说明：")
    print("1. 预处理：灰度化、高斯模糊、直方图均衡化")
    print("2. 边缘检测：使用Canny算子检测边缘")
    print("3. 形态学操作：闭操作和膨胀操作")
    print("4. 轮廓检测：检测可能是车牌的轮廓（绿色框）")
    print("5. 最佳轮廓：显示最终选择的最佳轮廓（红色框）")
    print("6. 车牌区域：提取并矫正后的车牌区域")
    print("7. 车牌区域预处理：对车牌区域进行进一步处理")
    print("8. 字符分割：分割车牌中的字符")