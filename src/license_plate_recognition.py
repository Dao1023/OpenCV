"""
车牌识别系统
包含12个核心函数，分别对应车牌定位、字符分割和字符识别的各个步骤
"""

import cv2
import numpy as np


# 车牌定位模块 (5个函数)

def preprocess_image(image):
    """
    图像预处理（灰度化、降噪）
    参数:
        image: 输入的原始图像
    返回:
        预处理后的图像
    """
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 高斯滤波降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 直方图均衡化增强对比度
    equalized = cv2.equalizeHist(blurred)
    
    return equalized


def detect_edges(image):
    """
    边缘检测
    参数:
        image: 预处理后的图像
    返回:
        边缘检测结果
    """
    # 使用Canny边缘检测
    # 自动计算阈值
    median = np.median(image)
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    
    # Canny边缘检测
    edges = cv2.Canny(image, lower, upper)
    
    return edges


def apply_morphology(edges):
    """
    形态学操作
    参数:
        edges: 边缘检测结果
    返回:
        形态学操作后的图像
    """
    # 创建一个较小的矩形结构元素，减少模糊效果
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))  # 从(15, 5)减小到(5, 2)
    
    # 进行闭操作（先膨胀后腐蚀），减少迭代次数
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)  # 减少迭代次数
    
    # 进行轻微的膨胀操作，减少迭代次数
    dilated = cv2.dilate(closed, kernel, iterations=1)  # 减少迭代次数
    
    return dilated


def detect_contours(morph_image):
    """
    轮廓检测
    参数:
        morph_image: 形态学操作后的图像
    返回:
        检测到的轮廓列表
    """
    # 查找轮廓
    contours, _ = cv2.findContours(morph_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选可能是车牌的轮廓
    plate_contours = []
    image_area = morph_image.shape[0] * morph_image.shape[1]
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 过滤掉太小的轮廓 - 提高最小面积阈值
        if area < image_area * 0.002:  # 面积小于整个图像的0.2%
            continue
            
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 车牌的宽高比通常在2.5到5.5之间 - 缩小范围
        aspect_ratio = w / h
        if aspect_ratio < 2.5 or aspect_ratio > 5.5:
            continue
            
        # 车牌的面积不能太小 - 提高最小面积阈值
        if area < image_area * 0.01:  # 面积小于整个图像的1%
            continue
            
        # 车牌的面积也不能太大 - 添加最大面积限制
        if area > image_area * 0.2:  # 面积大于整个图像的20%
            continue
            
        # 计算轮廓的矩形度（轮廓面积与边界矩形面积的比值）
        rect_area = w * h
        rectity = area / rect_area if rect_area > 0 else 0
        
        # 车牌通常是矩形，矩形度应该较高
        if rectity < 0.6:  # 矩形度小于60%
            continue
            
        # 计算轮廓的凸包面积比（轮廓面积与凸包面积的比值）
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # 车牌轮廓应该相对凸出
        if convexity < 0.8:  # 凸包面积比小于80%
            continue
            
        plate_contours.append(contour)
    
    # 如果没有找到合适的轮廓，返回面积最大的几个轮廓
    if not plate_contours:
        # 按面积排序，返回最大的3个轮廓
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        plate_contours = contours_sorted[:3]
    
    return plate_contours


def extract_plate_region(image, contours):
    """
    车牌区域提取
    参数:
        image: 原始图像
        contours: 检测到的轮廓列表
    返回:
        最可能是车牌的区域图像和对应的轮廓
    """
    if not contours:
        return None, None
    
    # 计算每个轮廓的评分
    best_contour = None
    best_score = 0
    
    for contour in contours:
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算宽高比
        aspect_ratio = w / h
        
        # 计算面积
        area = cv2.contourArea(contour)
        
        # 计算矩形度（轮廓面积与边界矩形面积的比值）
        rect_area = w * h
        rectity = area / rect_area if rect_area > 0 else 0
        
        # 计算评分（综合考虑宽高比、面积和矩形度）
        # 理想宽高比在2.5-4.5之间
        aspect_score = 1.0 - abs(aspect_ratio - 3.5) / 3.5 if aspect_ratio < 7 else 0
        
        # 面积评分（相对于图像面积）
        image_area = image.shape[0] * image.shape[1]
        area_score = min(area / image_area * 100, 1.0)
        
        # 矩形度评分（车牌通常是矩形）
        rectity_score = rectity
        
        # 综合评分
        total_score = (aspect_score * 0.4 + area_score * 0.3 + rectity_score * 0.3)
        
        # 更新最佳轮廓
        if total_score > best_score:
            best_score = total_score
            best_contour = contour
    
    if best_contour is None:
        return None, None
    
    # 提取车牌区域
    x, y, w, h = cv2.boundingRect(best_contour)
    
    # 扩展边界，确保包含完整的车牌
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    # 提取车牌区域
    plate_region = image[y1:y2, x1:x2]
    
    return plate_region, best_contour


# 字符分割模块 (4个函数)

def preprocess_plate_region(plate_image):
    """
    车牌区域预处理
    参数:
        plate_image: 车牌区域图像
    返回:
        预处理后的车牌图像
    """
    pass


def detect_character_boundaries(plate_image):
    """
    字符边界检测
    参数:
        plate_image: 预处理后的车牌图像
    返回:
        字符边界信息
    """
    pass


def extract_character_images(plate_image, boundaries):
    """
    字符图像提取
    参数:
        plate_image: 车牌图像
        boundaries: 字符边界信息
    返回:
        单个字符图像列表
    """
    pass


def normalize_characters(char_images):
    """
    字符标准化
    参数:
        char_images: 字符图像列表
    返回:
        标准化后的字符图像列表
    """
    pass


# 字符识别模块 (3个函数)

def extract_character_features(char_images):
    """
    字符特征提取
    参数:
        char_images: 标准化后的字符图像列表
    返回:
        字符特征列表
    """
    pass


def classify_characters(features):
    """
    字符分类器实现
    参数:
        features: 字符特征列表
    返回:
        识别结果列表
    """
    pass


def postprocess_recognition_results(results):
    """
    识别结果后处理
    参数:
        results: 字符识别结果列表
    返回:
        最终车牌号码
    """
    pass


def main(image_path):
    """
    主函数，整合所有步骤完成车牌识别
    参数:
        image_path: 输入图像路径
    返回:
        识别结果（车牌号）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 车牌定位流程
    preprocessed_image = preprocess_image(image)
    edge_image = detect_edges(preprocessed_image)
    morph_image = apply_morphology(edge_image)  # 修正函数名
    contours = detect_contours(morph_image)
    plate_image = extract_plate_region(image, contours)
    
    if plate_image is None:
        print("未能检测到车牌区域")
        return None
    
    # 字符分割流程
    preprocessed_plate = preprocess_plate_region(plate_image)
    boundaries = detect_character_boundaries(preprocessed_plate)
    char_images = extract_character_images(preprocessed_plate, boundaries)
    normalized_chars = normalize_characters(char_images)
    
    # 字符识别流程
    features = extract_character_features(normalized_chars)
    char_results = classify_characters(features)
    license_plate = postprocess_recognition_results(char_results)
    
    return license_plate


if __name__ == "__main__":
    # 测试代码
    test_image_path = "../assets/images/LPR/1.jpg"  # 测试图像路径
    result = main(test_image_path)
    if result:
        print(f"识别结果: {result}")
    else:
        print("识别失败")