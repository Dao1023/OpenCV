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


def extract_and_correct_plate_region(image, contour):
    """
    提取并矫正车牌区域
    参数:
        image: 原始图像
        contour: 最佳轮廓
    返回:
        矫正后的车牌区域图像
    """
    # 获取轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(contour)
    
    # 扩展边界，确保包含完整的车牌
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    # 提取车牌区域
    plate_region = image[y1:y2, x1:x2]
    
    # 尝试进行透视矫正
    try:
        # 获取轮廓的近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果近似多边形有4个顶点，尝试透视变换
        if len(approx) == 4:
            # 对顶点进行排序
            rect = order_points(approx.reshape(4, 2))
            
            # 计算目标矩形的宽度和高度
            width_a = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width_b = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            
            height_a = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
            height_b = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            
            # 定义目标矩形的四个顶点
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]], dtype="float32")
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # 应用透视变换
            warped = cv2.warpPerspective(image, M, (max_width, max_height))
            
            return warped
        else:
            # 如果无法进行透视变换，返回原始区域
            return plate_region
    except Exception as e:
        # 如果出错，返回原始区域
        print(f"透视变换失败: {e}")
        return plate_region


def order_points(pts):
    """
    对四个点进行排序，使其按照左上、右上、右下、左下的顺序排列
    参数:
        pts: 四个点的坐标
    返回:
        排序后的四个点
    """
    # 计算点的和
    s = pts.sum(axis=1)
    # 和最小的点是左上角
    tl = pts[np.argmin(s)]
    # 和最大的点是右下角
    br = pts[np.argmax(s)]
    
    # 计算点的差
    diff = np.diff(pts, axis=1)
    # 差最小的点是左上角
    tr = pts[np.argmin(diff)]
    # 差最大的点是右下角
    bl = pts[np.argmax(diff)]
    
    # 返回排序后的点
    return np.array([tl, tr, br, bl], dtype="float32")


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
    车牌区域预处理 - 改进版本
    参数:
        plate_image: 车牌区域图像
    返回:
        预处理后的车牌图像
    """
    # 转换为灰度图像
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image.copy()
    
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 高斯滤波去噪
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 自适应阈值二值化 - 使用更小的块大小以获得更精细的结果
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 7, 3)
    
    # 形态学操作 - 先闭操作连接断裂部分，再开操作去除小噪点
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    processed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return processed


def detect_character_boundaries(plate_image):
    """
    字符边界检测 - 简单但鲁棒的版本
    参数:
        plate_image: 预处理后的车牌图像
    返回:
        字符边界信息
    """
    height, width = plate_image.shape
    
    # 1. 图像增强 - 应用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(plate_image)
    
    # 2. 垂直投影分析
    # 计算垂直投影
    vertical_proj = np.sum(enhanced, axis=0)
    
    # 平滑处理
    vertical_proj = cv2.GaussianBlur(vertical_proj.astype(np.float32), (5, 1), 0)
    
    # 3. 自适应阈值计算 - 使用简单的百分位数方法
    # 使用较低的百分位数作为阈值，确保不漏掉字符
    threshold = np.percentile(vertical_proj, 30)  # 使用30%分位数作为阈值
    
    # 确保阈值不低于最小值
    min_threshold = np.mean(vertical_proj) * 0.5
    threshold = max(threshold, min_threshold)
    
    # 4. 寻找字符边界
    boundaries = []
    in_char = False
    start = 0
    
    # 动态最小字符宽度（基于图像宽度）
    min_char_width = max(5, int(width * 0.05))  # 增加最小宽度要求
    
    for x in range(width):
        if vertical_proj[x] > threshold and not in_char:
            # 进入字符区域
            start = x
            in_char = True
        elif vertical_proj[x] <= threshold and in_char:
            # 离开字符区域
            end = x
            in_char = False
            
            # 过滤太窄的区域
            if end - start > min_char_width:
                boundaries.append((start, end))
    
    # 处理最后一个字符
    if in_char:
        end = width - 1
        if end - start > min_char_width:
            boundaries.append((start, end))
    
    # 5. 如果字符太少，尝试轮廓检测
    if len(boundaries) < 5:
        # 应用形态学操作，连接断裂的字符
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 轮廓检测
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按x坐标排序
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        contour_boundaries = []
        min_char_width = max(5, int(width * 0.05))
        min_char_height = max(5, int(height * 0.3))
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤太小的轮廓
            if w < min_char_width or h < min_char_height:
                continue
            
            # 检查宽高比
            aspect_ratio = h / w
            if aspect_ratio < 0.5:  # 字符应该有一定的高度
                continue
            
            contour_boundaries.append((x, x + w))
        
        # 如果轮廓方法检测到更多字符，使用轮廓方法
        if len(contour_boundaries) >= len(boundaries):
            boundaries = contour_boundaries
    
    # 6. 边界后处理
    if boundaries:
        # 按x坐标排序
        boundaries.sort(key=lambda b: b[0])
        
        # 计算字符宽度的中位数
        widths = [b[1] - b[0] for b in boundaries]
        median_width = np.median(widths) if widths else width * 0.1
        
        # 合并过近的边界
        merged_boundaries = [boundaries[0]]
        
        for current in boundaries[1:]:
            last = merged_boundaries[-1]
            gap = current[0] - last[1]
            
            # 如果间距小于字符宽度的一半，合并
            if gap < median_width * 0.5:
                merged_boundaries[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_boundaries.append(current)
        
        # 分割过宽的边界
        final_boundaries = []
        
        for boundary in merged_boundaries:
            start, end = boundary
            boundary_width = end - start
            
            # 如果边界太宽，可能是多个字符合并了
            if boundary_width > median_width * 1.8:  # 降低分割阈值
                # 尝试分割
                # 计算该区域的垂直投影
                region_proj = vertical_proj[start:end]
                
                # 寻找局部最小值作为分割点
                local_minima = []
                for i in range(1, len(region_proj)-1):
                    if (region_proj[i] < region_proj[i-1] and 
                        region_proj[i] < region_proj[i+1] and
                        region_proj[i] < threshold):
                        local_minima.append(start + i)
                
                # 如果找到合适的分割点，进行分割
                if local_minima:
                    # 选择最佳分割点 - 限制分割数量
                    max_splits = min(3, int(boundary_width / median_width))  # 最多分割成3个字符
                    if len(local_minima) > max_splits:
                        # 选择间距大致相等的分割点
                        step = len(local_minima) // (max_splits + 1)
                        selected_minima = [local_minima[i] for i in range(step, len(local_minima), step)]
                    else:
                        selected_minima = local_minima
                    
                    split_points = [start] + selected_minima + [end]
                    for i in range(len(split_points)-1):
                        s, e = split_points[i], split_points[i+1]
                        if e - s > min_char_width:
                            final_boundaries.append((s, e))
                else:
                    final_boundaries.append(boundary)
            else:
                final_boundaries.append(boundary)
        
        boundaries = final_boundaries
    
    # 7. 最终验证和调整
    # 确保边界在图像范围内
    boundaries = [(max(0, start), min(width, end)) for start, end in boundaries]
    
    # 过滤掉过小的边界
    boundaries = [b for b in boundaries if b[1] - b[0] > min_char_width]
    
    # 按x坐标排序
    boundaries.sort(key=lambda b: b[0])
    
    print(f"初始检测到 {len(boundaries)} 个字符边界", flush=True)
    
    # 如果字符数量过多，选择最佳的7个字符
    if len(boundaries) > 7:
        print(f"检测到 {len(boundaries)} 个字符边界，选择最佳的7个", flush=True)
        # 计算每个边界的得分（基于宽度和高度）
        scores = []
        for i, (start, end) in enumerate(boundaries):
            width = end - start
            # 计算该区域的垂直投影总和
            region_proj = vertical_proj[start:end]
            proj_sum = np.sum(region_proj)
            # 得分 = 宽度 * 投影总和
            score = width * proj_sum
            scores.append((i, score))
        
        # 按得分排序，选择得分最高的7个
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in scores[:7]])
        boundaries = [boundaries[i] for i in selected_indices]
        # 重新排序
        boundaries.sort(key=lambda b: b[0])
        print(f"选择了 {len(boundaries)} 个字符边界", flush=True)
    else:
        print(f"字符边界数量为 {len(boundaries)}，无需减少", flush=True)
    
    # 如果仍然字符太少，尝试更宽松的条件
    if len(boundaries) < 5:
        print(f"字符数量太少({len(boundaries)})，尝试更宽松的条件", flush=True)
        # 降低阈值
        threshold = np.percentile(vertical_proj, 20)  # 使用20%分位数
        
        # 重新寻找字符边界
        new_boundaries = []
        in_char = False
        start = 0
        min_char_width = max(3, int(width * 0.03))  # 降低最小宽度要求
        
        for x in range(width):
            if vertical_proj[x] > threshold and not in_char:
                start = x
                in_char = True
            elif vertical_proj[x] <= threshold and in_char:
                end = x
                in_char = False
                if end - start > min_char_width:
                    new_boundaries.append((start, end))
        
        if in_char:
            end = width - 1
            if end - start > min_char_width:
                new_boundaries.append((start, end))
        
        print(f"宽松条件下检测到 {len(new_boundaries)} 个字符边界", flush=True)
        
        # 只有当新检测到的字符数量更多但不超过7个时，才使用新的边界
        if len(new_boundaries) > len(boundaries) and len(new_boundaries) <= 7:
            boundaries = new_boundaries
            print(f"使用宽松条件下的检测结果，共 {len(boundaries)} 个字符边界", flush=True)
        else:
            print(f"保持原检测结果，共 {len(boundaries)} 个字符边界", flush=True)
    
    # 最终检查，确保字符数量不超过7个
    if len(boundaries) > 7:
        print(f"最终检查：字符数量仍然过多({len(boundaries)})，选择最佳的7个", flush=True)
        # 计算每个边界的得分（基于宽度和高度）
        scores = []
        for i, (start, end) in enumerate(boundaries):
            width = end - start
            # 计算该区域的垂直投影总和
            region_proj = vertical_proj[start:end]
            proj_sum = np.sum(region_proj)
            # 得分 = 宽度 * 投影总和
            score = width * proj_sum
            scores.append((i, score))
        
        # 按得分排序，选择得分最高的7个
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in scores[:7]])
        boundaries = [boundaries[i] for i in selected_indices]
        # 重新排序
        boundaries.sort(key=lambda b: b[0])
        print(f"最终选择了 {len(boundaries)} 个字符边界", flush=True)
    
    return boundaries


def extract_character_images(plate_image, boundaries):
    """
    字符图像提取
    参数:
        plate_image: 车牌图像
        boundaries: 字符边界信息
    返回:
        单个字符图像列表
    """
    # 如果没有边界信息，返回空列表
    if not boundaries:
        return []
    
    # 获取原始车牌图像（如果输入是二值化图像，需要使用原始图像）
    # 这里假设plate_image是预处理后的二值化图像
    # 在实际应用中，可能需要传入原始车牌图像
    
    char_images = []
    
    # 提取每个字符区域
    for i, (start, end) in enumerate(boundaries):
        # 提取字符区域
        char_region = plate_image[:, start:end]
        
        # 添加边距，确保字符完整
        h, w = char_region.shape
        margin_x = max(2, w // 10)  # 水平边距
        margin_y = max(2, h // 10)  # 垂直边距
        
        # 创建带边距的图像
        char_with_margin = np.zeros((h + 2 * margin_y, w + 2 * margin_x), dtype=char_region.dtype)
        char_with_margin[margin_y:margin_y + h, margin_x:margin_x + w] = char_region
        
        char_images.append(char_with_margin)
    
    return char_images


def normalize_characters(char_images):
    """
    字符标准化
    参数:
        char_images: 字符图像列表
    返回:
        标准化后的字符图像列表
    """
    if not char_images:
        return []
    
    normalized_chars = []
    target_size = (32, 32)  # 标准化后的字符大小
    
    for char_img in char_images:
        # 确保图像是二值图像
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        
        # 调整大小，保持宽高比
        h, w = char_img.shape
        
        # 计算缩放比例，使图像适应目标大小
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 调整图像大小
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建目标大小的空白图像
        normalized = np.zeros(target_size, dtype=resized.dtype)
        
        # 计算居中位置
        start_y = (target_size[0] - new_h) // 2
        start_x = (target_size[1] - new_w) // 2
        
        # 将调整大小后的图像放入中心位置
        normalized[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        # 确保是二值图像（0和255）
        _, binary = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
        
        normalized_chars.append(binary)
    
    return normalized_chars


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