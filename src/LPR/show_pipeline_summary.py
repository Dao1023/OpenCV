import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def show_license_plate_recognition_pipeline():
    """展示车牌识别流程的优化效果"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    output_dir = "../output/LPR/pipeline_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择一个测试图像
    test_file = "3.jpg"  # 选择一个有代表性的图像
    
    # 读取所有处理步骤的图像
    steps = [
        ("1. 原始图像", f"../test_images/{test_file}"),
        ("2. 预处理", f"../output/LPR/{test_file}_1_preprocessed.jpg"),
        ("3. 边缘检测", f"../output/LPR/{test_file}_2_edge.jpg"),
        ("4. 形态学操作", f"../output/LPR/{test_file}_3_morphology.jpg"),
        ("5. 轮廓检测", f"../output/LPR/{test_file}_4_contours.jpg"),
        ("6. 最佳轮廓", f"../output/LPR/{test_file}_5_best_contour.jpg"),
        ("7. 车牌区域", f"../output/LPR/{test_file}_6_plate.jpg"),
        ("8. 车牌预处理", f"../output/LPR/{test_file}_7_preprocessed_plate.jpg"),
        ("9. 字符边界检测", f"../output/LPR/{test_file}_debug_boundaries.jpg")
    ]
    
    # 创建一个大图来显示所有处理步骤
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'车牌识别处理流程 - {test_file}', fontsize=16)
    
    for i, (title, path) in enumerate(steps):
        row = i // 3
        col = i % 3
        
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # 转换为RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img)
                axes[row, col].set_title(title)
                axes[row, col].axis('off')
            else:
                axes[row, col].text(0.5, 0.5, f"无法读取图像\n{path}", 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(title)
        else:
            axes[row, col].text(0.5, 0.5, f"文件不存在\n{path}", 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_file}_pipeline_summary.jpg'), dpi=300)
    plt.close()
    
    print(f"车牌识别处理流程图已保存到 {output_dir}/{test_file}_pipeline_summary.jpg")
    
    # 创建字符边界检测的详细说明
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'字符边界检测优化效果 - {test_file}', fontsize=16)
    
    # 优化前
    original_path = f"../output/LPR/{test_file}_8_character_boundaries.jpg"
    if os.path.exists(original_path):
        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_img)
        axes[0].set_title('优化前')
        axes[0].axis('off')
    
    # 优化后
    optimized_path = f"../output/LPR/{test_file}_debug_boundaries.jpg"
    if os.path.exists(optimized_path):
        optimized_img = cv2.imread(optimized_path)
        optimized_img = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(optimized_img)
        axes[1].set_title('优化后')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_file}_character_boundary_optimization.jpg'), dpi=300)
    plt.close()
    
    print(f"字符边界检测优化效果图已保存到 {output_dir}/{test_file}_character_boundary_optimization.jpg")

if __name__ == "__main__":
    show_license_plate_recognition_pipeline()