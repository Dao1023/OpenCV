import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def compare_character_boundaries():
    """比较优化前后的字符边界检测结果"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    output_dir = "../output/LPR/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试图像列表
    test_images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
    
    # 创建一个大图来显示所有比较结果
    fig, axes = plt.subplots(len(test_images), 2, figsize=(15, 20))
    fig.suptitle('字符边界检测结果比较', fontsize=16)
    
    for i, test_file in enumerate(test_images):
        # 读取优化后的结果
        optimized_path = f"../output/LPR/{test_file}_debug_boundaries.jpg"
        original_path = f"../output/LPR/{test_file}_8_character_boundaries.jpg"
        
        if os.path.exists(optimized_path) and os.path.exists(original_path):
            optimized_img = cv2.imread(optimized_path)
            original_img = cv2.imread(original_path)
            
            # 转换为RGB
            optimized_img = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # 显示图像
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f'{test_file} - 优化前')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(optimized_img)
            axes[i, 1].set_title(f'{test_file} - 优化后')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'character_boundaries_comparison.jpg'), dpi=300)
    plt.close()
    
    print(f"字符边界检测结果比较图已保存到 {output_dir}/character_boundaries_comparison.jpg")

if __name__ == "__main__":
    compare_character_boundaries()