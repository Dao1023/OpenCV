import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(use_grayscale=True):
    """加载图像"""
    # 尝试加载不同格式的图像
    image_paths = [
        'assets/images/1.jpg',
        'assets/images/1.png',
        'assets/images/1.gif'
    ]
    
    for path in image_paths:
        try:
            # 根据参数决定是否读取为灰度图像
            if use_grayscale:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(path)
            if img is not None:
                print(f"成功加载图像: {path}")
                print(f"图像尺寸: {img.shape}")
                return img
        except Exception as e:
            print(f"加载 {path} 失败: {e}")
            continue
    
    raise FileNotFoundError("无法加载任何图像文件")

def gaussian_filter_comparison(img, use_grayscale=True):
    """高斯滤波参数对比"""
    # 不同的高斯核大小和sigma值
    params = [
        (5, 1.0),
        (5, 3.0),
        (9, 1.0),
        (9, 3.0),
        (15, 5.0)
    ]
    
    filtered_images = []
    titles = []
    
    for ksize, sigma in params:
        # 确保ksize是奇数
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        if use_grayscale or len(img.shape) == 2:
            filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        else:
            # 对于彩色图像，需要分别对每个通道进行滤波
            filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        filtered_images.append(filtered)
        titles.append(f'Gaussian\nksize={ksize}, σ={sigma}')
    
    return filtered_images, titles

def median_filter_comparison(img, use_grayscale=True):
    """中值滤波参数对比"""
    # 不同的核大小
    k_sizes = [3, 5, 7, 9, 15]
    
    filtered_images = []
    titles = []
    
    for ksize in k_sizes:
        # 确保ksize是奇数
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        if use_grayscale or len(img.shape) == 2:
            filtered = cv2.medianBlur(img, ksize)
        else:
            # 对于彩色图像，需要分别对每个通道进行滤波
            filtered = cv2.medianBlur(img, ksize)
        filtered_images.append(filtered)
        titles.append(f'Median\nksize={ksize}')
    
    return filtered_images, titles

def bilateral_filter_comparison(img, use_grayscale=True):
    """双边滤波参数对比"""
    # 不同的参数组合
    params = [
        (9, 75, 75),
        (15, 80, 80),
        (25, 90, 90),
        (35, 100, 100),
        (50, 150, 150)
    ]
    
    filtered_images = []
    titles = []
    
    for d, sigma_color, sigma_space in params:
        if use_grayscale or len(img.shape) == 2:
            filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        else:
            # 对于彩色图像，需要分别对每个通道进行滤波
            filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        filtered_images.append(filtered)
        titles.append(f'Bilateral\nd={d}, σc={sigma_color}, σs={sigma_space}')
    
    return filtered_images, titles

def create_filter_comparison(use_grayscale=True):
    """创建滤波对比图"""
    # 加载图像
    img = load_image(use_grayscale)
    
    # 调整图像大小以便更好地显示
    if img.shape[0] > 800 or img.shape[1] > 800:
        scale = min(800/img.shape[0], 800/img.shape[1])
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    # 获取各种滤波结果
    gaussian_imgs, gaussian_titles = gaussian_filter_comparison(img, use_grayscale)
    median_imgs, median_titles = median_filter_comparison(img, use_grayscale)
    bilateral_imgs, bilateral_titles = bilateral_filter_comparison(img, use_grayscale)
    
    # 创建对比图 (改为3行5列)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('comparison of different filters', fontsize=16)
    
    # 显示高斯滤波结果 (现在在第0行)
    for i, (img_f, title) in enumerate(zip(gaussian_imgs, gaussian_titles)):
        if use_grayscale or len(img_f.shape) == 2:
            axes[0, i].imshow(img_f, cmap='gray')
        else:
            img_f_rgb = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img_f_rgb)
        axes[0, i].set_title(title, fontsize=8)
        axes[0, i].axis('off')
    
    # 显示中值滤波结果 (现在在第1行)
    for i, (img_f, title) in enumerate(zip(median_imgs, median_titles)):
        if use_grayscale or len(img_f.shape) == 2:
            axes[1, i].imshow(img_f, cmap='gray')
        else:
            img_f_rgb = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img_f_rgb)
        axes[1, i].set_title(title, fontsize=8)
        axes[1, i].axis('off')
    
    # 显示双边滤波结果 (现在在第2行)
    for i, (img_f, title) in enumerate(zip(bilateral_imgs, bilateral_titles)):
        if use_grayscale or len(img_f.shape) == 2:
            axes[2, i].imshow(img_f, cmap='gray')
        else:
            img_f_rgb = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            axes[2, i].imshow(img_f_rgb)
        axes[2, i].set_title(title, fontsize=8)
        axes[2, i].axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('output/filter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 可以通过修改这个参数来控制是否使用灰度图
    # True表示使用灰度图，False表示使用彩色图
    use_grayscale = False
    create_filter_comparison(use_grayscale)