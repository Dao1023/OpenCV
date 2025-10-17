import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 相机内参（可用标定得到，此处假设已知）
fx, fy = 800, 800
cx, cy = 320, 240
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

# 2. 创建一个虚拟棋盘格图像并检测角点
# 这里我们跳过真实图像，直接假设检测到了角点（模拟）
# 实际使用时可用 cv2.findChessboardCorners
image_points = np.array([
    [200, 150],
    [300, 150],
    [200, 250],
    [300, 250]
], dtype=np.float32)

# 对应的 3D 棋盘格坐标（单位：mm，Z=0 表示平面）
object_points = np.array([
    [0, 0, 0],
    [100, 0, 0],
    [0, 100, 0],
    [100, 100, 0]
], dtype=np.float32)

# 3. 求解位姿
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None)

if success:
    # 将旋转向量转为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    # 4. 构建坐标轴（用于可视化）
    axis_3d = np.float32([[0, 0, 0],
                          [50, 0, 0],   # X 轴（红）
                          [0, 50, 0],   # Y 轴（绿）
                          [0, 0, 50]])  # Z 轴（蓝）
    
    # 投影到相机坐标系下的 3D 位置
    axis_cam = (R @ axis_3d.T).T + tvec.T  # shape: (4, 3)
    
    # 5. 用 Matplotlib 可视化 3D 坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin = axis_cam[0]
    ax.quiver(*origin, *(axis_cam[1] - origin), color='r', label='X')
    ax.quiver(*origin, *(axis_cam[2] - origin), color='g', label='Y')
    ax.quiver(*origin, *(axis_cam[3] - origin), color='b', label='Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Estimated 3D Pose of Chessboard Plane')
    plt.show()
else:
    print("Pose estimation failed.")