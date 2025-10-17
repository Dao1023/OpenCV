import numpy as np
import open3d as o3d

# 模拟一个 200x200 的深度图（单位：米）
height, width = 200, 200
depth = np.ones((height, width), dtype=np.float32) * 2.0  # 所有点深度为 2 米
# 添加一个“凸起”模拟物体
depth[80:120, 80:120] = 1.5

# 相机内参（焦距 fx=fy=500，主点中心）
fx = fy = 500
cx, cy = width // 2, height // 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

# 生成网格坐标
u, v = np.meshgrid(np.arange(width), np.arange(height))
u = u.astype(np.float32)
v = v.astype(np.float32)

# 有效深度掩码
mask = depth > 0

# 转换为 3D 点
z = depth[mask]
x = (u[mask] - cx) * z / fx
y = (v[mask] - cy) * z / fy
points = np.stack([x, y, z], axis=1)

# 创建点云并可视化
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])