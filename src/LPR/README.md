# 车牌识别系统使用指南

## 简介
这是一个基于OpenCV的车牌识别系统，经过优化后能够准确地检测和分割车牌中的字符。

## 文件说明

### 核心文件
- `license_plate_recognition.py` - 车牌识别系统的主要实现
- `test_functions.py` - 完整的测试套件
- `test_character_boundaries.py` - 字符边界检测测试脚本

### 演示和可视化
- `demo_license_plate_recognition.py` - 车牌识别系统演示脚本
- `compare_character_boundaries.py` - 字符边界检测结果比较脚本
- `show_pipeline_summary.py` - 车牌识别流程总结脚本

### 文档
- `character_boundary_optimization_summary.md` - 字符边界检测优化总结
- `project_summary.md` - 项目总结报告

## 使用方法

### 1. 运行完整测试
```bash
cd src
python test_functions.py
```
这将运行完整的测试套件，包括图像预处理、边缘检测、形态学操作、轮廓检测和字符分割。

### 2. 测试字符边界检测
```bash
cd src
python test_character_boundaries.py
```
这将专门测试字符边界检测算法，并输出详细的调试信息。

### 3. 运行演示
```bash
cd src
python demo_license_plate_recognition.py
```
这将处理所有测试图像，并生成详细的演示结果图。

### 4. 生成比较图
```bash
cd src
python compare_character_boundaries.py
```
这将生成优化前后的字符边界检测结果比较图。

### 5. 生成流程总结
```bash
cd src
python show_pipeline_summary.py
```
这将生成车牌识别处理流程的总结图。

## 输出文件
所有输出文件保存在 `../output/LPR/` 目录下：
- `*_debug_boundaries.jpg` - 字符边界检测结果
- `comparison/` - 比较图
- `demo/` - 演示结果
- `pipeline_summary/` - 流程总结

## 测试图像
测试图像位于 `../assets/images/LPR/` 目录下，包含5张车牌图像。

## 优化特点
1. 自适应字符边界检测 - 根据初始检测结果自动调整检测参数
2. 字符数量控制 - 确保检测到的字符数量在合理范围内
3. 详细的调试信息 - 便于理解和调试算法
4. 鲁棒性强 - 适应不同质量的车牌图像

## 系统要求
- Python 3.7+
- OpenCV 4.x
- NumPy
- Matplotlib

## 注意事项
- 确保测试图像路径正确
- 输出目录需要有写入权限
- 如果遇到中文字体显示问题，可以安装中文字体支持