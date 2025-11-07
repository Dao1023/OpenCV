import numpy as np
import cv2

def optical_flow_optimized():
    # 1. 添加视频文件存在性检查
    video_path = "assets/test.avi"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 2. 优化特征点检测参数
    feature_params = dict(
        maxCorners=150,  # 增加特征点数量以提高跟踪稳定性
        qualityLevel=0.01,  # 降低质量阈值以检测更多特征点
        minDistance=10,    # 增加最小距离以避免特征点过于密集
        blockSize=7       # 添加块大小参数
    )

    # 3. 优化LK光流参数
    lk_params = dict(
        winSize=(21, 21),  # 增大窗口大小以提高跟踪精度
        maxLevel=3,        # 增加金字塔层数以处理更大位移
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # 4. 预分配颜色数组
    max_features = 200
    colors = np.random.randint(0, 255, (max_features, 3))

    # 5. 读取第一帧并初始化
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取视频第一帧")
        return
        
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # 6. 使用更稳定的特征点检测方法
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # 如果特征点太少，重新检测
    if p0 is None or len(p0) < 10:
        print("初始特征点不足，调整参数重新检测")
        feature_params['qualityLevel'] = 0.001
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 7. 创建掩码用于绘制轨迹
    mask = np.zeros_like(old_frame)
    
    # 8. 添加帧率控制
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 9. 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 10. 筛选有效的跟踪点
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        else:
            good_new = np.array([])
            good_old = np.array([])

        # 11. 绘制轨迹和特征点
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            
            # 使用预分配的颜色
            color_idx = i % len(colors)
            color = colors[color_idx].tolist()
            
            # 绘制轨迹线
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            # 绘制当前特征点
            frame = cv2.circle(frame, (a, b), 5, color, -1)

        # 12. 合并图像
        img = cv2.add(frame, mask)
        
        # 13. 添加信息显示
        info_text = f"Frame: {frame_count}, Features: {len(good_new)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)

        # 14. 显示结果
        cv2.imshow("Optical Flow (Optimized)", img)
        
        # 15. 改进退出机制
        k = cv2.waitKey(delay) & 0xff
        if k == 27:  # ESC键退出
            break
        elif k == ord('p'):  # P键暂停
            cv2.waitKey(0)
        elif k == ord('r'):  # R键重置轨迹
            mask = np.zeros_like(old_frame)
        
        # 16. 定期重新检测特征点（每30帧）
        if frame_count % 30 == 0 and len(good_new) < 20:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if p0 is not None:
                print(f"重新检测特征点: {len(p0)}个")
        else:
            # 更新下一帧的特征点
            old_gray = frame_gray.copy()
            if len(good_new) > 0:
                p0 = good_new.reshape(-1, 1, 2)
            else:
                # 如果没有有效特征点，重新检测
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # 17. 资源清理
    cap.release()
    cv2.destroyAllWindows()
    print("光流分析完成")

if __name__ == "__main__":
    optical_flow_optimized()
