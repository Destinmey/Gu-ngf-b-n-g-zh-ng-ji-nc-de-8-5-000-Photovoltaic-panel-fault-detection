import cv2
import os
from datetime import datetime
from rknnlite.api import RKNNLite
from func import object_detection_function

# 配置参数
MODEL_PATH = "./model.rknn"
INPUT_VIDEO = "./input.mp4"
OUTPUT_DIR = "./output_videos"
WINDOW_NAME = "Object Detection"
WINDOW_SIZE = (1200, 800)

def initialize_rknn_model():
    """初始化RKNN模型"""
    rknn_model = RKNNLite()
    
    # 加载模型
    if rknn_model.load_rknn(MODEL_PATH) != 0:
        print("加载RKNN模型失败")
        exit(1)
    
    # 初始化运行时环境
    if rknn_model.init_runtime() != 0:
        print("初始化运行环境失败")
        exit(1)
    
    return rknn_model

def create_output_directory():
    """创建输出目录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")
    return OUTPUT_DIR

def generate_output_filename():
    """生成输出文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"detection_{timestamp}.mp4")

def initialize_video_writer(cap):
    """初始化视频写入器"""
    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 创建输出目录
    create_output_directory()
    
    # 生成输出文件名
    output_file = generate_output_filename()
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    
    # 创建VideoWriter对象
    out = cv2.VideoWriter(
        output_file, 
        fourcc, 
        fps, 
        (width, height)
    )
    
    if not out.isOpened():
        print(f"无法创建视频文件: {output_file}")
        return None
    
    print(f"视频将保存至: {output_file}")
    return out

def process_video(rknn_model):
    """处理视频流并保存结果"""
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {INPUT_VIDEO}")
        return
    
    # 初始化视频写入器
    out = initialize_video_writer(cap)
    if out is None:
        return
    
    # 创建显示窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1])
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理当前帧
        processed_frame = object_detection_function(rknn_model, frame)
        
        # 保存处理后的帧
        out.write(processed_frame)
        
        # 显示结果
        cv2.imshow(WINDOW_NAME, processed_frame)
        
        frame_count += 1
        print(f"已处理帧: {frame_count}", end='\r')  # 实时更新处理进度
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n视频处理完成! 共处理 {frame_count} 帧")

if __name__ == "__main__":
    # 初始化模型
    rknn_model = initialize_rknn_model()
    
    try:
        # 处理视频
        process_video(rknn_model)
    finally:
        # 释放模型资源
        rknn_model.release()
        print("模型资源已释放")
