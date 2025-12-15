import cv2
import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm, trange

def imagesToVideo(
    image_folder_path: str,
    save_video_file_path: str,
    fps: int = 30,
) -> bool:
    """
    将图片文件夹中的图片按文件名中的数字顺序转换为视频
    
    Args:
        image_folder_path: 图片文件夹路径
        save_video_file_path: 保存视频文件路径
        fps: 视频帧率，默认30
        
    Returns:
        bool: 是否成功
    """
    try:
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图片文件
        image_files = []
        for file in os.listdir(image_folder_path):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)
        
        if not image_files:
            print(f"错误: 在 {image_folder_path} 中未找到图片文件")
            return False
        
        # 从文件名中提取数字并排序
        def extract_number(filename: str) -> int:
            """从文件名中用_分割的部分提取数字"""
            parts = filename.split('_')
            for part in parts:
                # 尝试提取数字
                numbers = re.findall(r'\d+', part)
                if numbers:
                    return int(numbers[0])
            # 如果找不到数字，返回0
            return 0
        
        # 按提取的数字排序
        image_files.sort(key=extract_number)
        
        # 读取第一张图片获取尺寸
        first_image_path = os.path.join(image_folder_path, image_files[0])
        first_frame = cv2.imread(first_image_path)
        if first_frame is None:
            print(f"错误: 无法读取图片 {first_image_path}")
            return False
        
        height, width, channels = first_frame.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video_file_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"错误: 无法创建视频文件 {save_video_file_path}")
            return False
        
        # 将每张图片写入视频
        print('[INFO][images::imagesToVideo]')
        print('\t start convert images to video...')
        for image_file in tqdm(image_files):
            image_path = os.path.join(image_folder_path, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"警告: 无法读取图片 {image_path}，跳过")
                continue
            
            # 如果尺寸不一致，调整大小
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            video_writer.write(frame)
        
        video_writer.release()
        print(f"成功: 已将 {len(image_files)} 张图片转换为视频 {save_video_file_path}")
        return True
        
    except Exception as e:
        print(f"错误: 转换图片到视频时发生异常: {str(e)}")
        return False


def catVideos(
    video_file_path_list: list,
    save_video_file_path: str,
    fps: int = 30,
) -> bool:
    """
    将多个视频按照最接近NxN的顺序拼接为一个视频
    
    Args:
        video_file_path_list: 视频文件路径列表
        save_video_file_path: 保存视频文件路径
        fps: 输出视频帧率，默认30
        
    Returns:
        bool: 是否成功
    """
    try:
        if not video_file_path_list:
            print("错误: 视频文件列表为空")
            return False
        
        # 读取所有视频
        video_caps = []
        video_properties = []
        
        for video_path in video_file_path_list:
            if not os.path.exists(video_path):
                print(f"警告: 视频文件不存在 {video_path}，跳过")
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告: 无法打开视频 {video_path}，跳过")
                continue
            
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video_caps.append(cap)
            video_properties.append({
                'width': width,
                'height': height,
                'frame_count': frame_count
            })
        
        if not video_caps:
            print("错误: 没有可用的视频文件")
            return False
        
        num_videos = len(video_caps)
        
        # 计算最接近的NxN布局
        def find_best_grid(num: int) -> Tuple[int, int]:
            """找到最接近NxN的网格布局"""
            sqrt_num = int(np.sqrt(num))
            # 尝试找到最接近的方形布局
            for rows in range(sqrt_num, 0, -1):
                cols = (num + rows - 1) // rows  # 向上取整
                if rows * cols >= num:
                    return rows, cols
            return 1, num
        
        rows, cols = find_best_grid(num_videos)
        print(f"视频布局: {rows}x{cols} (共{num_videos}个视频)")
        
        # 统一所有视频的尺寸（使用第一个视频的尺寸，或使用最小尺寸）
        # 这里使用第一个视频的尺寸作为标准
        target_width = video_properties[0]['width']
        target_height = video_properties[0]['height']
        
        # 计算输出视频尺寸
        output_width = target_width * cols
        output_height = target_height * rows
        
        # 获取最大帧数（所有视频中最长的）
        max_frames = max(prop['frame_count'] for prop in video_properties)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video_file_path, fourcc, fps, (output_width, output_height))
        
        if not video_writer.isOpened():
            print(f"错误: 无法创建视频文件 {save_video_file_path}")
            # 释放所有视频捕获器
            for cap in video_caps:
                cap.release()
            return False
        
        # 逐帧处理
        print('[INFO][images::catVideos]')
        print('\t start cat videos...')
        for frame_idx in trange(max_frames):
            # 创建输出画布
            output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            video_idx = 0
            for row in range(rows):
                for col in range(cols):
                    if video_idx >= num_videos:
                        break
                    
                    cap = video_caps[video_idx]
                    ret, frame = cap.read()
                    
                    if ret:
                        # 调整帧尺寸
                        if frame.shape[:2] != (target_height, target_width):
                            frame = cv2.resize(frame, (target_width, target_height))
                        
                        # 将帧放置到对应位置
                        y_start = row * target_height
                        y_end = y_start + target_height
                        x_start = col * target_width
                        x_end = x_start + target_width
                        
                        output_frame[y_start:y_end, x_start:x_end] = frame
                    else:
                        # 如果视频结束，使用黑色帧
                        pass
                    
                    video_idx += 1
                
                if video_idx >= num_videos:
                    break
            
            video_writer.write(output_frame)
        
        # 释放资源
        video_writer.release()
        for cap in video_caps:
            cap.release()
        
        print(f"成功: 已将 {num_videos} 个视频拼接为 {save_video_file_path}")
        return True
        
    except Exception as e:
        print(f"错误: 拼接视频时发生异常: {str(e)}")
        # 确保释放所有资源
        if 'video_caps' in locals():
            for cap in video_caps:
                if cap.isOpened():
                    cap.release()
        if 'video_writer' in locals() and video_writer.isOpened():
            video_writer.release()
        return False
