import cv2
import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm, trange

def catImages(
    image_folder_path_list: list,
    save_image_folder_path: str,
) -> bool:
    """
    将所有文件夹中名称相同的图片文件按照最接近16:9比例的方式拼接起来
    
    Args:
        image_folder_path_list: 图片文件夹路径列表
        save_image_folder_path: 保存拼接后图片的文件夹路径
        
    Returns:
        bool: 是否成功
    """
    try:
        if not image_folder_path_list:
            print("错误: 图片文件夹列表为空")
            return False
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 检查所有文件夹是否存在
        valid_folders = []
        for folder_path in image_folder_path_list:
            if not os.path.exists(folder_path):
                print(f"警告: 文件夹不存在 {folder_path}，跳过")
                continue
            valid_folders.append(folder_path)
        
        if not valid_folders:
            print("错误: 没有可用的图片文件夹")
            return False
        
        # 收集所有文件夹中的图片文件名
        all_image_files = {}  # {folder_path: set of filenames}
        for folder_path in valid_folders:
            image_files = set()
            for file in os.listdir(folder_path):
                if Path(file).suffix.lower() in image_extensions:
                    image_files.add(file)
            all_image_files[folder_path] = image_files
        
        # 找到所有文件夹中都存在的图片文件名（交集）
        common_image_names = None
        for folder_path, image_files in all_image_files.items():
            if common_image_names is None:
                common_image_names = image_files.copy()
            else:
                common_image_names &= image_files
        
        if not common_image_names:
            print("错误: 没有在所有文件夹中都存在的图片文件")
            return False
        
        # 创建保存文件夹
        os.makedirs(save_image_folder_path, exist_ok=True)
        
        # 读取第一张图片获取尺寸（假设所有图片尺寸一致）
        first_folder = valid_folders[0]
        first_image_name = sorted(common_image_names)[0]
        first_image_path = os.path.join(first_folder, first_image_name)
        first_image = cv2.imread(first_image_path)
        
        if first_image is None:
            print(f"错误: 无法读取图片 {first_image_path}")
            return False
        
        img_height, img_width, channels = first_image.shape
        
        # 计算最接近16:9的布局
        def find_best_16_9_layout(num: int) -> Tuple[int, int]:
            """找到最接近16:9比例的网格布局"""
            target_ratio = 16.0 / 9.0
            best_rows, best_cols = 1, num
            best_diff = abs((best_cols / best_rows) - target_ratio)
            
            # 尝试不同的行数
            for rows in range(1, num + 1):
                cols = (num + rows - 1) // rows  # 向上取整
                if rows * cols < num:
                    continue
                
                ratio = cols / rows
                diff = abs(ratio - target_ratio)
                
                if diff < best_diff:
                    best_diff = diff
                    best_rows = rows
                    best_cols = cols
            
            return best_rows, best_cols
        
        num_images = len(valid_folders)
        rows, cols = find_best_16_9_layout(num_images)
        
        print('[INFO][images::catImages]')
        print(f'\t 找到 {len(common_image_names)} 个同名图片文件')
        print(f'\t 图片布局: {rows}x{cols} (共{num_images}个文件夹)')
        print(f'\t 开始拼接图片...')
        
        # 对每个同名图片进行拼接
        for image_name in tqdm(sorted(common_image_names)):
            # 创建输出画布
            output_width = img_width * cols
            output_height = img_height * rows
            output_image = np.zeros((output_height, output_width, channels), dtype=np.uint8)
            
            # 从各个文件夹读取同名图片并拼接
            folder_idx = 0
            for row in range(rows):
                for col in range(cols):
                    if folder_idx >= num_images:
                        break
                    
                    folder_path = valid_folders[folder_idx]
                    image_path = os.path.join(folder_path, image_name)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        print(f"警告: 无法读取图片 {image_path}，跳过")
                        folder_idx += 1
                        continue
                    
                    # 如果尺寸不一致，调整大小
                    if image.shape[:2] != (img_height, img_width):
                        image = cv2.resize(image, (img_width, img_height))
                    
                    # 将图片放置到对应位置
                    y_start = row * img_height
                    y_end = y_start + img_height
                    x_start = col * img_width
                    x_end = x_start + img_width
                    
                    output_image[y_start:y_end, x_start:x_end] = image
                    folder_idx += 1
                
                if folder_idx >= num_images:
                    break
            
            # 保存拼接后的图片
            save_image_path = os.path.join(save_image_folder_path, image_name)
            cv2.imwrite(save_image_path, output_image)
        
        print(f"成功: 已将 {len(common_image_names)} 个同名图片拼接并保存到 {save_image_folder_path}")
        return True
        
    except Exception as e:
        print(f"错误: 拼接图片时发生异常: {str(e)}")
        return False

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
