from gs_texturing.Method.images import imagesToVideo, catVideos

def demo():
    image_folder_path_list = [
        '/Users/chli/chLi/Dataset/GS/haizei_1/train/point_cloud.ply_7000/gt/',
        '/Users/chli/chLi/Dataset/GS/haizei_1/train/point_cloud.ply_7000/renders/',
    ]

    video_file_path_list = []
    save_cat_video_file_path = '/Users/chli/chLi/Dataset/GS/haizei_1/train/point_cloud.ply_7000/cat_results.mp4'

    for image_folder_path in image_folder_path_list:
        save_video_file_path = image_folder_path[:-1] + '.mp4'
        imagesToVideo(image_folder_path, save_video_file_path)
        video_file_path_list.append(save_video_file_path)

    catVideos(video_file_path_list, save_cat_video_file_path)
    return True
