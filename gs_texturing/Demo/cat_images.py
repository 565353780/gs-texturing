from gs_texturing.Method.images import catImages

def demo():
    image_folder_path_list = [
        '/Users/chli/chLi/Dataset/GS/haizei_1/train/point_cloud.ply_7000/gt/',
        '/Users/chli/chLi/Dataset/GS/haizei_1/train/point_cloud.ply_7000/renders/',
    ]

    save_image_folder_path = '/Users/chli/chLi/Dataset/GS/haizei_1/train/point_cloud.ply_7000/cat_images/'

    catImages(image_folder_path_list, save_image_folder_path)
    return True
