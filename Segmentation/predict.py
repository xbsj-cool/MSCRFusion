from PIL import Image
import os
from tqdm import tqdm
from hrnet import HRnet_Segmentation

if __name__ == "__main__":

    hrnet = HRnet_Segmentation()
    vi_dir_origin_path = "test/MSRS/vi/"
    ir_dir_origin_path = "test/MSRS/ir/"
    dir_save_path = "outputs/"

    vi_img_names = os.listdir(vi_dir_origin_path)
    ir_img_names = os.listdir(ir_dir_origin_path)
    for img_name in tqdm(ir_img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            vi_image_path = os.path.join(vi_dir_origin_path, img_name)
            ir_image_path = os.path.join(ir_dir_origin_path, img_name)
            vi_image = Image.open(vi_image_path)
            ir_image = Image.open(ir_image_path)
            r_image = hrnet.get_miou_png(vi_image, ir_image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name))

