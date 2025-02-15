import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import glob
from utils.utils_metrics import compute_mIoU
from data_processing import labeltocolor

def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_names.append(filename)
    return image_names

if __name__ == "__main__":
    num_classes = 9
    name_classes = ["background","car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    dataset_path = 'test'
    image_ids = open(os.path.join(dataset_path, "test.txt"),'r').read().splitlines()
    gt_dir = os.path.join(dataset_path, "labels")
    pred_dir_vi = "./test/vi_result"
    pred_dir_ir = "./test/ir_result"
    mask_fuse_path = "./test"

    print("Get miou.")
    compute_mIoU(gt_dir, pred_dir_vi, pred_dir_ir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数

    vi_image_names = get_image_names(pred_dir_vi)
    vi_image_paths = glob.glob(pred_dir_vi + '/*.png')
    ir_image_names = get_image_names(pred_dir_ir)
    ir_image_paths = glob.glob(pred_dir_ir + '/*.png')
    num_images = len(vi_image_names)

    for idx in range(num_images):
        vi_savepath = mask_fuse_path + "/" + "vi_mask_1" + "/" + str(vi_image_names[idx])
        ir_savepath = mask_fuse_path + "/" + "ir_mask_1" + "/" + str(ir_image_names[idx])
        ccm_savepath = mask_fuse_path + "/" + "fuse_mask_1" + "/" + str(ir_image_names[idx])

        vi_mask = cv2.imread(vi_image_paths[idx], cv2.IMREAD_GRAYSCALE)
        vi_mask_color = labeltocolor(vi_mask)
        vi_mask_color = vi_mask_color[:, :, ::-1]
        vi_gray = cv2.cvtColor(vi_mask_color, cv2.COLOR_BGR2GRAY)
        vi_ret, vi_binary = cv2.threshold(vi_gray, 0, 255, cv2.THRESH_BINARY)
        cv2.imwrite(vi_savepath, vi_binary)

        ir_mask = cv2.imread(ir_image_paths[idx], cv2.IMREAD_GRAYSCALE)
        ir_mask_color = labeltocolor(ir_mask)
        ir_mask_color = ir_mask_color[:, :, ::-1]
        ir_gray = cv2.cvtColor(ir_mask_color, cv2.COLOR_BGR2GRAY)
        ir_ret, ir_binary = cv2.threshold(ir_gray, 0, 255, cv2.THRESH_BINARY)
        cv2.imwrite(ir_savepath, ir_binary)

        ccm = cv2.add(vi_binary, ir_binary)
        cv2.imwrite(ccm_savepath, ccm)

        idx += 1


