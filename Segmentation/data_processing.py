import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
import cv2
import numpy as np


color_map = np.zeros((256 * 3)).astype('uint8')
color_map[:3 * 9] = np.array([[0, 0, 0],   # 0像素还是得为0
                                  [64, 0, 128],  # 原像素值为1的
                                  [64, 64, 0],  # 原像素值为2的
                                  [0, 128, 192],
                                  # [192, 0, 192],
                                  # [128, 128, 0],
                                  # [64, 64, 128],
                                  # [192, 128, 128],
                                  # [192, 64, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  ],dtype='uint8').flatten()


def labeltocolor(mask):
    im=Image.fromarray(mask)
    im.putpalette(color_map)
    im=np.array(im.convert('RGB'))
    return im

def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    person = [64, 64, 0]
    bike = [0, 128, 192]
    curve = [192, 0, 192]
    car_stop = [128, 128, 0]
    guardrail = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump = [192, 64, 0]
    palette = np.array(
        [
            unlabelled,
            car,
            person,
            bike,
            curve,
            car_stop,
            guardrail,
            color_cone,
            bump,
        ]
    )
    return palette

def imread(path):
    label = np.array(Image.open(path))
    return label

def visualize(save_name, label):
    palette = get_palette()
    pred = label
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    print(label.max())
    for cid in range(1, int(label.max())):
        img[pred == cid] = palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(save_name)

def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_names.append(filename)
    return image_names

def pilConvertJPG(path):
    for a, _, c in os.walk(path):
        for n in c:
            if '.jpg' in n or '.png' in n or '.jpeg' in n:
                img = Image.open(os.path.join(a, n))
                rgb_im = img.convert('RGB')
                error_img_path = os.path.join(a, n)
                os.remove(error_img_path)
                n = ''.join(filter(lambda n: ord(n) < 256, n))
                jpg_img_path = os.path.splitext(os.path.join(a, n).replace('\\', '/'))[0]
                jpg_img_path += '.jpg'
                print(jpg_img_path)
                rgb_im.save(jpg_img_path)


