import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor

class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        vi_jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "MSRS/vi/JPEGImages"), name + ".png"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "MSRS/vi/SegmentationClass"), name + ".png"))
        ir_jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "MSRS/ir/JPEGImages"), name + ".png"))
        vi_jpg, ir_jpg, png = self.get_random_data(vi_jpg, ir_jpg, png, self.input_shape, random=self.train)
        vi_jpg = np.transpose(preprocess_input(np.array(vi_jpg, np.float64)), [2, 0, 1])
        ir_jpg = np.transpose(preprocess_input(np.array(ir_jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return vi_jpg, ir_jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, vi_image, ir_image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        vi_image = cvtColor(vi_image)
        ir_image = cvtColor(ir_image)
        label = Image.fromarray(np.array(label))
        iw, ih = vi_image.size
        h, w = input_shape

        if not random:
            iw, ih = vi_image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            vi_image = vi_image.resize((nw, nh), Image.BICUBIC)
            ir_image = ir_image.resize((nw, nh), Image.BICUBIC)
            vi_new_image = Image.new('RGB', [w, h], (128, 128, 128))
            vi_new_image.paste(vi_image, ((w - nw) // 2, (h - nh) // 2))
            ir_new_image = Image.new('RGB', [w, h], (128, 128, 128))
            ir_new_image.paste(ir_image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return vi_new_image, ir_new_image, new_label

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        vi_image = vi_image.resize((nw, nh), Image.BICUBIC)
        ir_image = ir_image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            vi_image = vi_image.transpose(Image.FLIP_LEFT_RIGHT)
            ir_image = ir_image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        vi_new_image = Image.new('RGB', (w, h), (128, 128, 128))
        ir_new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        vi_new_image.paste(vi_image, (dx, dy))
        ir_new_image.paste(ir_image, (dx, dy))
        new_label.paste(label, (dx, dy))
        vi_image = vi_new_image
        ir_image = ir_new_image
        label = new_label

        vi_image_data = np.array(vi_image, np.uint8)
        ir_image_data = np.array(ir_image, np.uint8)

        blur = self.rand() < 0.25
        if blur:
            vi_image_data = cv2.GaussianBlur(vi_image_data, (5, 5), 0)
            ir_image_data = cv2.GaussianBlur(ir_image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            vi_image_data = cv2.warpAffine(vi_image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            ir_image_data = cv2.warpAffine(ir_image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        vi_hue, vi_sat, vi_val = cv2.split(cv2.cvtColor(vi_image_data, cv2.COLOR_RGB2HSV))
        vi_dtype = vi_image_data.dtype
        ir_hue, ir_sat, ir_val = cv2.split(cv2.cvtColor(ir_image_data, cv2.COLOR_RGB2HSV))
        ir_dtype = ir_image_data.dtype

        vi_x = np.arange(0, 256, dtype=r.dtype)
        vi_lut_hue = ((vi_x * r[0]) % 180).astype(vi_dtype)
        vi_lut_sat = np.clip(vi_x * r[1], 0, 255).astype(vi_dtype)
        vi_lut_val = np.clip(vi_x * r[2], 0, 255).astype(vi_dtype)
        ir_x = np.arange(0, 256, dtype=r.dtype)
        ir_lut_hue = ((ir_x * r[0]) % 180).astype(ir_dtype)
        ir_lut_sat = np.clip(ir_x * r[1], 0, 255).astype(ir_dtype)
        ir_lut_val = np.clip(ir_x * r[2], 0, 255).astype(ir_dtype)

        vi_image_data = cv2.merge(
            (cv2.LUT(vi_hue, vi_lut_hue), cv2.LUT(vi_sat, vi_lut_sat), cv2.LUT(vi_val, vi_lut_val)))
        vi_image_data = cv2.cvtColor(vi_image_data, cv2.COLOR_HSV2RGB)

        ir_image_data = cv2.merge(
            (cv2.LUT(ir_hue, ir_lut_hue), cv2.LUT(ir_sat, ir_lut_sat), cv2.LUT(ir_val, ir_lut_val)))
        ir_image_data = cv2.cvtColor(ir_image_data, cv2.COLOR_HSV2RGB)
        return vi_image_data, ir_image_data, label

def seg_dataset_collate(batch):
    vi_images      = []
    ir_images      = []
    pngs        = []
    seg_labels  = []
    for vi_img, ir_img, png, labels in batch:
        vi_images.append(vi_img)
        ir_images.append(ir_img)
        pngs.append(png)
        seg_labels.append(labels)
    vi_images      = torch.from_numpy(np.array(vi_images)).type(torch.FloatTensor)
    ir_images      = torch.from_numpy(np.array(ir_images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return vi_images, ir_images, pngs, seg_labels

