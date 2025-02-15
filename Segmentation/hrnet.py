import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.hrnet import HRnet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class HRnet_Segmentation(object):
    _defaults = {
        "model_path": 'trained model_path',
        "backbone": "hrnetv2_w32",
        "input_shape": [480, 640],
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes <= 9:
            self.colors = [(0, 0, 0), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 192), (128, 128, 0),
                           (64, 64, 128), (192, 128, 128), (192, 64, 0)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self, onnx=False):
        self.net = HRnet(num_classes=self.num_classes, backbone=self.backbone, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, vi_image, ir_image, count=False, name_classes=None):
        vi_image = cvtColor(vi_image)
        ir_image = cvtColor(ir_image)
        orininal_h = np.array(vi_image).shape[0]
        orininal_w = np.array(vi_image).shape[1]
        vi_image_data, nw, nh = resize_image(vi_image, (self.input_shape[1], self.input_shape[0]))
        ir_image_data, nw, nh = resize_image(ir_image, (self.input_shape[1], self.input_shape[0]))
        vi_image_data = np.expand_dims(np.transpose(preprocess_input(np.array(vi_image_data, np.float32)), (2, 0, 1)), 0)
        ir_image_data = np.expand_dims(np.transpose(preprocess_input(np.array(ir_image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            vi_images = torch.from_numpy(vi_image_data)
            ir_images = torch.from_numpy(ir_image_data)
            if self.cuda:
                vi_images = vi_images.cuda()
                ir_images = ir_images.cuda()
            pr = self.net(vi_images, ir_images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))
        return image

    def get_miou_png(self, vi_image, ir_image):
        vi_image = cvtColor(vi_image)
        ir_image = cvtColor(ir_image)
        orininal_h = np.array(vi_image).shape[0]
        orininal_w = np.array(vi_image).shape[1]

        vi_image_data, nw, nh = resize_image(vi_image, (self.input_shape[1],self.input_shape[0]))
        ir_image_data, nw, nh = resize_image(ir_image, (self.input_shape[1], self.input_shape[0]))

        vi_image_data = np.expand_dims(np.transpose(preprocess_input(np.array(vi_image_data, np.float32)), (2, 0, 1)), 0)
        ir_image_data = np.expand_dims(np.transpose(preprocess_input(np.array(ir_image_data, np.float32)), (2, 0, 1)), 0)
        with torch.no_grad():
            vi_images = torch.from_numpy(vi_image_data)
            ir_images = torch.from_numpy(ir_image_data)
            if self.cuda:
                vi_images = vi_images.cuda()
                ir_images = ir_images.cuda()
            pr = self.net(vi_images, ir_images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
