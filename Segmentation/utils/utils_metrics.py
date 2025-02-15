import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import glob

def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs, axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1], axis=[0,1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)
def compute_mIoU(gt_dir, pred_dir_vi, pred_dir_ir, png_name_list, num_classes, name_classes=None):
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_vi_imgs = [join(pred_dir_vi, x + ".png") for x in png_name_list]
    pred_ir_imgs = [join(pred_dir_ir, x + ".png") for x in png_name_list]
    for ind in range(len(gt_imgs)):
        vi_image_id = os.path.splitext(os.path.basename(pred_vi_imgs[ind]))[0]
        ir_image_id = os.path.splitext(os.path.basename(pred_ir_imgs[ind]))[0]
        vi_pred = np.array(Image.open(pred_vi_imgs[ind]))
        ir_pred = np.array(Image.open(pred_ir_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        if len(label.flatten()) != len(vi_pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(vi_pred.flatten()), gt_imgs[ind],
                    pred_vi_imgs[ind]))
            continue
        if name_classes is not None and ind >= 0 :
            vi_hist = fast_hist(label.flatten(), vi_pred.flatten(), num_classes)
            ir_hist = fast_hist(label.flatten(), ir_pred.flatten(), num_classes)
            vi_IoUs = per_class_iu(vi_hist)
            ir_IoUs = per_class_iu(ir_hist)
            if name_classes is not None:
                for ind_class in range(num_classes):
                    # print('vi===>' + name_classes[ind_class] + ':\tIou-' + str(round(vi_IoUs[ind_class] * 100, 2)) \
                    #     + '; Recall (equal to the PA)-' + str(round(vi_PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(vi_Precision[ind_class] * 100, 2)))
                    # print('ir===>' + name_classes[ind_class] + ':\tIou-' + str(round(ir_IoUs[ind_class] * 100, 2)) \
                    #     + '; Recall (equal to the PA)-' + str(round(ir_PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(round(ir_Precision[ind_class] * 100, 2)))
                    if (vi_IoUs[ind_class] < ir_IoUs[ind_class] and ind_class > 0) or vi_IoUs[ind_class] < 0.5:
                    # if vi_IoUs[ind_class] < ir_IoUs[ind_class] and ind_class > 0:
                        height = vi_pred.shape[0]
                        width = vi_pred.shape[1]
                        for y in range(height):
                            for x in range(width):
                                if vi_pred[y, x] == ind_class:
                                    vi_pred[y, x] = 0
                        vi_image = Image.fromarray(np.uint8(vi_pred))
                        vi_image.save(os.path.join(pred_dir_vi, str(vi_image_id) + ".png"))
                    # elif (ir_IoUs[ind_class] < vi_IoUs[ind_class] and ind_class > 0) or ir_IoUs[ind_class] < 0.5:
                    elif ir_IoUs[ind_class] < vi_IoUs[ind_class] and ind_class > 0:
                        height = ir_pred.shape[0]
                        width = ir_pred.shape[1]
                        for y in range(height):
                            for x in range(width):
                                if ir_pred[y, x] == ind_class:
                                    ir_pred[y, x] = 0
                        ir_image = Image.fromarray(np.uint8(ir_pred))
                        ir_image.save(os.path.join(pred_dir_ir, str(ir_image_id) + ".png"))
                    # print("finish")
