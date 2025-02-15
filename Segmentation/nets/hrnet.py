import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .backbone import BN_MOMENTUM, hrnet_classification

class HRnet_Backbone(nn.Module):
    def __init__(self, backbone='hrnetv2_w18', pretrained=False):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def forward(self, x_vi, x_ir):
        x_vi = self.model.conv1(x_vi)
        x_ir = self.model.conv1(x_ir)
        x_vi = self.model.bn1(x_vi)
        x_ir = self.model.bn1(x_ir)
        x_vi = self.model.relu(x_vi)
        x_ir = self.model.relu(x_ir)
        x_vi = self.model.conv2(x_vi)
        x_ir = self.model.conv2(x_ir)
        x_vi = self.model.bn2(x_vi)
        x_ir = self.model.bn2(x_ir)
        x_vi = self.model.relu(x_vi)
        x_ir = self.model.relu(x_ir)
        x_vi = self.model.layer1(x_vi)
        x_ir = self.model.layer1(x_ir)

        x_list_1_vi = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list_1_vi.append(self.model.transition1[i](x_vi))
            else:
                x_list_1_vi.append(x_vi)
        y_list_1_vi = self.model.stage2(x_list_1_vi)

        x_list_1_ir = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list_1_ir.append(self.model.transition1[i](x_ir))
            else:
                x_list_1_ir.append(x_ir)
        y_list_1_ir = self.model.stage2(x_list_1_ir)

        y_list_1_combined = [y_vi + y_ir for y_vi, y_ir in zip(y_list_1_vi, y_list_1_ir)]

        x_list_2_combined = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list_2_combined.append(self.model.transition2[i](y_list_1_combined[i]))
                else:
                    x_list_2_combined.append(self.model.transition2[i](y_list_1_combined[-1]))
            else:
                x_list_2_combined.append(y_list_1_combined[i])
        y_list_2_combined = self.model.stage3(x_list_2_combined)

        x_list_2_ir = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list_2_ir.append(self.model.transition2[i](y_list_1_ir[i]))
                else:
                    x_list_2_ir.append(self.model.transition2[i](y_list_1_ir[-1]))
            else:
                x_list_2_ir.append(y_list_1_ir[i])
        y_list_2_ir = self.model.stage3(x_list_2_ir)

        y_list_2_combined = [y_combined + y_ir for y_combined, y_ir in zip(y_list_2_combined, y_list_2_ir)]

        x_list_3_combined = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list_3_combined.append(self.model.transition3[i](y_list_2_combined[i]))
                else:
                    x_list_3_combined.append(self.model.transition3[i](y_list_2_combined[-1]))
            else:
                x_list_3_combined.append(y_list_2_combined[i])
        y_list_3_combined = self.model.stage4(x_list_3_combined)

        x_list_3_ir = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list_3_ir.append(self.model.transition3[i](y_list_2_ir[i]))
                else:
                    x_list_3_ir.append(self.model.transition3[i](y_list_2_ir[-1]))
            else:
                x_list_3_ir.append(y_list_2_ir[i])
        y_list_3_ir = self.model.stage4(x_list_3_ir)
        y_list_combined = [y_combined + y_ir for y_combined, y_ir in zip(y_list_3_combined, y_list_3_ir)]

        return y_list_combined


class HRnet(nn.Module):
    def __init__(self, num_classes=9, backbone='hrnetv2_w18', pretrained=False):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(backbone=backbone, pretrained=pretrained)

        last_inp_channels = np.int64(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, vi_inputs, ir_inputs):
        H, W = vi_inputs.size(2), vi_inputs.size(3)
        x = self.backbone(vi_inputs, ir_inputs)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x