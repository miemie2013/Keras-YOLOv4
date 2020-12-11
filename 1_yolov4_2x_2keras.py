#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
from config import *
from model.custom_layers import DCNv2
from model.yolo import *
import paddle.fluid as fluid


use_gpu = True



import torch


def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('yolov4.pt')
print('============================================================')




cfg = YOLOv4_2x_Config()

# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)
Head = select_head(cfg.head_type)
cfg.head['num_classes'] = 80
cfg.head['iou_aware'] = False
head = Head(yolo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
yolo = YOLO(backbone, head)

x = keras.layers.Input(shape=(None, None, 3))
im_size = keras.layers.Input(shape=(2, ))
outputs = yolo.get_outputs(x)
preds = yolo.get_prediction(outputs, im_size)
predict_model = keras.models.Model(inputs=[x, im_size], outputs=preds)
predict_model.summary(line_length=130)
print('\nCopying...')


def copy_conv_bn(conv_unit, w, scale, offset, m, v):
    w = w.transpose(2, 3, 1, 0)
    conv_unit.conv.set_weights([w])
    conv_unit.bn.set_weights([scale, offset, m, v])

def copy_conv(conv_layer, w, b):
    w = w.transpose(2, 3, 1, 0)
    conv_layer.set_weights([w, b])


def get(idx):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'bn%d.weight' % idx
    keyword3 = 'bn%d.bias' % idx
    keyword4 = 'bn%d.running_mean' % idx
    keyword5 = 'bn%d.running_var' % idx
    w, scale, offset, m, v = None, None, None, None, None
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            scale = value
        elif keyword3 in key:
            offset = value
        elif keyword4 in key:
            m = value
        elif keyword5 in key:
            v = value
    return w, scale, offset, m, v

def get2(idx):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'conv%d.bias' % idx
    w, b = None, None
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            b = value
    return w, b


def copy_blocks(blocks, k, n):
    for i in range(n):
        w, scale, offset, m, v = get(k+i*2)
        copy_conv_bn(blocks.sequential[i].conv1, w, scale, offset, m, v)
        w, scale, offset, m, v = get(k+i*2+1)
        copy_conv_bn(blocks.sequential[i].conv2, w, scale, offset, m, v)



# CSPDarknet53
k = 1
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.conv1, w, scale, offset, m, v)

# stage1
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage1_conv1, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage1_conv2, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage1_conv3, w, scale, offset, m, v)

copy_blocks(backbone.stage1_blocks, k, n=1)
k += 2*1

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage1_conv4, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage1_conv5, w, scale, offset, m, v)

# stage2
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage2_conv1, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage2_conv2, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage2_conv3, w, scale, offset, m, v)

copy_blocks(backbone.stage2_blocks, k, n=2)
k += 2*2

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage2_conv4, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage2_conv5, w, scale, offset, m, v)

# stage3
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage3_conv1, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage3_conv2, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage3_conv3, w, scale, offset, m, v)

copy_blocks(backbone.stage3_blocks, k, n=8)
k += 2*8

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage3_conv4, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage3_conv5, w, scale, offset, m, v)

# stage4
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage4_conv1, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage4_conv2, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage4_conv3, w, scale, offset, m, v)

copy_blocks(backbone.stage4_blocks, k, n=8)
k += 2*8

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage4_conv4, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage4_conv5, w, scale, offset, m, v)

# stage5
w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage5_conv1, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage5_conv2, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage5_conv3, w, scale, offset, m, v)

copy_blocks(backbone.stage5_blocks, k, n=4)
k += 2*4

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage5_conv4, w, scale, offset, m, v)

w, scale, offset, m, v = get(k)
k += 1
copy_conv_bn(backbone.stage5_conv5, w, scale, offset, m, v)


# head

for i in range(73, 94, 1):
    conv2dunit = head.get_conv2dunit('conv%.3d' % i)
    w, scale, offset, m, v = get(i)
    copy_conv_bn(conv2dunit, w, scale, offset, m, v)
for i in range(95, 102, 1):
    conv2dunit = head.get_conv2dunit('conv%.3d' % i)
    w, scale, offset, m, v = get(i)
    copy_conv_bn(conv2dunit, w, scale, offset, m, v)
for i in range(103, 110, 1):
    conv2dunit = head.get_conv2dunit('conv%.3d' % i)
    w, scale, offset, m, v = get(i)
    copy_conv_bn(conv2dunit, w, scale, offset, m, v)


w, b = get2(94)
copy_conv(head.conv094.conv, w, b)
w, b = get2(102)
copy_conv(head.conv102.conv, w, b)
w, b = get2(110)
copy_conv(head.conv110.conv, w, b)


predict_model.save_weights('yolov4_2x.h5')
print('\nDone.')







