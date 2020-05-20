#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : keras_yolov4，复制权重
#
# ================================================================
import torch
import keras
import keras.layers as layers
from model.yolov4 import YOLOv4


def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('yolov4.pt')
print('============================================================')


def find(base_model, conv2d_name, batch_normalization_name):
    i1, i2 = -1, -1
    for i in range(len(base_model.layers)):
        if base_model.layers[i].name == conv2d_name:
            i1 = i
        if base_model.layers[i].name == batch_normalization_name:
            i2 = i
    return i1, i2

def copy1(conv, bn, idx):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'bn%d.weight' % idx
    keyword3 = 'bn%d.bias' % idx
    keyword4 = 'bn%d.running_mean' % idx
    keyword5 = 'bn%d.running_var' % idx
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            y = value
        elif keyword3 in key:
            b = value
        elif keyword4 in key:
            m = value
        elif keyword5 in key:
            v = value
    w = w.transpose(2, 3, 1, 0)
    conv.set_weights([w])
    bn.set_weights([y, b, m, v])

def copy2(conv, idx):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'conv%d.bias' % idx
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            b = value
    w = w.transpose(2, 3, 1, 0)
    conv.set_weights([w, b])



num_classes = 80
num_anchors = 3

inputs = layers.Input(shape=(None, None, 3))
model_body = YOLOv4(inputs, num_classes, num_anchors)
model_body.summary()
keras.utils.vis_utils.plot_model(model_body, to_file='yolov4.png', show_shapes=True)

print('\nCopying...')
for i in range(1, 94, 1):
    i1, i2 = find(model_body, 'conv2d_%d' % i, 'batch_normalization_%d' % i)
    copy1(model_body.layers[i1], model_body.layers[i2], i)
for i in range(95, 102, 1):
    i1, i2 = find(model_body, 'conv2d_%d' % i, 'batch_normalization_%d' % (i-1,))
    copy1(model_body.layers[i1], model_body.layers[i2], i)
for i in range(103, 110, 1):
    i1, i2 = find(model_body, 'conv2d_%d' % i, 'batch_normalization_%d' % (i-2,))
    copy1(model_body.layers[i1], model_body.layers[i2], i)

i1, _ = find(model_body, 'conv2d_94', 'aaa')
copy2(model_body.layers[i1], 94)
i1, _ = find(model_body, 'conv2d_102', 'aaa')
copy2(model_body.layers[i1], 102)
i1, _ = find(model_body, 'conv2d_110', 'aaa')
copy2(model_body.layers[i1], 110)

model_body.save('yolov4.h5')
print('\nDone.')


