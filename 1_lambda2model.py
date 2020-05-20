#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2019-12-20 14:29:47
#   Description : 将训练模型中yolov4的所有部分提取出来。
#                 需要修改'ep000001-loss36.509-val_loss34.633.h5'为你最后训练得到的文件名。
#
#================================================================
import keras
from model.yolov4 import YOLOv4
from train import decode, loss_layer

num_classes = 80
num_anchors = 3
inputs = keras.layers.Input(shape=(None, None, 3))
model_body = YOLOv4(inputs, num_classes, num_anchors)

model_body.load_weights('./weights/ep000001-loss36.509-val_loss34.633.h5', by_name=True)
model_body.save('yolov4.h5')

