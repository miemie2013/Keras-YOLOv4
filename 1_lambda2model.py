#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2019-12-20 14:29:47
#   Description : 将训练模型中yolov4的所有部分提取出来。
#                 需要修改'step00001000.h5'为你最后训练得到的文件名。
#
#================================================================
import os
import keras
import tensorflow as tf
from model.yolov4 import YOLOv4

# 显存分配。只用cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))



num_classes = 80
num_anchors = 3
inputs = keras.layers.Input(shape=(None, None, 3))
model_body = YOLOv4(inputs, num_classes, num_anchors)

model_body.load_weights('./weights/step00001000.h5', by_name=True)
model_body.save('yolov4.h5')

