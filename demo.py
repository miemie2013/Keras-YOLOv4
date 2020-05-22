#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : keras_yolov4
#
# ================================================================
import cv2
import os
import time
import tensorflow as tf
import keras.layers as layers
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode


# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。
use_gpu = False
use_gpu = True

# 显存分配。
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


if __name__ == '__main__':
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。
    # model_path = 'yolov4.h5'
    model_path = './weights/step00001000.h5'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    input_shape = (416, 416)
    # input_shape = (608, 608)


    num_anchors = 3
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)
    inputs = layers.Input(shape=(None, None, 3))
    yolo = YOLOv4(inputs, num_classes, num_anchors)
    yolo.load_weights(model_path, by_name=True)

    _decode = Decode(0.05, 0.45, input_shape, yolo, all_classes)

    # detect images in test floder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            start = time.time()
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image, boxes, scores, classes = _decode.detect_image(image, draw_image = True)
                cv2.imwrite('images/res/' + f, image)
            print('total time: {0:.6f}s'.format(time.time() - start))

