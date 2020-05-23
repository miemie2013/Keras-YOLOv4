#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : keras_yolov4
#
# ================================================================
import keras.layers as layers
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
import json
from tools.cocotools import eval

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。
    # model_path = 'yolov4.h5'
    model_path = './weights/step00040000.h5'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    # input_shape = (416, 416)
    input_shape = (608, 608)

    # 验证集图片的相对路径
    eval_pre_path = '../COCO/val2017/'
    anno_file = '../COCO/annotations/instances_val2017.json'
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    num_anchors = 3
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)
    inputs = layers.Input(shape=(None, None, 3))
    yolo = YOLOv4(inputs, num_classes, num_anchors)
    yolo.load_weights(model_path, by_name=True)

    _decode = Decode(0.05, 0.45, input_shape, yolo, all_classes)
    box_ap = eval(_decode, images, eval_pre_path, anno_file)

