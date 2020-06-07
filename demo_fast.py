#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-03 15:35:27
#   Description : keras_yolov4
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import colorsys
import random
import time
import numpy as np
import tensorflow as tf
import keras.layers as layers
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)



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


def process_image(img, input_shape):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale_x = float(input_shape[1]) / w
    scale_y = float(input_shape[0]) / h
    img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    pimage = img.astype(np.float32) / 255.
    pimage = np.expand_dims(pimage, axis=0)
    return pimage


def draw(image, boxes, scores, classes, all_classes, colors):
    image_h, image_w, _ = image.shape
    for box, score, cl in zip(boxes, scores, classes):
        x0, y0, x1, y1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
        bbox_color = colors[cl]
        # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
        bbox_thick = 1
        cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
        bbox_mess = '%s: %.2f' % (all_classes[cl], score)
        t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
        cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
        cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)



if __name__ == '__main__':
    # classes_path = 'data/voc_classes.txt'
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。
    # model_path = 'yolov4.h5'
    model_path = './weights/step00070000.h5'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    input_shape = (416, 416)
    # input_shape = (608, 608)

    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.05
    nms_thresh = 0.45
    keep_top_k = 100
    nms_top_k = 100

    # 是否给图片画框。不画可以提速。读图片、后处理还可以继续优化。
    draw_image = True
    # draw_image = False

    # 初始卷积核个数
    initial_filters = 32
    anchors = np.array([
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]]
    ])
    # 一些预处理
    anchors = anchors.astype(np.float32)
    num_anchors = len(anchors[0])  # 每个输出层有几个先验框

    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)
    inputs = layers.Input(shape=(None, None, 3))
    yolo = YOLOv4(inputs, num_classes, num_anchors, initial_filters, True, anchors, conf_thresh, nms_thresh, keep_top_k, nms_top_k)
    yolo.load_weights(model_path, by_name=True)

    # 定义颜色
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)


    path_dir = os.listdir('images/test')
    # warm up
    if use_gpu:
        for k, filename in enumerate(path_dir):
            image = cv2.imread('images/test/' + filename)

            # 预处理方式一
            pimage = process_image(np.copy(image), input_shape)
            # 预处理方式二
            # pimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # pimage = np.expand_dims(pimage, axis=0)

            outs = yolo.predict(pimage)
            if k == 10:
                break


    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = len(path_dir)
    start = time.time()
    for k, filename in enumerate(path_dir):
        image = cv2.imread('images/test/' + filename)

        # 预处理方式一
        pimage = process_image(np.copy(image), input_shape)
        # 预处理方式二
        # pimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pimage = np.expand_dims(pimage, axis=0)

        outs = yolo.predict(pimage)
        boxes, scores, classes = outs[0][0], outs[1][0], outs[2][0]

        img_h, img_w, _ = image.shape
        a = input_shape[0]
        boxes = boxes * [img_w/a, img_h/a, img_w/a, img_h/a]
        if boxes is not None and draw_image:
            draw(image, boxes, scores, classes, all_classes, colors)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - k) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
        if draw_image:
            cv2.imwrite('images/res/' + filename, image)
            logger.info("Detection bbox results save in images/res/{}".format(filename))
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / num_imgs), (num_imgs / cost)))


