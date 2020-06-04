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
import time
import numpy as np
import tensorflow as tf
import keras.layers as layers
import keras.backend as K

from model.fastnms import fast_nms
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode



dic = np.load('ooo.npz')
all_pred_boxes2 = dic['all_pred_boxes']
all_pred_scores2 = dic['all_pred_scores']

conf_thresh = 0.65
nms_thresh = 0.45
keep_top_k = 100
nms_top_k = 100



all_pred_boxes = layers.Input(name='all_pred_boxes', shape=(None, 4))
all_pred_scores = layers.Input(name='all_pred_scores', shape=(None, 80))



conf_preds = tf.transpose(all_pred_scores, perm=[0, 2, 1])  # [1, 80, -1]
cur_scores = conf_preds[0]  # [80, -1]
conf_scores = tf.reduce_max(cur_scores, axis=0)  # [-1, ]  每个预测框的所有类别的最高分数
keep = tf.where(conf_scores > conf_thresh)  # 最高分数大与阈值的保留
keep = tf.reshape(keep, (-1,))  # [-1, ]

# keep_extra = tf.where(conf_scores < conf_thresh)
# keep_extra = keep_extra[:1]
# keep = tf.concat([keep, keep_extra], axis=0)

scores = tf.gather(all_pred_scores[0], keep)  # [?, 80]
scores = tf.transpose(scores, perm=[1, 0])  # [80, ?]
boxes = tf.gather(all_pred_boxes[0], keep)  # [?, 4]
boxes, scores, classes = fast_nms(boxes, scores, conf_thresh, nms_thresh, keep_top_k, nms_top_k)

sess = K.get_session()
aaa00 = sess.run(scores, feed_dict={all_pred_boxes: all_pred_boxes2, all_pred_scores: all_pred_scores2,})


# aaa01 = np.mean(aaa00)573 405   372 394
# aaa02 = np.var(aaa00)

print()





