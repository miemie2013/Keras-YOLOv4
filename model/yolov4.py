#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : keras_yolov4
#
# ================================================================

import keras
import tensorflow as tf
import keras.layers as layers
from keras import backend as K
from keras.engine.topology import Layer
from model.fastnms import fastnms


# 对坐标解码
def decode(conv_output, anchors, stride, num_class):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_class))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))  # [-1, -1, 4]
    pred_conf = tf.reshape(pred_conf, (batch_size, -1, 1))  # [-1, -1, 1]
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, num_class))  # [-1, -1, 80]
    return pred_xywh, pred_conf, pred_prob


class PreLayer(Layer):
    def __init__(self):
        super(PreLayer, self).__init__()
    def compute_output_shape(self, input_shape):
        return (None, 416, 416, 3)
    def call(self, x):
        x = tf.image.resize_bicubic(x, (416, 416))
        x = x / 255.0
        return x

class Mish(Layer):
    def __init__(self):
        super(Mish, self).__init__()
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, x):
        return x * (K.tanh(K.softplus(x)))


def conv2d_unit(x, filters, kernels, strides=1, padding='valid', bn=1, act='mish'):
    use_bias = (bn != 1)
    x = layers.Conv2D(filters, kernels,
               padding=padding,
               strides=strides,
               use_bias=use_bias,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    if bn:
        x = layers.BatchNormalization()(x)
    if act == 'leaky':
        x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
    elif act == 'mish':
        x = Mish()(x)
    return x

def residual_block(inputs, filters_1, filters_2):
    x = conv2d_unit(inputs, filters_1, 1, strides=1, padding='valid')
    x = conv2d_unit(x, filters_2, 3, strides=1, padding='same')
    x = layers.add([inputs, x])
    # x = layers.Activation('linear')(x)
    return x

def stack_residual_block(inputs, filters_1, filters_2, n):
    x = residual_block(inputs, filters_1, filters_2)
    for i in range(n - 1):
        x = residual_block(x, filters_1, filters_2)
    return x

def spp(x):
    x_1 = x
    x_2 = layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
    x_3 = layers.MaxPooling2D(pool_size=9, strides=1, padding='same')(x)
    x_4 = layers.MaxPooling2D(pool_size=13, strides=1, padding='same')(x)
    out = layers.Concatenate()([x_4, x_3, x_2, x_1])
    return out

def YOLOv4(inputs, num_classes, num_anchors, initial_filters=32,
           fast=False, anchors=None, conf_thresh=0.05, nms_thresh=0.45, keep_top_k=100, nms_top_k=100):
    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8
    i512 = i32 * 16
    i1024 = i32 * 32

    if fast:
        # x = PreLayer()(inputs)
        x = inputs
    else:
        x = inputs

    # cspdarknet53部分
    x = conv2d_unit(x, i32, 3, strides=1, padding='same')

    # ============================= s2 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i64, 3, strides=2)
    s2 = conv2d_unit(x, i64, 1, strides=1)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = stack_residual_block(x, i32, i64, n=1)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = layers.Concatenate()([x, s2])
    x = conv2d_unit(x, i64, 1, strides=1)

    # ============================= s4 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i128, 3, strides=2)
    s4 = conv2d_unit(x, i64, 1, strides=1)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = stack_residual_block(x, i64, i64, n=2)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = layers.Concatenate()([x, s4])
    x = conv2d_unit(x, i128, 1, strides=1)

    # ============================= s8 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i256, 3, strides=2)
    s8 = conv2d_unit(x, i128, 1, strides=1)
    x = conv2d_unit(x, i128, 1, strides=1)
    x = stack_residual_block(x, i128, i128, n=8)
    x = conv2d_unit(x, i128, 1, strides=1)
    s8 = layers.Concatenate()([x, s8])
    x = conv2d_unit(s8, i256, 1, strides=1)

    # ============================= s16 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i512, 3, strides=2)
    s16 = conv2d_unit(x, i256, 1, strides=1)
    x = conv2d_unit(x, i256, 1, strides=1)
    x = stack_residual_block(x, i256, i256, n=8)
    x = conv2d_unit(x, i256, 1, strides=1)
    s16 = layers.Concatenate()([x, s16])
    x = conv2d_unit(s16, i512, 1, strides=1)

    # ============================= s32 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i1024, 3, strides=2)
    s32 = conv2d_unit(x, i512, 1, strides=1)
    x = conv2d_unit(x, i512, 1, strides=1)
    x = stack_residual_block(x, i512, i512, n=4)
    x = conv2d_unit(x, i512, 1, strides=1)
    x = layers.Concatenate()([x, s32])
    x = conv2d_unit(x, i1024, 1, strides=1)
    # cspdarknet53部分结束

    # fpn部分
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = spp(x)

    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    fpn_s32 = conv2d_unit(x, i512, 1, strides=1, act='leaky')

    x = conv2d_unit(fpn_s32, i256, 1, strides=1, act='leaky')
    x = layers.UpSampling2D(2)(x)
    s16 = conv2d_unit(s16, i256, 1, strides=1, act='leaky')
    x = layers.Concatenate()([s16, x])
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    fpn_s16 = conv2d_unit(x, i256, 1, strides=1, act='leaky')

    x = conv2d_unit(fpn_s16, i128, 1, strides=1, act='leaky')
    x = layers.UpSampling2D(2)(x)
    s8 = conv2d_unit(s8, i128, 1, strides=1, act='leaky')
    x = layers.Concatenate()([s8, x])

    # output_s
    x = conv2d_unit(x, i128, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i256, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i128, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i256, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i128, 1, strides=1, act='leaky')
    output_s = conv2d_unit(x, i256, 3, strides=1, padding='same', act='leaky')
    output_s = conv2d_unit(output_s, num_anchors * (num_classes + 5), 1, strides=1, bn=0, act=None)

    # output_m
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i256, 3, strides=2, act='leaky')
    x = layers.Concatenate()([x, fpn_s16])
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    output_m = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    output_m = conv2d_unit(output_m, num_anchors * (num_classes + 5), 1, strides=1, bn=0, act=None)

    # output_l
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i512, 3, strides=2, act='leaky')
    x = layers.Concatenate()([x, fpn_s32])
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    output_l = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    output_l = conv2d_unit(output_l, num_anchors * (num_classes + 5), 1, strides=1, bn=0, act=None)

    # 用张量操作实现后处理
    if fast:
        def output_layer(args):
            output_s, output_m, output_l = args

            # 先对坐标解码
            pred_xywh_s, pred_conf_s, pred_prob_s = decode(output_s, anchors[0], 8, num_classes)
            pred_xywh_m, pred_conf_m, pred_prob_m = decode(output_m, anchors[1], 16, num_classes)
            pred_xywh_l, pred_conf_l, pred_prob_l = decode(output_l, anchors[2], 32, num_classes)
            # 获取分数
            pred_score_s = pred_conf_s * pred_prob_s
            pred_score_m = pred_conf_m * pred_prob_m
            pred_score_l = pred_conf_l * pred_prob_l
            # 所有输出层的预测框集合后再执行nms
            all_pred_boxes = tf.concat([pred_xywh_s, pred_xywh_m, pred_xywh_l], axis=1)       # [batch_size, -1, 4]
            all_pred_scores = tf.concat([pred_score_s, pred_score_m, pred_score_l], axis=1)   # [batch_size, -1, 80]

            # 用fastnms
            output = fastnms(all_pred_boxes, all_pred_scores, conf_thresh, nms_thresh, keep_top_k, nms_top_k)

            return output
        output = layers.Lambda(output_layer)([output_s, output_m, output_l])
        model_body = keras.models.Model(inputs=inputs, outputs=output)
    else:
        model_body = keras.models.Model(inputs=inputs, outputs=[output_l, output_m, output_s])
    return model_body


