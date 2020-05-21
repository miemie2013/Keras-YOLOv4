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
import keras.layers as layers
from keras import backend as K
from keras.engine.topology import Layer


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

def YOLOv4(inputs, num_classes, num_anchors, initial_filters = 32):
    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8
    i512 = i32 * 16
    i1024 = i32 * 32


    # cspdarknet53部分
    x = conv2d_unit(inputs, i32, 3, strides=1, padding='same')

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

    model_body = keras.models.Model(inputs=inputs, outputs=[output_l, output_m, output_s])
    return model_body


