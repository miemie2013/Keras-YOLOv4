#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np



class Mish(Layer):
    def __init__(self):
        super(Mish, self).__init__()
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, x):
        return x * (K.tanh(K.softplus(x)))


class SPP(Layer):
    def __init__(self, out_c, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.out_c = out_c
        self.seq = seq
        self.pool1 = keras.layers.MaxPool2D(pool_size=5, strides=1, padding='same')
        self.pool2 = keras.layers.MaxPool2D(pool_size=9, strides=1, padding='same')
        self.pool3 = keras.layers.MaxPool2D(pool_size=13, strides=1, padding='same')

    def compute_output_shape(self, input_shape):
        return (None, None, None, self.out_c)

    def call(self, x):
        x_1 = x
        x_2 = self.pool1(x)
        x_3 = self.pool2(x)
        x_4 = self.pool3(x)
        if self.seq == 'desc':
            out = tf.concat([x_4, x_3, x_2, x_1], -1)
        else:
            out = tf.concat([x_1, x_2, x_3, x_4], -1)
        return out



# tf2教程 https://blog.csdn.net/qq_31456593/article/details/95040756
class Conv2dUnit(object):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 bn=0,
                 gn=0,
                 af=0,
                 groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 use_dcn=False,
                 name=''):
        super(Conv2dUnit, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.groups = groups
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn

        # conv
        pad = (filter_size - 1) // 2
        self.padding = pad
        padding = None
        self.zero_padding = None
        if pad == 0:
            padding = 'valid'
        else:
            if stride == 1:
                padding = 'same'
            elif stride == 2:
                if not use_dcn:
                    padding = 'valid'
                    self.zero_padding = keras.layers.ZeroPadding2D(padding=((pad, 0), (pad, 0)))
        kernel_initializer = 'glorot_uniform'
        bias_initializer = 'zeros'
        if use_dcn:
            # 咩酱自实现的DCNv2，咩酱的得意之作，tensorflow的纯python接口实现，效率极高。
            pad2 = (filter_size - 1) // 2
            self.conv = DCNv2(input_dim, filters, filter_size=filter_size, stride=stride, padding=pad2, bias_attr=False, name=name+'.conv')
        else:
            self.conv = keras.layers.Conv2D(filters, kernel_size=filter_size, strides=stride, padding=padding, use_bias=bias_attr,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name=name+'.conv')

        # norm
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            # PPYOLO使用的epsilon=1e-5，不设置的话会有点偏差。
            self.bn = keras.layers.BatchNormalization(name=name+'.bn', epsilon=1e-5)
        if gn:
            pass
            # self.gn = keras.layers.BatchNormalization()
        if af:
            pass
            # self.af = keras.layers.BatchNormalization()

        # act
        self.act = None
        if act == 'relu':
            self.act = keras.layers.ReLU()
        elif act == 'leaky':
            self.act = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)
        elif act == 'mish':
            self.act = Mish()


    def freeze(self):
        self.conv.trainable = False
        if self.bn is not None:
            self.bn.trainable = False

    def __call__(self, x):
        if self.zero_padding:
            x = self.zero_padding(x)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = self.af(x)
        if self.act:
            x = self.act(x)
        return x



class DCNv2(Layer):
    '''
    咩酱自实现的DCNv2，咩酱的得意之作，tensorflow的纯python接口实现，效率极高。
    '''
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 distribution='normal',
                 gain=1,
                 name=''):
        super(DCNv2, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias_attr = bias_attr

        self.conv_offset_padding = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
        self.zero_padding = keras.layers.ZeroPadding2D(padding=((padding, padding+1), (padding, padding+1)))

    def build(self, input_shape):
        input_dim = self.input_dim
        filters = self.filters
        filter_size = self.filter_size
        bias_attr = self.bias_attr
        self.offset_w = self.add_weight('offset_w', shape=[filter_size, filter_size, input_dim, filter_size * filter_size * 3], initializer='zeros')
        self.offset_b = self.add_weight('offset_b', shape=[1, 1, 1, filter_size * filter_size * 3], initializer='zeros')
        self.dcn_weight = self.add_weight('dcn_weight', shape=[filters, input_dim, filter_size, filter_size], initializer='uniform')
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = self.add_weight('dcn_bias', shape=[filters, ], initializer='zeros')

    def compute_output_shape(self, input_shape):
        filters = self.filters
        return (None, None, None, filters)

    def call(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias


        # 当filter_size = 3, stride = 2, padding = 1时， 设置padding2 = 'valid'，K.conv2d层前加一个self.conv_offset_padding
        # 当filter_size = 3, stride = 1, padding = 1时， 设置padding2 = 'same'，K.conv2d层前不用加一个self.conv_offset_padding
        # 无论什么条件，self.zero_padding层都是必须要加的。
        if stride == 2:
            temp = self.conv_offset_padding(x)
        else:
            temp = x
        padding2 = None
        if stride == 2:
            padding2 = 'valid'
        else:
            padding2 = 'same'
        offset_mask = K.conv2d(temp, self.offset_w, strides=(stride, stride), padding=padding2)
        offset_mask += self.offset_b

        offset_mask = tf.transpose(offset_mask, [0, 3, 1, 2])
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = tf.nn.sigmoid(mask)


        # ===================================
        N = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        out_C = tf.shape(dcn_weight)[0]
        in_C = tf.shape(dcn_weight)[1]
        kH = tf.shape(dcn_weight)[2]
        kW = tf.shape(dcn_weight)[3]
        W_f = tf.cast(W, tf.float32)
        H_f = tf.cast(H, tf.float32)
        kW_f = tf.cast(kW, tf.float32)
        kH_f = tf.cast(kH, tf.float32)

        out_W = (W_f + 2 * padding - (kW_f - 1)) // stride
        out_H = (H_f + 2 * padding - (kH_f - 1)) // stride
        out_W = tf.cast(out_W, tf.int32)
        out_H = tf.cast(out_H, tf.int32)
        out_W_f = tf.cast(out_W, tf.float32)
        out_H_f = tf.cast(out_H, tf.float32)

        # 1.先对图片x填充得到填充后的图片pad_x
        pad_x = self.zero_padding(x)
        pad_x = tf.transpose(pad_x, [0, 3, 1, 2])

        # 卷积核中心点在pad_x中的位置
        rows = tf.range(out_W_f, dtype=tf.float32) * stride + padding
        cols = tf.range(out_H_f, dtype=tf.float32) * stride + padding
        rows = tf.tile(rows[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis], [1, out_H, 1, 1, 1])
        cols = tf.tile(cols[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, out_W, 1, 1])
        start_pos_yx = tf.concat([cols, rows], axis=-1)  # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = tf.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置

        # 卷积核内部的偏移
        half_W = (kW_f - 1) / 2
        half_H = (kH_f - 1) / 2
        rows2 = tf.range(kW_f, dtype=tf.float32) - half_W
        cols2 = tf.range(kH_f, dtype=tf.float32) - half_H
        rows2 = tf.tile(rows2[tf.newaxis, :, tf.newaxis], [kH, 1, 1])
        cols2 = tf.tile(cols2[:, tf.newaxis, tf.newaxis], [1, kW, 1])
        filter_inner_offset_yx = tf.concat([cols2, rows2], axis=-1)  # [kH, kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = tf.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = tf.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移

        mask = tf.transpose(mask, [0, 2, 3, 1])       # [N, out_H, out_W, kH*kW*1]
        offset = tf.transpose(offset, [0, 2, 3, 1])   # [N, out_H, out_W, kH*kW*2]
        offset_yx = tf.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终位置
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = tf.maximum(pos_y, 0.0)
        pos_y = tf.minimum(pos_y, H_f + padding * 2 - 1.0)
        pos_x = tf.maximum(pos_x, 0.0)
        pos_x = tf.minimum(pos_x, W_f + padding * 2 - 1.0)
        ytxt = tf.concat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = tf.transpose(pad_x, [0, 2, 3, 1])  # [N, pad_x_H, pad_x_W, C]

        mask = tf.reshape(mask, (N, out_H, out_W, kH, kW))  # [N, out_H, out_W, kH, kW]

        def _process_sample(args):
            _pad_x, _mask, _ytxt = args
            # _pad_x:    [pad_x_H, pad_x_W, in_C]
            # _mask:     [out_H, out_W, kH, kW]
            # _ytxt:     [out_H, out_W, kH*kW, 2]

            _ytxt = tf.reshape(_ytxt, (out_H * out_W * kH * kW, 2))  # [out_H*out_W*kH*kW, 2]
            _yt = _ytxt[:, :1]
            _xt = _ytxt[:, 1:]
            _y1 = tf.floor(_yt)
            _x1 = tf.floor(_xt)
            _y2 = _y1 + 1.0
            _x2 = _x1 + 1.0
            _y1x1 = tf.concat([_y1, _x1], -1)
            _y1x2 = tf.concat([_y1, _x2], -1)
            _y2x1 = tf.concat([_y2, _x1], -1)
            _y2x2 = tf.concat([_y2, _x2], -1)

            _y1x1_int = tf.cast(_y1x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v1 = tf.gather_nd(_pad_x, _y1x1_int)  # [out_H*out_W*kH*kW, in_C]
            _y1x2_int = tf.cast(_y1x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v2 = tf.gather_nd(_pad_x, _y1x2_int)  # [out_H*out_W*kH*kW, in_C]
            _y2x1_int = tf.cast(_y2x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v3 = tf.gather_nd(_pad_x, _y2x1_int)  # [out_H*out_W*kH*kW, in_C]
            _y2x2_int = tf.cast(_y2x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
            v4 = tf.gather_nd(_pad_x, _y2x2_int)  # [out_H*out_W*kH*kW, in_C]

            lh = _yt - _y1  # [out_H*out_W*kH*kW, 1]
            lw = _xt - _x1
            hh = 1 - lh
            hw = 1 - lw
            w1 = hh * hw
            w2 = hh * lw
            w3 = lh * hw
            w4 = lh * lw
            value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [out_H*out_W*kH*kW, in_C]
            _mask = tf.reshape(_mask, (out_H * out_W * kH * kW, 1))
            value = value * _mask
            value = tf.reshape(value, (out_H, out_W, kH, kW, in_C))
            value = tf.transpose(value, [0, 1, 4, 2, 3])   # [out_H, out_W, in_C, kH, kW]
            return value

        # 旧的方案，使用逐元素相乘，慢！
        # new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
        # new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))   # [N, out_H, out_W, in_C * kH * kW]
        # new_x = tf.transpose(new_x, [0, 3, 1, 2])  # [N, in_C*kH*kW, out_H, out_W]
        # exp_new_x = tf.reshape(new_x, (N, 1, in_C*kH*kW, out_H, out_W))  # 增加1维，[N,      1, in_C*kH*kW, out_H, out_W]
        # reshape_w = tf.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))      # [1, out_C,  in_C*kH*kW,     1,     1]
        # out = exp_new_x * reshape_w                                   # 逐元素相乘，[N, out_C,  in_C*kH*kW, out_H, out_W]
        # out = tf.reduce_sum(out, axis=[2, ])                           # 第2维求和，[N, out_C, out_H, out_W]
        # out = tf.transpose(out, [0, 2, 3, 1])

        # 新的方案，用等价的1x1卷积代替逐元素相乘，快！
        new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
        new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))                # [N, out_H, out_W, in_C * kH * kW]
        tw = tf.transpose(dcn_weight, [1, 2, 3, 0])      # [out_C, in_C, kH, kW] -> [in_C, kH, kW, out_C]
        tw = tf.reshape(tw, (1, 1, in_C*kH*kW, out_C))   # [1, 1, in_C*kH*kW, out_C]  变成1x1卷积核
        out = K.conv2d(new_x, tw, strides=(1, 1), padding='valid')     # [N, out_H, out_W, out_C]
        return out



