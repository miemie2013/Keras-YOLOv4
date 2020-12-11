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
import numpy as np
import copy
import keras
import keras.backend as K
from keras.engine.topology import Layer

from model.custom_layers import Conv2dUnit, SPP
from model.fast_nms import fast_nms
from model.matrix_nms import matrix_nms


def yolo_box(conv_output, anchors, stride, num_classes, scale_x_y, im_size, clip_bbox, conf_thresh):
    conv_output = tf.transpose(conv_output, [0, 2, 3, 1])
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    output_size_f = tf.cast(output_size, tf.float32)
    rows = tf.range(output_size_f, dtype=tf.float32)
    cols = tf.range(output_size_f, dtype=tf.float32)
    rows = tf.tile(rows[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis], [1, output_size, 1, 1, 1])
    cols = tf.tile(cols[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, output_size, 1, 1])
    offset = tf.concat([rows, cols], -1)
    offset = tf.tile(offset, [batch_size, 1, 1, anchor_per_scale, 1])
    # Grid Sensitive
    pred_xy = (scale_x_y * tf.sigmoid(conv_raw_dxdy) + offset - (scale_x_y - 1.0) * 0.5 ) * stride

    _anchors = tf.Variable(anchors)
    pred_wh = (tf.exp(conv_raw_dwdh) * _anchors)

    pred_xyxy = tf.concat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], -1)   # 左上角xy + 右下角xy
    pred_conf = tf.sigmoid(conv_raw_conf)
    # mask = (pred_conf > conf_thresh).float()
    pred_prob = tf.sigmoid(conv_raw_prob)
    pred_scores = pred_conf * pred_prob
    # pred_scores = pred_scores * mask
    # pred_xyxy = pred_xyxy * mask

    # paddle中实际的顺序
    # pred_xyxy = tf.transpose(pred_xyxy, [0, 3, 1, 2, 4])
    # pred_scores = tf.transpose(pred_scores, [0, 3, 1, 2, 4])

    pred_xyxy = tf.reshape(pred_xyxy, (batch_size, output_size*output_size*anchor_per_scale, 4))
    pred_scores = tf.reshape(pred_scores, (batch_size, tf.shape(pred_xyxy)[1], num_classes))

    _im_size_h = im_size[:, 0:1]
    _im_size_w = im_size[:, 1:2]
    _im_size = tf.concat([_im_size_w, _im_size_h], 1)
    _im_size = tf.reshape(_im_size, (-1, 1, 2))
    _im_size = tf.tile(_im_size, [1, tf.shape(pred_xyxy)[1], 1])
    _im_size_f = tf.cast(_im_size, tf.float32)
    pred_x0y0 = pred_xyxy[:, :, 0:2] / output_size_f / stride * _im_size_f
    pred_x1y1 = pred_xyxy[:, :, 2:4] / output_size_f / stride * _im_size_f
    if clip_bbox:
        x0 = pred_x0y0[:, :, 0:1]
        y0 = pred_x0y0[:, :, 1:2]
        x1 = pred_x1y1[:, :, 0:1]
        y1 = pred_x1y1[:, :, 1:2]
        x0 = tf.where(x0 < 0, x0 * 0, x0)
        y0 = tf.where(y0 < 0, y0 * 0, y0)
        x1 = tf.where(x1 > _im_size_f[:, :, 0:1], _im_size_f[:, :, 0:1], x1)
        y1 = tf.where(y1 > _im_size_f[:, :, 1:2], _im_size_f[:, :, 1:2], y1)
        pred_xyxy = tf.concat([x0, y0, x1, y1], -1)
    else:
        pred_xyxy = tf.concat([pred_x0y0, pred_x1y1], -1)
    return pred_xyxy, pred_scores


def _split_ioup(output, an_num, num_classes):
    """
    Split new output feature map to output, predicted iou
    along channel dimension
    """
    ioup = output[:, :an_num, :, :]
    ioup = tf.sigmoid(ioup)

    oriout = output[:, an_num:, :, :]

    return (ioup, oriout)


# sigmoid()函数的反函数。先取倒数再减一，取对数再取相反数。
def _de_sigmoid(x, eps=1e-7):
    # x限制在区间[eps, 1 / eps]内
    x = tf.maximum(x, eps)
    x = tf.minimum(x, 1 / eps)

    # 先取倒数再减一
    x = 1.0 / x - 1.0

    # e^(-x)限制在区间[eps, 1 / eps]内
    x = tf.maximum(x, eps)
    x = tf.minimum(x, 1 / eps)

    # 取对数再取相反数
    x = -tf.keras.backend.log(x)
    return x


def _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor):
    """
    post process output objectness score
    """
    tensors = []
    stride = output.shape[1] // an_num
    for m in range(an_num):
        tensors.append(output[:, stride * m:stride * m + 4, :, :])
        obj = output[:, stride * m + 4:stride * m + 5, :, :]
        obj = tf.sigmoid(obj)

        ip = ioup[:, m:m + 1, :, :]

        new_obj = tf.pow(obj, (1 - iou_aware_factor)) * tf.pow(ip, iou_aware_factor)
        new_obj = _de_sigmoid(new_obj)   # 置信位未进行sigmoid()激活

        tensors.append(new_obj)

        tensors.append(output[:, stride * m + 5:stride * m + 5 + num_classes, :, :])

    output = tf.concat(tensors, 1)

    return output



def get_iou_aware_score(output, an_num, num_classes, iou_aware_factor):
    ioup, output = _split_ioup(output, an_num, num_classes)
    output = _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor)
    return output



class CoordConv(Layer):
    def __init__(self, in_c, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv
        self.in_c = in_c

    def compute_output_shape(self, input_shape):
        c = self.in_c
        if self.coord_conv:
            c += 2
        return (None, None, None, c)

    def call(self, input):
        if not self.coord_conv:
            return input
        b = tf.shape(input)[0]
        h = tf.shape(input)[1]
        w = tf.shape(input)[2]
        w_f = tf.cast(w, tf.float32)
        h_f = tf.cast(h, tf.float32)
        x_range = tf.range(w_f, dtype=tf.float32) / (w_f - 1) * 2.0 - 1
        y_range = tf.range(h_f, dtype=tf.float32) / (h_f - 1) * 2.0 - 1
        x_range = tf.tile(x_range[tf.newaxis, tf.newaxis, :, tf.newaxis], [b, h, 1, 1])
        y_range = tf.tile(y_range[tf.newaxis, :, tf.newaxis, tf.newaxis], [b, 1, w, 1])
        offset = tf.concat([input, x_range, y_range], -1)
        return offset


class DropBlock(Layer):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, training=None):
        def dropblock_inference():
            return input

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return dropblock_inference()

        def CalculateGamma(input, block_size, keep_prob):
            h = tf.shape(input)[1]
            h_f = tf.cast(h, tf.float32)
            feat_shape_t = tf.zeros((1, 1, 1, 1), dtype=tf.float32) + h_f  # shape: [1, 1, 1, 1]
            feat_area = tf.pow(feat_shape_t, 2)  # shape: [1, 1, 1, 1]

            block_shape_t = tf.zeros((1, 1, 1, 1), dtype=tf.float32) + float(block_size)
            block_area = tf.pow(block_shape_t, 2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = tf.pow(useful_shape_t, 2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = tf.shape(input)
        p = tf.tile(gamma, input_shape)

        input_shape_tmp = tf.shape(input)
        random_matrix = tf.random_uniform(input_shape_tmp)
        one_zero_m = tf.cast(random_matrix < p, tf.float32)

        mask_flag = K.pool2d(one_zero_m, (self.block_size, self.block_size), strides=(1, 1), padding='same', pool_mode='max')
        mask = 1.0 - mask_flag

        elem_numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        elem_numel_m = tf.cast(elem_numel, tf.float32)

        elem_sum = tf.reduce_sum(mask)

        output = input * mask * elem_numel_m / elem_sum
        return output




class DetectionBlock(object):
    def __init__(self,
                 in_c,
                 channel,
                 coord_conv=True,
                 bn=0,
                 gn=0,
                 af=0,
                 conv_block_num=2,
                 is_first=False,
                 use_spp=True,
                 drop_block=True,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=True,
                 name=''):
        super(DetectionBlock, self).__init__()
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        self.use_spp = use_spp
        self.coord_conv = coord_conv
        self.is_first = is_first
        self.is_test = is_test
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob

        self.layers = []
        self.tip_layers = []
        for j in range(conv_block_num):
            coordConv = CoordConv(in_c, coord_conv)
            input_c = in_c + 2 if coord_conv else in_c
            conv_unit1 = Conv2dUnit(input_c, channel, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='{}.{}.0'.format(name, j))
            self.layers.append(coordConv)
            self.layers.append(conv_unit1)
            if self.use_spp and is_first and j == 1:
                spp = SPP(channel * 4, seq='asc')
                conv_unit2 = Conv2dUnit(channel * 4, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='{}.{}.spp.conv'.format(name, j))
                conv_unit3 = Conv2dUnit(512, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='{}.{}.1'.format(name, j))
                self.layers.append(spp)
                self.layers.append(conv_unit2)
                self.layers.append(conv_unit3)
            else:
                conv_unit3 = Conv2dUnit(channel, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='{}.{}.1'.format(name, j))
                self.layers.append(conv_unit3)

            if self.drop_block and j == 0 and not is_first:
                dropBlock = DropBlock(
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)
                self.layers.append(dropBlock)
            in_c = channel * 2

        if self.drop_block and is_first:
            dropBlock = DropBlock(
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
            self.layers.append(dropBlock)
        if conv_block_num == 0:
            input_c = in_c
        else:
            input_c = channel * 2
        coordConv = CoordConv(input_c, coord_conv)
        input_c = input_c + 2 if coord_conv else input_c
        conv_unit = Conv2dUnit(input_c, channel, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='{}.2'.format(name))
        self.layers.append(coordConv)
        self.layers.append(conv_unit)

        coordConv = CoordConv(channel, coord_conv)
        input_c = channel + 2 if coord_conv else channel
        conv_unit = Conv2dUnit(input_c, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='{}.tip'.format(name))
        self.tip_layers.append(coordConv)
        self.tip_layers.append(conv_unit)

    def __call__(self, input):
        conv = input
        for ly in self.layers:
            conv = ly(conv)
        route = conv
        tip = conv
        for ly in self.tip_layers:
            tip = ly(tip)
        return route, tip


class YOLOv3Head(object):
    def __init__(self,
                 conv_block_num=2,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23],
                          [30, 61], [62, 45], [59, 119],
                          [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 norm_type="bn",
                 coord_conv=True,
                 iou_aware=True,
                 iou_aware_factor=0.4,
                 block_size=3,
                 scale_x_y=1.05,
                 spp=True,
                 drop_block=True,
                 keep_prob=0.9,
                 clip_bbox=True,
                 yolo_loss=None,
                 downsample=[32, 16, 8],
                 in_channels=[2048, 1024, 512],
                 nms_cfg=None,
                 is_train=False
                 ):
        super(YOLOv3Head, self).__init__()
        self.conv_block_num = conv_block_num
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.coord_conv = coord_conv
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.scale_x_y = scale_x_y
        self.use_spp = spp
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.clip_bbox = clip_bbox
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.block_size = block_size
        self.downsample = downsample
        self.in_channels = in_channels
        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg
        self.is_train = is_train

        _anchors = copy.deepcopy(anchors)
        _anchors = np.array(_anchors)
        _anchors = _anchors.astype(np.float32)
        self._anchors = _anchors   # [9, 2]

        self.mask_anchors = []
        for m in anchor_masks:
            temp = []
            for aid in m:
                temp += anchors[aid]
            self.mask_anchors.append(temp)

        bn = 0
        gn = 0
        af = 0
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        if norm_type == 'bn':
            bn = 1
        elif norm_type == 'sync_bn':
            bn = 1
        elif norm_type == 'gn':
            gn = 1
        elif norm_type == 'affine_channel':
            af = 1

        self.detection_blocks = []
        self.yolo_output_convs = []
        self.upsample_layers = []
        out_layer_num = len(downsample)
        for i in range(out_layer_num):
            in_c = self.in_channels[i]
            if i > 0:  # perform concat in first 2 detection_block
                in_c = self.in_channels[i] + 512 // (2**i)
            _detection_block = DetectionBlock(
                in_c=in_c,
                channel=64 * (2**out_layer_num) // (2**i),
                coord_conv=self.coord_conv,
                bn=bn,
                gn=gn,
                af=af,
                is_first=i == 0,
                conv_block_num=self.conv_block_num,
                use_spp=self.use_spp,
                drop_block=self.drop_block,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=(not self.is_train),
                name="yolo_block.{}".format(i)
            )
            # out channel number = mask_num * (5 + class_num)
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            yolo_output_conv = Conv2dUnit(64 * (2**out_layer_num) // (2**i) * 2, num_filters, 1, stride=1, bias_attr=True, bn=0, gn=0, af=0, act=None, name="yolo_output.{}.conv".format(i))
            self.detection_blocks.append(_detection_block)
            self.yolo_output_convs.append(yolo_output_conv)


            if i < out_layer_num - 1:
                # do not perform upsample in the last detection_block
                conv_unit = Conv2dUnit(64 * (2**out_layer_num) // (2**i), 256 // (2**i), 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name="yolo_transition.{}".format(i))
                # upsample
                upsample = keras.layers.UpSampling2D(2, interpolation='nearest')
                self.upsample_layers.append(conv_unit)
                self.upsample_layers.append(upsample)

    def set_dropblock(self, is_test):
        for detection_block in self.detection_blocks:
            for l in detection_block.layers:
                if isinstance(l, DropBlock):
                    l.is_test = is_test

    def _get_outputs(self, body_feats):
        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = body_feats[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                block = keras.layers.Concatenate(axis=-1)([route, block])
            route, tip = self.detection_blocks[i](block)
            block_out = self.yolo_output_convs[i](tip)
            outputs.append(block_out)
            if i < out_layer_num - 1:
                route = self.upsample_layers[i*2](route)
                route = self.upsample_layers[i*2+1](route)
        return outputs

    def get_loss(self, outputs, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            outputs (list): 大中小感受野的输出，为NHWC格式
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """

        outputs = [tf.transpose(o, [0, 3, 1, 2]) for o in outputs]

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes)

    def get_prediction(self, outputs, im_size):
        """
        Get prediction result of YOLOv3 network

        Args:
            outputs (list): 大中小感受野的输出，为NHWC格式
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): shape = [bs, keep_top_k, 6]

        """

        def output_layer(args):
            outputs = args
            outputs = [tf.transpose(o, [0, 3, 1, 2]) for o in outputs]

            boxes = []
            scores = []
            for i, output in enumerate(outputs):
                if self.iou_aware:
                    output = get_iou_aware_score(output,
                                                 len(self.anchor_masks[i]),
                                                 self.num_classes,
                                                 self.iou_aware_factor)
                box, score = yolo_box(output, self._anchors[self.anchor_masks[i]], self.downsample[i],
                                      self.num_classes, self.scale_x_y, im_size, self.clip_bbox,
                                      conf_thresh=self.nms_cfg['score_threshold'])
                boxes.append(box)
                scores.append(score)
            yolo_boxes = tf.concat(boxes, 1)
            yolo_scores = tf.concat(scores, 1)

            # nms
            nms_cfg = self.nms_cfg
            def _process_sample(args):
                _yolo_boxes, _yolo_scores = args
                # _yolo_boxes:   [?, 4]
                # _yolo_scores:  [?, 80]

                nms_type = self.nms_cfg['nms_type']
                if nms_type == 'matrix_nms':
                    pred = matrix_nms(_yolo_boxes, _yolo_scores,
                                      score_threshold=nms_cfg['score_threshold'],
                                      post_threshold=nms_cfg['post_threshold'],
                                      nms_top_k=nms_cfg['nms_top_k'],
                                      keep_top_k=nms_cfg['keep_top_k'],
                                      use_gaussian=nms_cfg['use_gaussian'],
                                      gaussian_sigma=nms_cfg['gaussian_sigma'])
                elif nms_type == 'fast_nms':
                    pred = fast_nms(_yolo_boxes, _yolo_scores,
                                    score_threshold=nms_cfg['score_threshold'],
                                    nms_threshold=nms_cfg['nms_threshold'],
                                    nms_top_k=nms_cfg['nms_top_k'],
                                    keep_top_k=nms_cfg['keep_top_k'])
                return pred
            preds = tf.map_fn(_process_sample, [yolo_boxes, yolo_scores], dtype=tf.float32)
            return preds
        preds = keras.layers.Lambda(output_layer)(outputs)
        return preds





