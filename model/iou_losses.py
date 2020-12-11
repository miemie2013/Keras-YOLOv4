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


class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 max_height=608,
                 max_width=608,
                 ciou_term=False,
                 loss_square=True):
        self._loss_weight = loss_weight
        self._MAX_HI = max_height
        self._MAX_WI = max_width
        self.ciou_term = ciou_term
        self.loss_square = loss_square

    def __call__(self,
                 x,
                 y,
                 w,
                 h,
                 tx,
                 ty,
                 tw,
                 th,
                 anchors,
                 downsample_ratio,
                 batch_size,
                 scale_x_y=1.,
                 ioup=None,
                 eps=1.e-10):
        '''
        Args:
            x  | y | w | h  ([Variables]): the output of yolov3 for encoded x|y|w|h
            tx |ty |tw |th  ([Variables]): the target of yolov3 for encoded x|y|w|h
            anchors ([float]): list of anchors for current output layer
            downsample_ratio (float): the downsample ratio for current output layer
            batch_size (int): training batch size
            eps (float): the decimal to prevent the denominator eqaul zero
        '''
        pred = self._bbox_transform(x, y, w, h, anchors, downsample_ratio,
                                    batch_size, False, scale_x_y, eps)
        gt = self._bbox_transform(tx, ty, tw, th, anchors, downsample_ratio,
                                  batch_size, True, scale_x_y, eps)
        iouk = self._iou(pred, gt, ioup, eps)
        if self.loss_square:
            loss_iou = 1. - iouk * iouk
        else:
            loss_iou = 1. - iouk
        loss_iou = loss_iou * self._loss_weight

        return loss_iou

    def _iou(self, pred, gt, ioup=None, eps=1.e-10):
        x1, y1, x2, y2 = pred
        x1g, y1g, x2g, y2g = gt
        x2 = tf.maximum(x1, x2)
        y2 = tf.maximum(y1, y2)

        xkis1 = tf.maximum(x1, x1g)
        ykis1 = tf.maximum(y1, y1g)
        xkis2 = tf.minimum(x2, x2g)
        ykis2 = tf.minimum(y2, y2g)

        inter_w = (xkis2 - xkis1)
        inter_h = (ykis2 - ykis1)
        inter_w = tf.maximum(inter_w, 0.0)
        inter_h = tf.maximum(inter_h, 0.0)
        intsctk = inter_w * inter_h

        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + eps
        iouk = intsctk / unionk
        if self.ciou_term:
            ciou = self.get_ciou_term(pred, gt, iouk, eps)
            iouk = iouk - ciou
        return iouk

    def get_ciou_term(self, pred, gt, iouk, eps):
        x1, y1, x2, y2 = pred
        x1g, y1g, x2g, y2g = gt

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) + 1e-9
        h = (y2 - y1) + 1e-9

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        # A or B
        xc1 = tf.minimum(x1, x1g)
        yc1 = tf.minimum(y1, y1g)
        xc2 = tf.maximum(x2, x2g)
        yc2 = tf.maximum(y2, y2g)

        # DIOU term
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + eps) / (dist_union + eps)
        # CIOU term
        ciou_term = 0
        ar_gt = wg / hg
        ar_pred = w / h
        arctan = tf.atan(ar_gt) - tf.atan(ar_pred)
        ar_loss = 4. / np.pi / np.pi * arctan * arctan
        alpha = ar_loss / (1 - iouk + ar_loss + eps)
        ciou_term = alpha * ar_loss
        return diou_term + ciou_term

    def _bbox_transform(self, dcx, dcy, dw, dh, anchors, downsample_ratio,
                        batch_size, is_gt, scale_x_y, eps):
        shape_fmp = tf.shape(dcx)
        # batch_size = shape_fmp[0]
        anchor_per_scale = shape_fmp[1]
        output_size = shape_fmp[2]
        output_size_f = tf.cast(output_size, tf.float32)
        rows = tf.range(output_size_f, dtype=tf.float32)
        cols = tf.range(output_size_f, dtype=tf.float32)
        rows = tf.tile(rows[tf.newaxis, tf.newaxis, tf.newaxis, :], [batch_size, anchor_per_scale, output_size, 1])
        cols = tf.tile(cols[tf.newaxis, tf.newaxis, :, tf.newaxis], [batch_size, anchor_per_scale, 1, output_size])

        if is_gt:
            cx = (dcx + rows) / output_size_f
            cy = (dcy + cols) / output_size_f
        else:
            dcx_sig = tf.sigmoid(dcx)
            dcy_sig = tf.sigmoid(dcy)
            if (abs(scale_x_y - 1.0) > eps):
                dcx_sig = scale_x_y * dcx_sig - 0.5 * (scale_x_y - 1)
                dcy_sig = scale_x_y * dcy_sig - 0.5 * (scale_x_y - 1)
            cx = (dcx_sig + rows) / output_size_f
            cy = (dcy_sig + cols) / output_size_f

        anchor_w_ = [anchors[i] for i in range(0, len(anchors)) if i % 2 == 0]
        anchor_w_np = np.array(anchor_w_)
        anchor_w_ = tf.ones(anchor_w_np.shape, dtype=tf.float32) * anchor_w_np
        anchor_w = tf.tile(anchor_w_[tf.newaxis, :, tf.newaxis, tf.newaxis], [batch_size, 1, output_size, output_size])

        anchor_h_ = [anchors[i] for i in range(0, len(anchors)) if i % 2 == 1]
        anchor_h_np = np.array(anchor_h_)
        anchor_h_ = tf.ones(anchor_h_np.shape, dtype=tf.float32) * anchor_h_np
        anchor_h = tf.tile(anchor_h_[tf.newaxis, :, tf.newaxis, tf.newaxis], [batch_size, 1, output_size, output_size])

        # e^tw e^th
        exp_dw = tf.exp(dw)
        exp_dh = tf.exp(dh)
        pw = (exp_dw * anchor_w) / (output_size_f * downsample_ratio)
        ph = (exp_dh * anchor_h) / (output_size_f * downsample_ratio)

        x1 = cx - 0.5 * pw
        y1 = cy - 0.5 * ph
        x2 = cx + 0.5 * pw
        y2 = cy + 0.5 * ph

        return x1, y1, x2, y2


class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=1.0, max_height=608, max_width=608):
        super(IouAwareLoss, self).__init__(
            loss_weight=loss_weight, max_height=max_height, max_width=max_width)

    def __call__(self,
                 ioup,
                 x,
                 y,
                 w,
                 h,
                 tx,
                 ty,
                 tw,
                 th,
                 anchors,
                 downsample_ratio,
                 batch_size,
                 scale_x_y,
                 eps=1.e-10):
        '''
        Args:
            ioup ([Variables]): the predicted iou
            x  | y | w | h  ([Variables]): the output of yolov3 for encoded x|y|w|h
            tx |ty |tw |th  ([Variables]): the target of yolov3 for encoded x|y|w|h
            anchors ([float]): list of anchors for current output layer
            downsample_ratio (float): the downsample ratio for current output layer
            batch_size (int): training batch size
            eps (float): the decimal to prevent the denominator eqaul zero
        '''

        pred = self._bbox_transform(x, y, w, h, anchors, downsample_ratio,
                                    batch_size, False, scale_x_y, eps)
        gt = self._bbox_transform(tx, ty, tw, th, anchors, downsample_ratio,
                                  batch_size, True, scale_x_y, eps)
        iouk = self._iou(pred, gt, ioup, eps)

        # cross_entropy
        # loss_iou_aware = fluid.layers.cross_entropy(ioup, iouk, soft_label=True)
        loss_iou_aware = iouk * (0 - tf.log(ioup + 1e-9))
        loss_iou_aware = tf.reduce_sum(loss_iou_aware, axis=[-1, ], keepdims=True)

        loss_iou_aware = loss_iou_aware * self._loss_weight
        return loss_iou_aware




