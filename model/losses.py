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

from model.matrix_nms import jaccard

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


def paddle_yolo_box(conv_output, anchors, stride, num_classes, scale_x_y, im_size, clip_bbox, conf_thresh):
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
    pred_xyxy = tf.transpose(pred_xyxy, [0, 3, 1, 2, 4])
    pred_scores = tf.transpose(pred_scores, [0, 3, 1, 2, 4])

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



class YOLOv3Loss(object):
    """
    Combined loss for YOLOv3 network

    Args:
        batch_size (int): training batch size
        ignore_thresh (float): threshold to ignore confidence loss
        label_smooth (bool): whether to use label smoothing
        use_fine_grained_loss (bool): whether use fine grained YOLOv3 loss
                                      instead of fluid.layers.yolov3_loss
    """

    def __init__(self,
                 ignore_thresh=0.7,
                 label_smooth=True,
                 use_fine_grained_loss=False,
                 iou_loss=None,
                 iou_aware_loss=None,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 match_score=False):
        self._ignore_thresh = ignore_thresh
        self._label_smooth = label_smooth
        self._use_fine_grained_loss = use_fine_grained_loss
        self._iou_loss = iou_loss
        self._iou_aware_loss = iou_aware_loss
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.match_score = match_score

    def __call__(self, outputs, gt_box, gt_label, gt_score, targets, anchors,
                 anchor_masks, mask_anchors, num_classes):
        return self._get_fine_grained_loss(
            outputs, targets, gt_box, num_classes,
            mask_anchors, self._ignore_thresh)

    def _get_fine_grained_loss(self,
                               outputs,
                               targets,
                               gt_box,
                               num_classes,
                               mask_anchors,
                               ignore_thresh,
                               eps=1.e-10):
        """
        Calculate fine grained YOLOv3 loss

        Args:
            outputs ([Variables]): List of Variables, output of backbone stages
            targets ([Variables]): List of Variables, The targets for yolo
                                   loss calculatation.
            gt_box (Variable): The ground-truth boudding boxes.
            batch_size (int): The training batch size
            num_classes (int): class num of dataset
            mask_anchors ([[float]]): list of anchors in each output layer
            ignore_thresh (float): prediction bbox overlap any gt_box greater
                                   than ignore_thresh, objectness loss will
                                   be ignored.

        Returns:
            Type: dict
                xy_loss (Variable): YOLOv3 (x, y) coordinates loss
                wh_loss (Variable): YOLOv3 (w, h) coordinates loss
                obj_loss (Variable): YOLOv3 objectness score loss
                cls_loss (Variable): YOLOv3 classification loss

        """

        assert len(outputs) == len(targets), \
            "YOLOv3 output layer number not equal target number"

        batch_size = tf.shape(gt_box)[0]
        loss_xys, loss_whs, loss_objs, loss_clss = 0.0, 0.0, 0.0, 0.0
        if self._iou_loss is not None:
            loss_ious = 0.0
        if self._iou_aware_loss is not None:
            loss_iou_awares = 0.0
        for i, (output, target,
                anchors) in enumerate(zip(outputs, targets, mask_anchors)):
            downsample = self.downsample[i]
            an_num = len(anchors) // 2
            if self._iou_aware_loss is not None:
                ioup, output = self._split_ioup(output, an_num, num_classes)
            x, y, w, h, obj, cls = self._split_output(output, an_num,
                                                      num_classes)
            tx, ty, tw, th, tscale, tobj, tcls = self._split_target(target)

            tscale_tobj = tscale * tobj

            scale_x_y = self.scale_x_y if not isinstance(
                self.scale_x_y, Sequence) else self.scale_x_y[i]

            if (abs(scale_x_y - 1.0) < eps):
                sigmoid_x = tf.sigmoid(x)
                loss_x = tx * (0 - tf.log(sigmoid_x + 1e-9)) + (1 - tx) * (0 - tf.log(1 - sigmoid_x + 1e-9))
                loss_x *= tscale_tobj
                loss_x = tf.reduce_sum(loss_x, axis=[1, 2, 3])
                sigmoid_y = tf.sigmoid(y)
                loss_y = ty * (0 - tf.log(sigmoid_y + 1e-9)) + (1 - ty) * (0 - tf.log(1 - sigmoid_y + 1e-9))
                loss_y *= tscale_tobj
                loss_y = tf.reduce_sum(loss_y, axis=[1, 2, 3])
            else:
                # Grid Sensitive
                dx = scale_x_y * tf.sigmoid(x) - 0.5 * (scale_x_y - 1.0)
                dy = scale_x_y * tf.sigmoid(y) - 0.5 * (scale_x_y - 1.0)
                loss_x = tf.abs(dx - tx) * tscale_tobj
                loss_x = tf.reduce_sum(loss_x, axis=[1, 2, 3])
                loss_y = tf.abs(dy - ty) * tscale_tobj
                loss_y = tf.reduce_sum(loss_y, axis=[1, 2, 3])

            # NOTE: we refined loss function of (w, h) as L1Loss
            loss_w = tf.abs(w - tw) * tscale_tobj
            loss_w = tf.reduce_sum(loss_w, axis=[1, 2, 3])
            loss_h = tf.abs(h - th) * tscale_tobj
            loss_h = tf.reduce_sum(loss_h, axis=[1, 2, 3])
            if self._iou_loss is not None:
                loss_iou = self._iou_loss(x, y, w, h, tx, ty, tw, th, anchors,
                                          downsample, batch_size,
                                          scale_x_y)
                loss_iou = loss_iou * tscale_tobj
                loss_iou = tf.reduce_sum(loss_iou, axis=[1, 2, 3])
                loss_ious += tf.reduce_mean(loss_iou)

            if self._iou_aware_loss is not None:
                loss_iou_aware = self._iou_aware_loss(
                    ioup, x, y, w, h, tx, ty, tw, th, anchors, downsample,
                    batch_size, scale_x_y)
                loss_iou_aware = loss_iou_aware * tobj
                loss_iou_aware = tf.reduce_sum(loss_iou_aware, axis=[1, 2, 3])
                loss_iou_awares += tf.reduce_mean(loss_iou_aware)

            loss_obj_pos, loss_obj_neg = self._calc_obj_loss(
                output, obj, tobj, gt_box, batch_size, anchors,
                num_classes, downsample, self._ignore_thresh, scale_x_y)

            sigmoid_cls = tf.sigmoid(cls)
            loss_cls = tcls * (0 - tf.log(sigmoid_cls + 1e-9)) + (1 - tcls) * (0 - tf.log(1 - sigmoid_cls + 1e-9))
            loss_cls = tf.reduce_sum(loss_cls, axis=[4, ])
            loss_cls *= tobj
            loss_cls = tf.reduce_sum(loss_cls, axis=[1, 2, 3])

            loss_xys += tf.reduce_mean(loss_x + loss_y)
            loss_whs += tf.reduce_mean(loss_w + loss_h)
            loss_objs += tf.reduce_mean(loss_obj_pos + loss_obj_neg)
            loss_clss += tf.reduce_mean(loss_cls)

        loss_list = [loss_xys, loss_whs, loss_objs, loss_clss]
        if self._iou_loss is not None:
            loss_list.append(loss_ious)
        if self._iou_aware_loss is not None:
            loss_list.append(loss_iou_awares)
        return loss_list

    def _split_ioup(self, output, an_num, num_classes):
        """
        Split output feature map to output, predicted iou
        along channel dimension
        """
        ioup = output[:, :an_num, :, :]
        ioup = tf.sigmoid(ioup)

        oriout = output[:, an_num:, :, :]
        return (ioup, oriout)

    def _split_output(self, output, an_num, num_classes):
        """
        Split output feature map to x, y, w, h, objectness, classification
        along channel dimension
        """
        batch_size = tf.shape(output)[0]
        output_size = tf.shape(output)[2]
        output = tf.reshape(output, (batch_size, an_num, 5 + num_classes, output_size, output_size))
        x = output[:, :, 0, :, :]
        y = output[:, :, 1, :, :]
        w = output[:, :, 2, :, :]
        h = output[:, :, 3, :, :]
        obj = output[:, :, 4, :, :]
        cls = output[:, :, 5:, :, :]
        cls = tf.transpose(cls, [0, 1, 3, 4, 2])
        return (x, y, w, h, obj, cls)

    def _split_target(self, target):
        """
        split target to x, y, w, h, objectness, classification
        along dimension 2

        target is in shape [N, an_num, 6 + class_num, H, W]
        """
        tx = target[:, :, 0, :, :]
        ty = target[:, :, 1, :, :]
        tw = target[:, :, 2, :, :]
        th = target[:, :, 3, :, :]

        tscale = target[:, :, 4, :, :]
        tobj = target[:, :, 5, :, :]

        tcls = target[:, :, 6:, :, :]
        tcls = tf.transpose(tcls, [0, 1, 3, 4, 2])

        return (tx, ty, tw, th, tscale, tobj, tcls)

    def _calc_obj_loss(self, output, obj, tobj, gt_box, batch_size, anchors,
                       num_classes, downsample, ignore_thresh, scale_x_y):
        # A prediction bbox overlap any gt_bbox over ignore_thresh,
        # objectness loss will be ignored, process as follows:

        _anchors = np.array(anchors)
        _anchors = np.reshape(_anchors, (-1, 2)).astype(np.float32)

        im_size = tf.ones((batch_size, 2), dtype=tf.float32)
        bbox, prob = paddle_yolo_box(output, _anchors, downsample,
                                     num_classes, scale_x_y, im_size, clip_bbox=False,
                                     conf_thresh=0.0)

        # 2. split pred bbox and gt bbox by sample, calculate IoU between pred bbox
        #    and gt bbox in each sample
        def _process_sample(args):
            pred, gt = args
            # pred:   [3*13*13, 4]
            # gt:     [50, 4]

            def box_xywh2xyxy(box):
                x = box[:, 0:1]
                y = box[:, 1:2]
                w = box[:, 2:3]
                h = box[:, 3:4]
                return tf.concat(
                    [
                        x - w / 2.,
                        y - h / 2.,
                        x + w / 2.,
                        y + h / 2.,
                    ], 1)

            gt = box_xywh2xyxy(gt)   # [50, 4]
            iou = jaccard(pred, gt)   # [3*13*13, 50]
            return iou

        # [bz, 3*13*13, 50]   每张图片的这个输出层的所有预测框（比如3*13*13个）与所有gt（50个）两两之间的iou
        iou = tf.map_fn(_process_sample, [bbox, gt_box], dtype=tf.float32)

        # 3. Get iou_mask by IoU between gt bbox and prediction bbox,
        #    Get obj_mask by tobj(holds gt_score), calculate objectness loss
        max_iou = tf.reduce_max(iou, axis=[-1, ])   # [bz, 3*13*13]   预测框与所有gt最高的iou
        iou_mask = tf.cast(max_iou <= ignore_thresh, tf.float32)   # [bz, 3*13*13]   候选负样本处为1
        if self.match_score:
            max_prob = tf.reduce_max(prob, axis=[-1, ])   # [bz, 3*13*13]   预测框所有类别最高分数
            iou_mask = iou_mask * tf.cast(max_prob <= 0.25, tf.float32)   # 最高分数低于0.25的预测框，被视作负样本或者忽略样本，虽然在训练初期该分数不可信。
        output_shape = tf.shape(output)
        an_num = len(anchors) // 2
        iou_mask = tf.reshape(iou_mask, (output_shape[0], an_num, output_shape[2], output_shape[3]))   # [bz, 3, 13, 13]   候选负样本处为1

        # NOTE: tobj holds gt_score, obj_mask holds object existence mask
        obj_mask = tf.cast(tobj > 0., tf.float32)   # [bz, 3, 13, 13]  正样本处为1

        # 候选负样本 中的 非正样本 才是负样本。所有样本中，正样本和负样本之外的样本是忽略样本。
        noobj_mask = (1.0 - obj_mask) * iou_mask   # [N, 3, n_grid, n_grid]  负样本处为1

        # For positive objectness grids, objectness loss should be calculated
        # For negative objectness grids, objectness loss is calculated only iou_mask == 1.0
        sigmoid_obj = tf.sigmoid(obj)
        loss_obj_pos = tobj * (0 - tf.log(sigmoid_obj + 1e-9))   # 由于有mixup增强，tobj正样本处不一定为1.0
        loss_obj_neg = noobj_mask * (0 - tf.log(1 - sigmoid_obj + 1e-9))   # 负样本的损失
        loss_obj_pos = tf.reduce_sum(loss_obj_pos, axis=[1, 2, 3])
        loss_obj_neg = tf.reduce_sum(loss_obj_neg, axis=[1, 2, 3])

        return loss_obj_pos, loss_obj_neg




