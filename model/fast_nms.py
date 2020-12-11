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


def _iou(box_a, box_b):
    '''
    :param box_a:    [c, A, 4]
    :param box_b:    [c, B, 4]
    :return:   [c, A, B]  两两之间的iou
    '''

    c = tf.shape(box_a)[0]
    A = tf.shape(box_a)[1]
    B = tf.shape(box_b)[1]

    _box_a = tf.reshape(box_a, (c, A, 1, 4))
    _box_b = tf.reshape(box_b, (c, 1, B, 4))
    expand_box_a = tf.tile(_box_a, [1, 1, B, 1])
    expand_box_b = tf.tile(_box_b, [1, A, 1, 1])

    # 两个矩形的面积
    boxes1_area = (expand_box_a[..., 2] - expand_box_a[..., 0]) * (
            expand_box_a[..., 3] - expand_box_a[..., 1])
    boxes2_area = (expand_box_b[..., 2] - expand_box_b[..., 0]) * (
            expand_box_b[..., 3] - expand_box_b[..., 1])

    # 相交矩形的左上角坐标、右下角坐标
    left_up = tf.maximum(expand_box_a[:, :, :, :2], expand_box_b[:, :, :, :2])
    right_down = tf.minimum(expand_box_a[:, :, :, 2:], expand_box_b[:, :, :, 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)
    return iou

def _fast_nms(boxes, scores, nms_thresh, keep_top_k, nms_top_k):
    '''
    :param boxes:    [?, 4]
    :param scores:   [80, ?]
    '''

    # 同类方框根据得分降序排列
    # idx = tf.argsort(scores, axis=1, direction='DESCENDING')
    # scores = tf.sort(scores, axis=1, direction='DESCENDING')
    k = tf.shape(scores)[1]
    scores, idx = tf.nn.top_k(scores, k=k, sorted=True)

    idx = idx[:, :keep_top_k]
    scores = scores[:, :keep_top_k]

    num_classes, num_dets = tf.shape(idx)[0], tf.shape(idx)[1]

    idx = tf.reshape(idx, (-1, ))
    boxes = tf.gather(boxes, idx)
    boxes = tf.reshape(boxes, (num_classes, num_dets, 4))

    # 计算一个c×n×n的IOU矩阵，其中每个n×n矩阵表示对该类n个候选框，两两之间的IOU
    iou = _iou(boxes, boxes)

    # 因为自己与自己的IOU=1，IOU(A,B)=IOU(B,A)，所以对上一步得到的IOU矩阵
    # 进行一次处理。具体做法是将每一个通道，的对角线元素和下三角部分置为0
    rows = tf.range(0, num_dets, 1, 'int32')
    cols = tf.range(0, num_dets, 1, 'int32')
    rows = tf.tile(tf.reshape(rows, (1, -1)), [num_dets, 1])
    cols = tf.tile(tf.reshape(cols, (-1, 1)), [1, num_dets])
    tri_mask = tf.cast(rows > cols, 'float32')
    tri_mask = tf.tile(tf.reshape(tri_mask, (1, num_dets, num_dets)), [num_classes, 1, 1])
    iou = tri_mask * iou   # [c, n, n]
    iou_max = tf.reduce_max(iou, axis=1)  # [c, n]  同一类别，n个框与“分数比它高的框”的最高iou


    # 同一类别，n个框与“分数比它高的框”的最高iou超过nms_thresh的话，就丢弃。下标是0的框肯定被保留。
    keep = tf.where(iou_max <= nms_thresh)

    # Assign each kept detection to its corresponding class
    classes = tf.range(0, num_classes, 1, 'int32')
    classes = tf.tile(tf.reshape(classes, (-1, 1)), [1, num_dets])
    classes = tf.gather_nd(classes, keep)

    boxes = tf.gather_nd(boxes, keep)
    scores = tf.gather_nd(scores, keep)

    # Only keep the top cfg.max_num_detections highest scores across all classes
    # idx = tf.argsort(scores, axis=0, direction='DESCENDING')
    # scores = tf.sort(scores, axis=0, direction='DESCENDING')
    k = tf.shape(scores)[0]
    scores, idx = tf.nn.top_k(scores, k=k, sorted=True)

    idx = idx[:nms_top_k]
    scores = scores[:nms_top_k]

    classes = tf.gather(classes, idx)
    boxes = tf.gather(boxes, idx)

    return boxes, scores, classes


def fast_nms(bboxes,
             scores,
             score_threshold,
             nms_threshold,
             nms_top_k,
             keep_top_k):
    '''
    :param bboxes:  [-1, 4]
    :param scores:  [-1, 80]
    '''
    cur_scores = tf.transpose(scores, perm=[1, 0])  # [80, -1]
    conf_scores = tf.reduce_max(cur_scores, axis=0)  # [-1, ]  每个预测框的所有类别的最高分数
    keep = tf.where(conf_scores > score_threshold)  # 最高分数大与阈值的保留
    keep = tf.reshape(keep, (-1,))  # [-1, ]

    # I hate tensorflow.
    def exist_objs_1(keep, scores, bboxes):
        scores = tf.gather(scores, keep)  # [?, 80]
        scores = tf.transpose(scores, perm=[1, 0])  # [80, ?]
        boxes = tf.gather(bboxes, keep)  # [?, 4]
        boxes, scores, classes = _fast_nms(boxes, scores, nms_threshold, keep_top_k, nms_top_k)

        # 再做一次分数过滤。前面提到，只要某个框最高分数>阈值就保留，
        # 然而计算上面那个矩阵时，这个框其实重复了80次，每一个分身代表是不同类的物品。
        # 非最高分数的其它类别，它的得分可能小于阈值，要过滤。
        # 所以fastnms存在这么一个现象：某个框它最高分数 > 阈值，它有一个非最高分数类的得分也超过了阈值，
        # 那么最后有可能两个框都保留，而且这两个框有相同的xywh
        keep = tf.where(scores > score_threshold)  # 最高分数大与阈值的保留
        keep = tf.reshape(keep, (-1,))  # [-1, ]

        # I hate tensorflow.
        def exist_objs_2(keep, boxes, scores, classes):
            boxes = tf.gather(boxes, keep)
            scores = tf.gather(scores, keep)
            classes = tf.gather(classes, keep)

            # sort and keep keep_top_k
            k = tf.shape(scores)[0]
            _, sort_inds = tf.nn.top_k(scores, k=k, sorted=True)
            sort_inds = sort_inds[:keep_top_k]
            boxes = tf.gather(boxes, sort_inds)
            scores = tf.gather(scores, sort_inds)
            classes = tf.gather(classes, sort_inds)

            scores = tf.reshape(scores, (-1, 1))
            classes = tf.reshape(classes, (-1, 1))
            classes = tf.cast(classes, tf.float32)
            pred = tf.concat([classes, scores, boxes], 1)

            obj_num = tf.shape(pred)[0]
            pad_pred = tf.zeros((keep_top_k-obj_num, 6), tf.float32) - 1.0
            pred = tf.concat([pred, pad_pred], 0)
            return pred

        # I hate tensorflow.
        def no_objs_2():
            pred = tf.zeros((keep_top_k, 6), tf.float32) - 1.0
            return pred

        # 是否有物体
        # I hate tensorflow.
        pred = tf.cond(tf.equal(tf.shape(keep)[0], 0),
                       no_objs_2,
                       lambda: exist_objs_2(keep, boxes, scores, classes))
        return pred

    # I hate tensorflow.
    def no_objs_1():
        pred = tf.zeros((keep_top_k, 6), tf.float32) - 1.0
        return pred

    # 是否有物体
    # I hate tensorflow.
    pred = tf.cond(tf.equal(tf.shape(keep)[0], 0),
                   no_objs_1,
                   lambda: exist_objs_1(keep, scores, bboxes))
    return pred

