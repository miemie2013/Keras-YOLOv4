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



# 相交矩形的面积
def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = tf.shape(box_a)[0]
    B = tf.shape(box_b)[0]

    box_a_rb = tf.reshape(box_a[:, 2:], (A, 1, 2))
    box_a_rb = tf.tile(box_a_rb, [1, B, 1])
    box_b_rb = tf.reshape(box_b[:, 2:], (1, B, 2))
    box_b_rb = tf.tile(box_b_rb, [A, 1, 1])
    max_xy = tf.minimum(box_a_rb, box_b_rb)

    box_a_lu = tf.reshape(box_a[:, :2], (A, 1, 2))
    box_a_lu = tf.tile(box_a_lu, [1, B, 1])
    box_b_lu = tf.reshape(box_b[:, :2], (1, B, 2))
    box_b_lu = tf.tile(box_b_lu, [A, 1, 1])
    min_xy = tf.maximum(box_a_lu, box_b_lu)

    inter = tf.maximum(max_xy - min_xy, 0.0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)

    A = tf.shape(box_a)[0]
    B = tf.shape(box_b)[0]

    area_a = (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])
    area_a = tf.reshape(area_a, (A, 1))
    area_a = tf.tile(area_a, [1, B])  # [A, B]

    area_b = (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])
    area_b = tf.reshape(area_b, (1, B))
    area_b = tf.tile(area_b, [A, 1])  # [A, B]


    union = area_a + area_b - inter
    return inter / union  # [A, B]



def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = tf.shape(cate_labels)[0]

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = jaccard(bboxes, bboxes)   # shape: [n_samples, n_samples]
    # 只取上三角部分
    rows = tf.range(0, n_samples, 1, 'int32')
    cols = tf.range(0, n_samples, 1, 'int32')
    rows = tf.tile(tf.reshape(rows, (1, -1)), [n_samples, 1])
    cols = tf.tile(tf.reshape(cols, (-1, 1)), [1, n_samples])
    tri_mask = tf.cast(rows > cols, 'float32')
    iou_matrix = iou_matrix * tri_mask

    # label_specific matrix.
    cate_labels_x = tf.tile(tf.reshape(cate_labels, (1, -1)), [n_samples, 1])   # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    d = cate_labels_x - tf.transpose(cate_labels_x, [1, 0])
    d = tf.pow(d, 2)   # 同类处为0，非同类处>0。 tf中用 == 0比较无效，所以用 < 1
    label_matrix = tf.cast(d < 1, tf.float32) * tri_mask   # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou = tf.reduce_max(iou_matrix * label_matrix, axis=[0, ])   # shape: [n_samples, ]
    compensate_iou = tf.transpose(tf.tile(tf.reshape(compensate_iou, (1, -1)), [n_samples, 1]), [1, 0])   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = tf.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = tf.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = tf.reduce_sum(decay_matrix / compensate_matrix, axis=[0, ])
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = tf.reduce_min(decay_matrix, axis=[0, ])
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update




def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.):
    inds = tf.where(scores > score_threshold)

    # I hate tensorflow.
    def exist_objs_1(inds, scores, bboxes):
        cate_scores = tf.gather_nd(scores, inds)

        cate_labels = inds[:, 1]
        bboxes = tf.gather(bboxes, inds[:, 0])

        # sort and keep top nms_top_k
        # sort_inds = tf.argsort(cate_scores, axis=-1, direction='DESCENDING')
        k = tf.shape(cate_scores)[0]
        _, sort_inds = tf.nn.top_k(cate_scores, k=k, sorted=True)
        sort_inds = sort_inds[:nms_top_k]
        bboxes = tf.gather(bboxes, sort_inds)
        cate_scores = tf.gather(cate_scores, sort_inds)
        cate_labels = tf.gather(cate_labels, sort_inds)

        # Matrix NMS
        kernel = 'gaussian' if use_gaussian else 'linear'
        cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

        # filter.
        keep = tf.where(cate_scores >= post_threshold)
        keep = tf.reshape(keep, (-1,))

        # I hate tensorflow.
        def exist_objs_2(keep, bboxes, cate_scores, cate_labels):
            bboxes = tf.gather(bboxes, keep)
            cate_scores = tf.gather(cate_scores, keep)
            cate_labels = tf.gather(cate_labels, keep)

            # sort and keep keep_top_k
            # sort_inds = tf.argsort(cate_scores, axis=-1, direction='DESCENDING')
            k = tf.shape(cate_scores)[0]
            _, sort_inds = tf.nn.top_k(cate_scores, k=k, sorted=True)
            sort_inds = sort_inds[:keep_top_k]
            bboxes = tf.gather(bboxes, sort_inds)
            cate_scores = tf.gather(cate_scores, sort_inds)
            cate_labels = tf.gather(cate_labels, sort_inds)

            cate_scores = tf.reshape(cate_scores, (-1, 1))
            cate_labels = tf.reshape(cate_labels, (-1, 1))
            cate_labels = tf.cast(cate_labels, tf.float32)
            pred = tf.concat([cate_labels, cate_scores, bboxes], 1)

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
                       lambda: exist_objs_2(keep, bboxes, cate_scores, cate_labels))
        return pred

    # I hate tensorflow.
    def no_objs_1():
        pred = tf.zeros((keep_top_k, 6), tf.float32) - 1.0
        return pred

    # 是否有物体
    # I hate tensorflow.
    pred = tf.cond(tf.equal(tf.shape(inds)[0], 0),
                   no_objs_1,
                   lambda: exist_objs_1(inds, scores, bboxes))

    return pred



