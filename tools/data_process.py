#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-05 15:35:27
#   Description : 数据处理
#
# ================================================================
import os
import copy
import numpy as np

import logging
logger = logging.getLogger(__name__)


# 数据清洗
def data_clean(coco, img_ids, catid2clsid, image_dir):
    records = []
    ct = 0
    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        im_fname = img_anno['file_name']
        im_w = float(img_anno['width'])
        im_h = float(img_anno['height'])

        ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        instances = coco.loadAnns(ins_anno_ids)   # 这张图片所有标注anno。每个标注有'segmentation'、'bbox'、...

        bboxes = []
        anno_id = []    # 注解id
        for inst in instances:
            x, y, box_w, box_h = inst['bbox']   # 读取物体的包围框
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))
            if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                inst['clean_bbox'] = [x1, y1, x2, y2]   # inst增加一个键值对
                bboxes.append(inst)   # 这张图片的这个物体标注保留
                anno_id.append(inst['id'])
            else:
                logger.warn(
                    'Found an invalid bbox in annotations: im_id: {}, '
                    'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                        img_id, float(inst['area']), x1, y1, x2, y2))
        num_bbox = len(bboxes)   # 这张图片的物体数

        # 左上角坐标+右下角坐标+类别id
        gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
        gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_score = np.ones((num_bbox, 1), dtype=np.float32)   # 得分的标注都是1
        is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
        difficult = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_poly = [None] * num_bbox

        for i, box in enumerate(bboxes):
            catid = box['category_id']
            gt_class[i][0] = catid2clsid[catid]
            gt_bbox[i, :] = box['clean_bbox']
            is_crowd[i][0] = box['iscrowd']
            if 'segmentation' in box:
                gt_poly[i] = box['segmentation']

        im_fname = os.path.join(image_dir,
                                im_fname) if image_dir else im_fname
        coco_rec = {
            'im_file': im_fname,
            'im_id': np.array([img_id]),
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'anno_id': anno_id,
            'gt_bbox': gt_bbox,
            'gt_score': gt_score,
            'gt_poly': gt_poly,
        }

        logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
            im_fname, img_id, im_h, im_w))
        records.append(coco_rec)   # 注解文件。
        ct += 1
    logger.info('{} samples in train set.'.format(ct))
    return records

def get_samples(train_records, train_indexes, step, batch_size, with_mixup):
    indexes = train_indexes[step * batch_size:(step + 1) * batch_size]
    samples = []
    for i in range(batch_size):
        pos = indexes[i]
        sample = copy.deepcopy(train_records[pos])

        # 为mixup数据增强做准备
        if with_mixup:
            num = len(train_indexes)
            mix_idx = np.random.randint(1, num)
            mix_idx = train_indexes[(mix_idx + step * batch_size + i) % num]   # 为了不选到自己
            sample['mixup'] = copy.deepcopy(train_records[mix_idx])

        samples.append(sample)
    return samples


