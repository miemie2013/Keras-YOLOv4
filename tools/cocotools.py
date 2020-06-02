#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : coco评测
#
# ================================================================
import os
import json
import sys
import cv2
import shutil
import logging
logger = logging.getLogger(__name__)


clsid2catid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
               15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
               27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
               39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
               51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
               63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
               75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

catid2clsid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
               17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26,
               32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
               44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50,
               57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62,
               73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
               86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt != None or anno_file != None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def bbox_eval(anno_file):
    from pycocotools.coco import COCO

    coco_gt = COCO(anno_file)

    outfile = 'eval_results/bbox_detections.json'
    logger.info('Generating json file...')
    bbox_list = []
    path_dir = os.listdir('eval_results/bbox/')
    for name in path_dir:
        with open('eval_results/bbox/' + name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                r_list = json.loads(line)
                bbox_list += r_list
    with open(outfile, 'w') as f:
        json.dump(bbox_list, f)

    map_stats = cocoapi_eval(outfile, 'bbox', coco_gt=coco_gt)
    # flush coco evaluation result
    sys.stdout.flush()
    return map_stats

def eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image):
    # 8G内存的电脑并不能装下所有结果，所以把结果写进文件里。
    if os.path.exists('eval_results/bbox/'): shutil.rmtree('eval_results/bbox/')
    if draw_image:
        if os.path.exists('eval_results/images/'): shutil.rmtree('eval_results/images/')
    if not os.path.exists('eval_results/'): os.mkdir('eval_results/')
    os.mkdir('eval_results/bbox/')
    if draw_image:
        os.mkdir('eval_results/images/')

    count = 0
    n = len(images)
    batch_im_id = []
    batch_im_name = []
    batch_img = []
    for i, im in enumerate(images):
        im_id = im['id']
        file_name = im['file_name']
        image = cv2.imread(eval_pre_path + file_name)
        if i % eval_batch_size == 0:
            batch_im_id = []
            batch_im_name = []
            batch_img = []
        batch_im_id.append(im_id)
        batch_im_name.append(file_name)
        batch_img.append(image)

        # 收集够一个batch的图片
        if i != n - 1 and len(batch_img) != eval_batch_size:
            continue

        result_image, result_boxes, result_scores, result_classes = _decode.detect_batch(batch_img, draw_image=draw_image)
        k = 0
        for image, boxes, scores, classes in zip(result_image, result_boxes, result_scores, result_classes):
            if boxes is not None:
                im_id = batch_im_id[k]
                im_name = batch_im_name[k]
                n = len(boxes)
                bbox_data = []
                for p in range(n):
                    clsid = classes[p]
                    score = scores[p]
                    xmin, ymin, xmax, ymax = boxes[p]
                    catid = (_clsid2catid[int(clsid)])
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                    bbox = [xmin, ymin, w, h]
                    # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                    bbox = [round(float(x) * 10) / 10 for x in bbox]
                    bbox_res = {
                        'image_id': im_id,
                        'category_id': catid,
                        'bbox': bbox,
                        'score': float(score)
                    }
                    bbox_data.append(bbox_res)
                path = 'eval_results/bbox/%s.json' % im_name.split('.')[0]
                if draw_image:
                    cv2.imwrite('eval_results/images/%s' % im_name, image)
                with open(path, 'w') as f:
                    json.dump(bbox_data, f)
            count += 1
            k += 1
            if count % 100 == 0:
                logger.info('Test iter {}'.format(count))
    # 开始评测
    box_ap_stats = bbox_eval(anno_file)
    return box_ap_stats


def test_dev(_decode, images, test_pre_path, test_batch_size, draw_image):
    # 8G内存的电脑并不能装下所有结果，所以把结果写进文件里。
    if os.path.exists('results/bbox/'): shutil.rmtree('results/bbox/')
    if draw_image:
        if os.path.exists('results/images/'): shutil.rmtree('results/images/')
    if not os.path.exists('results/'): os.mkdir('results/')
    os.mkdir('results/bbox/')
    if draw_image:
        os.mkdir('results/images/')

    count = 0
    n = len(images)
    batch_im_id = []
    batch_img = []
    for i, im in enumerate(images):
        im_id = im['id']
        file_name = im['file_name']
        image = cv2.imread(test_pre_path + file_name)
        if i % test_batch_size == 0:
            batch_im_id = []
            batch_img = []
        batch_im_id.append(im_id)
        batch_img.append(image)

        # 收集够一个batch的图片
        if i != n - 1 and len(batch_img) != test_batch_size:
            continue

        result_image, result_boxes, result_scores, result_classes = _decode.detect_batch(batch_img, draw_image=draw_image)
        k = 0
        for image, boxes, scores, classes in zip(result_image, result_boxes, result_scores, result_classes):
            if boxes is not None:
                im_id = batch_im_id[k]
                n = len(boxes)
                bbox_data = []
                for p in range(n):
                    clsid = classes[p]
                    score = scores[p]
                    xmin, ymin, xmax, ymax = boxes[p]
                    catid = (clsid2catid[int(clsid)])
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                    bbox = [xmin, ymin, w, h]
                    # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                    bbox = [round(float(x) * 10) / 10 for x in bbox]
                    bbox_res = {
                        'image_id': im_id,
                        'category_id': catid,
                        'bbox': bbox,
                        'score': float(score)
                    }
                    bbox_data.append(bbox_res)
                path = 'results/bbox/%.12d.json' % im_id
                if draw_image:
                    cv2.imwrite('results/images/%.12d.jpg' % im_id, image)
                with open(path, 'w') as f:
                    json.dump(bbox_data, f)
            count += 1
            k += 1
            if count % 100 == 0:
                logger.info('Test iter {}'.format(count))
    # 生成json文件
    logger.info('Generating json file...')
    bbox_list = []
    path_dir = os.listdir('results/bbox/')
    for name in path_dir:
        with open('results/bbox/' + name, 'r', encoding='utf-8') as f2:
            for line in f2:
                line = line.strip()
                r_list = json.loads(line)
                bbox_list += r_list
    # 提交到网站的文件
    with open('results/bbox_detections.json', 'w') as f2:
        json.dump(bbox_list, f2)
    logger.info('Done.')

