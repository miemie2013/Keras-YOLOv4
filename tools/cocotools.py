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
import time
import threading
import numpy as np
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


def multi_thread_read(j, images, _decode, offset, eval_pre_path, batch_im_id, batch_im_name, batch_img, batch_pimage, batch_im_size):
    im = images[offset + j]
    im_id = im['id']
    file_name = im['file_name']
    image = cv2.imread(eval_pre_path + file_name)
    batch_im_id[j] = im_id
    batch_im_name[j] = file_name
    batch_img[j] = image
    pimage, im_size = _decode.process_image(np.copy(image))
    batch_pimage[j] = pimage
    batch_im_size[j] = im_size

def read_eval_data(images,
                   _decode,
                   eval_pre_path,
                   eval_batch_size,
                   num_steps,
                   eval_dic):
    n = len(images)
    for i in range(num_steps):
        key_list = list(eval_dic.keys())
        key_len = len(key_list)
        while key_len >= 3:
            time.sleep(0.01)
            key_list = list(eval_dic.keys())
            key_len = len(key_list)


        batch_size = eval_batch_size
        if i == num_steps - 1:
            batch_size = n - (num_steps - 1) * eval_batch_size

        batch_im_id = [None] * batch_size
        batch_im_name = [None] * batch_size
        batch_img = [None] * batch_size
        batch_pimage = [None] * batch_size
        batch_im_size = [None] * batch_size
        threads = []
        offset = i * eval_batch_size
        for j in range(batch_size):
            t = threading.Thread(target=multi_thread_read,
                                 args=(j, images, _decode, offset, eval_pre_path, batch_im_id, batch_im_name, batch_img, batch_pimage, batch_im_size))
            threads.append(t)
            t.start()
        # 等待所有线程任务结束。
        for t in threads:
            t.join()

        batch_pimage = np.concatenate(batch_pimage, axis=0)
        batch_im_size = np.concatenate(batch_im_size, axis=0)
        dic = {}
        dic['batch_im_id'] = batch_im_id
        dic['batch_im_name'] = batch_im_name
        dic['batch_img'] = batch_img
        dic['batch_pimage'] = batch_pimage
        dic['batch_im_size'] = batch_im_size
        eval_dic['%.8d' % i] = dic

def multi_thread_write_json(j, result_image, result_boxes, result_scores, result_classes, batch_im_id, batch_im_name, _clsid2catid, draw_image, result_dir):
    image = result_image[j]
    boxes = result_boxes[j]
    scores = result_scores[j]
    classes = result_classes[j]
    if boxes is not None:
        im_id = batch_im_id[j]
        im_name = batch_im_name[j]
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
        path = '%s/bbox/%s.json' % (result_dir, im_name.split('.')[0])
        if draw_image:
            cv2.imwrite('%s/images/%s' % (result_dir, im_name), image)
        with open(path, 'w') as f:
            json.dump(bbox_data, f)



def eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh, type='eval'):
    assert type in ['eval', 'test_dev']
    result_dir = 'eval_results'
    if type == 'test_dev':
        result_dir = 'results'

    # 8G内存的电脑并不能装下所有结果，所以把结果写进文件里。
    if os.path.exists('%s/bbox/' % result_dir): shutil.rmtree('%s/bbox/' % result_dir)
    if draw_image:
        if os.path.exists('%s/images/' % result_dir): shutil.rmtree('%s/images/' % result_dir)
    if not os.path.exists('%s/' % result_dir): os.mkdir('%s/' % result_dir)
    os.mkdir('%s/bbox/' % result_dir)
    if draw_image:
        os.mkdir('%s/images/' % result_dir)


    n = len(images)
    num_steps = n // eval_batch_size   # 总步数
    if n % eval_batch_size != 0:
        num_steps += 1

    logger.info('Total iter: {}'.format(num_steps))
    start = time.time()

    # 读数据的线程
    eval_dic = {}
    thr = threading.Thread(target=read_eval_data,
                           args=(images,
                                 _decode,
                                 eval_pre_path,
                                 eval_batch_size,
                                 num_steps,
                                 eval_dic))
    thr.start()
    for i in range(num_steps):
        key_list = list(eval_dic.keys())
        key_len = len(key_list)
        while key_len == 0:
            time.sleep(0.01)
            key_list = list(eval_dic.keys())
            key_len = len(key_list)
        dic = eval_dic.pop('%.8d' % i)
        batch_im_id = dic['batch_im_id']
        batch_im_name = dic['batch_im_name']
        batch_img = dic['batch_img']
        batch_pimage = dic['batch_pimage']
        batch_im_size = dic['batch_im_size']

        result_image, result_boxes, result_scores, result_classes = _decode.detect_batch(batch_img, batch_pimage, batch_im_size, draw_image=draw_image, draw_thresh=draw_thresh)
        batch_size = eval_batch_size
        if i == num_steps - 1:
            batch_size = n - (num_steps - 1) * eval_batch_size

        for j in range(batch_size):
            t = threading.Thread(target=multi_thread_write_json,
                                 args=(j, result_image, result_boxes, result_scores, result_classes, batch_im_id, batch_im_name, _clsid2catid, draw_image, result_dir))
            t.start()
        if i % 100 == 0:
            logger.info('Test iter {}'.format(i))
    logger.info('Test Done.')
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / n), (n / cost)))
    if type == 'eval':
        # 开始评测
        box_ap_stats = bbox_eval(anno_file)
        return box_ap_stats
    elif type == 'test_dev':
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
        return 1


