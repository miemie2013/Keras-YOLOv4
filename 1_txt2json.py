#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : Convert annotation files (txt format) into coco json format.
#                 自定义数据集的注解转换成coco的注解格式。生成的json注解文件在annotation_json目录下。
#
# ================================================================
import os
import cv2
import json
import copy
import shutil
from tools.cocotools import get_classes


if __name__ == '__main__':
    # 自定义数据集的注解转换成coco的注解格式。只需改下面5个即可。
    train_path = 'annotation/voc2012_train.txt'
    val_path = 'annotation/voc2012_val.txt'
    classes_path = 'data/voc_classes.txt'
    train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
    val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径


    # 创建json注解目录
    if os.path.exists('annotation_json/'): shutil.rmtree('annotation_json/')
    os.mkdir('annotation_json/')

    anno_name = train_path.split('.')
    val_anno_name = val_path.split('.')
    print('Convert annotation files (txt format) into coco json format...')
    info = {
        'description': 'My dataset',
        'url': 'https://github.com/miemie2013',
        'version': '1.0',
        'year': '2020',
        'contributor': 'miemie2013',
        'date_created': '2020/06/01',
    }
    licenses_0 = {
        'url': 'https://github.com/miemie2013',
        'id': 1,
        'name': 'miemie2013 license',
    }
    licenses = [licenses_0]
    categories = []
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    for cid in range(num_classes):
        cate = {
            'supercategory': 'object',
            'id': cid,
            'name': class_names[cid],
        }
        categories.append(cate)
    base_json = {
        'info': info,
        'licenses': licenses,
        'categories': categories,
    }
    train_json = copy.deepcopy(base_json)
    val_json = copy.deepcopy(base_json)

    # train set
    with open(train_path) as f:
        train_lines = f.readlines()
    images = []
    annos = []
    im_id = 0
    anno_id = 0
    for line in train_lines:
        anno_list = line.split()
        ndarr = cv2.imread(train_pre_path + anno_list[0])
        img_h, img_w, _ = ndarr.shape
        image = {
            'license': 1,
            'file_name': anno_list[0],
            'coco_url': 'a',
            'height': img_h,
            'width': img_w,
            'date_captured': 'a',
            'flickr_url': 'a',
            'id': im_id,
        }
        images.append(image)
        for p in range(1, len(anno_list), 1):
            bbox = anno_list[p].split(',')
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])
            cid = int(bbox[4])
            w = x2 - x1
            h = y2 - y1
            anno = {
                'segmentation': [[]],
                'area': w*h,
                'iscrowd': 0,
                'image_id': im_id,
                'bbox': [x1, y1, w, h],
                'category_id': cid,
                'id': anno_id,
            }
            annos.append(anno)
            anno_id += 1
        im_id += 1
    train_json['annotations'] = annos
    train_json['images'] = images
    with open('annotation_json/%s.json' % anno_name[0].split('/')[1], 'w') as f2:
        json.dump(train_json, f2)

    # val set
    with open(val_path) as f:
        val_lines = f.readlines()
    images = []
    annos = []
    im_id = 0
    anno_id = 0
    for line in val_lines:
        anno_list = line.split()
        ndarr = cv2.imread(val_pre_path + anno_list[0])
        img_h, img_w, _ = ndarr.shape
        image = {
            'license': 1,
            'file_name': anno_list[0],
            'coco_url': 'a',
            'height': img_h,
            'width': img_w,
            'date_captured': 'a',
            'flickr_url': 'a',
            'id': im_id,
        }
        images.append(image)
        for p in range(1, len(anno_list), 1):
            bbox = anno_list[p].split(',')
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])
            cid = int(bbox[4])
            w = x2 - x1
            h = y2 - y1
            anno = {
                'segmentation': [[]],
                'area': w*h,
                'iscrowd': 0,
                'image_id': im_id,
                'bbox': [x1, y1, w, h],
                'category_id': cid,
                'id': anno_id,
            }
            annos.append(anno)
            anno_id += 1
        im_id += 1
    val_json['annotations'] = annos
    val_json['images'] = images
    with open('annotation_json/%s.json' % val_anno_name[0].split('/')[1], 'w') as f2:
        json.dump(val_json, f2)

    print('Done.')

