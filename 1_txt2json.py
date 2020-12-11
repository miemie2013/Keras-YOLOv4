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

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def write_json(txt_path, img_path, base_json, anno_name, im_id, anno_id):
    target_json = copy.deepcopy(base_json)
    with open(txt_path) as f:
        txt_lines = f.readlines()
    images = []
    annos = []
    for line in txt_lines:
        anno_list = line.split()
        ndarr = cv2.imread(img_path + anno_list[0])
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
                'segmentation': [[x2, y2, x2, y1, x1, y1, x1, y2, x2, y2]],
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
    target_json['annotations'] = annos
    target_json['images'] = images
    filename = anno_name[0]
    if '/' in anno_name[0]:
        filename = anno_name[0].split('/')[-1]
    with open('annotation_json/%s.json' % filename, 'w') as f2:
        json.dump(target_json, f2)
    return im_id, anno_id


if __name__ == '__main__':
    # 自定义数据集的注解转换成coco的注解格式。只需改下面7个即可。文件夹下的子目录（子文件）用/隔开，而不能用\或\\。
    train_path = 'annotation/voc2012_train.txt'
    val_path = 'annotation/voc2012_val.txt'
    test_path = None   # 如果没有测试集，填None；如果有测试集，填txt注解文件的路径。
    classes_path = 'data/voc_classes.txt'
    train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
    val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径
    test_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'    # 测试集图片相对路径


    # 创建json注解目录
    if os.path.exists('annotation_json/'): shutil.rmtree('annotation_json/')
    os.mkdir('annotation_json/')

    train_anno_name = train_path.split('.')
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
    im_id = 0
    anno_id = 0

    # train set
    im_id, anno_id = write_json(train_path, train_pre_path, base_json, train_anno_name, im_id, anno_id)

    # val set
    im_id, anno_id = write_json(val_path, val_pre_path, base_json, val_anno_name, im_id, anno_id)

    # test set
    if test_path is not None:
        test_anno_name = test_path.split('.')
        im_id, anno_id = write_json(test_path, test_pre_path, base_json, test_anno_name, im_id, anno_id)

    print('Done.')

