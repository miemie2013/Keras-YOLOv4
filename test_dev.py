#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
from config import *
import argparse

from tools.cocotools import *
from model.decode_np import Decode
from model.yolo import *
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='YOLO Eval Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=1,
                    choices=[0, 1, 2],
                    help='0 -- yolov4_2x.py;  1 -- ppyolo_2x.py;  2 -- ppyolo_r18vd.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu

if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = YOLOv4_2x_Config()
    elif config_file == 1:
        cfg = PPYOLO_2x_Config()
    elif config_file == 2:
        cfg = PPYOLO_r18vd_Config()


    # 读取的模型
    model_path = cfg.eval_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.eval_cfg['draw_image']
    draw_thresh = cfg.eval_cfg['draw_thresh']

    # 验证时的批大小
    eval_batch_size = cfg.eval_cfg['eval_batch_size']

    # test集图片的相对路径
    test_pre_path = '../COCO/test2017/'
    anno_file = '../COCO/annotations/image_info_test-dev2017.json'
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    all_classes = get_classes(cfg.classes_path)
    num_classes = len(all_classes)


    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    Head = select_head(cfg.head_type)
    cfg.head['drop_block'] = False   # 预测时关闭DropBlock，以获得一致的推理结果。
    head = Head(yolo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
    yolo = YOLO(backbone, head)

    x = keras.layers.Input(shape=(None, None, 3), name='x', dtype='float32')
    im_size = keras.layers.Input(shape=(2,), name='im_size', dtype='int32')
    outputs = yolo.get_outputs(x)
    preds = yolo.get_prediction(outputs, im_size)
    predict_model = keras.models.Model(inputs=[x, im_size], outputs=preds)
    predict_model.load_weights(model_path, by_name=True, skip_mismatch=True)
    predict_model.summary(line_length=130)

    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k

    _decode = Decode(predict_model, all_classes, use_gpu, cfg, for_test=False)
    eval(_decode, images, test_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh, type='test_dev')

