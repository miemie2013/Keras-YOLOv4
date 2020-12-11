#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import time
import threading
import argparse

from config import *
from model.decode_np import Decode
from model.yolo import *
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='YOLO Infer Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=1,
                    choices=[0, 1, 2],
                    help='0 -- yolov4_2x.py;  1 -- ppyolo_2x.py;  2 -- ppyolo_r18vd.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu


if __name__ == '__main__':
    video_path = 'D://PycharmProjects/moviepy/che3.mp4'
    output_dir = './video_out'   # 生成的视频存放路径
    fps = 60   # 生成的视频的帧率。应该和原视频一样。

    cfg = None
    if config_file == 0:
        cfg = YOLOv4_2x_Config()
    elif config_file == 1:
        cfg = PPYOLO_2x_Config()
    elif config_file == 2:
        cfg = PPYOLO_r18vd_Config()


    # 读取的模型
    model_path = cfg.test_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.test_cfg['draw_image']
    draw_thresh = cfg.test_cfg['draw_thresh']

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

    _decode = Decode(predict_model, all_classes, use_gpu, cfg, for_test=True)

    if not os.path.exists('images/res/'): os.mkdir('images/res/')
    path_dir = os.listdir('images/test')

    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.split(video_path)[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    start = time.time()
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame:%d' % (index))
        index += 1

        pimage, im_size = _decode.process_image(np.copy(frame))
        image, boxes, scores, classes = _decode.detect_image(frame, pimage, im_size, draw_image, draw_thresh)

        cv2.imshow("detection", frame)
        writer.write(frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    writer.release()


