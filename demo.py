#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : keras_yolov4
#
# ================================================================
import cv2
import os
import time

from model.decode_np import Decode

if __name__ == '__main__':
    file = 'data/coco_classes.txt'
    model_path = 'yolov4.h5'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    input_shape = (416, 416)
    # input_shape = (608, 608)

    _decode = Decode(0.05, 0.45, input_shape, model_path, file)

    # detect images in test floder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            start = time.time()
            for f in files:
                # print(f)
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = _decode.detect_image(image)
                cv2.imwrite('images/res/' + f, image)
            print('total time: {0:.6f}s'.format(time.time() - start))

    # detect videos one at a time in videos/test folder
    # video = 'library1.mp4'
    # _decode.detect_video(video)

