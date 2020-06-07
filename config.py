#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-05 15:35:27
#   Description : 配置文件。
#
# ================================================================


class TrainConfig(object):
    """
    train.py里需要的配置
    """
    def __init__(self):
        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径

        # COCO数据集
        self.train_path = '../COCO/annotations/instances_train2017.json'
        self.val_path = '../COCO/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        self.val_pre_path = '../COCO/val2017/'  # 验证集图片相对路径

        # 训练时若预测框与所有的gt小于阈值self.iou_loss_thresh时视为反例
        self.iou_loss_thresh = 0.7

        # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。）
        self.pattern = 1
        self.lr = 0.0001
        self.batch_size = 8
        # 如果self.pattern = 1，需要指定self.model_path表示从哪个模型读取权重继续训练。
        self.model_path = 'yolov4.h5'
        # self.model_path = './weights/step00001000.h5'

        # ========= 一些设置 =========
        # 每隔几步保存一次模型
        self.save_iter = 1000
        # 每隔几步计算一次eval集的mAP
        self.eval_iter = 5000
        # 训练多少步
        self.max_iters = 800000


        # 验证
        # self.input_shape越大，精度会上升，但速度会下降。
        # self.input_shape = (320, 320)
        # self.input_shape = (416, 416)
        self.input_shape = (608, 608)
        # 验证时的分数阈值和nms_iou阈值
        self.conf_thresh = 0.001
        self.nms_thresh = 0.45
        # 是否画出验证集图片
        self.draw_image = False
        # 验证时的批大小
        self.eval_batch_size = 4


        # ============= 数据增强相关 =============
        self.with_mixup = False
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # PadBox
        self.num_max_boxes = 70
        # Gt2YoloTarget
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = [[12, 16], [19, 36], [40, 28],
                        [36, 75], [76, 55], [72, 146],
                        [142, 110], [192, 243], [459, 401]]
        self.downsample_ratios = [32, 16, 8]



class TrainConfig_2(object):
    """
    其它配置
    """
    def __init__(self):
        pass




