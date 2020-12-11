#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
from model.losses import *
from model.iou_losses import *
from model.head import *
from model.resnet import *
from model.cspdarknet import *
from model.yolov4_head import *


def select_backbone(name):
    if name == 'Resnet50Vd':
        return Resnet50Vd
    if name == 'Resnet18Vd':
        return Resnet18Vd
    if name == 'CSPDarknet53':
        return CSPDarknet53

def select_head(name):
    if name == 'YOLOv3Head':
        return YOLOv3Head
    if name == 'YOLOv4Head':
        return YOLOv4Head

def select_loss(name):
    if name == 'YOLOv3Loss':
        return YOLOv3Loss
    if name == 'IouLoss':
        return IouLoss
    if name == 'IouAwareLoss':
        return IouAwareLoss




