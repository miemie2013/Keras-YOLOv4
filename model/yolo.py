#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================


class YOLO(object):
    def __init__(self, backbone, head):
        super(YOLO, self).__init__()
        self.backbone = backbone
        self.head = head

    def get_outputs(self, x):
        body_feats = self.backbone(x)
        outputs = self.head._get_outputs(body_feats)
        return outputs

    def get_prediction(self, outputs, im_size):
        preds = self.head.get_prediction(outputs, im_size)
        return preds

    def get_loss(self, args, target_num):
        if target_num == 3:
            output0 = args[0]
            output1 = args[1]
            output2 = args[2]
            gt_box = args[3]
            target0 = args[4]
            target1 = args[5]
            target2 = args[6]
            outputs = [output0, output1, output2]
            targets = [target0, target1, target2]
        elif target_num == 2:
            output0 = args[0]
            output1 = args[1]
            gt_box = args[2]
            target0 = args[3]
            target1 = args[4]
            outputs = [output0, output1]
            targets = [target0, target1]
        loss = self.head.get_loss(outputs, gt_box, None, None, targets)
        return loss




