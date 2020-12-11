#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-30 21:08:11
#   Description : keras_ppyolo
#
# ================================================================
from model.fast_nms import fast_nms
from model.head import *
from model.matrix_nms import matrix_nms



class YOLOv4Head(object):
    def __init__(self,
                 conv_block_num=2,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23],
                          [30, 61], [62, 45], [59, 119],
                          [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 norm_type="bn",
                 coord_conv=True,
                 iou_aware=True,
                 iou_aware_factor=0.4,
                 block_size=3,
                 scale_x_y=1.05,
                 spp=True,
                 drop_block=True,
                 keep_prob=0.9,
                 clip_bbox=True,
                 yolo_loss=None,
                 downsample=[32, 16, 8],
                 in_channels=[2048, 1024, 512],
                 nms_cfg=None,
                 is_train=False
                 ):
        super(YOLOv4Head, self).__init__()
        self.conv_block_num = conv_block_num
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.coord_conv = coord_conv
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.scale_x_y = scale_x_y
        self.use_spp = spp
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.clip_bbox = clip_bbox
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.block_size = block_size
        self.downsample = downsample
        self.in_channels = in_channels
        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg
        self.is_train = is_train

        _anchors = copy.deepcopy(anchors)
        _anchors = np.array(_anchors)
        _anchors = _anchors.astype(np.float32)
        self._anchors = _anchors   # [9, 2]

        self.mask_anchors = []
        for m in anchor_masks:
            temp = []
            for aid in m:
                temp += anchors[aid]
            self.mask_anchors.append(temp)

        bn = 0
        gn = 0
        af = 0
        if norm_type == 'bn':
            bn = 1
        elif norm_type == 'gn':
            gn = 1
        elif norm_type == 'affine_channel':
            af = 1

        num_anchors = len(anchor_masks[0])
        self.num_anchors = num_anchors

        self.conv073 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv073')
        self.conv074 = Conv2dUnit(512, 1024, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv074')
        self.conv075 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv075')
        self.spp = SPP(512 * 4, seq='desc')
        self.conv076 = Conv2dUnit(512 * 4, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv076')
        self.conv077 = Conv2dUnit(512, 1024, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv077')
        self.conv078 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv078')

        # pan01
        self.conv079 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv079')
        self.upsample1 = keras.layers.UpSampling2D(2, interpolation='nearest')
        self.conv080 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv080')
        self.conv081 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv081')
        self.conv082 = Conv2dUnit(256, 512, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv082')
        self.conv083 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv083')
        self.conv084 = Conv2dUnit(256, 512, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv084')
        self.conv085 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv085')
        # pan01结束

        # pan02
        self.conv086 = Conv2dUnit(256, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv086')
        self.upsample2 = keras.layers.UpSampling2D(2, interpolation='nearest')
        self.conv087 = Conv2dUnit(256, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv087')
        self.conv088 = Conv2dUnit(256, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv088')
        self.conv089 = Conv2dUnit(128, 256, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv089')
        self.conv090 = Conv2dUnit(256, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv090')
        self.conv091 = Conv2dUnit(128, 256, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv091')
        self.conv092 = Conv2dUnit(256, 128, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv092')
        # pan02结束

        # output_s, 不用concat()
        self.conv093 = Conv2dUnit(128, 256, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv093')
        self.conv094 = Conv2dUnit(256, num_anchors * (num_classes + 5), 1, stride=1, bias_attr=True, act=None, name='head.conv094')


        # output_m, 需要concat()
        self.conv095 = Conv2dUnit(128, 256, 3, stride=2, bn=bn, gn=gn, af=af, act='leaky', name='head.conv095')

        self.conv096 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv096')
        self.conv097 = Conv2dUnit(256, 512, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv097')
        self.conv098 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv098')
        self.conv099 = Conv2dUnit(256, 512, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv099')
        self.conv100 = Conv2dUnit(512, 256, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv100')

        self.conv101 = Conv2dUnit(256, 512, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv101')
        self.conv102 = Conv2dUnit(512, num_anchors * (num_classes + 5), 1, stride=1, bias_attr=True, act=None, name='head.conv102')

        # output_l, 需要concat()
        self.conv103 = Conv2dUnit(256, 512, 3, stride=2, bn=bn, gn=gn, af=af, act='leaky', name='head.conv103')

        self.conv104 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv104')
        self.conv105 = Conv2dUnit(512, 1024, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv105')
        self.conv106 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv106')
        self.conv107 = Conv2dUnit(512, 1024, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv107')
        self.conv108 = Conv2dUnit(1024, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv108')

        self.conv109 = Conv2dUnit(512, 1024, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', name='head.conv109')
        self.conv110 = Conv2dUnit(1024, num_anchors * (num_classes + 5), 1, stride=1, bias_attr=True, act=None, name='head.conv110')


    def get_conv2dunit(self, name):
        layer = getattr(self, name)
        return layer

    def set_dropblock(self, is_test):
        pass

    def _get_outputs(self, body_feats):
        s8, s16, s32 = body_feats

        x = self.conv073(s32)
        x = self.conv074(x)
        x = self.conv075(x)
        x = self.spp(x)

        x = self.conv076(x)
        x = self.conv077(x)
        fpn_s32 = self.conv078(x)

        # pan01
        x = self.conv079(fpn_s32)
        x = self.upsample1(x)
        s16 = self.conv080(s16)
        x = keras.layers.Concatenate(axis=-1)([s16, x])
        x = self.conv081(x)
        x = self.conv082(x)
        x = self.conv083(x)
        x = self.conv084(x)
        fpn_s16 = self.conv085(x)
        # pan01结束

        # pan02
        x = self.conv086(fpn_s16)
        x = self.upsample2(x)
        s8 = self.conv087(s8)
        x = keras.layers.Concatenate(axis=-1)([s8, x])
        x = self.conv088(x)
        x = self.conv089(x)
        x = self.conv090(x)
        x = self.conv091(x)
        x = self.conv092(x)
        # pan02结束

        # output_s, 不用concat()
        output_s = self.conv093(x)
        output_s = self.conv094(output_s)

        # output_m, 需要concat()
        x = self.conv095(x)
        x = keras.layers.Concatenate(axis=-1)([x, fpn_s16])
        x = self.conv096(x)
        x = self.conv097(x)
        x = self.conv098(x)
        x = self.conv099(x)
        x = self.conv100(x)
        output_m = self.conv101(x)
        output_m = self.conv102(output_m)

        # output_l, 需要concat()
        x = self.conv103(x)
        x = keras.layers.Concatenate(axis=-1)([x, fpn_s32])
        x = self.conv104(x)
        x = self.conv105(x)
        x = self.conv106(x)
        x = self.conv107(x)
        x = self.conv108(x)
        output_l = self.conv109(x)
        output_l = self.conv110(output_l)

        outputs = [output_l, output_m, output_s]
        return outputs

    def get_loss(self, outputs, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            outputs (list): 大中小感受野的输出，为NHWC格式
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """

        outputs = [tf.transpose(o, [0, 3, 1, 2]) for o in outputs]

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes)

    def get_prediction(self, outputs, im_size):
        """
        Get prediction result of YOLOv3 network

        Args:
            outputs (list): 大中小感受野的输出，为NHWC格式
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): shape = [bs, keep_top_k, 6]

        """

        def output_layer(args):
            outputs = args
            outputs = [tf.transpose(o, [0, 3, 1, 2]) for o in outputs]

            boxes = []
            scores = []
            for i, output in enumerate(outputs):
                if self.iou_aware:
                    output = get_iou_aware_score(output,
                                                 len(self.anchor_masks[i]),
                                                 self.num_classes,
                                                 self.iou_aware_factor)
                box, score = yolo_box(output, self._anchors[self.anchor_masks[i]], self.downsample[i],
                                      self.num_classes, self.scale_x_y, im_size, self.clip_bbox,
                                      conf_thresh=self.nms_cfg['score_threshold'])
                boxes.append(box)
                scores.append(score)
            yolo_boxes = tf.concat(boxes, 1)
            yolo_scores = tf.concat(scores, 1)

            # nms
            nms_cfg = self.nms_cfg
            def _process_sample(args):
                _yolo_boxes, _yolo_scores = args
                # _yolo_boxes:   [?, 4]
                # _yolo_scores:  [?, 80]

                nms_type = self.nms_cfg['nms_type']
                if nms_type == 'matrix_nms':
                    pred = matrix_nms(_yolo_boxes, _yolo_scores,
                                      score_threshold=nms_cfg['score_threshold'],
                                      post_threshold=nms_cfg['post_threshold'],
                                      nms_top_k=nms_cfg['nms_top_k'],
                                      keep_top_k=nms_cfg['keep_top_k'],
                                      use_gaussian=nms_cfg['use_gaussian'],
                                      gaussian_sigma=nms_cfg['gaussian_sigma'])
                elif nms_type == 'fast_nms':
                    pred = fast_nms(_yolo_boxes, _yolo_scores,
                                    score_threshold=nms_cfg['score_threshold'],
                                    nms_threshold=nms_cfg['nms_threshold'],
                                    nms_top_k=nms_cfg['nms_top_k'],
                                    keep_top_k=nms_cfg['keep_top_k'])
                return pred
            preds = tf.map_fn(_process_sample, [yolo_boxes, yolo_scores], dtype=tf.float32)
            return preds
        preds = keras.layers.Lambda(output_layer)(outputs)
        return preds





