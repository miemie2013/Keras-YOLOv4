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
import time
import threading
import datetime
from collections import OrderedDict
import os
import argparse

from config import *
from model.EMA import ExponentialMovingAverage

from model.yolo import YOLO
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decode_np import Decode
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='YOLO Training Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=1,
                    choices=[0, 1, 2],
                    help='0 -- yolov4_2x.py;  1 -- ppyolo_2x.py;  2 -- ppyolo_r18vd.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu


# 显存分配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))



def multi_thread_op(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms, batch_transforms,
                    shape, images, gt_bbox, gt_score, gt_class, target0, target1, target2, target_num):
    for k in range(i, batch_size, num_threads):
        for sample_transform in sample_transforms:
            if isinstance(sample_transform, MixupImage):
                if with_mixup:
                    samples[k] = sample_transform(samples[k], context)
            else:
                samples[k] = sample_transform(samples[k], context)

        for batch_transform in batch_transforms:
            if isinstance(batch_transform, RandomShapeSingle):
                samples[k] = batch_transform(shape, samples[k], context)
            else:
                samples[k] = batch_transform(samples[k], context)

        # 整理成ndarray
        images[k] = np.expand_dims(samples[k]['image'].astype(np.float32), 0)
        gt_bbox[k] = np.expand_dims(samples[k]['gt_bbox'].astype(np.float32), 0)
        gt_score[k] = np.expand_dims(samples[k]['gt_score'].astype(np.float32), 0)
        gt_class[k] = np.expand_dims(samples[k]['gt_class'].astype(np.int32), 0)
        target0[k] = np.expand_dims(samples[k]['target0'].astype(np.float32), 0)
        target1[k] = np.expand_dims(samples[k]['target1'].astype(np.float32), 0)
        if target_num > 2:
            target2[k] = np.expand_dims(samples[k]['target2'].astype(np.float32), 0)


def read_train_data(cfg,
                    train_indexes,
                    train_steps,
                    train_records,
                    batch_size,
                    _iter_id,
                    train_dic,
                    use_gpu,
                    context, with_mixup, sample_transforms, batch_transforms, target_num):
    iter_id = _iter_id
    num_threads = cfg.train_cfg['num_threads']
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            key_list = list(train_dic.keys())
            key_len = len(key_list)
            while key_len >= cfg.train_cfg['max_batch']:
                time.sleep(0.01)
                key_list = list(train_dic.keys())
                key_len = len(key_list)

            # ==================== train ====================
            sizes = cfg.randomShape['sizes']
            shape = np.random.choice(sizes)
            images = [None] * batch_size
            gt_bbox = [None] * batch_size
            gt_score = [None] * batch_size
            gt_class = [None] * batch_size
            target0 = [None] * batch_size
            target1 = [None] * batch_size
            target2 = [None] * batch_size

            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=multi_thread_op, args=(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms, batch_transforms,
                                                                   shape, images, gt_bbox, gt_score, gt_class, target0, target1, target2, target_num))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            images = np.concatenate(images, 0)
            gt_bbox = np.concatenate(gt_bbox, 0)
            target0 = np.concatenate(target0, 0)
            target1 = np.concatenate(target1, 0)
            if target_num > 2:
                target2 = np.concatenate(target2, 0)

            dic = {}
            dic['images'] = images.transpose(0, 2, 3, 1)
            dic['gt_bbox'] = gt_bbox
            dic['target0'] = target0
            dic['target1'] = target1
            if target_num > 2:
                dic['target2'] = target2
            train_dic['%.8d'%iter_id] = dic

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                return 0




if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = YOLOv4_2x_Config()
    elif config_file == 1:
        cfg = PPYOLO_2x_Config()
    elif config_file == 2:
        cfg = PPYOLO_r18vd_Config()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)

    # 步id，无需设置，会自动读。
    iter_id = 0

    # 创建模型
    target_num = len(cfg.head['anchor_masks'])
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    IouLoss = select_loss(cfg.iou_loss_type)
    iou_loss = IouLoss(**cfg.iou_loss)
    iou_aware_loss = None
    if cfg.head['iou_aware']:
        IouAwareLoss = select_loss(cfg.iou_aware_loss_type)
        iou_aware_loss = IouAwareLoss(**cfg.iou_aware_loss)
    Loss = select_loss(cfg.yolo_loss_type)
    yolo_loss = Loss(iou_loss=iou_loss, iou_aware_loss=iou_aware_loss, **cfg.yolo_loss)
    Head = select_head(cfg.head_type)
    head = Head(yolo_loss=yolo_loss, is_train=True, nms_cfg=cfg.nms_cfg, **cfg.head)   # 评测时还是会使用了DropBlock，所以用eval.py评测模型时与训练时评测得到的mAP有一点不同。
    yolo = YOLO(backbone, head)

    # predict_model
    x = keras.layers.Input(shape=(None, None, 3), name='x', dtype='float32')
    im_size = keras.layers.Input(shape=(2,), name='im_size', dtype='int32')
    outputs = yolo.get_outputs(x)
    preds = yolo.get_prediction(outputs, im_size)
    predict_model = keras.models.Model(inputs=[x, im_size], outputs=preds)

    # train_model
    anchor_masks = cfg.gt2YoloTarget['anchor_masks']
    anchor_num_per_layer = len(anchor_masks[0])
    num_filters = (num_classes + 6)
    gt_bbox_tensor = keras.layers.Input(shape=(None, 4), name='gt_bbox', dtype='float32')
    target0_tensor = keras.layers.Input(shape=(anchor_num_per_layer, num_filters, None, None), name='target0', dtype='float32')
    target1_tensor = keras.layers.Input(shape=(anchor_num_per_layer, num_filters, None, None), name='target1', dtype='float32')
    if target_num > 2:
        target2_tensor = keras.layers.Input(shape=(anchor_num_per_layer, num_filters, None, None), name='target2', dtype='float32')
        targets = [target0_tensor, target1_tensor, target2_tensor]
    else:
        targets = [target0_tensor, target1_tensor]
    loss_list = keras.layers.Lambda(yolo.get_loss, name='yolo_loss',
                                    arguments={'target_num': target_num, })([*outputs, gt_bbox_tensor, *targets])
    train_model = keras.models.Model(inputs=[x, gt_bbox_tensor, *targets], outputs=loss_list)
    loss_n = len(loss_list)

    _decode = Decode(predict_model, class_names, use_gpu, cfg, for_test=False)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        train_model.load_weights(cfg.train_cfg['model_path'], by_name=True, skip_mismatch=True)

        strs = cfg.train_cfg['model_path'].split('step')
        if len(strs) == 2:
            iter_id = int(strs[1][:8])

        # 冻结，使得需要的显存减少。低显存的卡建议这样配置。
        backbone.freeze()

    ema = None
    if cfg.use_ema:
        ema = ExponentialMovingAverage(predict_model, cfg.ema_decay)
        ema.register()

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

    batch_size = cfg.train_cfg['batch_size']
    with_mixup = cfg.decodeImage['with_mixup']
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage(**cfg.mixupImage)      # mixup增强
    colorDistort = ColorDistort(**cfg.colorDistort)  # 颜色扰动
    randomExpand = RandomExpand(**cfg.randomExpand)  # 随机填充
    randomCrop = RandomCrop(**cfg.randomCrop)        # 随机裁剪
    randomFlipImage = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
    normalizeBox = NormalizeBox(**cfg.normalizeBox)        # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
    padBox = PadBox(**cfg.padBox)                          # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
    bboxXYXY2XYWH = BboxXYXY2XYWH(**cfg.bboxXYXY2XYWH)     # sample['gt_bbox']被改写为cx_cy_w_h格式。
    # batch_transforms改sample_transforms
    randomShape = RandomShapeSingle(random_inter=cfg.randomShape['random_inter'])     # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
    normalizeImage = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。先除以255归一化，再减均值除以标准差
    permute = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
    gt2YoloTarget = Gt2YoloTargetSingle(**cfg.gt2YoloTarget)   # 填写target张量。

    sample_transforms = []
    sample_transforms.append(decodeImage)
    sample_transforms.append(mixupImage)
    sample_transforms.append(colorDistort)
    sample_transforms.append(randomExpand)
    sample_transforms.append(randomCrop)
    sample_transforms.append(randomFlipImage)
    sample_transforms.append(normalizeBox)
    sample_transforms.append(padBox)
    sample_transforms.append(bboxXYXY2XYWH)

    batch_transforms = []
    batch_transforms.append(randomShape)
    batch_transforms.append(normalizeImage)
    batch_transforms.append(permute)
    batch_transforms.append(gt2YoloTarget)

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    train_model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(lr=cfg.train_cfg['lr']))
    train_model.summary(line_length=130)

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size

    # 读数据的线程
    train_dic ={}
    thr = threading.Thread(target=read_train_data,
                           args=(cfg,
                                 train_indexes,
                                 train_steps,
                                 train_records,
                                 batch_size,
                                 iter_id,
                                 train_dic,
                                 use_gpu,
                                 context, with_mixup, sample_transforms, batch_transforms, target_num))
    thr.start()


    best_ap_list = [0.0, 0]  #[map, iter]
    train_speed_count = 0
    train_speed_start = 0.0
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            key_list = list(train_dic.keys())
            key_len = len(key_list)
            while key_len == 0:
                time.sleep(0.01)
                key_list = list(train_dic.keys())
                key_len = len(key_list)
            dic = train_dic.pop('%.8d'%iter_id)

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            images = dic['images']
            gt_bbox = dic['gt_bbox']
            target0 = dic['target0']
            target1 = dic['target1']
            if target_num > 2:
                target2 = dic['target2']
                targets = [target0, target1, target2]
            else:
                targets = [target0, target1]
            batch_xs = [images, gt_bbox, *targets]
            y_true = [np.zeros(batch_size) for _ in range(loss_n)]
            losses = train_model.train_on_batch(batch_xs, y_true)

            _all_loss = losses[0]
            _loss_xy = losses[1]
            _loss_wh = losses[2]
            _loss_obj = losses[3]
            _loss_cls = losses[4]
            _loss_iou = -10.0
            _loss_iou_aware = -10.0
            if yolo_loss._iou_loss is not None:
                _loss_iou = losses[5]
            if yolo_loss._iou_aware_loss is not None:
                _loss_iou_aware = losses[6]

            if cfg.use_ema:
                ema.update()   # 更新ema字典

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = ''
                if _loss_iou > 0.0 and _loss_iou_aware > 0.0:
                    strs = 'Train iter: {}, all_loss: {:.6f}, loss_xy: {:.6f}, loss_wh: {:.6f}, loss_obj: {:.6f}, loss_cls: {:.6f}, loss_iou: {:.6f}, loss_iou_aware: {:.6f}, eta: {}'.format(
                        iter_id, _all_loss, _loss_xy, _loss_wh, _loss_obj, _loss_cls, _loss_iou, _loss_iou_aware, eta)
                elif _loss_iou <= 0.0 and _loss_iou_aware > 0.0:
                    strs = 'Train iter: {}, all_loss: {:.6f}, loss_xy: {:.6f}, loss_wh: {:.6f}, loss_obj: {:.6f}, loss_cls: {:.6f}, loss_iou_aware: {:.6f}, eta: {}'.format(
                        iter_id, _all_loss, _loss_xy, _loss_wh, _loss_obj, _loss_cls, _loss_iou_aware, eta)
                elif _loss_iou > 0.0 and _loss_iou_aware <= 0.0:
                    strs = 'Train iter: {}, all_loss: {:.6f}, loss_xy: {:.6f}, loss_wh: {:.6f}, loss_obj: {:.6f}, loss_cls: {:.6f}, loss_iou: {:.6f}, eta: {}'.format(
                        iter_id, _all_loss, _loss_xy, _loss_wh, _loss_obj, _loss_cls, _loss_iou, eta)
                elif _loss_iou <= 0.0 and _loss_iou_aware <= 0.0:
                    strs = 'Train iter: {}, all_loss: {:.6f}, loss_xy: {:.6f}, loss_wh: {:.6f}, loss_obj: {:.6f}, loss_cls: {:.6f}, eta: {}'.format(
                        iter_id, _all_loss, _loss_xy, _loss_wh, _loss_obj, _loss_cls, eta)
                logger.info(strs)

            # ==================== train_speed ====================
            mod_iter_id = iter_id % 1000
            step_iter = 200   # 每隔200步计算一下训练速度。
            if mod_iter_id >= 20:   # 前20步热身。
                if mod_iter_id == 20:
                    train_speed_count = 0
                    train_speed_start = time.time()
                elif mod_iter_id > 825:
                    pass
                else:
                    train_speed_count += 1
                    if train_speed_count % step_iter == 0:
                        sts = train_speed_count // step_iter
                        sts *= step_iter
                        cost = time.time() - train_speed_start
                        logger.info('Train Speed: %.3f steps per second.' % ((sts / cost), ))

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:
                if cfg.use_ema:
                    ema.apply()
                save_path = './weights/step%.8d.h5' % iter_id
                predict_model.save_weights(save_path)
                if cfg.use_ema:
                    ema.restore()
                path_dir = os.listdir('./weights')
                steps = []
                names = []
                for name in path_dir:
                    if name[len(name) - 2:len(name)] == 'h5' and name[0:4] == 'step':
                        step = int(name[4:12])
                        steps.append(step)
                        names.append(name)
                if len(steps) > 10:
                    i = steps.index(min(steps))
                    os.remove('./weights/'+names[i])
                logger.info('Save model to {}'.format(save_path))

            # ==================== eval ====================
            if iter_id % cfg.train_cfg['eval_iter'] == 0:
                if cfg.use_ema:
                    ema.apply()
                head.set_dropblock(is_test=True)   # 没卵用，因为是静态图。为了和Pytorch版保持风格一致，故保留。
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])
                logger.info("box ap: %.3f" % (box_ap[0], ))
                head.set_dropblock(is_test=False)  # 没卵用，因为是静态图。为了和Pytorch版保持风格一致，故保留。

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    predict_model.save_weights('./weights/best_model.h5')
                if cfg.use_ema:
                    ema.restore()
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                logger.info('Done.')
                exit(0)

