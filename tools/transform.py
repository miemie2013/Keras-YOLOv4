#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-05 15:35:27
#   Description : 数据增强。凑不要脸地搬运了百度PaddleDetection的部分代码（我的另一个仓库https://github.com/miemie2013/yolact）。
#                 gt_mask相关操作被注释掉了。其实稍微改造一下YOLOv4即可支持实例分割。
#
# ================================================================
import cv2
import uuid
import numpy as np

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass



class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False):
        """ Transform the image data to numpy format.
        对图片解码。最开始的一步。把图片读出来（rgb格式），加入到sample['image']。一维数组[h, w, 1]加入到sample['im_info']
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None, coco=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()   # 增加一对键值对'image'。

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(   # 增加一对键值对'im_info'。
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)
        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context, coco)

        # 掩码图片。如果是训练集的话。
        '''if 'gt_poly' in sample:
            gt_poly = sample['gt_poly']
            assert len(gt_poly) == len(sample['gt_bbox']), "Poly Numbers Error."
            gt_mask = []
            if len(gt_poly) > 0:
                # 最初的方案，用cv2.fillPoly()画出真实掩码。因为有取整，发现有点偏差。
                # for obj_polys in gt_poly:
                #     mask = np.zeros((im.shape[0], im.shape[1]), dtype="uint8")
                #     for poly in obj_polys:  # coco数据集里，一个物体由多个多边形表示时（比如物体被挡住，不连通时）
                #         points = np.array(poly)
                #         points = np.reshape(points, (-1, 2))
                #         points = points.astype(np.int32)
                #         vertices = np.array([points], dtype=np.int32)
                #         cv2.fillPoly(mask, vertices, 1)   # 一定要是这个API而不是fillConvexPoly()，后者只能填充凸多边形而不支持凹多边形。
                #         mask = np.reshape(mask, (im.shape[0], im.shape[1], 1))
                #     gt_mask.append(mask)
                # gt_mask = np.concatenate(gt_mask, axis=-1)

                # 现在的方案，用annToMask()得到真实掩码
                anno_id = sample['anno_id']
                target = coco.loadAnns(anno_id)
                # mask是一个list，里面每一个元素是一个shape=(height*width,)的ndarray，1代表是这个注解注明的物体，0代表其他物体和背景。
                masks = [coco.annToMask(obj).reshape(-1) for obj in target]
                masks = np.vstack(masks)  # 设N=len(mask)=注解数，这一步将mask转换成一个ndarray，shape=(N, height*width)
                masks = masks.reshape(-1, im.shape[0], im.shape[1])  # shape=(N, height, width)
                gt_mask = masks.transpose(1, 2, 0)     # shape=(height, width, N)
            else:   # 对于没有gt的纯背景图，弄1个方便后面的增强跟随sample['image']
                gt_mask = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.int32)
            sample['gt_mask'] = gt_mask'''
        return sample


class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def _concat_mask(self, mask1, mask2, gt_score1, gt_score2):
        h = max(mask1.shape[0], mask2.shape[0])
        w = max(mask1.shape[1], mask2.shape[1])
        expand_mask1 = np.zeros((h, w, mask1.shape[2]), 'float32')
        expand_mask2 = np.zeros((h, w, mask2.shape[2]), 'float32')
        expand_mask1[:mask1.shape[0], :mask1.shape[1], :] = mask1
        expand_mask2[:mask2.shape[0], :mask2.shape[1], :] = mask2
        l1 = len(gt_score1)
        l2 = len(gt_score2)
        if l2 == 0:
            return expand_mask1
        elif l1 == 0:
            return expand_mask2
        mask = np.concatenate((expand_mask1, expand_mask2), axis=-1)
        return mask

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample

        # 一定概率触发mixup
        if np.random.uniform(0., 1.) < 0.5:
            sample.pop('mixup')
            return sample

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['mixup']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        # mask = self._concat_mask(sample['gt_mask'], sample['mixup']['gt_mask'], gt_score1, gt_score2)
        sample['image'] = im
        # sample['gt_mask'] = mask
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
        return sample


class PhotometricDistort(BaseOperator):
    def __init__(self):
        super(PhotometricDistort, self).__init__()

    def __call__(self, sample, context=None):
        im = sample['image']

        image = im.astype(np.float32)

        # RandomBrightness
        if np.random.randint(2):
            delta = 32
            delta = np.random.uniform(-delta, delta)
            image += delta

        state = np.random.randint(2)
        if state == 0:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if np.random.randint(2):
            lower = 0.5
            upper = 1.5
            image[:, :, 1] *= np.random.uniform(lower, upper)

        if np.random.randint(2):
            delta = 18.0
            image[:, :, 0] += np.random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if state == 1:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        sample['image'] = image
        return sample


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.

    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                min_ar, max_ar = self.aspect_ratio
                aspect_ratio = np.random.uniform(
                    max(min_ar, scale**2), min(max_ar, scale**-2))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['image'] = self._crop_image(sample['image'], crop_box)
                # gt_mask = self._crop_image(sample['gt_mask'], crop_box)    # 掩码裁剪
                # sample['gt_mask'] = np.take(gt_mask, valid_ids, axis=-1)   # 掩码筛选
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool) and
                isinstance(self.is_mask_flip, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects([rle], height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1, :]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        def is_poly(segm):
            assert isinstance(segm, (list, dict)), \
                "Invalid segm type: {}".format(type(segm))
            return isinstance(segm, list)

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            # gt_mask = sample['gt_mask']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                # gt_mask = gt_mask[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                if self.is_mask_flip and len(sample['gt_poly']) != 0:
                    sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                        height, width)
                sample['flipped'] = True
                sample['image'] = im
                # sample['gt_mask'] = gt_mask
        sample = samples if batch_input else samples[0]
        return sample

class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox
        return sample

class PadBox(BaseOperator):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes
        super(PadBox, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox

        # 掩码
        # mask = sample['gt_mask']
        # pad_mask = np.zeros((mask.shape[0], mask.shape[1], num_max), dtype=np.float32)
        # if gt_num > 0:
        #     pad_mask[:, :, :gt_num] = mask[:, :, :gt_num]
        # sample['gt_mask'] = pad_mask

        if 'gt_class' in fields:
            pad_class = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in fields:
            pad_score = np.zeros((num_max), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'is_difficult' in fields:
            pad_diff = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        return sample

class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample

class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.

    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608], random_inter=True):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        # mask_shape = shape // 4
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im

            # gt_mask = samples[i]['gt_mask']
            # 4倍下采样。与4倍下采样的特征图计算损失。
            # 不能随机插值方法，有的方法不适合50个通道。
            # gt_mask = cv2.resize(gt_mask, (mask_shape, mask_shape), interpolation=cv2.INTER_LINEAR)
            # gt_mask = (gt_mask > 0.5).astype(np.float32)
            # samples[i]['gt_mask'] = gt_mask
        return samples

class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0., 0., 0.],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    # if self.is_channel_first:
                    #     mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                    #     std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    # else:
                    #     mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                    #     std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    # im -= mean
                    # im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height

def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
        sample_bbox[2] <= object_bbox[0] or \
        sample_bbox[1] >= object_bbox[3] or \
        sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap

class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[:2]
        an_hw = np.array(self.anchors) / np.array([[w, h]])


        batch_size = len(samples)
        batch_image = np.zeros((batch_size, h, w, 3))
        # 准备标记
        batch_label_sbbox = np.zeros((batch_size, int(h / self.downsample_ratios[2]), int(w / self.downsample_ratios[2]),
                                      len(self.anchor_masks[0]), 5 + self.num_classes))
        batch_label_mbbox = np.zeros((batch_size, int(h / self.downsample_ratios[1]), int(w / self.downsample_ratios[1]),
                                      len(self.anchor_masks[0]), 5 + self.num_classes))
        batch_label_lbbox = np.zeros((batch_size, int(h / self.downsample_ratios[0]), int(w / self.downsample_ratios[0]),
                                      len(self.anchor_masks[0]), 5 + self.num_classes))
        batch_gt_bbox = np.zeros((batch_size, samples[0]['gt_bbox'].shape[0], 4))
        batch_label = [batch_label_lbbox, batch_label_mbbox, batch_label_sbbox]

        p = 0
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (grid_h, grid_w, len(mask), 5 + self.num_classes),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[gj, gi, best_n, 0] = gx * w
                        target[gj, gi, best_n, 1] = gy * h
                        target[gj, gi, best_n, 2] = gw * w
                        target[gj, gi, best_n, 3] = gh * h

                        # objectness record gt_score
                        target[gj, gi, best_n, 4] = score

                        # classification
                        onehot = np.zeros(self.num_classes, dtype=np.float)
                        onehot[cls] = 1.0
                        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                        deta = 0.01
                        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
                        target[gj, gi, best_n, 5:] = smooth_onehot

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[gj, gi, idx, 0] = gx * w
                                target[gj, gi, idx, 1] = gy * h
                                target[gj, gi, idx, 2] = gw * w
                                target[gj, gi, idx, 3] = gh * h

                                # objectness record gt_score
                                target[gj, gi, idx, 4] = score

                                # classification
                                onehot = np.zeros(self.num_classes, dtype=np.float)
                                onehot[cls] = 1.0
                                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                                deta = 0.01
                                smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
                                target[gj, gi, idx, 5:] = smooth_onehot
                # sample['target{}'.format(i)] = target
                batch_label[i][p, :, :, :, :] = target
                batch_gt_bbox[p, :, :] = gt_bbox * [w, h, w, h]
            batch_image[p, :, :, :] = im
            p += 1
        return batch_image, batch_label, batch_gt_bbox



