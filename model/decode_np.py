# -*- coding: utf-8 -*-

import random
import colorsys
import cv2
import time
import os
import keras
import numpy as np
import keras.layers as layers
from model.yolov4 import YOLOv4


class Decode(object):
    def __init__(self, obj_threshold, nms_threshold, input_shape, model_path, file_path):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self.input_shape = input_shape
        self.all_classes = self.get_classes(file_path)
        self.num_classes = len(self.all_classes)

        self.num_anchors = 3
        inputs = layers.Input(shape=(None, None, 3))
        self._yolo = YOLOv4(inputs, self.num_classes, self.num_anchors)
        self._yolo.load_weights(model_path, by_name=True)

    # 处理一张图片
    def detect_image(self, image):
        pimage = self.process_image(np.copy(image))

        start = time.time()
        boxes, scores, classes = self.predict(pimage, image.shape)
        print('time: {0:.6f}s'.format(time.time() - start))
        if boxes is not None:
            self.draw(image, boxes, scores, classes)
        return image

    # 处理视频
    def detect_video(self, video):
        video_path = os.path.join("videos", "test", video)
        camera = cv2.VideoCapture(video_path)
        cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

        # Prepare for saving the detected video
        sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mpeg')

        vout = cv2.VideoWriter()
        vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

        while True:
            res, frame = camera.read()

            if not res:
                break

            image = self.detect_image(frame)
            cv2.imshow("detection", image)

            # Save the video frame by frame
            vout.write(image)

            if cv2.waitKey(110) & 0xff == 27:
                break

        vout.release()
        camera.release()

    def get_classes(self, file):
        with open(file) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        return class_names

    def draw(self, image, boxes, scores, classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def training_transform(self, height, width, output_height, output_width):
        height_scale, width_scale = output_height / height, output_width / width
        scale = min(height_scale, width_scale)
        resize_height, resize_width = round(height * scale), round(width * scale)
        pad_top = (output_height - resize_height) // 2
        pad_left = (output_width - resize_width) // 2
        A = np.float32([[scale, 0.0], [0.0, scale]])
        B = np.float32([[pad_left], [pad_top]])
        M = np.hstack([A, B])
        return M, output_height, output_width

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        M, h_out, w_out = self.training_transform(h, w, self.input_shape[0], self.input_shape[1])
        # 填充黑边缩放
        letterbox = cv2.warpAffine(img, M, (w_out, h_out))
        pimage = np.float32(letterbox) / 255.
        pimage = np.expand_dims(pimage, axis=0)
        return pimage

    def predict(self, image, shape):
        start = time.time()
        outs = self._yolo.predict(image)
        print('\ndarknet time: {0:.6f}s'.format(time.time() - start))

        # numpy后处理
        start = time.time()
        a1 = np.reshape(outs[0], (1, self.input_shape[0]//32, self.input_shape[1]//32, 3, 5+self.num_classes))
        a2 = np.reshape(outs[1], (1, self.input_shape[0]//16, self.input_shape[1]//16, 3, 5+self.num_classes))
        a3 = np.reshape(outs[2], (1, self.input_shape[0]//8, self.input_shape[1]//8, 3, 5+self.num_classes))
        boxes, scores, classes = self._yolo_out([a1, a2, a3], shape)
        print('post process time: {0:.6f}s'.format(time.time() - start))

        return boxes, scores, classes


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _process_feats(self, out, anchors, mask):
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

        anchors = [anchors[i] for i in mask]
        anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

        # Reshape to batch, height, width, num_anchors, box_params.
        out = out[0]
        box_xy = self._sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4])
        box_wh = box_wh * anchors_tensor

        box_confidence = self._sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self._sigmoid(out[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_shape
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._t1)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep


    def _yolo_out(self, outs, shape):
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                   [72, 146], [142, 110], [192, 243], [459, 401]]

        boxes, classes, scores = [], [], []

        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        # Scale boxes back to original image shape.
        h, w = self.input_shape
        iw, ih = shape[1], shape[0]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) / (2*scale)
        dy = (h - nh) / (2*scale)
        sc = max(iw, ih)
        image_dims = [sc, sc, sc, sc]
        dd = [dx, dy, 0, 0]
        boxes = boxes * image_dims - dd

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        # 换坐标
        boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

        return boxes, scores, classes


