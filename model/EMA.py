#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : PaddleDetection
#   Created date:
#   Description :
#
# ================================================================
import keras
import numpy as np
import time
import threading


class ExponentialMovingAverage():
    def __init__(self, model, decay, thres_steps=True):
        self._model = model
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    def register(self):
        self._update_step = 0
        for i in range(len(self._model.layers)):
            ly = self._model.layers[i]
            if ly.trainable is False:   # 只记录可训练参数。
                continue
            weights = ly.get_weights()
            if len(weights) == 0:
                continue
            layer_name = ly.name
            if isinstance(ly, keras.layers.BatchNormalization):   # bn层的均值、方差不会被记录，它们有自己的滑动平均。
                scale, offset, m, v = weights
                self._shadow['%s.scale' % layer_name] = scale.copy()
                self._shadow['%s.offset' % layer_name] = offset.copy()
            else:
                for k in range(len(weights)):
                    w = weights[k]
                    self._shadow['%s.%d' % (layer_name, k)] = w.copy()

    def update(self):
        start = time.time()
        decay = min(self._decay, (1 + self._update_step) / (10 + self._update_step)) if self._thres_steps else self._decay
        for i in range(len(self._model.layers)):
            ly = self._model.layers[i]
            if ly.trainable is False:
                continue
            weights = ly.get_weights()
            if len(weights) == 0:
                continue
            layer_name = ly.name
            if isinstance(ly, keras.layers.BatchNormalization):
                scale, offset, m, v = weights

                name = '%s.scale' % layer_name
                new_val = scale.copy()
                old_val = np.array(self._shadow[name])
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[name] = new_average

                name = '%s.offset' % layer_name
                new_val = offset.copy()
                old_val = np.array(self._shadow[name])
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[name] = new_average
            else:
                for k in range(len(weights)):
                    name = '%s.%d' % (layer_name, k)
                    w = weights[k]
                    new_val = w.copy()
                    old_val = np.array(self._shadow[name])
                    new_average = decay * old_val + (1 - decay) * new_val
                    self._shadow[name] = new_average
        self._update_step += 1
        cost = time.time() - start
        # print('cost time: {0:.6f}s'.format(cost))
        return decay

    def apply(self):
        for i in range(len(self._model.layers)):
            ly = self._model.layers[i]
            if ly.trainable is False:
                continue
            weights = ly.get_weights()
            if len(weights) == 0:
                continue
            layer_name = ly.name
            if isinstance(ly, keras.layers.BatchNormalization):
                scale, offset, m, v = weights

                name = '%s.scale' % layer_name
                self._backup[name] = scale.copy()
                scale2 = self._shadow[name]

                name = '%s.offset' % layer_name
                self._backup[name] = offset.copy()
                offset2 = self._shadow[name]

                ly.set_weights([scale2, offset2, m, v])
            else:
                ws = []
                for k in range(len(weights)):
                    name = '%s.%d' % (layer_name, k)
                    w = weights[k]
                    self._backup[name] = w.copy()
                    w2 = self._shadow[name]
                    ws.append(w2.copy())
                ly.set_weights(ws)

    def restore(self):
        for i in range(len(self._model.layers)):
            ly = self._model.layers[i]
            if ly.trainable is False:
                continue
            weights = ly.get_weights()
            if len(weights) == 0:
                continue
            layer_name = ly.name
            if isinstance(ly, keras.layers.BatchNormalization):
                scale, offset, m, v = weights

                name = '%s.scale' % layer_name
                scale2 = self._backup[name]

                name = '%s.offset' % layer_name
                offset2 = self._backup[name]

                ly.set_weights([scale2, offset2, m, v])
            else:
                ws = []
                for k in range(len(weights)):
                    name = '%s.%d' % (layer_name, k)
                    w2 = self._backup[name]
                    ws.append(w2.copy())
                ly.set_weights(ws)
        self._backup = {}
