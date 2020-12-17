# PPYOLO AND YOLOv4

## 概述

PP-YOLO是[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型。

2020/11/05:经过不懈努力，咩酱终于在Keras上实现了可变形卷积DCNv2！这应该是咩酱最自豪的工作了。之前的种种算法（如CenterNet）因为使用了可变形卷积，而Keras、tensorflow官方没有实现可变形卷积，使得这些算法无缘在Keras平台大显身手。
而咩酱不才，一直都无法实现这一算法。经过差不多两年对算法的学习，对深度学习框架的理解，这次我再次挑战实现可变形卷积，终于大获全胜！
值得一提的是这次的DCNv2并不需要读者编译什么c、c++、cuda、自定义op这些玩意！因为这是用tensorflow的纯python接口实现，效率极高，是咩酱的得意之作！
带有DCNv2的PPYOLO，速度超过了不带有DCNv2的YOLOv4，咩酱也亲自与Pytorch版的PPYOLO（https://github.com/miemie2013/Pytorch-PPYOLO ）测过FPS，速度持平，可见实现的DCNv2效率极高。
其实一开始我并不想干这么费脑子的事情，但是抬头不见低头见，多造轮子其实是件好事，自己就会得到锻炼。下面我们来一览PPYOLO与YOLOv4的神采吧：



| 算法 | 骨干网络 | 图片输入大小 | mAP(COCO val2017) | mAP(COCO test2017) | FPS  |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|
| YOLOv4    | CSPDarkNet53 | (608,608)  | 0.491  | 0.420  | 10.3 |
| PPYOLO    | ResNet50-vd | (608,608)  | 0.448  | 0.451  | 11.9 |
| PPYOLO_r18vd    | ResNet18-vd | (608,608)  | 0.286  | -  | 33.7 |
| PPYOLO_r18vd    | ResNet18-vd | (416,416)  | 0.286  | -  | 50.8 |
| PPYOLO_r18vd    | ResNet18-vd | (320,320)  | 0.262  | -  | 65.0 |

**注意:**

- 测速环境为： win10, i5-9400F, 8GB RAM, GTX1660Ti(6GB), cuda9, tensorflow-gpu==1.12.2。若使用Linux系统FPS还能再提高。
- FPS由demo.py测得。预测50张图片，预测之前会有一个热身(warm up)阶段使速度稳定。
- 由于原版YOLOv4使用coco trainval2014进行训练，训练样本中包含部分评估样本，若使用val2017集会导致精度虚高。所以表中的0.491的精度并不可信。
- PPYOLO使用了matrix_nms进行后处理，本仓库的YOLOv4亦使用了matrix_nms进行后处理。matrix_nms拥有和fast_nms一样的速度，mAP却比后者高。
- PPYOLO_r18vd(416,416) mAP(IoU=0.50)(COCO val2017)为0.470，表中的0.286指的是mAP(IoU=0.50:0.95)(COCO val2017)。
- PPYOLO_r18vd(608,608) mAP(IoU=0.50)(COCO val2017)为0.478。不建议使用608x608输入大小。
- PPYOLO_r18vd(320,320) mAP(IoU=0.50)(COCO val2017)为0.437。


yolov4_2x.h5在val2017下的mAP:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.491
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.720
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.546
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.554
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.610
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.357
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.599
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.450
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.719
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.803
```

ppyolo_2x.h5在val2017下的mAP:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.492
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.420
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.665
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773
```

yolov4_2x.h5在test2017下的mAP:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.624
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.245
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.457
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.595
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.369
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.648
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755
```

ppyolo_2x.h5在test2017下的mAP:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.655
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.498
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.343
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.578
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.630
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.787
```


## 已实现的部分

EMA(指数滑动平均)：修改config/ppyolo_2x.py中self.use_ema = True打开。修改config/ppyolo_2x.py中self.use_ema = False关闭。打开ema会拖慢训练速度。咩酱暂时想不到好办法优化这一部分。

DropBlock：随机丢弃特征图上的像素。

IoU Loss：iou损失。

IoU Aware：预测预测框和gt的iou。并作用在objness上。

Grid Sensitive：预测框中心点的xy可以出到网格之外，应付gt中心点在网格线上这种情况。

Matrix NMS：SOLOv2中提出的算法，在soft-nms等基础上进行并行化加速，若预测框与同类高分框有iou，减小预测框的分数而不是直接丢弃。这里用box iou代替mask iou。

CoordConv：特征图带上像素的坐标信息（通道数+2）。

SPP：3个池化层的输出和原图拼接。


## 未实现的部分

多卡训练（由于咩酱只有一张6G的卡，也不是硕士生没有实验室，这部分可能不会实现）。

L2权重衰减、学习率warm up和学习率分段衰减。这些在Pytorch版和Paddle版PPYOLO中都已经实现了：

https://github.com/miemie2013/Pytorch-PPYOLO

https://github.com/miemie2013/Paddle-PPYOLO


## 咩酱刷屏时刻

Keras版YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch版YOLOv3：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle版YOLOv3：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle完美复刻版版yolact: https://github.com/miemie2013/PaddlePaddle_yolact

Keras版YOLOv4: https://github.com/miemie2013/Keras-YOLOv4 (mAP 41%+)

Pytorch版YOLOv4: https://github.com/miemie2013/Pytorch-YOLOv4 (mAP 41%+)

Paddle版YOLOv4：https://github.com/miemie2013/Paddle-YOLOv4 (mAP 41%+)

PaddleDetection版SOLOv2: https://github.com/miemie2013/PaddleDetection-SOLOv2

Pytorch实时版FCOS，跑得比YOLOv4快: https://github.com/miemie2013/Pytorch-FCOS

Paddle实时版FCOS，跑得比YOLOv4快: https://github.com/miemie2013/Paddle-FCOS

Keras版CartoonGAN: https://github.com/miemie2013/keras_CartoonGAN

纯python实现一个深度学习框架: https://github.com/miemie2013/Pure_Python_Deep_Learning

Pytorch版PPYOLO: https://github.com/miemie2013/Pytorch-PPYOLO (mAP 44.8%)

## 更新日记

2020/11/05:初次见面

## 未来工作

加入轻量级模型，如PP-YOLO_r18vd、PP-YOLO-tiny模型。

## 快速开始

(1)环境搭建

需要安装cuda9，其余见requirements.txt。

(2)下载预训练模型

下载yolov4.pt
链接：https://pan.baidu.com/s/152poRrQW9Na_C8rkhNEh3g
提取码：09ou

将它放在项目根目录下。然后运行1_yolov4_2x_2keras.py得到一个yolov4_2x.h5，它也位于根目录下。


下载PaddleDetection的ppyolo.pdparams。如果你使用Linux，请使用以下命令：
```
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
```

如果你使用Windows，请复制以下网址到浏览器或迅雷下载：
```
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
```
下载好后将它放在项目根目录下。然后运行1_ppyolo_2x_2keras.py得到一个ppyolo_2x.h5，它也位于根目录下。


下载PaddleDetection的ppyolo_r18vd.pdparams。如果你使用Linux，请使用以下命令：
```
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
```

如果你使用Windows，请复制以下网址到浏览器或迅雷下载：
```
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
```
下载好后将它放在项目根目录下。然后运行1_ppyolo_r18vd_2keras.py得到一个ppyolo_r18vd.h5，它也位于根目录下。

(3)预测图片、获取FPS（预测images/test/里的图片，结果保存在images/res/）

(如果使用yolov4_2x.py配置文件)
```
python demo.py --config=0
```

(如果使用ppyolo_2x.py配置文件)
```
python demo.py --config=1
```

(如果使用ppyolo_r18vd.py配置文件)
```
python demo.py --config=2
```


## 数据集的放置位置
数据集应该和本项目位于同一级目录。一个示例：
```
D://GitHub
     |------COCO
     |        |------annotations
     |        |------test2017
     |        |------train2017
     |        |------val2017
     |
     |------VOCdevkit
     |        |------VOC2007
     |        |        |------Annotations
     |        |        |------ImageSets
     |        |        |------JPEGImages
     |        |        |------SegmentationClass
     |        |        |------SegmentationObject
     |        |
     |        |------VOC2012
     |                 |------Annotations
     |                 |------ImageSets
     |                 |------JPEGImages
     |                 |------SegmentationClass
     |                 |------SegmentationObject
     |
     |------Keras-YOLOv4-master
              |------annotation
              |------config
              |------data
              |------model
              |------...
```


## 训练

(如果使用yolov4_2x.py配置文件)
```
python train.py --config=0
```


(如果使用ppyolo_2x.py配置文件)
```
python train.py --config=1
```

通过修改config/xxxxxxx.py的代码来进行更换数据集、更改超参数以及训练参数。

训练时如果发现mAP很稳定了，就停掉，修改学习率为原来的十分之一，接着继续训练，mAP还会再上升。暂时是这样手动操作。

## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
注意：xxx.jpg仅仅是文件名而不是文件的路径！xxx.jpg仅仅是文件名而不是文件的路径！xxx.jpg仅仅是文件名而不是文件的路径！

运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。 
在config/xxxxxxx.py里修改train_path、val_path、classes_path、train_pre_path、val_pre_path、num_classes这6个变量（自带的voc2012数据集直接解除注释就ok了）,就可以开始训练自己的数据集了。 
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。

## 评估
(如果使用yolov4_2x.py配置文件)
```
python eval.py --config=0
```

(如果使用ppyolo_2x.py配置文件)
```
python eval.py --config=1
```


## test-dev
(如果使用yolov4_2x.py配置文件)
```
python test_dev.py --config=0
```

(如果使用ppyolo_2x.py配置文件)
```
python test_dev.py --config=1
```


运行完之后，进入results目录，把bbox_detections.json压缩成bbox_detections.zip，提交到
https://competitions.codalab.org/competitions/20794#participate
获得bbox mAP。该mAP是test集的结果，也就是大部分检测算法论文的标准指标。


## 预测
(如果使用yolov4_2x.py配置文件)
```
python demo.py --config=0
```

(如果使用ppyolo_2x.py配置文件)
```
python demo.py --config=1
```


## 预测视频
(如果使用yolov4_2x.py配置文件)
```
python demo_video.py --config=0
```

(如果使用ppyolo_2x.py配置文件)
```
python demo_video.py --config=1
```
（按esc键停止播放）


## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或AIStudio上关注我（求粉）~

## 打赏

如果你觉得这个仓库对你很有帮助，可以给我打钱↓
![Example 0](weixin/sk.png)

咩酱爱你哟！另外，有偿接私活，可联系微信wer186259，金主快点来吧！
