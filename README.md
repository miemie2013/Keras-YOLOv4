# Keras-YOLOv4

## 概述
今天是2020.05.20，脱单太难了！

6G的卡也可训练，前提是必须要冻结网络前部分的层。以下是检测效果：

![Example 0](images/res/000000014038.jpg)

参考自https://github.com/miemie2013/Keras-DIOU-YOLOv3
和https://github.com/Tianxiaomo/pytorch-YOLOv4

## 传送门
yolov3魔改成yolact：https://github.com/miemie2013/yolact

Pytorch版：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle版：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

## 更新日记

2020/05/20:初次见面

## FBI WARNING

大部分代码搬运了Keras-DIOU-YOLOv3的代码，除了网络结构大改。YOLOv4的许多料都还没有堆上去，请给我一点时间。

## 需要补充

后处理改为用张量操作实现；更多调优。

## 文件下载
一个没有训练充分的模型：链接：https://pan.baidu.com/s/1TQ1AGfylIqjAhDNaoWW9mA 
提取码：2aot

## 训练
下载我从Tianxiaomo的仓库保存下来的pytorch模型yolov4.pt
链接：https://pan.baidu.com/s/152poRrQW9Na_C8rkhNEh3g
提取码：09ou

将它放在项目根目录下。然后运行1_pytorch2keras.py得到一个yolov4.h5，它也位于根目录下。
运行train.py进行训练。train.py不支持命令行参数设置使用的数据集、超参数。
而是通过修改train.py源代码来进行更换数据集、更改超参数（减少冗余代码）。

数据集注解文件的格式如下：
```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```

或者你不下载yolov4.pt，而是下载上面提到的训练不充分的yolov4.h5继续训练也可以。
追求更高的精度，你需要把冻结层的代码删除，也就是train.py中ly.trainable = False那一部分。但是需要你有一块高显存的显卡。
训练时默认每5000步计算一次验证集的mAP。

## 评估
训练时默认每5000步计算一次验证集的mAP。或者运行eval.py评估指定模型的mAP。

## 预测
运行demo.py。

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。


## 广告位招租
可联系微信wer186259
