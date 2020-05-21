# Keras-YOLOv4

## 概述
今天是2020.05.20，脱单太难了！

6G的卡也可训练，前提是必须要冻结网络前部分的层。以下是检测效果：

![Example 0](images/res/000000014038.jpg)

参考自https://github.com/miemie2013/Keras-DIOU-YOLOv3
和https://github.com/Tianxiaomo/pytorch-YOLOv4

## 传送门
3款yolov3，总有一款适合你。paddlepaddle版，可以使用百度Tesla V100的算力。Pytorch，学术方面最火的深度学习框架，动态图，不必开sess，调试最方便。Keras，对新手友好的深度学习框架。

Pytorch兄弟版：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle姐妹版：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

## 更新日记

2020/05/20:初次见面

## FBI WARNING

大部分代码搬运了Keras-DIOU-YOLOv3的代码，除了网络结构大改。YOLOv4的许多料都还没有堆上去，请给我一点时间。

## 需要补充

验证的代码；后处理改为用张量操作实现；更多调优。

## 文件下载
一个没有训练充分的模型：链接：https://pan.baidu.com/s/1TQ1AGfylIqjAhDNaoWW9mA 
提取码：2aot

## 仓库文件介绍

```
train.py            训练yolov4。
1_lambda2model.py   将训练模型中yolov4的所有部分提取出来。
1_pytorch2keras.py  将Tianxiaomo的pytorch模型导出为keras模型。
demo.py             用keras模型进行预测。


annotation/  存放训练集、验证集的注解文件。
data/        存放数据集物品类别名称文件（一行一个类别名称），类别名称最好不要有空格、斜杠、反斜杠，不然后面计算mAP时会报错。
images/      用于测试的图片，放在子目录test/下。预测输出在子目录res/下。
mAP/         对模型评估时产生的中间临时文件。
model/       存放yolov4核心代码。
```

## 训练
使用train.py进行训练。train.py不支持命令行参数设置使用的数据集、超参数。
而是通过修改train.py源代码来进行更换数据集、更改超参数（减少冗余代码）。
1.如果你要使用自己的数据集训练，那么请修改
```
train_path = 'annotation/coco2017_train.txt'
val_path = 'annotation/coco2017_val.txt'
classes_path = 'data/coco_classes.txt'
```

注解文件的格式如下：
```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```
和YunYang1994的注解文件格式是完全一样的，这里再次致敬大佬！

2.本仓库有pattern=0、pattern=1、pattern=2这3种训练模式。
0-从头训练，1-读取model_body继续训练（包括解冻，但需要先运行1_lambda2model.py脚本取得model_body），2-读取coco预训练模型训练
你只需要修改pattern的值即可指定训练模式。
然后在这3种模式的if-else分支下，你再指定批大小batch_size、学习率lr等超参数。

3.如果你决定从头训练一个模型（即pattern=0），而且你的显卡显存比较小，比如说只有6G。
又或者说你想训练一个小模型，因为你的数据集比较小。
那么你可以设置initial_filters为一个比较小的值，比如说8。
initial_filters会影响到后面的卷积层的卷积核个数（除了最后面3个卷积层的卷积核个数不受影响）。
yolov3的initial_filters默认是32，你调小initial_filters会使得模型变小，运算量减少，适合在小数据集上训练。


## 评估
训练完成后，用1_lambda2model.py将训练模型中yolov4的所有部分提取出来。
运行evaluate_kr.py对keras模型（1_lambda2model.py提取出来的模型）评估，跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。


## 预测
运行demo.py。

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。


## 广告位招租
可联系微信wer186259
