English | [简体中文](README.md)

# Keras-YOLOv4

## Introduction
YOu can train this yolov4 with a 6GB memory GPU with freeze some layers.

Some codes are copy from https://github.com/miemie2013/Keras-DIOU-YOLOv3


## My another Repos

Keras version YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch version YOLOv3：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle version YOLOv3：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle version yolact: https://github.com/miemie2013/PaddlePaddle_yolact

yolov3 version yolact: https://github.com/miemie2013/yolact

Keras version YOLOv4: https://github.com/miemie2013/Keras-YOLOv4

Pytorch version YOLOv4: coming soon...

## Updates

2020/05/20:This is the first time I met you.

2020/06/02:Support training custom datasets.

2020/06/05:Support fastnms mentioned in yolact.Run demo_fast.py for a try. Win10 + cuda9 + 1660Ti(6GB)，about 10 FPS if you use numpy, about 14 FPS if you use fastnms.

2020/06/08:Copy some codes from PaddleDetection.

## Something I wanna do

add the rest tricks mentioned in YOLOv4.Oh, my terrible English.

## Model Zoo
A poorly trained model step00070000.h5, I train it with a 6GB GPU, with freezing the layers before conv2d_86,I train it 70000 steps.

Link: https://pan.baidu.com/s/17R9pmdsxLo2cx-0M-EVfyg 
Extraction code: ib2u

when you have downloaded it, run eval.py to calculate mAP（with the conditions of input_shape = (608, 608), conf_thresh=0.001, nms_thresh=0.45）：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.394
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.508
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.334
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
```
To get higher mAP, you need to delete the codes to freeze layers in train.py. i.e. ly.trainable = False. But you need to have a GPU with larger memory.


## Train
Download the pytorch model yolov4.pt that I saved from tianxiaomo's repo.
Link: https://pan.baidu.com/s/152poRrQW9Na_C8rkhNEh3g
Extraction code: 09ou

Place it in this repo's root directory. Then run 1_pytorch2keras.py to get yolov4.h5, which is also in the root directory.
Run train.py to train.Modify config.py to change data set, change super parameters and training parameters.
Or you don't download yolov4.pt, but download step00070000.h5 which is not fully trained as mentioned above.
To get higher mAP, you need to delete the codes to freeze layers in train.py. i.e. ly.trainable = False. But you need to have a GPU with larger memory.

During training, the mAP of val set is calculated every 5000 steps by default.
When training, if you find that the mAP is stable, stop it, change the learning rate to one tenth of the original, then continue training, and the mAP will rise again. This is the manual operation for the time being.

## Train custom datasets
The voc2012 data set is a good example.
Put the txt annotation file of your dataset into the annotation directory.The format of the txt annotation file is as follows:
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# img_name obj1_x1,obj1_y1,obj1_x2,obj1_y2,obj1_cid obj2_x1,obj2_y1,obj2_x2,obj2_y2,obj2_cid ...
```
Run 1_txt2json.py, you will get 2 json annotation files with coco annotation style under the annotation_json directory, which are supported in train.py.
Modify train_path,val_path,classes_path,train_pre_path,val_pre_path in config.py to start training your own dataset.
In addition, you can load the weights in yolov4.h5.At this time, keras does not load only six weights of three output convolutions (because the number of output channels is different due to different categories).
If you need to run demo.py,demo_fast.py,eval.py, the variables related to the dataset also need to be modified, which should be easy to understand.


## Eval
During training, the mAP of val set is calculated every 5000 steps by default.
Or run eval.py Evaluates the mAP for the specified model. This mAP is the result of the val set.

## Test-dev
Run test_dev.py.
After running, enter the results directory and zip bbox_detections.json into bbox_detections.zip, submit to
https://competitions.codalab.org/competitions/20794#participate
to get bbox mAP.


Here is the mAP of step00070000.h5 in COCO test-dev（with the conditions of input_shape = (608, 608), conf_thresh=0.001, nms_thresh=0.45）：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.362
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.360
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.445
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.445
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.284
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
```

## Infer
run demo.py.run demo_fast.py.


