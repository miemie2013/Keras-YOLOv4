


安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0b0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*


下载预训练模型ppyolo.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
python 1_ppyolo_2x_2paddle.py



下载预训练模型ResNet50_vd_ssld_pretrained.tar
cd ~/w*
wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
tar -xf ResNet50_vd_ssld_pretrained.tar
python 1_r50vd_ssld_2paddle.py
rm -f ResNet50_vd_ssld_pretrained.tar
rm -rf ResNet50_vd_ssld_pretrained



-------------------------------- PPYOLO --------------------------------
=============== 训练 ===============
(如果使用yolov4_2x.py配置文件)
python train.py --config=0

(如果使用ppyolo_2x.py配置文件)
python train.py --config=1

(如果使用ppyolo_r18vd.py配置文件)
python train.py --config=2



=============== 预测 ===============
(如果使用yolov4_2x.py配置文件)
python demo.py --config=0

(如果使用ppyolo_2x.py配置文件)
python demo.py --config=1

(如果使用ppyolo_r18vd.py配置文件)
python demo.py --config=2


=============== 预测视频 ===============
(如果使用yolov4_2x.py配置文件)
python demo_video.py --config=0

(如果使用ppyolo_2x.py配置文件)
python demo_video.py --config=1

(如果使用ppyolo_r18vd.py配置文件)
python demo_video.py --config=2


=============== 验证 ===============
(如果使用yolov4_2x.py配置文件)
python eval.py --config=0

(如果使用ppyolo_2x.py配置文件)
python eval.py --config=1

(如果使用ppyolo_r18vd.py配置文件)
python eval.py --config=2



=============== 跑test_dev ===============
(如果使用yolov4_2x.py配置文件)
python test_dev.py --config=0

(如果使用ppyolo_2x.py配置文件)
python test_dev.py --config=1

(如果使用ppyolo_r18vd.py配置文件)
python test_dev.py --config=2






















