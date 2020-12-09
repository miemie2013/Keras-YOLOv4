#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-05-20 15:35:27
#   Description : Convert annotation files (voc format) into txt format.
#                 将voc注解格式数据集的注解转换成txt注解格式。生成的txt注解文件在annotation目录下。
#
# ================================================================
import os
import shutil



'''
将 dataset_dir 改为你的数据集的路径。
生成的txt注解文件格式为：
图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...

train_difficult控制是否训练难例。use_default_label控制是否使用默认的类别文件。
'''


# 是否训练难例。
train_difficult = True
# train_difficult = False


# 是否使用默认的类别文件。
use_default_label = True
# use_default_label = False


dataset_dir = '../VOCdevkit/VOC2012/'
train_path = dataset_dir + 'ImageSets/Main/train.txt'
val_path = dataset_dir + 'ImageSets/Main/val.txt'
# test_path = dataset_dir + 'ImageSets/Main/test.txt'
test_path = None
annos_dir = dataset_dir + 'Annotations/'


# 保存的txt注解文件的文件名
train_txt_name = 'voc2012_train.txt'
val_txt_name = 'voc2012_val.txt'
test_txt_name = 'voc2012_test.txt'



class_names = []
class_names_ids = {}
cid_index = 0


if use_default_label:
    # class_txt_name指向已有的类别文件，一行一个类别名。类别id根据这个类别文件中类别名在第几行确定。
    # 如果只训练该数据集的部分类别，那么编辑该类别文件，只留下所需类别的类别名即可。
    class_txt_name = 'data/voc_classes.txt'
    if not os.path.exists(class_txt_name):
        raise FileNotFoundError("%s does not exist!" % class_txt_name)
    with open(class_txt_name, 'r', encoding='utf-8') as f:
        for line in f:
            cname = line.strip()
            class_names.append(cname)
            class_names_ids[cname] = cid_index
            cid_index += 1
else:   # 如果不使用默认的类别文件。则会分析出有几个类别，生成一个类别文件。
    # 保存的类别文件名
    class_txt_name = 'data/class_names.txt'



train_names = []
val_names = []
test_names = []

with open(train_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        train_names.append(line)
with open(val_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        val_names.append(line)
if test_path is not None:
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            test_names.append(line)




# 创建txt注解目录
if os.path.exists('annotation/'): shutil.rmtree('annotation/')
os.mkdir('annotation/')


def write_txt(xml_names, annos_dir, txt_name, use_default_label, train_difficult, class_names, class_names_ids, cid_index):
    content = ''
    for xml_name in xml_names:
        xml_file = '%s%s.xml'%(annos_dir, xml_name)
        enter_gt = False
        enter_part = False
        x0, y0, x1, y1, cid = '', '', '', '', -10
        difficult = 0
        img_name = ''
        bboxes = ''
        with open(xml_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '<filename>' in line:
                    if '</filename>' in line:
                        ss = line.split('name>')
                        sss = ss[1].split('</file')
                        img_name = sss[0]
                    else:
                        print('Error 1.')
                if '<object>' in line:
                    if '</object>' in line:
                        print('Error 2.')
                    else:
                        enter_gt = True
                if '</object>' in line:
                    if cid > -5:
                        if train_difficult:
                            bboxes += ' %s,%s,%s,%s,%d'%(x0, y0, x1, y1, cid)
                        else:
                            if difficult == 0:
                                bboxes += ' %s,%s,%s,%s,%d'%(x0, y0, x1, y1, cid)
                    x0, y0, x1, y1, cid = '', '', '', '', -10
                    difficult = 0
                    enter_gt = False
                    enter_part = False
                if enter_gt:
                    if '<part>' in line:   # <object>里会有<part>节点，我们要忽略<part>节点。
                        if '</part>' in line:
                            print('Error part.')
                        else:
                            enter_part = True
                    if '</part>' in line:
                        enter_part = False
                    if not enter_part:
                        if '<name>' in line:
                            if '</name>' in line:
                                ss = line.split('name>')
                                sss = ss[1].split('</')
                                cname = sss[0]
                                if use_default_label:
                                    if cname not in class_names:
                                        cid = -10
                                    else:
                                        cid = class_names_ids[cname]
                                else:
                                    if cname not in class_names:
                                        class_names.append(cname)
                                        class_names_ids[cname] = cid_index
                                        cid_index += 1
                                    cid = class_names_ids[cname]
                            else:
                                print('Error 3.')
                        if '<xmin>' in line:
                            if '</xmin>' in line:
                                ss = line.split('xmin>')
                                sss = ss[1].split('</')
                                x0 = sss[0]
                            else:
                                print('Error 4.')
                        if '<ymin>' in line:
                            if '</ymin>' in line:
                                ss = line.split('ymin>')
                                sss = ss[1].split('</')
                                y0 = sss[0]
                            else:
                                print('Error 5.')
                        if '<xmax>' in line:
                            if '</xmax>' in line:
                                ss = line.split('xmax>')
                                sss = ss[1].split('</')
                                x1 = sss[0]
                            else:
                                print('Error 6.')
                        if '<ymax>' in line:
                            if '</ymax>' in line:
                                ss = line.split('ymax>')
                                sss = ss[1].split('</')
                                y1 = sss[0]
                            else:
                                print('Error 7.')
                        if '<difficult>' in line:
                            if '</difficult>' in line:
                                ss = line.split('difficult>')
                                sss = ss[1].split('</')
                                difficult = int(sss[0])
                            else:
                                print('Error 8.')
        content += img_name + bboxes + '\n'
    with open('annotation/%s' % txt_name, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()
    return class_names, class_names_ids, cid_index


# train set
class_names, class_names_ids, cid_index = write_txt(train_names, annos_dir, train_txt_name,
                                                    use_default_label, train_difficult, class_names, class_names_ids, cid_index)

# val set
class_names, class_names_ids, cid_index = write_txt(val_names, annos_dir, val_txt_name,
                                                    use_default_label, train_difficult, class_names, class_names_ids, cid_index)

# test set
if test_path is not None:
    class_names, class_names_ids, cid_index = write_txt(test_names, annos_dir, test_txt_name,
                                                        use_default_label, train_difficult, class_names, class_names_ids, cid_index)


if not use_default_label:
    num_classes = len(class_names)
    content = ''
    for cid in range(num_classes):
        for cname in class_names_ids.keys():
            if cid == class_names_ids[cname]:
                content += cname + '\n'
                break

    if not os.path.exists('data/'): os.mkdir('data/')
    with open(class_txt_name, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()

print('Done.')











