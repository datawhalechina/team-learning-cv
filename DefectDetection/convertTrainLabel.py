import numpy as np # linear algebra
import os
import json
from tqdm.auto import tqdm
import shutil as sh
import cv2

josn_path = "./train_data/guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json"
image_path = "./train_data/guangdong1_round2_train2_20191004_images/defect/"

name_list = []
image_h_list = []
image_w_list = []
c_list = []
w_list = []
h_list = []
x_center_list = []
y_center_list = []

with open(josn_path, 'r') as f:
    temps = tqdm(json.loads(f.read()))
    for temp in temps:
        # image_w = temp["image_width"]
        # image_h = temp["image_height"]
        name = temp["name"].split('.')[0]
        path = os.path.join(image_path, name, temp["name"])
        # print('path: ',path)
        im = cv2.imread(path)
        sp = im.shape
        image_h, image_w = sp[0], sp[1]
        # print("image_h, image_w: ", image_h, image_w)
        # print("defect_name: ",temp["defect_name"])
        #bboxs
        x_l, y_l, x_r, y_r = temp["bbox"]
        # print(temp["name"], temp["bbox"])
        if temp["defect_name"]=="沾污":
            defect_name = '0'
        elif temp["defect_name"]=="错花":
            defect_name = '1'
        elif temp["defect_name"] == "水印":
            defect_name = '2'
        elif temp["defect_name"] == "花毛":
            defect_name = '3'
        elif temp["defect_name"] == "缝头":
            defect_name = '4'
        elif temp["defect_name"] == "缝头印":
            defect_name = '5'
        elif temp["defect_name"] == "虫粘":
            defect_name = '6'
        elif temp["defect_name"] == "破洞":
            defect_name = '7'
        elif temp["defect_name"] == "褶子":
            defect_name = '8'
        elif temp["defect_name"] == "织疵":
            defect_name = '9'
        elif temp["defect_name"] == "漏印":
            defect_name = '10'
        elif temp["defect_name"] == "蜡斑":
            defect_name = '11'
        elif temp["defect_name"] == "色差":
            defect_name = '12'
        elif temp["defect_name"] == "网折":
            defect_name = '13'
        elif temp["defect_name"] == "其他":
            defect_name = '14'
        else:
            defect_name = '15'
            print("----------------------------------error---------------------------")
            raise("erro")
        # print(image_w, image_h)
        # print(defect_name)
        x_center = (x_l + x_r)/(2*image_w)
        y_center = (y_l + y_r)/(2*image_h)
        w = (x_r - x_l)/(image_w)
        h = (y_r - y_l)/(image_h)
        # print(x_center, y_center, w, h)
        name_list.append(temp["name"])
        c_list.append(defect_name)
        image_h_list.append(image_w)
        image_w_list.append(image_h)
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        w_list.append(w)
        h_list.append(h)

    index = list(set(name_list))
    print(len(index))
    for fold in [0]:
        val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
        print(len(val_index))
        for num, name in enumerate(name_list):
            print(c_list[num], x_center_list[num], y_center_list[num], w_list[num], h_list[num])
            row = [c_list[num], x_center_list[num], y_center_list[num], w_list[num], h_list[num]]
            if name in val_index:
                path2save = 'val/'
            else:
                path2save = 'train/'
            # print('convertor\\fold{}\\labels\\'.format(fold) + path2save)
            # print('convertor\\fold{}/labels\\'.format(fold) + path2save + name.split('.')[0] + ".txt")
            # print("{}/{}".format(image_path, name))
            # print('convertor\\fold{}\\images\\{}\\{}'.format(fold, path2save, name))
            if not os.path.exists('convertor/fold{}/labels/'.format(fold) + path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold) + path2save)
            with open('convertor/fold{}/labels/'.format(fold) + path2save + name.split('.')[0] + ".txt", 'a+') as f:
                for data in row:
                    f.write('{} '.format(data))
                f.write('\n')
                if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
                    os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
                sh.copy(os.path.join(image_path, name.split('.')[0], name),
                        'convertor/fold{}/images/{}/{}'.format(fold, path2save, name))


