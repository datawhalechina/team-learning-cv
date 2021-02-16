#-*- coding: utf-8 -*-
'''
@use:将图片和对应的xml生成为裁剪后两张的图片及数据集
'''

from __future__ import division
import os.path
from PIL import Image
import numpy as np
import shutil
import cv2
from tqdm import tqdm

ImgPath = './convertor/fold0/images/val/'  #原始图片
path = './convertor/fold0/labels/val/'  #原始标注

ProcessedPath = './process_data/'  #生成后数据

txtfiles = os.listdir(path)
print(txtfiles)
#patch img_size
patch_size = 1024
#slide window stride
stride = 600

txtfiles = tqdm(txtfiles)
for file in txtfiles: #遍历txt进行操作
    image_pre, ext = os.path.splitext(file)
    imgfile = ImgPath + image_pre + '.jpg'
    txtfile = path + image_pre + '.txt'
    # if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
    #     print(file)

    img = cv2.imread(imgfile)
    sp = img.shape
    img_h, img_w = sp[0], sp[1]

    f = open(os.path.join(path, file), "r")
    lines = f.readlines()
    savepath_img = ProcessedPath + 'images' + '/val'  #处理完的图片保存路径
    savepath_txt = ProcessedPath + 'labels' + '/val'  #处理完的图片标签路径
    if not os.path.exists(savepath_img):
        os.makedirs(savepath_img)
    if not os.path.exists(savepath_txt):
        os.makedirs(savepath_txt)

    bndbox = []
    boxname = []
    for line in lines:
        c, x_c, y_c, w, h, _ = line.split(' ')
        c, x_c, y_c, w, h = float(c), float(x_c), float(y_c), float(w), float(h)
        bndbox.append([x_c, y_c, w, h])
        boxname.append([c])
    # print("boxname: ", boxname)
    # b = bndbox[1]
    # print(b.nodeName)
    #a: x起点, b: y起点, w: 宽, h: 高

    a = []
    b = []
    for a_ in range(0, img_w, stride):
        a.append(a_)
    for b_ in range(0, img_h, stride):
        b.append(b_)


    cropboxes = []
    for i in a:
        for j in b:
            cropboxes.append([i, j, i + patch_size, j + patch_size])
    i = 1
    top_size, bottom_size, left_size, right_size = (150, 0, 0, 0)

    def select(m, n, w, h):
        # m: x起点, n: y起点, w: 宽, h: 高
        bbox = []
        # 查找图片中所有的 box 框
        for index in range(0, len(bndbox)):
            boxcls = boxname[index]#获取回归框的类别
            # print(bndbox[index])
            # x min
            x1 = float(bndbox[index][0] * img_w - bndbox[index][2] * img_w/2)
            # y min
            y1 = float(bndbox[index][1] * img_h - bndbox[index][3] * img_h/2)
            # x max
            x2 = float(bndbox[index][0] * img_w + bndbox[index][2] * img_w/2)
            # y max
            y2 = float(bndbox[index][1] * img_h + bndbox[index][3] * img_h/2)
            # print("the index of the box is", index)
            # print("the box cls is",boxcls[0])
            # print("the xy", x1, y1, x2, y2)
            #如果标记框在第一个范围内则存入bbox[] 并转换成新的格式
            if x1 >= m and x2 <= m + w and y1 >= n and y2 <= n + h:
                a1 = x1 - m
                b1 = y1 - n
                a2 = x2 - m
                b2 = y2 - n
                box_w = a2 - a1
                box_h = b2 - b1
                x_c = (a1 + box_w/2)/w
                y_c = (b1 + box_h/2)/h
                box_w = box_w / w
                box_h = box_h / h
                bbox.append([boxcls[0], x_c, y_c, box_w, box_h])  # 更新后的标记框
        if bbox is not None:
            return bbox
        else:
            return 0

    img = Image.open(imgfile)
    for j in range(0, len(cropboxes)):
        # print("the img number is :", j)
        # 获取在 patch 的 box
        Bboxes = select(cropboxes[j][0], cropboxes[j][1], patch_size, patch_size)
        if len(Bboxes):
            with open(savepath_txt + '/' + image_pre + '_' + '{}'.format(j) + '.txt', 'w') as f:
                for Bbox in Bboxes:
                    for data in Bbox:
                        f.write('{} '.format(data))
                    f.write('\n')

        #图片裁剪
            try:
                cropedimg = img.crop(cropboxes[j])
                # print(np.array(cropedimg).shape)
                cropedimg.save(savepath_img + '/' + image_pre + '_' + str(j) + '.jpg')
                # print("done!")
            except:
                continue








