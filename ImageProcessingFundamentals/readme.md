﻿# 计算机视觉基础：图像处理（上）


## 基本信息
- 贡献人员：王程伟、任乔牧、张强、李芝翔
- 学习周期：12天，每天平均花费时间2小时-5小时不等，根据个人学习接受能力强弱有所浮动。
- 学习形式：理论学习 + 练习
- 人群定位：具备一定编程基础，了解OpenCV，有学习和梳理图像处理算法的需求。
- 先修内容：无
- 难度系数：中



## 任务安排

### Task01：OpenCV框架、图像插值算法—图像缩放（2天）

**理论部分**

* 了解OpenCV的框架组成
* 掌握基本的图像插值算法
  * 最近邻插值算法：掌握OpenCV的API、理解算法原理
  * 双线性插值算法：掌握OpenCV的API、理解算法原理

**练习部分**

* 调用OpenCV插值算法的API，使用不同的插值算法完成图像的缩放
* 不调用OpenCV插值算法的API，基于OpenCV自己实现两种插值算法并完成图像的缩放（可选）



### Task02：几何变换（2天）

**理论部分**

* 掌握图像几何变换（平移、旋转）的原理

**练习部分**

* 利用OpenCV实现图像的几何变换（平移、旋转）
  * 调用OpenCV对应的API
  * 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
    

    
### Task03：彩色空间互转（2天）

**理论部分**

- 掌握RGB与灰度图互转的原理
- 掌握RGB与HSV空间互转的原理

**练习部分**

- 利用OpenCV实现图像的RGB与灰度图互转
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
- 利用OpenCV实现图像的RGB与HSV空间互转
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
  

 
### Task04：图像滤波（2天）

**理论部分**

- 掌握均值滤波和方框滤波的原理
- 掌握高斯滤波的原理

**练习部分**

- 利用OpenCV对图像进行均值滤波和方框滤波
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
- 利用OpenCV对图像进行高斯滤波
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现 （可选）  
- 分析和理解不同滤波算法的适用场合和性能



### Task05：图像分割/二值化（2天）

**理论部分**

- 掌握大津法(最大类间方差法)的原理
- 掌握自适应阈值分割法（adaptiveThreshold）的原理

**练习部分**

- 利用OpenCV实现大津法(最大类间方差法)，对图像进行阈值分割
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
- 利用OpenCV实现自适应阈值分割法，对图像进行阈值分割
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）



### Task06：边缘检测（2天）

**理论部分**

- 掌握Sobel边缘检测的原理
- 掌握Canny边缘检测的原理

**练习部分**

- 利用OpenCV实现Sobel边缘检测
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
- 利用OpenCV实现Canny边缘检测
    - 调用OpenCV对应的API
    - 不调用OpenCV对应的API，利用OpenCV自己实现（可选）
- 分析和理解不同边缘检测算法的适用场合和性能

### Task07：兴趣扩展项目(可选)

- 根据个人需求和兴趣，可以结合项目进行实操，为该项目添加图像处理的功能：https://github.com/QiangZiBro/opencv-pyqt5


##  参考资料
- 软件包及安装
    - [OpenCV各版本下载](https://opencv.org/releases/)
    - [OpenCV+Python3.x以上，Window版和Linux版](https://github.com/vipstone/faceai/blob/master/doc/settingup.md)
    - [OpenCV+VS版1](https://blog.csdn.net/weixin_40647819/article/details/79938325)
    - [OpenCV+VS版2](http://notes.maxwi.com/2016/12/05/opencv-windows-env/)
- 相关文档
    - [OpenCV官方文档1](https://docs.opencv.org/3.0-last-rst/)
    - [OpenCV官方文档2](https://docs.opencv.org/3.1.0/index.html)
    - [OpenCV中文网站](http://wiki.opencv.org.cn/index.php/%E9%A6%96%E9%A1%B5)
    - [OpenCV中文文档](http://www.woshicver.com/)
    - [OpenCV中文教程](https://www.kancloud.cn/aollo/aolloopencv/269602)



# 计算机视觉基础：图像处理（下）



## 基本信息
- 贡献人员：王程伟、张强、李芝翔
- 学习周期：15天，每天平均花费时间 2小时-5小时不等，根据个人学习接受能力强弱有所浮动。
- 学习形式：理论学习 + 练习
- 人群定位：具备一定编程基础，了解 OpenCV，有学习和梳理图像处理算法的需求，参与过图像处理（上）组队学习者优先。
- 先修内容：[计算机视觉基础：图像处理（上）](https://github.com/datawhalechina/team-learning-cv/tree/master/ImageProcessingFundamentals)
- 难度系数：中


## 任务安排

### Task01：Harris特征点检测器-兴趣点检测（3天）

**理论部分**
- 掌握Harris特征点检测的原理
 
**练习部分**

- 使用OpenCV集成的Harris特征点检测器实现图像兴趣点检测


    
### Task02：LBP特征描述算子-人脸检测（4天）

**理论部分**

- 掌握LBP特征描述算子原理

**练习部分**

- 使用OpenCV的LBP检测器完成人脸检测任务


 
### Task03：Harr特征描述算子-人脸检测（4天）

**理论部分**

- 掌握Harr特征描述算子原理

**练习部分**

- 使用OpenCV的Harr检测器完成人脸检测任务



### Task04：HOG特征描述算子-行人检测（4天）

**理论部分**

- 掌握HOG特征描述算子的原理

**练习部分**

- 使用OpenCV预训练的HOG+SVM检测器完成行人检测任务

##  参考资料
- 软件包及安装
    - [OpenCV各版本下载](https://opencv.org/releases/)
    - [OpenCV+Python3.x以上，Window版和Linux版](https://github.com/vipstone/faceai/blob/master/doc/settingup.md)
    - [OpenCV+VS版1](https://blog.csdn.net/weixin_40647819/article/details/79938325)
    - [OpenCV+VS版2](http://notes.maxwi.com/2016/12/05/opencv-windows-env/)
- 相关文档
    - [OpenCV官方文档1](https://docs.opencv.org/3.0-last-rst/)
    - [OpenCV官方文档2](https://docs.opencv.org/3.1.0/index.html)
    - [OpenCV中文网站](http://wiki.opencv.org.cn/index.php/%E9%A6%96%E9%A1%B5)
    - [OpenCV中文文档](http://www.woshicver.com/)
    - [OpenCV中文教程](https://www.kancloud.cn/aollo/aolloopencv/269602)

