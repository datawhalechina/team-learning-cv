## 任务安排

开营时间：02月16日21:00

- 比赛题目：天池创新大赛：热身赛  布匹缺陷检测，内容：根据给出的布匹图片标注出其中的缺陷

- 比赛链接：https://tianchi.aliyun.com/competition/entrance/531864/introduction?spm=5176.12281976.0.0.506441a6dTFHF3


### Task00：熟悉规则（1天）

- 组队、修改群昵称。
- 熟悉打卡规则。
- 打卡截止时间：02月18日03:00

### Task01：比赛全流程体验（3天）

- 学习如何使用Docker提交代码及比赛上分。
- 记录比赛中遇到的问题，并在学习笔记中插入初始分数截图。
- 打卡截止时间：02月21日03:00
- 学习资料：
  - [Docker环境配置指南！](https://tianchi.aliyun.com/competition/entrance/231759/tab/226)
  - [比赛Docker相关操作](https://github.com/datawhalechina/team-learning-cv/blob/master/DefectDetection/docker%E6%8F%90%E4%BA%A4%E6%95%99%E7%A8%8B.pdf)

### Task02：Baseline学习及改进（5天）

- 学习baseline，并提出自己的改进策略，提交代码并更新自己的分数排名。
- 在学习笔记中插入改进baseline之后分数排名截图。
- 打卡截止时间：02月26日03:00
- 学习资料：
  - [Baseline学习及上分技巧](https://github.com/datawhalechina/team-learning-cv/blob/master/DefectDetection/README.md)

### Task03：学习者分享（2天）

- 我们根据截图，邀请提分比较多的学习者进行分享。


## 文件说明
- code : 存放所有相关代码的文件夹
    - train_data : 存放原始数据文件 guangdong1_round2_train2_20191004_Annotations  guangdong1_round2_train2_20191004_images
    - tcdata: 存放官方测试数据文件，docker 提交后会自动生成
    - data :训练数据路径设置 coco128.yaml中设置训练数据路径
    - models ： 网络相关的代码文件夹
    - weights ： 保存训练模型的文件夹，best.pt last.pt
    - convertTrainLabel.py：将官方的数据集转换成yolo数据的格式 运行生成convertor数据文件夹
    - process_data_yolo.py：滑动窗口处理convertor数据文件夹里面数据，将大图变成1024*1024小图，生成数据文件夹process_data
    - train.py :  训练代码， 运行该函数进行模型的训练，可以得到模型
    - detect.py : 预测代码
    - test.py :测试模型代码
    - run.sh : 预测测试集，生成结果的脚本   sh run.sh
    - train.sh : 训练脚本  sh trian.sh 
 
    


## 操作说明
- step1 : 将官方训练数据集解压后放入train_data 文件夹
- step2 : 训练运行  sh train.sh  
    - train.sh 有四步
        -python convertTrainLabel.py
        -python process_data_yolo.py
        -rm -rf ./convertor
        -python train.py
- step3 : 生成结果 sh run.sh

## 思路说明
- 本方案采用了yolov5作为baseline
- 数据处理：滑动窗口分割训练图片


## 改进思路
- 数据扩增：训练样本扩增随机竖直/水平翻折，色彩空间增强，使缺陷样本均匀
- 自适应anchor策略
- 适当减少回归框损失的权重
- 正负样本类别
- 多尺度训练
- 空洞卷积替换FPN最后一层
- FPN改进尝试：NAS-FPN、AC-PFN
- Anchor 匹配策略
