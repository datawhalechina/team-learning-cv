# 零基础入门语义分割-Task1 赛题理解

本章将对语义分割赛题进行赛题背景讲解，对赛题数据读取进行说明，并给出解题思路。

## 1 赛题理解
- 赛题名称：零基础入门语义分割-地表建筑物识别
- 赛题目标：通过本次赛题可以引导大家熟练掌握语义分割任务的定义，具体的解题流程和相应的模型，并掌握语义分割任务的发展。
- 赛题任务：赛题以计算机视觉为背景，要求选手使用给定的航拍图像训练模型并完成地表建筑物识别任务。

### 1.1 学习目标
- 理解赛题背景和赛题数据
- 完成赛题报名和数据下载，理解赛题的解题思路

### 1.2 赛题数据

遥感技术已成为获取地表覆盖信息最为行之有效的手段，遥感技术已经成功应用于地表覆盖检测、植被面积检测和建筑物检测任务。本赛题使用航拍数据，需要参赛选手完成地表建筑物识别，将地表航拍图像素划分为有建筑物和无建筑物两类。

如下图，左边为原始航拍图，右边为对应的建筑物标注。

![](./img/data-example.png)

赛题数据来源（Inria Aerial Image Labeling），并进行拆分处理。数据集报名后可见并可下载。赛题数据为航拍图，需要参赛选手识别图片中的地表建筑具体像素位置。

### 1.3 数据标签

赛题为语义分割任务，因此具体的标签为图像像素类别。在赛题数据中像素属于2类（无建筑物和有建筑物），因此标签为有建筑物的像素。赛题原始图片为jpg格式，标签为RLE编码的字符串。

RLE全称（run-length encoding），翻译为游程编码或行程长度编码，对连续的黑、白像素数以不同的码字进行编码。RLE是一种简单的非破坏性资料压缩法，经常用在在语义分割比赛中对标签进行编码。

RLE与图片之间的转换如下：

```python
import numpy as np
import pandas as pd
import cv2

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
```

### 1.4 评价指标

赛题使用Dice coefficient来衡量选手结果与真实标签的差异性，Dice coefficient可以按像素差异性来比较结果的差异性。Dice coefficient的具体计算方式如下：

$$
\frac{2 * |X \cap Y|}{|X| + |Y|} 
$$

其中$X$是预测结果，$Y$为真实标签的结果。当$X$与$Y$完全相同时Dice coefficient为1，排行榜使用所有测试集图片的平均Dice coefficient来衡量，分数值越大越好。

### 1.5 读取数据

| FileName                | Size     |                                                         含义 |
| :---------------------- | :------- | -----------------------------------------------------------: |
| test_a.zip              | 314.49MB | 测试集A榜图片 |
| test_a_samplesubmit.csv | 46.39KB  | 测试集A榜提交样例 |
| train.zip               | 3.68GB   | 训练集图片 |
| train_mask.csv.zip      | 97.52MB  | 训练集图片标注 |

具体数据读取案例：

```
import pandas as pd
import cv2
train_mask = pd.read_csv('train_mask.csv', sep='\t', names=['name', 'mask'])

# 读取第一张图，并将对于的rle解码为mask矩阵
img = cv2.imread('train/'+ train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])

print(rle_encode(mask) == train_mask['mask'].iloc[0])
# 结果为True
```

### 1.6 解题思路

由于本次赛题是一个典型的语义分割任务，因此可以直接使用语义分割的模型来完成：
- 步骤1：使用FCN模型模型跑通具体模型训练过程，并对结果进行预测提交；
- 步骤2：在现有基础上加入数据扩增方法，并划分验证集以监督模型精度；
- 步骤3：使用更加强大模型结构（如Unet和PSPNet）或尺寸更大的输入完成训练；
- 步骤4：训练多个模型完成模型集成操作；

### 1.7 本章小结

本章主要对赛题背景和主要任务进行讲解，并多对赛题数据和标注读取方式进行介绍，最后列举了赛题解题思路。

### 1.8 课后作业

1. 理解RLE编码过程，并完成赛题数据读取并可视化；
2. 统计所有图片整图中没有任何建筑物像素占所有训练集图片的比例；
3. 统计所有图片中建筑物像素占所有像素的比例；
4. 统计所有图片中建筑物区域平均区域大小；