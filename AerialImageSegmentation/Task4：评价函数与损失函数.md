# Task4：评价函数与损失函数（3天）

学习主题：语义分割模型各种评价函数与损失函数

学习内容：Dice、IoU、BCE、Focal Loss、Lovász-Softmax

学习成果：评价/损失函数的实践



## TP TN FP FN

在讲解语义分割中常用的评价函数和损失函数之前，先补充一**TP(真正例 true positive) TN(真反例 true negative) FP(假正例 false positive) FN(假反例 false negative)**的知识。在分类问题中，我们经常看到上述的表述方式，以二分类为例，我们可以将所有的样本预测结果分成TP、TN、 FP、FN四类，并且每一类含有的样本数量之和为总样本数量，即TP+FP+FN+TN=总样本数量。其混淆矩阵如下：

![image-20210115164322758](语义分割评价指标和损失函数_image/image-20210115164322758.png)

上述的概念都是通过以预测结果的视角定义的，可以依据下面方式理解：

预测结果中的正例 → 在实际中是正例 → 的所有样本被称为真正例（TP）<预测正确>

预测结果中的正例 → 在实际中是反例 → 的所有样本被称为假正例（FP）<预测错误>

预测结果中的反例 → 在实际中是正例 → 的所有样本被称为假反例（FN）<预测错误>

预测结果中的反例 → 在实际中是反例 → 的所有样本被称为真反例（TN）<预测正确>

这里就不得不提及精确率（precision）和召回率（recall）：
$$
Precision=\frac{TP}{TP+FP} \\
Recall=\frac{TP}{TP+FN}
$$
$Precision$代表了预测的正例中真正的正例所占比例；$Recall$代表了真正的正例中被正确预测出来的比例。

转移到语义分割任务中来，我们可以将语义分割看作是对每一个图像像素的的分类问题。根据混淆矩阵中的定义，我们亦可以将特定像素所属的集合或区域划分成TP、TN、 FP、FN四类。

![img](语义分割评价指标和损失函数_image/20190224105100991.jpg)

以上面的图片为例，图中左子图中的人物区域（黄色像素集合）是我们**真实标注的前景信息（target）**，其他区域（紫色像素集合）为背景信息。当经过预测之后，我们会得到的一张预测结果，图中右子图中的黄色像素为**预测的前景（prediction）**，紫色像素为预测的背景区域。此时，我们便能够将预测结果分成4个部分：

预测结果中的黄色无线区域 → 真实的前景 → 的所有像素集合被称为真正例（TP）<预测正确>

预测结果中的蓝色斜线区域 → 真实的背景 → 的所有像素集合被称为假正例（FP）<预测错误>

预测结果中的红色斜线区域 → 真实的前景 → 的所有像素集合被称为假反例（FN）<预测错误>

预测结果中的白色斜线区域 → 真实的背景 → 的所有像素集合被称为真反例（TN）<预测正确>

## Dice评价指标

**Dice系数**

Dice系数（Dice coefficient）是常见的评价分割效果的方法之一，同样也可以改写成损失函数用来度量prediction和target之间的距离。Dice系数定义如下：

$$
Dice (T, P) = \frac{2 |T \cap P|}{|T| \cup |P|} = \frac{2TP}{FP+2TP+FN}
$$
式中：$T$表示真实前景（target），$P$表示预测前景（prediction）。Dice系数取值范围为$[0,1]$，其中值为1时代表预测与真实完全一致。仔细观察，Dice系数与分类评价指标中的F1 score很相似：
$$
\frac{1}{F1} = \frac{1}{Precision} + \frac{1}{Recall} \\
F1 = \frac{2TP}{FP+2TP+FN}
$$
所以，Dice系数不仅在直观上体现了target与prediction的相似程度，同时其本质上还隐含了精确率和召回率两个重要指标。

计算Dice时，将$|T \cap P|$近似为prediction与target对应元素相乘再相加的结果。$|T|$ 和$|P|$的计算直接进行简单的元素求和（也有一些做法是取平方求和），如下示例：
$$
|T \cap P| = 
\begin{bmatrix}
0.01 & 0.03 & 0.02 & 0.02 \\ 
0.05 & 0.12 & 0.09 & 0.07 \\
0.89 & 0.85 & 0.88 & 0.91 \\
0.99 & 0.97 & 0.95 & 0.97 \\
\end{bmatrix} * 
\begin{bmatrix}
0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
\end{bmatrix} \stackrel{}{\rightarrow} 
\begin{bmatrix}
0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 \\
0.89 & 0.85 & 0.88 & 0.91 \\
0.99 & 0.97 & 0.95 & 0.97 \\
\end{bmatrix} \stackrel{sum}{\rightarrow} 7.41
$$

$$
|T| = 
\begin{bmatrix}
0.01 & 0.03 & 0.02 & 0.02 \\ 
0.05 & 0.12 & 0.09 & 0.07 \\
0.89 & 0.85 & 0.88 & 0.91 \\
0.99 & 0.97 & 0.95 & 0.97 \\
\end{bmatrix} \stackrel{sum}{\rightarrow} 7.82
$$

$$
|P| = 
\begin{bmatrix}
0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
\end{bmatrix}  \stackrel{sum}{\rightarrow} 8
$$

**Dice Loss**

为了能够实现最小化的损失函数，以方便模型训练，因此常使用$1 - Dice$形式作为损失函数：
$$
L = 1-\frac{2 |T \cap P|}{|T| \cup |P|}
$$
在一些场合还可以添加上**Laplace smoothing**减少过拟合：
$$
L = 1-\frac{2 |T \cap P| + 1}{|T| \cup |P|+1}
$$
**代码实现**

```python
import numpy as np

def dice(output, target):
    '''计算Dice系数'''
    smooth = 1e-6 # 避免0为除数
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

# 生成随机两个矩阵测试
target = np.random.randint(0, 2, (3, 3))
output = np.random.randint(0, 2, (3, 3))

d = dice(output, target)
# ----------------------------
target = array([[1, 0, 0],
       			[0, 1, 1],
			    [0, 0, 1]])
output = array([[1, 0, 1],
       			[0, 1, 0],
       			[0, 0, 0]])
d = 0.5714286326530524
```

## IoU

IoU（intersection over union）指标就是常说的交并比，不仅在语义分割评价中经常被使用，在目标检测中也是常用的评价指标。顾名思义，交并比就是指target与prediction两者之间交集与并集的比值：
$$
IoU=\frac{T \cap P}{T \cup P}=\frac{TP}{FP+TP+FN}
$$
仍然以人物前景分割为例，如下图，其IoU的计算左子图中的黄色区域除以右子图中的黄色区域。

|           Intersection( $target \cap prediction$)            |               Union($target \cup prediction$)                |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210115230310670](语义分割评价指标和损失函数_image/image-20210115230310670.png) | ![image-20210115230326073](语义分割评价指标和损失函数_image/image-20210115230326073.png) |

**代码实现**

```python
def iou_score(output, target):
    '''计算IoU指标'''
	intersection = np.logical_and(target, output) 
    union = np.logical_or(target, output) 
    return np.sum(intersection) / np.sum(union)

# 生成随机两个矩阵测试
target = np.random.randint(0, 2, (3, 3))
output = np.random.randint(0, 2, (3, 3))

d = iou_score(output, target)
# ----------------------------
target = array([[1, 0, 0],
       			[0, 1, 1],
			    [0, 0, 1]])
output = array([[1, 0, 1],
       			[0, 1, 0],
       			[0, 0, 0]])
d = 0.4
```

## BCE损失函数

二分类任务时的交叉熵计算函数。此函数可以认为是nn.CrossEntropyLoss函数的特例。其分类限定为二分类，y必须是{0,1}。还需要注意的是，input应该为概率分布的形式，这样才符合交叉熵的应用。所以在BCELoss之前，input一般为sigmoid激活层的输出，官方例子也是这样给的。该损失函数在自编码器中常用。

![image-20210116000309326](语义分割评价指标和损失函数_image/image-20210116000309326.png)

```python
class torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='elementwise_mean')

参数：
weight(Tensor)- 为每个类别的loss设置权值，常用于类别不均衡问题。
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
reduce(bool)- 返回值是否为标量，默认为True
```





## 参考

[语义分割的评价指标IoU](https://blog.csdn.net/lingzhou33/article/details/87901365)

[医学图像分割常用的损失函数](https://blog.csdn.net/Biyoner/article/details/84728417)

[What is "Dice loss" for image segmentation?](https://www.jianshu.com/p/0998e6560288)
