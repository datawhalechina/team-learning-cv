

#                      天池竞赛docker快速复现



## 问题核心：制作image供比赛平台pull



## 需要三步：

- 获取镜像库公网网址
- 制作镜像并推送至镜像库
- 在提交页面提交镜像地址



### 一、获取镜像库公网网址

```reStructuredText
1）在  https://cr.console.aliyun.com/ 上新建镜像库（请参考文档，同时直接建public库即可），查找镜像库公网网址，如：
registry.cn-hangzhou.aliyuncs.com/coggle/coggle

2）如果嫌阿里云的麻烦，直接在 https://hub.docker.com/ 上注册账号新建repository即可，要方便很多（但是刚制作好的镜像push跟pull要慢一些），而且亲测可用，如：
xuxml/tianchi-nlp
```



### 二、制作镜像并推送至镜像库

```
以下步骤均在本机上使用root权限操作：su - root
```



##### 1.login docker（请将tianchi替换成自己的账号）

```reStructuredText
docker login --username=tianchi registry.cn-hangzhou.aliyuncs.com
```



##### 2.在文件需要copy至镜像内的目录下新建Dockerfile文件，如：

```reStructuredText
|--tianchi-nlp
	|-- env_check.py
	|-- run.sh
	|-- result.zip
	|-- Dockerfile

注：tianchi目录下可以添加自己需要的内容
```



中Dockerfile文件内容为：

```reStructuredText
# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/keras:latest-cuda10.0-py3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 安装keras-bert包
RUN pip install keras-bert -i https://mirrors.aliyun.com/pypi/simple/

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
```

run.sh          参考：

```
python env_check.py
```

env_check.py参考：【先只写导入包的代码，确保镜像环境没问题再加其他代码】

```
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

print('done......')
```



##### 3.build image

```
docker build -t coggle:v1 .

注意最后的  .  号

如果成功会显示：Successfully built 76c643fb44ee，其中 76c643fb44ee 即为镜像id。
```



##### 4.tag and push image

```
docker tag 76c643fb44ee registry.cn-hangzhou.aliyuncs.com/coggle/coggle:v1
docker push registry.cn-hangzhou.aliyuncs.com/coggle/coggle:v1

如果提交至dockerhub：
docker login --username=xuxml  				   #xuxml需替换成自己的账号
docker tag 76c643fb44ee xuxml/tianchi-nlp:v1   #xuxml/tianchi-nlp需替换成自己的repository
docker push xuxml/tianchi-nlp:v1

注意：
1.将76c643fb44ee替换成自己的镜像ID
2.registry.cn-hangzhou.aliyuncs.com/coggle/coggle替换成自己在步骤一中的网址，其中冒号后的字符串（本例为V1）为版本号，可自行修改。
```



### 三、在提交页面提交镜像地址及版本号

```
1）在提交页面填入：registry.cn-hangzhou.aliyuncs.com/coggle/coggle:v1  即可。
2）如果是镜像push至dockerhub，填入：docker.io/xuxml/tianchi-nlp:v1      即可。（会慢一些，因为执行的时候要从dockerhub pull镜像，国内镜像源也还没同步）
```

