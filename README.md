# Bert-NLU-Joint
一个使用tensorflow2.3开发的基于Bert的意图识别的自然语言理解任务Demo

## 环境

* python 3.6.7
* transformers==3.2.0
* tensorflow==2.3.0
* tensorflow-addons==0.11.2
* transformers==3.2.0
* sklearn==0.0

其他环境见requirements.txt

## 数据集

SMP2019比赛数据集

* json格式

![data](https://img-blog.csdnimg.cn/20201026005515144.png)

## 原理

![model](https://img-blog.csdnimg.cn/20201026003332667.png)

## 训练
在system.config中配置好参数然后运行main.py文件

* 训练中的结果

![results](https://img-blog.csdnimg.cn/20201026010230511.png)

## 参考
* [出门问问基于BERT的联合NLU模型：让机器更懂你的意思](https://zhuanlan.zhihu.com/p/93522464)
* [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)