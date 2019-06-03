---
title: "Deep-Neural-Network-for-YouTube-Recommendations"
comments: true
share: true
toc: true
categories:
  - tensorflow
tags:
  - tensorflow
---


整个推荐系统由两个NN组成，一个用于用户候选集召回，另一个用于精准排序。

召回阶段主要由用户的历史行为作为输入，将候选集的量级缩小至百的量级，这一步关注的high precision，即选出来的都是用户想要点的item；而在排序阶段，该NN模型更关注的是high recall，运用丰富的user与item特征对候选集的item进行打分，推荐给用户得分最高的item。

在线下验证阶段，会运用precision，recall，rank loss等指标来进行验证，同时配合线上的A/B实验，来观察CTR等指标是否有显著的变化。

![useful image]({{ site.url }}/assets/images/image-20190505172929916.png)

### candidate generation

#### 样本构造

* 利用用户的历史对商品的行为序列，预测下一时刻用户会点击/购买的商品（youtube正样本定义为观看完视频），等同于一个多分类问题。

* 用户历史商品序列作为一行样本，序列长度需要固定，防止模型被活跃用户dominated

* 对于每一行样本，需要在**不同channel**抽取负样本item，即同一个序列扩大n倍负样本数

  Negative sample sampling for multiclass classification problem

  - Background distribution and correct sampling via importance weighting
  - Up to serveral thousand negative samples
  - Altenatives include *hierarchical softmax* [TBU] (can speed up training)

#### 特征类别

* 用户特征：用户历史item embedding的average，历史search query embedding的average，用户demographic与device特征embedding，concat至一起
* Example age, to reflect time-dependent popularity, at serving time, set to zero

#### Loss的设计

![useful image]({{ site.url }}/assets/images/image-20190505174226337.png)

以上concat至一起user embedding vector与n个item embedding vector (one positive item + n-1 sampling negative items) 做向量乘法，接softmax

(这里的item embedding vector是item各类属性embedding concat至一起，共同学习的；user vector为了能与其相乘，会接隐藏层，输出维度与item concat后的维度相同)

```python
loss = tf.reduce_mean(-tf.cast(y, dtype=tf.float32) * tf.log(yhat + 1e-24))
# e.g. y = [[1, 0, 0, 0, 0]]  y_hat = [[0.5, 0.1, 0.1, 0.1, 0.2]]
```


#### FAQ

1. 线上serving阶段，模型如何运作？

   在training阶段，通过模型学到了user vector和item vector。

   而在serving阶段，DNN输入的特征是<user, context>类的特征，及根据用户每次刷新的状态去输入到网络中得到的最后一层的向量作为user vector，可以做到实时反馈，每次最新的浏览点击都能反馈到输入层进而得到不同的user vector。

   而video vector则是学完的item embedding table与item特征 embedding concat至一起的item vector，最终通过user_vec与item_vec的内积最大索引就能快速得到结果。

   

2. 线上serving阶段用TF-model预测几亿个item，还是用ANN？

   TF model即使能够一次计算出几百万个score，这样的时间复杂度还是过高，softmax逐个进行向量内积运算加排序，复杂度应该是nlogn级别的。但使用LSH这种ANN（Approximate Nearest Neighbo）的方法，建立索引后的复杂度甚至可以低到 logN甚至常数级别的。

   

3. example age的作用？

   example age和消除ad position bias做法类似，example age为样本产生的时间，即用户点击item的时间，而不是item上架的时间。

   用户在2.21号晚上8点0分这个时刻点击了某视频，产生了一条样本。后续训练的时候这条样本的Example Age=训练时的时刻 - 2.21号20:00时刻。而当线上serving阶段，将age至为0。


### attention version
…………………



#### Reference

1. Paper: [https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
2. 工程问题：https://zhuanlan.zhihu.com/p/52504407?utm_source=wechat_session&utm_medium=social&utm_oi=618410682702630912
3. 利用DNN做推荐的实现过程总结: [https://zhuanlan.zhihu.com/p/38638747]( https://zhuanlan.zhihu.com/p/38638747)
4. Yoho实现的例子: [https://github.com/sladesha/deep_learning/tree/master/YoutubeNetwork](https://github.com/sladesha/deep_learning/tree/master/YoutubeNetwork)
5. 一个实现例子: [https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow](https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow)
6. Scaling Up To Large Vocabulary Image Annotation, described WARP loss and online optimization: [https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37180.pdf](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37180.pdf)
7. An implementation of hierarchical softmax: [https://talbaumel.github.io/blog/softmax/](https://talbaumel.github.io/blog/softmax/)

