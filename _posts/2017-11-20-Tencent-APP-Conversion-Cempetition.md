---
layout: post

title: "Tencent-APP-Convertion-Prediction"

author: "Dylan"
---



## 主要特征

<br>

#### APP安装表类特征

每个APP的安装用户数，每个APP类别的安装用户数

每个用户安装APP的个数，每种教育等级、性别、年龄安装APP的个数

**7天滑窗**：过去7天用户安装的APP个数、过去7天APP被安装的用户数

<br>

#### 统计类特征

过去一段时间窗口中每个用户在每个创意ID的点击数、转化数

**组合特征**：过去一段时间窗口中广告在每一个广告位、用户在每一个站点、每一个广告位在不同网络环境、用户在不同网络环境等等的点击数、转化数（两类、三类组合特征都包括在内）

<br>

**特征的组合方式非常多，选取哪些特征来做组合？**

<br>

**穷举后用卡方检验筛特征**

参考文献[描述量选择及特征的组合优化](http://202.197.191.206:8080/30/text/chapter04/4_8.htm)提到，由于任何非穷举的算法都不能确保所得结果是最优的，因此要得最优解，就必需采用穷举法，只是在搜索技术上采用一些技巧，使计算量有可能降低。

<br>

**循环特征消减和特征重要性评级**

参考文献[scikit-learn系列之特征选择](http://www.jianshu.com/p/8f6f94f1d275)中提到，在scikit-learn中有两种特征选择的方法，一种叫做循环特征消减(Recursive Feature Elimination)和特征重要性评级 (feature importance ranking)。

-  循环特征消减：其实就是循环地移除变量和建立模型，通过模型的准确率来评估变量对模型的贡献。这种方式很暴力，但也很准确。但是问题是我们没有那么多的时间来等待模型训练这么多次。
-  特征重要性评级：“组合决策树算法”（例如Random Forest or Extra Trees）可以计算每一个属性的重要性。重要性的值可以帮助我们选择出重要的特征。

<br>

**用gbdt筛选特征**

主要思想：
GBDT每棵树的路径直接作为LR输入特征使用。

原理：
用已有特征训练GBDT模型，然后利用GBDT模型学习到的树来构造新特征，最后把这些新特征加入原有特征一起训练模型。构造的新特征向量是取值0/1的，向量的每个元素对应于GBDT模型中树的叶子结点。当一个样本点通过某棵树最终落在这棵树的一个叶子结点上，那么在新特征向量中这个叶子结点对应的元素值为1，而这棵树的其他叶子结点对应的元素值为0。新特征向量的长度等于GBDT模型里所有树包含的叶子结点数之和。

步骤：

1. 首先要切分数据集，一部分用于训练GBDT，另一部分使用训练好的GBDT模型
2. GBDT模型的apply方法生成x在GBDT每个树中的index，然后通过onehot编码做成特征。
3. 新的特征输入到分类（如LR）模型中训练分类器。

实现：
参考文献[GBDT原理及利用GBDT构造新的特征-Python实现](http://blog.csdn.net/shine19930820/article/details/71713680)的末尾有一个调用GBDT训练模型构建树，调用[apply()](http://blog.csdn.net/shine19930820/article/details/71713680)方法得到特征，然后将特征通过one-hot编码后作为新的模型输入LR进行训练。[feature trainsformation with ensembles of tree官方文档](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py)

<br>

#### APP回流时间特征

*回流时间：转化时间与点击时间的间隔*

每个app_id的平均回流时间，回流时间缺失的用该app类别的平均回流时间来代替

<br>

#### 数据Trick特征

**重复数据trick特征：**通过观察原始数据是不难发现的,有很多只有clickTime和label不一样的重复数据，按时间排序发现重复数据如果转化，label一般标在头或尾，少部分在中间，在训练集上出现的情况在测试集上也会出现，所以标记这些位置后onehot，让模型去学习

**时间差trick特征：**与重复第一条的时间差和重复最后一条的时间差

**重复行个数trick特征：**是否重复数据行数大于2

<br>

#### 均值特征

每个app_id、app_category、广告位ID的小时均值、年龄均值特征

<br>

#### 活跃度特征

**活跃app数特征：**每个app类别下的app_id个数、每一种网络环境下的app_id个数

**活跃position数特征：**每个app_id下的广告位数、每个app类别下的广告位数

**活跃user数特征：**每个app_id下的用户个数、每个广告位下的用户个数、每个用户点击的广告创意个数等

 <br>

#### 平滑特征

<br>

#### Stacking加权技巧

