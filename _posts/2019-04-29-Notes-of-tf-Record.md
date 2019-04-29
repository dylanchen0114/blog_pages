---
title: "Notes of tf-Record"
comments: true
share: true
toc: true
categories:
  - tensorflow
tags:
  - tensorflow
---

#### 如何用tf-record进行读写操作？

创建TFRecordWriter

```python
wtiter = tf.python_io.TFRecordWriter(path)
```

将每一行样本数据封装成tf.train.Example或tf.train.SequenceExample，如下

```python
example = tf.train.Example(features=)
example = tf.train.SequenceExample(context=, featurelists=)
```

首先介绍features与featurelists的区别，再区分何时使用example与sequenceExample：

其中features与featurelists，其组成都为feature，feature的构造方式如下：

```python
tf.train.Feature(int64_list=tf.train.Int64List(value=[uid]))  # int
tf.train.Feature(float_list=tf.train.FloatList(value=[number]))  # float
tf.train.Feature(bytes_list=tf.train.BytesList(value=[string]))  # string
```

当样本中含有的数据都为定长时，只需要用features来映射，如每行样本的二分类结果y，或单值特征；而当样本中含不定长特征时，就需要用featurelists来映射，如一个sequence，每行样本的长度不一：

创建features与featurelists非常相似，传入的都为一个dict，key为name，value不同，features中dict的value为feature，而featurelists中dict的value为featurelist，也即多个feature组成的list

```python
# 构建featurelist
frame_hist = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), hist))
frame_sub_sample = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), sub_sample))
```

```python
# 创建features与FeatureLists
tf.train.features(feature={
  'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
  'number': tf.train.Feature(float_list=tf.train.FloatList(value=[number]))
})  # features

tf.train.FeatureLists(feature_lists={
  'hist': tf.train.FeatureList(feature=frame_hist),
  'sub_sample': tf.train.FeatureList(feature=frame_sub_sample)
})
```

所以，回到最初，当有不定长特征时，用tf.train.SequenceExample，没有时用tf.train.Example

```python
example = tf.train.SequenceExample(
  context=tf.train.Features(
    feature={
      'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
      'sl': tf.train.Feature(int64_list=tf.train.Int64List(value=[sl])),
      'last': tf.train.Feature(int64_list=tf.train.Int64List(value=[last]))

    }),
  feature_lists=tf.train.FeatureLists(feature_list={
    'hist': tf.train.FeatureList(feature=frame_hist),
    'sub_sample': tf.train.FeatureList(feature=frame_sub_sample),
    'mask': tf.train.FeatureList(feature=frame_mask)
  })
)
```

最后将example序列化，用wtiter写出：

```python
wtiter.write(example.SerializeToString())
```

