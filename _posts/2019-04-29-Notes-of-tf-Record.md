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

## 如何用tf-record进行读写操作？

### 写出tf-record

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



### 读取与解析tf-reocrd

创建TFRecordDataset，读入tf-record文件

```python
dataset = tf.data.TFRecordDataset(tfrecord_filename)
```

运用map API对每行样本进行处理，一般在map中定义single_example_parser自定义函数，该函数配合map来解析record。single_example_parser中需要做的

1. 用dict定义每个tf-record中key对应value的类型，其中value可以为tf.FixedLenFeature、tf.FixedLenSequenceFeature或tf.VarLenFeature

```python
# 定长特征
context_features = {
  "uid": tf.FixedLenFeature([], dtype=tf.int64),
  "sl": tf.FixedLenFeature([], dtype=tf.int64),
  "last": tf.FixedLenFeature([], dtype=tf.int64)
}

# 定长的序列特征
sequence_features = {
  "hist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
  "sub_sample": tf.FixedLenSequenceFeature([], dtype=tf.int64),
  "mask": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

# 不定长的多值特征
tf.VarLenFeature([], dtype=tf.int64)
```

2. 使用tf.parse_single_sequence_example或tf.parse_single_example将record进行解析，解析后的结果为一个dict，value即可为训练使用

```python
# 含sequence特征时
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
  serialized=serialized_example,
  context_features=context_features,
  sequence_features=sequence_features
)

# 没有sequence特征时
feature_parsed = tf.parse_single_example(
  serialized=serialized_example,
  features=
)

uid = context_parsed['uid']
sl = context_parsed['sl']
last = context_parsed['last']
sequences = sequence_parsed['hist']
sub_sample = sequence_parsed['sub_sample']
mask = sequence_parsed['mask']
```

用batch或padded_batch来对定长或不定长的的样本进行batch的操作，将连续的几个example合并成一个，并通过repeat来达到epoch的数据量

(其中padded_shapes对seq设置为None，e.g. padded_shapes = ([None], [None], [None], [], [], []))

```python
dataset = tf.contrib.data.TFRecordDataset(tfrecord_filename) \
    .map(single_example_parser) \
    .padded_batch(batch_size, padded_shapes=padded_shapes) \
    .repeat(num_epochs)
```

最后，对dataset生成iterator

```python
dataset.make_one_shot_iterator().get_next()
```



