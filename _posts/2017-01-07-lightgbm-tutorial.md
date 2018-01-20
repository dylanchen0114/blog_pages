---
title: "LightGBM-Tutorial"
comments: true
share: true
toc: true
categories:
  - model
tags:
  - model
---

Parameters Tuning
=================

https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-Tuning.rst

Tune Parameters for the Leaf-wise (Best-first) Tree
---------------------------------------------------

LightGBM uses the `leaf-wise` tree growth algorithm, while many other popular tools use depth-wise tree growth.
Compared with depth-wise growth, the leaf-wise algorithm can converge much faster.
However, the leaf-wise growth may be over-fitting if not used with the appropriate parameters.

To get good results using a leaf-wise tree, these are some important parameters:

1. ``num_leaves``. This is the main parameter to control the complexity of the tree model.
   Theoretically, we can set ``num_leaves = 2^(max_depth)`` to convert from depth-wise tree.
   However, this simple conversion is not good in practice.
   The reason is, when number of leaves are the same, the leaf-wise tree is much deeper than depth-wise tree. As a result, it may be over-fitting.
   Thus, when trying to tune the ``num_leaves``, we should let it be smaller than ``2^(max_depth)``.
   For example, when the ``max_depth=6`` the depth-wise tree can get good accuracy,
   but setting ``num_leaves`` to ``127`` may cause over-fitting, and setting it to ``70`` or ``80`` may get better accuracy than depth-wise.
   Actually, the concept ``depth`` can be forgotten in leaf-wise tree, since it doesn't have a correct mapping from ``leaves`` to ``depth``.

2. ``min_data_in_leaf``. This is a very important parameter to deal with over-fitting in leaf-wise tree.
   Its value depends on the number of training data and ``num_leaves``.
   Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting.
   In practice, setting it to hundreds or thousands is enough for a large dataset.

3. ``max_depth``. You also can use ``max_depth`` to limit the tree depth explicitly.

For Faster Speed
----------------

-  Use bagging by setting ``bagging_fraction`` and ``bagging_freq``

-  Use feature sub-sampling by setting ``feature_fraction``

-  Use small ``max_bin``

-  Use ``save_binary`` to speed up data loading in future learning

-  Use parallel learning, refer to `Parallel Learning Guide`


For Better Accuracy
-------------------

-  Use large ``max_bin`` (may be slower)

-  Use small ``learning_rate`` with large ``num_iterations``

-  Use large ``num_leaves`` (may cause over-fitting)

-  Use bigger training data

-  Try ``dart``

Deal with Over-fitting
----------------------

-  Use small ``max_bin``

-  Use small ``num_leaves``

-  Use ``min_data_in_leaf`` and ``min_sum_hessian_in_leaf``

-  Use bagging by set ``bagging_fraction`` and ``bagging_freq``

-  Use feature sub-sampling by set ``feature_fraction``

-  Use bigger training data

-  Try ``lambda_l1``, ``lambda_l2`` and ``min_gain_to_split`` for regularization

-  Try ``max_depth`` to avoid growing deep tree

# Parameter API

Python API: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

-  ``config``, default= ``""``, type=string, alias= ``config_file``

   -  path of config file

   -  **Note**: Only can be used in CLI version.

-  ``task``, default= ``train``, type=enum, options= ``train``, ``predict``, ``convert_model``, ``refit``

   -  ``train``, alias= ``training``, for training

   -  ``predict``, alias= ``prediction``, ``test``, for prediction.

   -  ``convert_model``, for converting model file into if-else format, see more information in `Convert model parameters `

   -  ``refit``, alias= ``refit_tree``, refit existing models with new data.

   -  **Note**: Only can be used in CLI version.

-  ``application``, default= ``regression``, type=enum,
   options= ``regression``, ``regression_l1``, ``huber``, ``fair``, ``poisson``, ``quantile``, ``mape``,
   ``binary``, ``multiclass``, ``multiclassova``, ``xentropy``, ``xentlambda``, ``lambdarank``,
   alias= ``objective``, ``app``

   -  regression application

      -  ``regression_l2``, L2 loss, alias= ``regression``, ``mean_squared_error``, ``mse``

      -  ``regression_l1``, L1 loss, alias= ``mean_absolute_error``, ``mae``

      -  ``huber``, `Huber loss`_

      -  ``fair``, `Fair loss`_

      -  ``poisson``, `Poisson regression`_

      -  ``quantile``, `Quantile regression`_

      -  ``mape``, `MAPE loss`_, alias= ``mean_absolute_percentage_error``

   -  ``binary``, binary `log loss`_ classification application

   -  multi-class classification application

      -  ``multiclass``, `softmax`_ objective function, ``num_class`` should be set as well

      -  ``multiclassova``, `One-vs-All` binary objective function, ``num_class`` should be set as well

   -  cross-entropy application

      -  ``xentropy``, objective function for cross-entropy (with optional linear weights), alias= ``cross_entropy``

      -  ``xentlambda``, alternative parameterization of cross-entropy, alias= ``cross_entropy_lambda``

      -  the label is anything in interval [0, 1]

   -  ``lambdarank``, `lambdarank`_ application

      -  the label should be ``int`` type in lambdarank tasks, and larger number represent the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)

      -  `label_gain <#objective-parameters>`can be used to set the gain(weight) of ``int`` label

      -  all values in ``label`` must be smaller than number of elements in ``label_gain``

-  ``boosting``, default= ``gbdt``, type=enum,
   options= ``gbdt``, ``rf``, ``dart``, ``goss``,
   alias= ``boost``, ``boosting_type``

   -  ``gbdt``, traditional Gradient Boosting Decision Tree

   -  ``rf``, Random Forest

   -  ``dart``, `Dropouts meet Multiple Additive Regression Trees`

   -  ``goss``, Gradient-based One-Side Sampling

-  ``data``, default= ``""``, type=string, alias= ``train``, ``train_data``

   -  training data, LightGBM will train from this data

-  ``valid``, default= ``""``, type=multi-string, alias= ``test``, ``valid_data``, ``test_data``

   -  validation/test data, LightGBM will output metrics for these data

   -  support multi validation data, separate by ``,``

-  ``num_iterations``, default= ``100``, type=int,
   alias= ``num_iteration``, ``num_tree``, ``num_trees``, ``num_round``, ``num_rounds``, ``num_boost_round``, ``n_estimators``

   -  number of boosting iterations

   -  **Note**: for Python/R package, **this parameter is ignored**,
      use ``num_boost_round`` (Python) or ``nrounds`` (R) input arguments of ``train`` and ``cv`` methods instead

   -  **Note**: internally, LightGBM constructs ``num_class * num_iterations`` trees for ``multiclass`` problems

-  ``learning_rate``, default= ``0.1``, type=double, alias= ``shrinkage_rate``

   -  shrinkage rate

   -  in ``dart``, it also affects on normalization weights of dropped trees

-  ``num_leaves``, default= ``31``, type=int, alias= ``num_leaf``

   -  number of leaves in one tree

-  ``tree_learner``, default= ``serial``, type=enum, options= ``serial``, ``feature``, ``data``, ``voting``, alias= ``tree``

   -  ``serial``, single machine tree learner

   -  ``feature``, alias= ``feature_parallel``, feature parallel tree learner

   -  ``data``, alias= ``data_parallel``, data parallel tree learner

   -  ``voting``, alias= ``voting_parallel``, voting parallel tree learner

   -  refer to `Parallel Learning Guide` to get more details

-  ``num_threads``, default= ``OpenMP_default``, type=int, alias= ``num_thread``, ``nthread``

   -  number of threads for LightGBM

   -  for the best speed, set this to the number of **real CPU cores**,
      not the number of threads (most CPU using `hyper-threading` to generate 2 threads per CPU core)

   -  do not set it too large if your dataset is small (do not use 64 threads for a dataset with 10,000 rows for instance)

   -  be aware a task manager or any similar CPU monitoring tool might report cores not being fully utilized. **This is normal**

   -  for parallel learning, should not use full CPU cores since this will cause poor performance for the network

-  ``device``, default= ``cpu``, options= ``cpu``, ``gpu``

   -  choose device for the tree learning, you can use GPU to achieve the faster learning

   -  **Note**: it is recommended to use the smaller ``max_bin`` (e.g. 63) to get the better speed up

   -  **Note**: for the faster speed, GPU use 32-bit float point to sum up by default, may affect the accuracy for some tasks.
      You can set ``gpu_use_dp=true`` to enable 64-bit float point, but it will slow down the training

   -  **Note**: refer to `Installation Guide ` to build with GPU

Learning Control Parameters
---------------------------

-  ``max_depth``, default= ``-1``, type=int

   -  limit the max depth for tree model. This is used to deal with over-fitting when ``#data`` is small. Tree still grows by leaf-wise

   -  ``< 0`` means no limit

-  ``min_data_in_leaf``, default= ``20``, type=int, alias= ``min_data_per_leaf`` , ``min_data``, ``min_child_samples``

   -  minimal number of data in one leaf. Can be used to deal with over-fitting

-  ``min_sum_hessian_in_leaf``, default= ``1e-3``, type=double,
   alias= ``min_sum_hessian_per_leaf``, ``min_sum_hessian``, ``min_hessian``, ``min_child_weight``

   -  minimal sum hessian in one leaf. Like ``min_data_in_leaf``, it can be used to deal with over-fitting

-  ``feature_fraction``, default= ``1.0``, type=double, ``0.0 < feature_fraction < 1.0``, alias= ``sub_feature``, ``colsample_bytree``

   -  LightGBM will randomly select part of features on each iteration if ``feature_fraction`` smaller than ``1.0``.
      For example, if set to ``0.8``, will select 80% features before training each tree

   -  can be used to speed up training

   -  can be used to deal with over-fitting

-  ``feature_fraction_seed``, default= ``2``, type=int

   -  random seed for ``feature_fraction``

-  ``bagging_fraction``, default= ``1.0``, type=double, ``0.0 < bagging_fraction < 1.0``, alias= ``sub_row``, ``subsample``

   -  like ``feature_fraction``, but this will randomly select part of data without resampling

   -  can be used to speed up training

   -  can be used to deal with over-fitting

   -  **Note**: To enable bagging, ``bagging_freq`` should be set to a non zero value as well

-  ``bagging_freq``, default= ``0``, type=int, alias= ``subsample_freq``

   -  frequency for bagging, ``0`` means disable bagging. ``k`` means will perform bagging at every ``k`` iteration

   -  **Note**: to enable bagging, ``bagging_fraction`` should be set as well

-  ``bagging_seed`` , default= ``3``, type=int, alias= ``bagging_fraction_seed``

   -  random seed for bagging

-  ``early_stopping_round``, default= ``0``, type=int, alias= ``early_stopping_rounds``, ``early_stopping``

   -  will stop training if one metric of one validation data doesn't improve in last ``early_stopping_round`` rounds

-  ``lambda_l1``, default= ``0``, type=double, alias= ``reg_alpha``

   -  L1 regularization

-  ``lambda_l2``, default= ``0``, type=double, alias= ``reg_lambda``

   -  L2 regularization

-  ``min_split_gain``, default= ``0``, type=double, alias= ``min_gain_to_split``

   -  the minimal gain to perform split

-  ``drop_rate``, default= ``0.1``, type=double

   -  only used in ``dart``

-  ``skip_drop``, default= ``0.5``, type=double

   -  only used in ``dart``, probability of skipping drop

-  ``max_drop``, default= ``50``, type=int

   -  only used in ``dart``, max number of dropped trees on one iteration

   -  ``<=0`` means no limit

-  ``uniform_drop``, default= ``false``, type=bool

   -  only used in ``dart``, set this to ``true`` if want to use uniform drop

-  ``xgboost_dart_mode``, default= ``false``, type=bool

   -  only used in ``dart``, set this to ``true`` if want to use xgboost dart mode

-  ``drop_seed``, default= ``4``, type=int

   -  only used in ``dart``, random seed to choose dropping models

-  ``top_rate``, default= ``0.2``, type=double

   -  only used in ``goss``, the retain ratio of large gradient data

-  ``other_rate``, default= ``0.1``, type=int

   -  only used in ``goss``, the retain ratio of small gradient data

-  ``min_data_per_group``, default= ``100``, type=int

   -  min number of data per categorical group

-  ``max_cat_threshold``, default= ``32``, type=int

   -  use for the categorical features

   -  limit the max threshold points in categorical features

-  ``cat_smooth``, default= ``10``, type=double

   -  used for the categorical features

   -  this can reduce the effect of noises in categorical features, especially for categories with few data

-  ``cat_l2``, default= ``10``, type=double

   -  L2 regularization in categorcial split

-  ``max_cat_to_onehot``, default= ``4``, type=int

   -  when number of categories of one feature smaller than or equal to ``max_cat_to_onehot``, one-vs-other split algorithm will be used

-  ``top_k``, default= ``20``, type=int, alias= ``topk``

   -  used in `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__

   -  set this to larger value for more accurate result, but it will slow down the training speed

IO Parameters
-------------

-  ``max_bin``, default= ``255``, type=int

   -  max number of bins that feature values will be bucketed in.
      Small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)

   -  LightGBM will auto compress memory according ``max_bin``.
      For example, LightGBM will use ``uint8_t`` for feature value if ``max_bin=255``

-  ``min_data_in_bin``, default= ``3``, type=int

   -  min number of data inside one bin, use this to avoid one-data-one-bin (may over-fitting)

-  ``data_random_seed``, default= ``1``, type=int

   -  random seed for data partition in parallel learning (not include feature parallel)

-  ``output_model``, default= ``LightGBM_model.txt``, type=string, alias= ``model_output``, ``model_out``

   -  file name of output model in training

-  ``input_model``, default= ``""``, type=string, alias= ``model_input``, ``model_in``

   -  file name of input model

   -  for ``prediction`` task, this model will be used for prediction data

   -  for ``train`` task, training will be continued from this model

-  ``output_result``, default= ``LightGBM_predict_result.txt``,
   type=string, alias= ``predict_result``, ``prediction_result``

   -  file name of prediction result in ``prediction`` task

-  ``pre_partition``, default= ``false``, type=bool, alias= ``is_pre_partition``

   -  used for parallel learning (not include feature parallel)

   -  ``true`` if training data are pre-partitioned, and different machines use different partitions

-  ``is_sparse``, default= ``true``, type=bool, alias= ``is_enable_sparse``, ``enable_sparse``

   -  used to enable/disable sparse optimization. Set to ``false`` to disable sparse optimization

-  ``two_round``, default= ``false``, type=bool, alias= ``two_round_loading``, ``use_two_round_loading``

   -  by default, LightGBM will map data file to memory and load features from memory.
      This will provide faster data loading speed. But it may run out of memory when the data file is very big

   -  set this to ``true`` if data file is too big to fit in memory

-  ``save_binary``, default= ``false``, type=bool, alias= ``is_save_binary``, ``is_save_binary_file``

   -  if ``true`` LightGBM will save the dataset (include validation data) to a binary file.
      Speed up the data loading for the next time

-  ``verbosity``, default= ``1``, type=int, alias= ``verbose``

   -  ``<0`` = Fatal,
      ``=0`` = Error (Warn),
      ``>0`` = Info

-  ``header``, default= ``false``, type=bool, alias= ``has_header``

   -  set this to ``true`` if input data has header

-  ``label``, default= ``""``, type=string, alias= ``label_column``

   -  specify the label column

   -  use number for index, e.g. ``label=0`` means column_0 is the label

   -  add a prefix ``name:`` for column name, e.g. ``label=name:is_click``

-  ``weight``, default= ``""``, type=string, alias= ``weight_column``

   -  specify the weight column

   -  use number for index, e.g. ``weight=0`` means column_0 is the weight

   -  add a prefix ``name:`` for column name, e.g. ``weight=name:weight``

   -  **Note**: index starts from ``0``.
      And it doesn't count the label column when passing type is Index, e.g. when label is column_0, and weight is column_1, the correct parameter is ``weight=0``

-  ``query``, default= ``""``, type=string, alias= ``query_column``, ``group``, ``group_column``

   -  specify the query/group id column

   -  use number for index, e.g. ``query=0`` means column_0 is the query id

   -  add a prefix ``name:`` for column name, e.g. ``query=name:query_id``

   -  **Note**: data should be grouped by query_id.
      Index starts from ``0``.
      And it doesn't count the label column when passing type is Index, e.g. when label is column_0 and query_id is column_1, the correct parameter is ``query=0``

-  ``ignore_column``, default= ``""``, type=string, alias= ``ignore_feature``, ``blacklist``

   -  specify some ignoring columns in training

   -  use number for index, e.g. ``ignore_column=0,1,2`` means column_0, column_1 and column_2 will be ignored

   -  add a prefix ``name:`` for column name, e.g. ``ignore_column=name:c1,c2,c3`` means c1, c2 and c3 will be ignored

   -  **Note**: works only in case of loading data directly from file

   -  **Note**: index starts from ``0``. And it doesn't count the label column

-  ``categorical_feature``, default= ``""``, type=string, alias= ``categorical_column``, ``cat_feature``, ``cat_column``

   -  specify categorical features

   -  use number for index, e.g. ``categorical_feature=0,1,2`` means column_0, column_1 and column_2 are categorical features

   -  add a prefix ``name:`` for column name, e.g. ``categorical_feature=name:c1,c2,c3`` means c1, c2 and c3 are categorical features

   -  **Note**: only supports categorical with ``int`` type. Index starts from ``0``. And it doesn't count the label column

   -  **Note**: the negative values will be treated as **missing values**

-  ``predict_raw_score``, default= ``false``, type=bool, alias= ``raw_score``, ``is_predict_raw_score``

   -  only used in ``prediction`` task

   -  set to ``true`` to predict only the raw scores

   -  set to ``false`` to predict transformed scores

-  ``predict_leaf_index``, default= ``false``, type=bool, alias= ``leaf_index``, ``is_predict_leaf_index``

   -  only used in ``prediction`` task

   -  set to ``true`` to predict with leaf index of all trees

-  ``predict_contrib``, default= ``false``, type=bool, alias= ``contrib``, ``is_predict_contrib``

   -  only used in ``prediction`` task

   -  set to ``true`` to estimate `SHAP values`_, which represent how each feature contributs to each prediction.
      Produces number of features + 1 values where the last value is the expected value of the model output over the training data

-  ``bin_construct_sample_cnt``, default= ``200000``, type=int, alias= ``subsample_for_bin``

   -  number of data that sampled to construct histogram bins

   -  will give better training result when set this larger, but will increase data loading time

   -  set this to larger value if data is very sparse

-  ``num_iteration_predict``, default= ``-1``, type=int

   -  only used in ``prediction`` task
   -  use to specify how many trained iterations will be used in prediction

   -  ``<= 0`` means no limit

-  ``pred_early_stop``, default= ``false``, type=bool

   -  if ``true`` will use early-stopping to speed up the prediction. May affect the accuracy

-  ``pred_early_stop_freq``, default= ``10``, type=int

   -  the frequency of checking early-stopping prediction

-  ``pred_early_stop_margin``, default= ``10.0``, type=double

   -  the threshold of margin in early-stopping prediction

-  ``use_missing``, default= ``true``, type=bool

   -  set to ``false`` to disable the special handle of missing value

-  ``zero_as_missing``, default= ``false``, type=bool

   -  set to ``true`` to treat all zero as missing values (including the unshown values in libsvm/sparse matrics)

   -  set to ``false`` to use ``na`` to represent missing values

-  ``init_score_file``, default= ``""``, type=string

   -  path to training initial score file, ``""`` will use ``train_data_file`` + ``.init`` (if exists)

-  ``valid_init_score_file``, default= ``""``, type=multi-string

   -  path to validation initial score file, ``""`` will use ``valid_data_file`` + ``.init`` (if exists)

   -  separate by ``,`` for multi-validation data

Objective Parameters
--------------------

-  ``sigmoid``, default= ``1.0``, type=double

   -  parameter for sigmoid function. Will be used in ``binary`` classification and ``lambdarank``

-  ``alpha``, default= ``0.9``, type=double

   -  parameter for `Huber loss`_ and `Quantile regression`_. Will be used in ``regression`` task

-  ``fair_c``, default= ``1.0``, type=double

   -  parameter for `Fair loss`_. Will be used in ``regression`` task

-  ``poisson_max_delta_step``, default= ``0.7``, type=double

   -  parameter for `Poisson regression`_ to safeguard optimization

-  ``scale_pos_weight``, default= ``1.0``, type=double

   -  weight of positive class in ``binary`` classification task

-  ``boost_from_average``, default= ``true``, type=bool

   -  only used in ``regression`` task

   -  adjust initial score to the mean of labels for faster convergence

-  ``is_unbalance``, default= ``false``, type=bool, alias= ``unbalanced_sets``

   -  used in ``binary`` classification

   -  set this to ``true`` if training data are unbalance

-  ``max_position``, default= ``20``, type=int

   -  used in ``lambdarank``

   -  will optimize `NDCG`_ at this position

-  ``label_gain``, default= ``0,1,3,7,15,31,63,...,2^30-1``, type=multi-double

   -  used in ``lambdarank``

   -  relevant gain for labels. For example, the gain of label ``2`` is ``3`` if using default label gains

   -  separate by ``,``

-  ``num_class``, default= ``1``, type=int, alias= ``num_classes``

   -  only used in multi-class classification

-  ``reg_sqrt``, default= ``false``, type=bool

   -  only used in ``regression``

   -  will fit ``sqrt(label)`` instead and prediction result will be also automatically converted to ``pow2(prediction)``

Metric Parameters
-----------------

-  ``metric``, default={``l2`` for regression, ``binary_logloss`` for binary classification, ``ndcg`` for lambdarank}, type=multi-enum

   -  ``l1``, absolute loss, alias= ``mean_absolute_error``, ``mae``

   -  ``l2``, square loss, alias= ``mean_squared_error``, ``mse``

   -  ``l2_root``, root square loss, alias= ``root_mean_squared_error``, ``rmse``

   -  ``quantile``, `Quantile regression`_

   -  ``mape``, `MAPE loss`_, alias= ``mean_absolute_percentage_error``

   -  ``huber``, `Huber loss`_

   -  ``fair``, `Fair loss`_

   -  ``poisson``, `Poisson regression`_

   -  ``ndcg``, `NDCG`_

   -  ``map``, `MAP`_, alias= ``mean_average_precision``

   -  ``auc``, `AUC`_

   -  ``binary_logloss``, `log loss`_

   -  ``binary_error``, for one sample: ``0`` for correct classification, ``1`` for error classification

   -  ``multi_logloss``, log loss for mulit-class classification

   -  ``multi_error``, error rate for mulit-class classification

   -  ``xentropy``, cross-entropy (with optional linear weights), alias= ``cross_entropy``

   -  ``xentlambda``, "intensity-weighted" cross-entropy, alias= ``cross_entropy_lambda``

   -  ``kldiv``, `Kullback-Leibler divergence`_, alias= ``kullback_leibler``

   -  support multi metrics, separated by ``,``

-  ``metric_freq``, default= ``1``, type=int, alias= ``output_freq``

   -  frequency for metric output

-  ``train_metric``, default= ``false``, type=bool, alias= ``training_metric``, ``is_training_metric``

   -  set this to ``true`` if you need to output metric result of training

-  ``ndcg_at``, default= ``1,2,3,4,5``, type=multi-int, alias= ``ndcg_eval_at``, ``eval_at``

   -  `NDCG`_ evaluation positions, separated by ``,``

Network Parameters
------------------

Following parameters are used for parallel learning, and only used for base (socket) version.

-  ``num_machines``, default= ``1``, type=int, alias= ``num_machine``

   -  used for parallel learning, the number of machines for parallel learning application

   -  need to set this in both socket and mpi versions

-  ``local_listen_port``, default= ``12400``, type=int, alias= ``local_port``

   -  TCP listen port for local machines

   -  you should allow this port in firewall settings before training

-  ``time_out``, default= ``120``, type=int

   -  socket time-out in minutes

-  ``machine_list_file``, default= ``""``, type=string, alias= ``mlist``

   -  file that lists machines for this parallel learning application

   -  each line contains one IP and one port for one machine. The format is ``ip port``, separate by space

GPU Parameters
--------------

-  ``gpu_platform_id``, default= ``-1``, type=int

   -  OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform.

   -  default value is ``-1``, means the system-wide default platform

-  ``gpu_device_id``, default= ``-1``, type=int

   -  OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID

   -  default value is ``-1``, means the default device in the selected platform

-  ``gpu_use_dp``, default= ``false``, type=bool

   -  set to ``true`` to use double precision math on GPU (default using single precision)

Convert Model Parameters
------------------------

This feature is only supported in command line version yet.

-  ``convert_model_language``, default= ``""``, type=string

   -  only ``cpp`` is supported yet

   -  if ``convert_model_language`` is set when ``task`` is set to ``train``, the model will also be converted

-  ``convert_model``, default= ``"gbdt_prediction.cpp"``, type=string

   -  output file name of converted model

Others
------

Continued Training with Input Score


LightGBM supports continued training with initial scores. It uses an additional file to store these initial scores, like the following:

::

​    0.5
​    -0.1
​    0.9
​    ...

It means the initial score of the first data row is ``0.5``, second is ``-0.1``, and so on.
The initial score file corresponds with data file line by line, and has per score per line.
And if the name of data file is ``train.txt``, the initial score file should be named as ``train.txt.init`` and in the same folder as the data file.
In this case LightGBM will auto load initial score file if it exists.

Weight Data


LightGBM supporta weighted training. It uses an additional file to store weight data, like the following:

::

​    1.0
​    0.5
​    0.8
​    ...

It means the weight of the first data row is ``1.0``, second is ``0.5``, and so on.
The weight file corresponds with data file line by line, and has per weight per line.
And if the name of data file is ``train.txt``, the weight file should be named as ``train.txt.weight`` and in the same folder as the data file.
In this case LightGBM will auto load weight file if it exists.

**update**:
You can specific weight column in data file now. Please refer to parameter ``weight`` in above.

Query Data


For LambdaRank learning, it needs query information for training data.
LightGBM use an additional file to store query data, like the following:

::

​    27
​    18
​    67
​    ...

It means first ``27`` lines samples belong one query and next ``18`` lines belong to another, and so on.

**Note**: data should be ordered by the query.

If the name of data file is ``train.txt``, the query file should be named as ``train.txt.query`` and in same folder of training data.
In this case LightGBM will load the query file automatically if it exists.

**update**:
You can specific query/group id in data file now. Please refer to parameter ``group`` in above.



Laurae++ Interactive Documentation: [https://sites.google.com/view/lauraepp/parameters](https://sites.google.com/view/lauraepp/parameters)

Huber loss: [https://en.wikipedia.org/wiki/Huber_loss](https://en.wikipedia.org/wiki/Huber_loss)

Quantile regression: [https://en.wikipedia.org/wiki/Quantile_regression](https://en.wikipedia.org/wiki/Quantile_regression)

MAPE loss: [https://en.wikipedia.org/wiki/Mean_absolute_percentage_error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)

Fair loss: [https://www.kaggle.com/c/allstate-claims-severity/discussion/24520](https://www.kaggle.com/c/allstate-claims-severity/discussion/24520)

Poisson regression: [https://en.wikipedia.org/wiki/Poisson_regression](https://en.wikipedia.org/wiki/Poisson_regression)

lambdarank: [https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf](https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf)

Dropouts meet Multiple Additive Regression Trees: [https://arxiv.org/abs/1505.01866](https://arxiv.org/abs/1505.01866)

hyper-threading: [https://en.wikipedia.org/wiki/Hyper-threading](https://en.wikipedia.org/wiki/Hyper-threading)

SHAP values: [https://arxiv.org/abs/1706.06060](https://arxiv.org/abs/1706.06060)

NDCG: [https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)

MAP: [https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision](https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision)

AUC: [https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)

log loss: [https://en.wikipedia.org/wiki/Cross_entropy](https://en.wikipedia.org/wiki/Cross_entropy)

softmax: [https://en.wikipedia.org/wiki/Softmax_function](https://en.wikipedia.org/wiki/Softmax_function)

One-vs-All: [https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest)

Kullback-Leibler divergence: [https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

# Python Practice

```python
import numpy as np
import pandas as pd
import lightgbm as lgb


# define dataset
train_data = lgb.Dataset(train, label=train_y, max_bin=255, free_raw_data=True)
test_data = lgb.Dataset(test, label=test_y, reference=train_data, max_bin=255, free_raw_data=True)

"""
reference: If this is Dataset for validation, training data should be used as reference.

free_raw_data: bool, optional (default=True)
If True, raw data is freed after constructing inner Dataset.
"""

# para searching
params = {
    'boosting_type': 'gbdt',  # np.random.choice(['dart', 'gbdt']),
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'], 
    
    'learning_rate': 0.35,
    
    'num_leaves': np.random.randint(64, 128),
    'max_depth': np.random.randint(6, 12),
    'min_data_in_leaf': int(2 ** (np.random.rand()*3.5 + 9)),
    
    'feature_fraction': np.random.rand()*0.35+0.65,
    'bagging_fraction': np.random.rand()*0.35+0.65,
    'bagging_freq': 1,
    
    'lambda_l1': 10 ** (np.random.rand() * 4),
    'lambda_l2': 10 ** (np.random.rand() * 3 + 2),
    'min_gain_to_split': 0.0,
    'min_sum_hessian_in_leaf': 0.1,
    
    'num_threads': 16,
    'verbose': 0,
    'is_training_metric': 'True'
}
print('Hyper-parameters:')
print(params)
  
evals_result = {}  # dict to store evaluation results of all the items in `valid_sets`

gbm = lgb.train(params, train_data, 5000, valid_sets=[train_data, test_data], \
                valid_names = ['train', 'valid'], evals_result=evals_result, \
                early_stopping_rounds=250, verbose_eval=100, 
                categorical_feature='auto')
"""
categorical_feature : list of strings or int, or 'auto', optional (default="auto")
Categorical features.
If list of int, interpreted as indices.
If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
If 'auto' and data is pandas DataFrame, pandas categorical columns are used.
"""
  
# save best round results to csv
bst_round = np.argmax(evals_result['valid']['auc'])
trn_auc = evals_result['train']['auc'][bst_round]
trn_loss = evals_result['train']['binary_logloss'][bst_round]
val_auc = evals_result['valid']['auc'][bst_round]
val_loss = evals_result['valid']['binary_logloss'][bst_round]

print('Best Round: %d'%bst_round)
print('Training loss: %.5f, Validation loss: %.5f'%(trn_loss, val_loss))
print('Training AUC : %.5f, Validation AUC : %.5f'%(trn_auc, val_auc))
  
feature_importance = pd.DataFrame({'name':gbm.feature_name(), 'importance':gbm.feature_importance()}).sort_values(by='importance', ascending=False)
feature_importance.to_csv('./feat_importance.csv', index=False)
    
res = '%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%d,%.5f,%.5f,%.5f,%.5f\n' % \
(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
 'LightGBM_baseline_song_context_prob', 
feat_cnt, 
params['boosting_type'],params['learning_rate'],
params['num_leaves'],params['max_depth'],params['min_data_in_leaf'], params['feature_fraction'], params['bagging_fraction'], params['bagging_freq'], 
params['lambda_l1'], params['lambda_l2'], params['min_gain_to_split'],
params['min_sum_hessian_in_leaf'], 
0.0, # sample rate
bst_round+1, trn_loss, trn_auc, val_loss, val_auc)

f = open('./lgb_record.csv', 'a')
f.write(res)
f.close()
```



</br>

| time             | model                               | feat_cnt | type | lr   | n_leaf | n_depth | min_data | feature_frac | bagging_frac | bagging_freq | l1   | l2    | min_gain | hessian | sample_rate | bst_rnd | trn_loss | trn_auc | val_loss | val_auc |
| ---------------- | ----------------------------------- | -------- | ---- | ---- | ------ | ------- | -------- | ------------ | ------------ | ------------ | ---- | ----- | -------- | ------- | ----------- | ------- | -------- | ------- | -------- | ------- |
| 2017/12/12 22:20 | LightGBM_baseline_new               | 385      | gbdt | 0.1  | 99     | 10      | 1306     | 0.6866       | 0.9054       | 1            | 6.37 | 65200 | 0        | 0.1     | 0           | 7732    | 0.46055  | 0.86678 | 0.58584  | 0.73963 |
| 2017/12/15 10:42 | LightGBM_baseline_new               | 385      | dart | 0.5  | 99     | 10      | 1306     | 0.6866       | 0.9054       | 1            | 6.37 | 65200 | 0        | 0.1     | 0           | 4998    | 0.47348  | 0.85819 | 0.58506  | 0.7386  |
| 2017/12/17 12:02 | LightGBM_baseline_song_context_prob | 390      | gbdt | 0.5  | 99     | 10      | 1306     | 0.6866       | 0.9054       | 1            | 6.37 | 65200 | 0        | 0.1     | 0           | 1302    | 0.46954  | 0.85925 | 0.59024  | 0.73556 |
| 2017/12/12 11:49 | LightGBM_baseline_new               | 385      | gbdt | 0.5  | 99     | 10      | 1306     | 0.6866       | 0.9054       | 1            | 6.37 | 65200 | 0        | 0.1     | 0           | 1304    | 0.4704   | 0.8586  | 0.59103  | 0.73458 |