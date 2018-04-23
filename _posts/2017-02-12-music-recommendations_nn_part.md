---
title: "Music-Recommendations_NN_Part"
comments: true
mathjax: true
share: true
toc: true
---

### FEATURE SUMMARY 

1. embedding features: Raw categorical features, including high cardinality ones

   ```python
   embedding_features = [
   'msno', 'city', 'gender', 'registered_via',
   'artist_name', 'language', 'cc',
   'source_type', 'source_screen_name', 'source_system_tab',
   'before_source_type', 'after_source_type', 'before_source_screen_name',
   'after_source_screen_name', 'before_language', 'after_language',
   # 'song_id', 'before_song_id', 'after_song_id'
   ]
   ```
   * create Keras input for embedding features via Input

     ```python
     tmp_input = Input(shape=(1,), dtype='int32', name=embedding_features[i]+'_input')
     ```

   * create embedding layers for the above input layer, define input_dim, output_dim, embeddings_initializer, embeddings_regularizer and input_length

     - input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
     - output_dim：大于0的整数，代表全连接嵌入的维度
     - embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](https://keras-cn.readthedocs.io/en/latest/other/initializations)
     - embeddings_regularizer: 嵌入矩阵的正则项，为[Regularizer](https://keras-cn.readthedocs.io/en/latest/other/regularizers)对象
     - embeddings_constraint: 嵌入矩阵的约束项，为[Constraints](https://keras-cn.readthedocs.io/en/latest/other/constraints)对象
     - mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用[递归层](https://keras-cn.readthedocs.io/en/latest/layers/recurrent_layer)处理变长输入时有用。设置为`True`的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 1。
     - input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接`Flatten`层，然后接`Dense`层，则必须指定该参数，否则`Dense`层的输出维度无法自动推断。

     ```python
     tmp_embeddings = Embedding(int(train_embeddings[i].max()+1),
                                K if i == 0 else K0,  # output_dim
                                embeddings_initializer=
                                RandomUniform(minval=-val_bound, maxval=val_bound),
                                embeddings_regularizer=l2(lw),
                                input_length=1,
                                trainable=True,
                                name=embedding_features[i]+'_embeddings')(tmp_input)
     ```

   * create Flatten layer on above embedding layer 

     ```python
     tmp_embeddings = Flatten(name=embedding_features[i]+'_flatten')(tmp_embeddings)
     ```

   * All the embedding input layer and embedding output layer then be collected by list

2. input layer for song_id related features

   * create Keras input for embedding features via Input

     ```python
     song_id_input = Input(shape=(1,), dtype='int32', name='song_id_input')
     before_song_id_input = Input(shape=(1,), dtype='int32', name='before_song_id_input')
     after_song_id_input = Input(shape=(1,), dtype='int32', name='after_song_id_input')
     ```

   * the song_id input layer also collected by the 1st embedding input list

3. the same embedding process as 1st for the song category features

   ```python
   genre_inputs = []
   genre_outputs = []
   genre_embeddings = Embedding(int(np.max(train_genre)+1), K0,
                                embeddings_initializer=
                                RandomUniform(minval=-0.05,maxval=0.05),
                                embeddings_regularizer=l2(lw),
                                input_length=1,
                                trainable=True,
                                name='genre_embeddings')
   for i in range(len(genre_features)):
       tmp_input = Input(shape=(1,), dtype='int32', name=genre_features[i]+'_input')
       tmp_embeddings = genre_embeddings(tmp_input)
       tmp_embeddings = Flatten(name=genre_features[i]+'_flatten')(tmp_embeddings)

   genre_inputs.append(tmp_input)
   genre_outputs.append(tmp_embeddings)
   ```

4. create numerical user features based on user embedding input layer

   ```python
   usr_features = ['bd', 'expiration_date', 'msno_rec_cnt', 'msno_source_screen_name_0', \
           'msno_source_screen_name_1', 'msno_source_screen_name_10', 'msno_source_screen_name_11', \
           'msno_source_screen_name_12', 'msno_source_screen_name_13', 'msno_source_screen_name_14', \
           'msno_source_screen_name_17', \
           'msno_source_screen_name_18', 'msno_source_screen_name_19', 'msno_source_screen_name_2', \
           'msno_source_screen_name_20', 'msno_source_screen_name_21', 'msno_source_screen_name_3', 'msno_source_screen_name_4', \
           'msno_source_screen_name_5', 'msno_source_screen_name_6', 'msno_source_screen_name_7', \
           'msno_source_screen_name_8', 'msno_source_screen_name_9', 'msno_source_system_tab_0', \
           'msno_source_system_tab_1', 'msno_source_system_tab_2', 'msno_source_system_tab_3', \
           'msno_source_system_tab_4', 'msno_source_system_tab_5', 'msno_source_system_tab_6', \
           'msno_source_system_tab_7', 'msno_source_system_tab_8', \
           'msno_source_type_0', 'msno_source_type_1', 'msno_source_type_10', \
           'msno_source_type_11', 'msno_source_type_2', \
           'msno_source_type_3', 'msno_source_type_4', 'msno_source_type_5', \
           'msno_source_type_6', \
           'msno_source_type_7', 'msno_source_type_8', 'msno_source_type_9', \
           'msno_timestamp_mean', 'msno_timestamp_std', 'registration_init_time', \
           'msno_song_length_mean', 'msno_artist_song_cnt_mean', 'msno_artist_rec_cnt_mean', \
           'msno_song_rec_cnt_mean', 'msno_yy_mean', 'msno_song_length_std', \
           'msno_artist_song_cnt_std', 'msno_artist_rec_cnt_std', 'msno_song_rec_cnt_std', \
           'msno_yy_std', 'artist_msno_cnt']
   for col in ['msno_song_length_mean', 'msno_artist_song_cnt_mean', 
               'msno_artist_rec_cnt_mean', 'msno_song_rec_cnt_mean', 
               'msno_yy_mean', 'msno_song_length_std', 'msno_artist_song_cnt_std', 
               'msno_artist_rec_cnt_std', 'msno_song_rec_cnt_std', 'msno_yy_std', 'artist_msno_cnt']:
       usr_features.remove(col)
   ```

   ​

   ```python
   # user feature array itself as intial value weights
   usr_input = Embedding(usr_feat.shape[0],
                         usr_feat.shape[1],
                         weights=[usr_feat],
                         input_length=1,
                         trainable=False,
                         name='usr_feat')(embedding_inputs[0])
   usr_input = Flatten(name='usr_feat_flatten')(usr_input)
   ```

    

5. create numerical song features layer as the above embedding process, based on song_id input layer

6. user_component, song_component, user_artist_component and song_artist_component as above

7. create context numerical features

   ```python
   context_features = ['after_artist_same', 'after_song_rec_cnt', 'after_timestamp', \
           'after_type_same', 'before_artist_same', 'before_song_rec_cnt', \
           'before_timestamp', 'before_type_same', 'msno_10000_after_cnt', \
           'msno_10000_before_cnt', 'msno_10_after_cnt', 'msno_10_before_cnt', \
           'msno_25_after_cnt', 'msno_25_before_cnt', 'msno_50000_after_cnt', \
           'msno_50000_before_cnt', 'msno_5000_after_cnt', 'msno_5000_before_cnt', \
           'msno_500_after_cnt', 'msno_500_before_cnt', 'msno_source_screen_name_prob', \
           'msno_source_system_tab_prob', 'msno_source_type_prob', 'msno_till_now_cnt', \
           'registration_init_time', 'song_50000_after_cnt', 'song_50000_before_cnt', \
           'song_till_now_cnt', 'timestamp', 'msno_artist_name_prob', 'msno_first_genre_id_prob', \
           'msno_xxx_prob', 'msno_language_prob', 'msno_yy_prob', 'msno_source_prob', \
           'song_source_system_tab_prob', 'song_source_screen_name_prob', 'song_source_type_prob']
   ```

   ​

   ```python
   context_input = Input(shape=(len(context_features),), name='context_feat')
   ```



### FIELD GROUP 

All features embedding input and output layer include:

embedding_inputs: 'msno', 'city', 'gender', 'registered_via', 'artist_name', 'language', 'cc', 'source_type', 'source_screen_name', 'source_system_tab', 'before_source_type', 'after_source_type', 'before_source_screen_name', 'after_source_screen_name', 'before_language', 'after_language', **'song_id_input', 'before_song_id_input', 'after_song_id_input'**

embedding_outputs: 'msno', 'city', 'gender', 'registered_via', 'artist_name', 'language', 'cc', 'source_type', 'source_screen_name', 'source_system_tab', 'before_source_type', 'after_source_type', 'before_source_screen_name', 'after_source_screen_name', 'before_language', 'after_language'

genre_inputs: 'first_genre_id', 'second_genre_id'

genre_outputs: 'first_genre_id', 'second_genre_id'

user_input: user_related_features, numerical features based on user embedding input

song_input: song_related_features, numerical features based on song embedding input

user_component_input: user_svds_features, numerical features based on user embedding input

song_component_input: song_svds_features, numerical features based on song embedding input

user_artist_component_input: user_artist_svds_features, numerical features based on user embedding input

song_artist_component_input: song_artist_svds_features, numerical features based on song embedding input

context_input: context_features



#### Into Fields

user_profile: embedding_outputs[1:4] + [user_input, user_component_input, user_artist_component_input]

song_profile: embedding_outputs[4:7] + genre_outputs + [song_input, song_component_input, song_artist_component_input]

```python
multiply_component = dot([usr_component_input, song_component_input],
                         axes=1, name='component_dot')

multiply_artist_component = dot([usr_artist_component_input, song_artist_component_input],
                                axes=1, name='artist_component_dot')
```

context_profile: embedding_outputs[7:] + [context_input, multiply_component, multiply_artist_component]

```python

# user field
usr_embeddings = FunctionalDense(K*2, usr_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='usr_profile')
usr_embeddings = Dense(K, name='usr_profile_output')(usr_embeddings)
usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')

# song field
song_embeddings = FunctionalDense(K*2, song_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='song_profile')
song_embeddings = Dense(K, name='song_profile_output')(song_embeddings)
# song_embeddings = add([song_embeddings, embedding_outputs[4]], name='song_embeddings')

# context field
context_embeddings = FunctionalDense(K, context_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='context_profile')

# joint embeddings
joint = dot([usr_embeddings, song_embeddings], axes=1, normalize=False, name='pred_cross')
joint_embeddings = concatenate([usr_embeddings, song_embeddings, context_embeddings, joint], name='joint_embeddings')
```



### TOP MODEL

```python
# top model

# add three fully connected layer
preds0 = FunctionalDense(K*2, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
preds1 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')

preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
preds = Dropout(0.5, name='prediction_dropout')(preds)
preds = Dense(1, activation='sigmoid', name='prediction')(preds)

model = Model(inputs=embedding_inputs+genre_inputs+[context_input], outputs=preds)

opt = RMSprop(lr=lr)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
```



![foo](https://kaggle2.blob.core.windows.net/forum-message-attachments/259372/8131/model.png)

