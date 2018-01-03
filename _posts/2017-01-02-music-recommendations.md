---
title: "KKBos-Music-Recommendations-3rd-place-solution"
comments: true
share: true
toc: true
categories:
  - Kaggle
tags:
  - Kaggel
---

This post describes the 3rd place solution of "WSDM-KKBos-Music-Recommendations"

## Data Description

In this task, you will be asked to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

KKBOX provides a training data set consists of information of the first observable listening event for each unique user-song pair within a specific time duration. Metadata of each unique user and song pair is also provided. The use of public data to increase the level of accuracy of your prediction is encouraged.

The train and the test data are selected from users listening history in a given time period. Note that this time period is chosen to be before the [WSDM-KKBox Churn Prediction](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) time period. The train and test sets are split based on time, and the split of public/private are based on unique user/song pairs.

### train.csv

-  msno: user id
-  song_id: song id
-  source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab `my library` contains functions to manipulate the local storage, and tab `search` contains functions relating to search.
-  source_screen_name: name of the layout a user sees.
-  source_type: an entry point a user first plays music on mobile apps. An entry point could be `album`, `online-playlist`, `song` .. etc.
-  target: this is the target variable. `target=1` means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, `target=0` otherwise .

### test.csv

-  id: row id (will be used for submission)
-  msno: user id
-  song_id: song id
-  source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab `my library` contains functions to manipulate the local storage, and tab `search` contains functions relating to search.
-  source_screen_name: name of the layout a user sees.
-  source_type: an entry point a user first plays music on mobile apps. An entry point could be `album`, `online-playlist`, `song` .. etc.

### sample_submission.csv

sample submission file in the format that we expect you to submit

-  id: same as `id` in `test.csv`
-  target: this is the target variable. `target=1` means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, `target=0` otherwise .

### songs.csv

The songs. Note that data is in unicode.

-  song_id
-  song_length: in ms
-  genre_ids: genre category. Some songs have multiple genres and they are separated by `|`
-  artist_name
-  composer
-  lyricist
-  language

### members.csv

user information.

-  msno
-  city
-  bd: age. Note: this column has outlier values, please use your judgement.
-  gender
-  registered_via: registration method
-  registration_init_time: format `%Y%m%d`
-  expiration_date: format `%Y%m%d`

### song_extra_info.csv

-  song_id
-  song name - the name of the song.
-  isrc - [International Standard Recording Code](https://en.wikipedia.org/wiki/International_Standard_Recording_Code), theoretically can be used as an identity of a song. However, what worth to note is, ISRCs generated from providers have not been officially verified; therefore the information in ISRC, such as country code and reference year, can be misleading/incorrect. Multiple songs could share one ISRC since a single recording could be re-published several times.

## Preprocessing

Simply label encoding on categorical features, note that LabelEncoder cannot process columns with mixed type, so nan must be changed to string or int if exist.

## Creating Features

Splitting train into two parts, where last 35% of train for generation table of features, and the rest of the data was used as a history. 

Note: Diff between df columns and a list can be done below:

```python
not_categorical_columns = [
  
'target', 
'song_length', 
'registration_init_time', 
'expiration_date', 
'time', 
'bd',

]
categorical_columns = all_data.columns.difference(not_categorical_columns)
```

1. target mean encode from history data by all categorical features, including pairs and triples combinations, fill nan as -1

2. count from future by on all categorical features, including pairs and triples combinations, note that the last appearance this value is set to be 0

3. count from past by on all categorical features, including pairs and triples combinations, note that the first appearance this value is set to be 0

4. time diff between current and next heard by all categorical features, including pairs and triples combinations, note that the last appearance of each categorical value is set to be -1

5. time diff between current and last(only history data here) heard by user and (genre, composer, language, artist_name …), note that if this categorical value has not been seen before, set to be -1

6. time diff between current and last(not history data here) heard, by all categorical features, including pairs and triples 

7. Turn date-time columns to ordinal 

   ```python
   for col in ['expiration_date', 'registration_init_time']:
       X[col] = df[col].apply(lambda x: x.toordinal())
   ```

8. the nunique song user heard on this artist / the nunique song of this artist

9. regression

10. matrix_factorization: https://github.com/lyst/lightfm

    ```python
    def matrix_factorization(df, df_history):
        cols = ['msno', 'source_type']
        group = get_group(df, cols)
        group_history = get_group(df_history, cols)

        encoder = LabelEncoder()
        encoder.fit(pd.concat([group, group_history]))

        df['user_id'] = encoder.transform(group)
        df_history['user_id'] = encoder.transform(group_history)

        num_users = max(df.user_id.max(), df_history.user_id.max()) + 1
        num_items = max(df.song_id.max(), df_history.song_id.max()) + 1
        num_msno = max(df.msno.max(), df_history.msno.max()) + 1

        M = coo_matrix(
            (df_history.target, ( df_history.user_id, df_history.song_id)),
            shape=(num_users, num_items)
        )

        user_features = pd.concat([df, df_history])[['msno', 'user_id']].drop_duplicates()

        user_features = coo_matrix(
            (np.ones(len(user_features)), (user_features.user_id, user_features.msno)),
            shape=(num_users, num_msno)
        )

        user_features = sp.hstack([sp.eye(num_users), user_features])

        model = LightFM(no_components=50, learning_rate=0.1)

        model.fit(
            M, 
            epochs=2, 
            num_threads=50, 
            user_features=user_features,
        )
        result = model.predict(
            df.user_id.values, 
            df.song_id.values, 
            user_features=user_features,
        )

        return result
    ```

##Modeling

Blending of xgboost and catboost as following, where "_mf" means with matrix_factorization

```python
p0_xgb_mf = joblib.load('p0_xgb_mf')
p0_xgb = joblib.load('p0_xgb')
p1_xgb_mf = joblib.load('p1_xgb_mf')
p1_xgb = joblib.load('p1_xgb')

p0_cb_mf = joblib.load('p0_cb_mf')
p0_cb = joblib.load('p0_cb')
p1_cb_mf = joblib.load('p1_cb_mf')
p1_cb = joblib.load('p1_cb')


p_cb = 0.6 * p0_cb + 0.4 * p1_cb
p_cb_mf = 0.6 * p0_cb_mf + 0.4 * p1_cb_mf
p_xgb = 0.6 * p0_xgb + 0.4 * p1_xgb
p_xgb_mf = 0.6 * p0_xgb_mf + 0.4 * p1_xgb_mf

p_c = 0.6 * p_cb_mf + 0.4 * p_cb
p_x = 0.6 * p_xgb_mf + 0.4 * p_xgb

p = 0.6 * p_c + 0.4 * p_x
```

