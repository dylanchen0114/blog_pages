---
title: "KKBos-Music-Recommendations-1st-place-solution"
comments: true
share: true
toc: true
categories:
  - Kaggle
tags:
  - Kaggel
---

## Preprocessing

### id_process

1. collect song_id_set in both train and test data, tag song id as True in "songs.csv" and "song_extra_info" if it exists in song_id_set. And the similar process to msno (user_id), tag msno as True in "member.csv" if it appeared in train or test data.
2. label encoding to categorical id columns, **fitted by train + test**, transform and replace the original ones.
3. count how many artists, lyricist and composer each song has, **get first item of each**, split multi-lyricist to corresponding columns and label encoding on this columns.

### cnt_log_process

1. how many songs each user **(train + test)** /artist/composer/lyricist/genre_id **("song.csv")** has.  Log1p transform
2. how many users each song/artist/composer/lyricist/genre_id has **(train + test)**. Log1p transform
3. Conditional probability features: for categorical features, P(source_type given msno), P(source_type given song_id) and so on. e.g, P(source_type given msno), given the user_id, calculate the probability that each source_type will occur.

### isrc_process

1. split [International Standard Recording Code](https://en.wikipedia.org/wiki/International_Standard_Recording_Code) to cc, xxx and yy (categorical features)
2. count how many users and songs this split codes have.  Log1p transform

### svd_process **

1. create sparse coo_matrix for user-song pairs, the following is the basic ways

   in this practice, user array as row and song array as column, the data is np.ones(len(user-song pair))

   ```python
   >>> from scipy.sparse import coo_matrix
   >>> coo_matrix((3, 4), dtype=np.int8).toarray()
   array([[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]], dtype=int8)

   >>> row  = np.array([0, 3, 1, 0])
   >>> col  = np.array([0, 3, 1, 2])
   >>> data = np.array([4, 5, 7, 9])
   >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
   array([[4, 0, 9, 0],
          [0, 7, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 5]])

   >>> # example with duplicates
   >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
   >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
   >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
   >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
   array([[3, 0, 1, 0],
          [0, 2, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1]])

   """
   practice
   """

   concat = tr[['msno', 'song_id']].append(te[['msno', 'song_id']])

   print(len(concat))

   data = np.ones(len(concat))

   msno = concat['msno'].values

   song_id = concat['song_id'].values

   rating = sparse.coo_matrix((data, (msno, song_id)))

   rating = (rating > 0) * 1.0  # drop duplicates

   ```

2. fit svds with the above sparse matrix and specific n_componets

   ```python

   [u, s, vt] = svds(rating, k=n_component)

   """
   output
   u: row component, length: n_unique row contents
   vt: column component, length: n_unique column contents
   s: sigma, singular value 
   """
   ```


3. using u, vt as features, and adding to the data-frame

   ```python
   members_topics = pd.DataFrame(u[:, ::-1])
   members_topics.columns = ['member_component_%d'%i for i in range(n_component)]
   members_topics['msno'] = range(member_cnt)
   member = member.merge(members_topics, on='msno', how='right')

   song_topics = pd.DataFrame(vt.transpose()[:, ::-1])
   song_topics.columns = ['song_component_%d'%i for i in range(n_component)]
   song_topics['song_id'] = range(song_cnt)
   song = song.merge(song_topics, on='song_id', how='right')
   ```

4. dot embedding features as following

   ```python
   for i in range(len(tr)):
       msno_idx = tr['msno'].values[i]
       song_idx = tr['song_id'].values[i]
       
       train_dot[i, 0] = np.dot(member_embeddings[msno_idx], np.dot(s_song, song_embeddings[song_idx]))
       train_dot[i, 1] = np.dot(member_artist_embeddings[msno_idx], np.dot(s_artist, song_artist_embeddings[song_idx]))

   for i in range(len(te)):
       msno_idx = te['msno'].values[i]
       song_idx = te['song_id'].values[i]
       
       test_dot[i, 0] = np.dot(member_embeddings[msno_idx], np.dot(s_song, song_embeddings[song_idx]))
       test_dot[i, 1] = np.dot(member_artist_embeddings[msno_idx], np.dot(s_artist, song_artist_embeddings[song_idx]))

   tr['song_embeddings_dot'] = train_dot[:, 0]
   tr['artist_embeddings_dot'] = train_dot[:, 1]

   te['song_embeddings_dot'] = test_dot[:, 0]
   te['artist_embeddings_dot'] = test_dot[:, 1]
   ```


### time_process

since the data is time-sensitive, the row number index can be treated as time-stamp.

1. calculate how many times the song_id/user_id occurs before or after; define a window size [10, 25, 500, 5000, 10000, 50000]
2. until now, how many times the song_id/user_id has occurred
3. the mean, std of time-stamp by user_id or song_id

### before_after_process

1. the value of ['song_id', 'source_type', 'source_screen_name', 'timestamp'] before this time
2. the value of ['song_id', 'source_type', 'source_screen_name', 'timestamp'] after this time
3. diff between this time and last time
4. diff between this time and next time