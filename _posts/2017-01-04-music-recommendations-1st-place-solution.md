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
2. Count how many users and songs this split codes have.  Log1p transform

 





