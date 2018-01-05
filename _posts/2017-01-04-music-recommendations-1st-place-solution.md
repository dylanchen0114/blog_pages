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

Collect song_id_set in both train and test data, tag song id as True in "songs.csv" and "song_extra_info" if it exists in song_id_set. And the similar process to msno (user_id), tag msno as True in "member.csv" if it appeared in train or test data.

Label encoding to categorical id columns, **fitted by train + test**, transform and replace the original ones.

Count how many artists, lyricist and composer each song has, **get first item of each**, split multi-lyricist to corresponding columns and label encoding on this columns.

### cnt_log_process

1. how many songs each user **(train + test)** /artist/composer/lyricist/genre_id **("song.csv")** has
2. how many users each song/artist/composer/lyricist/genre_id has **(train + test)**.
3. the prob that each user will re-listening in terms of 'source_system_tab', 'source_screen_name' or 'source_type'
4. Conditional probability features: for categorical features, P(source_type|msno), P(source_screen_name|msno), P(source_system_tab|msno), P(source_type|song_id), P(source_screen_name|song_id), P(source_system_tab|song_id), P(artist_name|msno), P(language|msno), P(first_genre_id|msno) and so on. e.g, P(source_type|msno), given the user_id, calculate the probability that each source_type will occur





