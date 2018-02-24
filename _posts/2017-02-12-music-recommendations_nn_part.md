---
title: "Music-Recommendations_NN_Part"
comments: true
mathjax: true
share: true
toc: true
---

### FEATURE SUMMARY 

1. Embedding features: Raw categorical features, including high cardinality ones

   ```python
   embedding_features = [
   'msno', 'city', 'gender', 'registered_via',
   'artist_name', 'language', 'cc',
   'source_type', 'source_screen_name', 'source_system_tab',
   'before_source_type', 'after_source_type', 'before_source_screen_name',
   'after_source_screen_name', 'before_language', 'after_language',
   'song_id', 'before_song_id', 'after_song_id'
   ]
   ```

2. Genre features

3. Before and after context features

   *  conditional probability features: given msno, calculate source_system_tab, source_screen_name, source, source_type, artist_name, first_genre_id, xxx, language, yy; given song, calculate source_system_tab, source_screen_name, source_type
   *  until now, how many times the song_id/user_id has occurred
   *  calculate how many times the song_id/user_id occurs before or after; define a window size [10, 25, 500, 5000, 10000, 50000]
   *  is the after/before artist_name or type the same with the current one

4. User features

5. Song features

6. User and song component features

7. Artist and song component features