---
title: "Lessons-Learned-from-TalkingData"
comments: true
share: true
toc: true
categories:
  - Kaggle
tags:
  - Kaggel
---

### Sampling Methods

1. Negative down-sampling: use all positive examples and down-sampled negative examples such that their size becomes equal to the number of positive ones. It discards about 99.8% of negative examples, but didn't see much performance deterioration when testing with initial features. 

   ​

   What can this down-sampling methods improve?

   "Thanks to negative down-sampling, the many-features was not a so big problem to us, so we didn't do fine-grained feature selection. For example, when I came up with the LDA features, I added 100 new features to my model. Luckily, it worked very well, so I added all of the 100 features. These 100 features were used for the final submission model without any further testing.

   I haven't cared about which one of them contribute the most though ideally I might have had to remove useless features."

   ​

### Features

Basically Sampling Methods just applied when training the model, when creating features, all sample data must be used.

Features were mostly tested by adding them one by one, and keeping them if local validation score improved by at least 0.00005. I also added several of them at once, then removed them one by one to see if validation score decreased. I basically did feature selection full time for the competition, preparing experiments to be run while I was away during day, or while I was sleeping. The machine never stopped.



1. Different methods to capture ID type categorical features information 

   1. Use categorical feature embedding by LDA/NMF/LSA (low or high dimension)

      ```python
      apps_of_ip = {}
      for sample in data_samples:
      	apps_of_ip.setdefault(sample['ip'], []).append(str(sample['app']))
      ips = list(apps_of_ip.keys())
      apps_as_sentence = [' '.join(apps_of_ip[ip]) for ip in ips]
      apps_as_matrix = CountTokenizer().fit_transform(apps_as_sentence)
      topics_of_ips = LDA(n_components=5).fit_transform(apps_as_matrix)
      ```

   2. "Matrix factorization. This was to capture the similarity between users and app. I use several of them. They all start with the same approach; construct a matrix with log of click counts. I used: ip x app, user x app, and os x device x app. These matrices are extremely sparse (most values are 0). For the first two I used truncated svd from sklearn, which gives me latent vectors (embeddings) for ip and user. For the last one, given there are 3 factors, I implemented libfm in Keras and used the embeddings it computes. I used between 3 and 5 latent factors. All in all, these embeddings gave me a boost over 0.0010. I think this is what led me in top 10. I got some variety of models by varying which embeddings I was using."

      ​

2. Treat day and hour as categorical features

3. Click count, cum-count, next **window size** (window size is a must trying feature kinds)

4. Forward and backward time delta

5. Average target rate



### Ensemble

1. Rank-based weighted averaging of several bagging



[https://zhuanlan.zhihu.com/p/36580283](https://zhuanlan.zhihu.com/p/36580283)

[https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56481](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56481)