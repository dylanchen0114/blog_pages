---
title: "Instacart-Market-Basket-Analysis-part2"
comments: true
share: true
toc: true
categories:
  - Kaggle
tags:
  - Kaggel
---

[http://blog.kaggle.com/2017/09/21/instacart-market-basket-analysis-winners-interview-2nd-place-kazuki-onodera/](http://blog.kaggle.com/2017/09/21/instacart-market-basket-analysis-winners-interview-2nd-place-kazuki-onodera/)

Key takeaway of the interview:

1. The problem is a little different from the general recommendation problem, where we often face a cold start issue of making predictions for new users and new items that we’ve never seen before. For example, a movie site may need to recommend new movies and make recommendations for new users.

2. Split the problem into two parts: 

   *  Predicting reorders: users, items and user*item features are used to predict the probability that a user will repurchase a product.
   *  Prediction None: only user-based features are used, **note that user*item features can be converted (max, min, sum, mean…) to be used here**, to predict the probability that the user's next order will contain any previously purchased products.

3. Enlarge the train and validation dataset

   *  Using each user's last 3 orders to be train or validation dataset as well

   *  The validation strategy split the dataset **by user**

      ![foo](http://5047-presscdn.pagely.netdna-cdn.com/wp-content/uploads/2017/09/Screen-Shot-2017-09-20-at-11.33.51-AM.png)

4. Top features for both reorder and none

   *  Reorder

      ![foo](http://5047-presscdn.pagely.netdna-cdn.com/wp-content/uploads/2017/09/p12.png)

   *  None

      ![foo](http://5047-presscdn.pagely.netdna-cdn.com/wp-content/uploads/2017/09/p13.png)

5. Feature insights and F1 Maximization, please refer to blog part1 and interview page