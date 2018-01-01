---
title: "Instacart-Market-Basket-Analysis"
comments: true
share: true
toc: true
categories:
  - Kaggle
tags:
  - Kaggel
---



This post generally describes what validation strategy, what kinds of features and models were made by konoder to win the second place of this competition.

Instead of giving a detailed explanation about the competition challenge, EDA and how he considered the problem, this post just directly dive into the code of feature engineering and model parts.



## User Feature

### repeat_previous_ratio 

*  repeat product in previous orders ratio:

   *intersection(previous & current) / current*

   counts = how many products in previous **W** order set are repeated in current order (where W means window size: 1~5)

   chance = how many products in current order

*  reordered and non-reordered products ratio in current order

*  non-reordered products counts, non-reordered_cumsum and this cumsum/order_sequence_number

*  order's size, cumsum, and this cumsum/order_sequence_number

### orderspan_average

*  the mean of days_since_prior_order

### visit_time

```python
def timezone(s):
    if s < 6:
        return 'midnight'
    elif s < 12:
        return 'morning'
    elif s < 18:
        return 'noon'
    else:
        return 'night'
```

*  day_of_week (normalized) visit frequency
*  timezone (normalized) visit frequency
*  timezone*day_of_week (normalized) visit frequency

### organic

*  organic products count / total orders count
*  gluten-free products count / total orders count
*  is_Asian products count / total orders count

### delta_time

*  delta hours of order_hour_of_day between current order and previous **t** order (t: 1~3)

### order_size

*  min, max, median, mean, std order size of each user

### have_you_bought

*  whether or not users have bought high-frequency products; as one-hot encoding

### None

*  tag previous t order whether or not is **None** order (t: 1~20)
*  None order: all products in this order are all new items (not bought before by this user)

## Item Feature



