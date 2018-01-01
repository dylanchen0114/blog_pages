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

*  Repeat product in previous orders ratio:

   *intersection(previous & current) / current*

   counts = how many products in previous **W** order set are repeated in current order (where W means window size: 1~5)

   chance = how many products in current order

*  Reordered and non-reordered products ratio in current order

*  Each order's non-reordered products counts, non-reordered_cumsum and this cumsum/order_sequence_number

*  Each order's size, cumsum, and this cumsum/order_sequence_number

### orderspan_average

*  The mean of days_since_prior_order

### visit_time

timezone are defined as following:

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

*  User's visit frequency and normalized frequency in each day_of_week
*  User's visit frequency and normalized frequency in each timezone
*  User's visit frequency and normalized frequency in each timezone*day_of_week

### organic

*  Each user's total count of organic products / Each user's total orders count
*  Each user's total count of gluten-free products / Each user's total orders count
*  Each user's total count of is_Asian products / Each user's total orders count