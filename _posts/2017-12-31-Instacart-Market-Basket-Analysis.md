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



# Features

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

### buy_time

*  (unique) count and ratio by product and hour

   *where unique means dropping duplicates by user, product and hour*

*  (unique) count and ratio by product and day_of_week

*  (unique) count and ratio by product and timezone

*  (unique) count and ratio by product and timezone*day_of_week

### cycle

*  min, max, median, mean, std of dat_since_prior_order
*  total order count of products, total reorder count of products, ratio
*  total user count of products
*  order count / user count

### co-occur

### mean_pos_cart

*  min, max, mean, median, std of add_to_cart_order

### one_shot

*  count of users that buy this product only once
*  ratio: above count / total users this product

### together

*  min, max, mean, median, std of total order size when this product is bought

### streak

### 1to1

*  ratio of 1to1, 11to1, 10to1, 111to1, 110to1, 101to1, 100to1

   ```python
   # calculate by user
   pd.crosstab(tmp.order_number, tmp.product_id)
   ```

   ratio = counts / chance

   If previous order match the pattern before 'to', chance += 1

   If current order match the pattern after 'to', counts += 1

### within-N

*  the probability each product will be reordered within **N** orders (N: 2~5)

   ratio = counts / chance

   chance: how many times user buy this product (except the last N orders)

   counts: how many times user reorder this product within N

### dow_diff

*  ratio = ratio1 - ratio2

   ratio1: normalized value counts by day_of_week

   ratio2: count by product and day_of_week / total count of each product

### first_order

### one_diff

## User x Item Feature

### total_buy

