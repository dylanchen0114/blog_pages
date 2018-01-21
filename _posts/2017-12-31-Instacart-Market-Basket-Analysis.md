---
title: "Instacart-Market-Basket-Analysis-part1"
comments: true
share: true
toc: true
categories:
  - Kaggle
tags:
  - Kaggel
---



This post generally describes what validation strategy, what kinds of features and models were made by konoder to win the second place of this competition.

Instead of giving a detailed explanation about the competition challenge, EDA and how he considered the problem, this post just directly dive into the code of feature engineering part.



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

*  whether or not users have bought high-frequency products (one-hot encoding)

### None

*  tag previous t order whether or not is **None** order (t: 1~20)

   None order: all products in this order are all new items (not bought before by this user)

## Item Feature

### buy_time

*  (unique) count and ratio by product and hour

   *where unique means dropping duplicates by user, product and hour*

*  (unique) count and ratio by product and day_of_week

*  (unique) count and ratio by product and timezone

*  (unique) count and ratio by product and timezone*day_of_week

### cycle

*  min, max, median, mean, std of dat_since_prior_order
*  total order count
*  total reorder count and ratio
*  total user count
*  order count / user count

### co-occur

### mean_pos_cart

*  min, max, mean, median, std of add_to_cart_order

### one_shot

*  count of users that buy this product only once
*  ratio: above count / total users this product

### together

*  min, max, mean, median, std of order size the product is bought

### streak

*  mean, max and median of count that users buy this product in a row

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

*  probability product will be reordered within **N** orders (N: 2~5)

   ratio = counts / chance

   chance: times user buy this product (except the last N orders)

   counts: times user reorder this product within N

### dow_diff

*  ratio = ratio1 - ratio2

   ratio1: normalized value counts by dow

   ratio2: count by product and dow / total count of each product

### first_order

*  probability that this product will be repurchased after first order of each user

   ratio = counts / chance

   chance: count of each user buy this product first time

   counts: count of each user repurchase this product after first buy

### onb_diff

*  diff of order number of each user buy this product
*  mean, std, min, max, median, skew of the above diff

## User x Item Feature

### total_buy

*  order count by user and product
*  ratio: above count / order_sequence_number
*  near 5 orders above features

### reorder_all

*  reorder situation in previous **t** order (t: 1~20)

### last_order_date

*  days since last order this item

### buy_item_in_a_row

*  count of orders users have bought this item in a row
*  above count / order_sequence_number

### last_order_number

*  the last order_sequence_number user bought this item
*  diff between above number and current order_sequence_number

### mean_pos_cart

*  min, max, mean, median, std of user add to cart this item
*  near 5 orders above statistics features

### timezone_dow

*  count by user, product and tz (or dow)
*  ratio1: above count / count by user and product
*  ratio2: above count / count by user and tz (or dow)

### order_ratio_by_chance

*  ratio = count / chance

   count: order count by user and product

   chance: max order_sequence_number by user - min order_sequence_number by user and product

*  above ratio in near 5 orders

### repeat_within_today

*  how many times user buy this item within today

### cycle

*  min, max, mean, median, std of days since last order this item
*  near 5 orders above features

### aisle_dep

*  count by user, aisle/dep
*  normalized count by user, aisle/dep

### co-occur

*  min, max, median, mean, std of order size when user buy this item

   ```python
   tbl['useritem_cooccur-min-min'] = tbl['user_order_size-min']  - tbl['useritem_cooccur-min']
   tbl['useritem_cooccur-max-min'] = tbl['useritem_cooccur-max'] - tbl['useritem_cooccur-min']
   tbl['useritem_cooccur-max-max'] = tbl['user_order_size-max'] - tbl['useritem_cooccur-max']
   ```

   *user_order_size-min(max) means min/max order size by user*

### streak

*  count of user buy each product in a row

### replacement **

*  replacement ratio of each product
*  the min, max, mean, median of the above ratio (code 316&011)

## Day Time Feature

### how_many_come

*  normalized order and user count by day_of_week

*  normalized order and user count by hour_of_day

   ```python
   dow['dow_rank_diff'] = dow.dow_order_cnt.rank() - dow.dow_item_cnt.rank()
   hour['hour_rank_diff'] = hour.hour_order_cnt.rank() - hour.hour_item_cnt.rank()
   ```

   ​