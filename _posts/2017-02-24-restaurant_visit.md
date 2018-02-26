---
title: " Recruit-Restaurant-Visitor-Forecasting-1st-Solution"
comments: true
mathjax: true
share: true
toc: true
---

### Validation Strategy

Data description: [https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data)

The dataset period begins on 2016-01-01 and ends on 2017-05-31. (2017-04-23 — 2017-05-31 as submission test set)

Take the last 42 days (2017-03-12 — 2017-04-22) as the offline test set.

To enrich the train data,  split the train as following: create different types of features before begin date, and predict the visitors amount during begin date and end date. 

#### Split #1

| Begin_Date | End_Date (39 days) |
| :--------: | :----------------: |
| 2017-03-12 |     2017-04-19     |
| 2017-03-05 |     2017-04-12     |
| 2017-02-26 |     2017-04-05     |
| 2017-02-19 |     2017-03-29     |
|   …………..   |       …………..       |
|   …………..   |       …………..       |
| 2016-02-21 |     2016-03-30     |
| 2016-02-14 |     2016-03-23     |
| 2016-02-07 |     2016-03-16     |

#### Split #2

Other train periods as following (as offline validation set, also trained in model when in final submission):

| Begin_Date |  End_Date  |
| :--------: | :--------: |
| 2017-03-19 | 2017-04-22 |
| 2017-03-26 | 2017-04-22 |
| 2017-04-02 | 2017-04-22 |
| 2017-04-09 | 2017-04-22 |
| 2017-04-16 | 2017-04-22 |

p.s. 

```python
data['visitors'] = np.log1p(data['visitors'])
# when submit 
np.expm1(pred)
```

### Store Features

1. group by store_id, calculate past all/56/28/14 days min/max/median/mean/count/std/skew visitors; fill nan with 0

2. group by store_id, expected weighted visitors: weighted sum and weighted mean (past all days); fill nan with 0

   *  calculate day difference between visit date and begin date
   *  take 0.985 ** day_diff as weight, recent behavior has a larger weight
   *  new visitors = weight * visitors

3. group by store_id and dow, calculate past all/56/28/14 days min/max/median/mean/count/std/skew visitors; fill nan with 0

4. group by store_id, calculate each day's visitors amount diff between today and last day

   mean, std, max and min of the above diff

   abs diff mean:  if the time long enough simple diff mean will be colse to 0 that would be no distinction.

   window size: past all and 58 days

   ```python
   result = data_temp.set_index(['store_id','visit_date'])['visitors'].unstack()
   result = result.diff(axis=1).iloc[:,1:]
   c = result.columns
   result['store_diff_mean'] = np.abs(result[c]).mean(axis=1)
   result['store_diff_std'] = result[c].std(axis=1)
   result['store_diff_max'] = result[c].max(axis=1)
   result['store_diff_min'] = result[c].min(axis=1)
   ```

5. group by store_id and dow, expected weighted visitors: weighted sum and weighted mean (past all days); fill nan with 0

   -  calculate day difference between visit date and begin date
   -  take 0.985 ** day_diff as weight, recent behavior has a larger weight; try different coef within [0.9,0.95,0.97,0.98,0.985,0.99,0.999,0.9999] 
   -  new visitors = weight * visitors

6. group by store_id and holiday flag1/2, calculate past all days min/max/median/mean/count/std/skew visitors; fill nan with 0

7. group by store_id, calculate the day diff between begin date and first/last visit date

### Visitor Features

1. group by genre name, calculate past all/56/28 days min/max/median/mean/count/std/skew visitors; fill nan with 0
2. group by genre name, expected weighted visitors: weighted sum and weighted mean (past all days); fill nan with 0
   -  calculate day difference between visit date and begin date
   -  take 0.985 ** day_diff as weight, recent behavior has a larger weight
   -  new visitors = weight * visitors
3. group by genre name, calculate past all/56/28 days min/max/median/mean/count/std/skew visitors; fill nan with 0
4. group by genre name and dow, expected weighted visitors: weighted sum and weighted mean (past all days); fill nan with 0
   -  calculate day difference between visit date and begin date
   -  take 0.985 ** day_diff as weight, recent behavior has a larger weight
   -  new visitors = weight * visitors

### Reserve Features

1. day diff between reserve visit datetime and reserve datetime (pre-processing)
2. group by store_id and reserve visit_date, calculate sum/count reserve_visitors
3. group by store_id and reserve visit_date, calculate mean diff time (1)
4. group by reserve visit_date, calculate sum/count reserve_visitors and mean diff time (1) (date features)

### Additional Rate Features

```python
result['store_mean_14_28_rate'] = result['store_mean14']/(result['store_mean28']+0.01)
result['store_mean_28_56_rate'] = result['store_mean28'] / (result['store_mean56'] + 0.01)
result['store_mean_56_1000_rate'] = result['store_mean56'] / (result['store_mean1000'] + 0.01)
result['genre_mean_28_56_rate'] = result['genre_mean28'] / (result['genre_mean56'] + 0.01)
result['genre_mean_56_1000_rate'] = result['genre_mean56'] / (result['genre_mean1000'] + 0.01)
```