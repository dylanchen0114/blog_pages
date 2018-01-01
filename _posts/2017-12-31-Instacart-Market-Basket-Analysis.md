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

*  Repeat product in previous orders ratio:

   *intersection(previous & current) / current*

   counts = how many products in previous **W** order set are repeated in current order (where W means window size: 1~5)

   chance = how many products in current order (all)

*  Reordered and non-reordered products ratio in current order

*  Non-reordered products counts









