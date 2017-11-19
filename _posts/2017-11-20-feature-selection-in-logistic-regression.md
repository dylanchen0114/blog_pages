---
layout: post

title: "Feature Selection in Linear Model"

author: "Dylan"
---

### Practice Example One

Get first 500 features based on mean, std, missing values' percentage.

Select top 250 features using information value, chi2-square or correlation with target variable.

Convert the categorical variables into indicator variables, so just need to deal with numeric variables after this stage.

Use step wise regression (forward selection ) to get to top 150 variables, do the directional sense check of variables. Say 140 variables are left after this stage.

Use stepwise regression with step wise option (which is combination of forward and backward selection) to get top 60 variables.

Use VIF to get rid of multi collinearity in scientific way.

Say y20 variables are left - then use stepwise again along with out of time validation procedures (rank ordering, scoring, co efficient stability, KS, GINI stability) to select most powerful variable for the model.





### Practice Example Two

Univariate Feature Selection

A more robust but more "brute-force" method where the impact of combined features are evaluated together is named "Recursive Feature Elimination":

1. First, train a model with all the feature and evaluate its performance on held out data.
2. Then drop let‘s say the 10% weakest features (e.g. the feature with least absolute coefficients in a linear model) and retrain on the remaining features.
3. Iterate until you observe **a sharp drop in the predictive accuracy** of the model.



a simple wrapper method: Forward Feature Selection (FFS). The way FFS works is quite simple:

-  Choose a evaluation metric associated with/or accuracy. Lets call it '**score**'. In each iteration we seek to find feature/s that maximises this **score**.
-  In first iteration, lets say out of four features [A,B,C,D], turns out B is the best performer. We save this feature to a **feature_subset**, [B], and its corresponding score, as **max_score**.
-  In the next iteration, we process our **feature_subset** [B] with one of the remaining features [A,C,D]. So processing,  [B with A], [B with C] and [B with D], out of these three, lets say we find the best score for  [B with C]. If the best score > **max_score**, we add the corresponding feature to our **feature_subset **which is now [B, C]**.**
-  We continue the iterations until our feature subset optimises for the **max_score**. We then declare that this feature subset is the best we can do.





To account for the instability of Lasso when dealing with highly correlated features, you should either consider combining the L1 penalty with L2 (the compound penalty is called Elastic Net) which will globally squash the coefficients but avoid randomly zeroing one out of 2 highly correlated relevant features



Stability in feature selection models for the pure Lasso can also be achieved by bootstrapping several Lasso models on dataset folds and selecting the intersection (or union, I am not sure) of the non zero-ed features. 