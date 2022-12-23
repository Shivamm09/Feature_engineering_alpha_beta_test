# Feature_engineering_alpha_beta_test

## Introduction

In this experiment, I conducted an A/B Test. The main goal was to understand what is the impact of feature engineering on the performance of classifiers.

Generally, when we deal with machine learning, we try out a couple of machine learning models (once we have the dataset prepared and cleaned), to understand what are the baseline results before we fine-tune the models. I wanted to investigate the effect of removing the noise from the dataset using statistics and variance threshold and compare the performance of the classifiers to that on a dataset without modifications, both without fine-tuning. 

## Project Structure

The Framework python file contains all the methods which I used such as running the statistical tests, the analysis, and the classifications. 

I ran the tests on two datasets, one is the heart attack project where the dataset is from https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset and the other is for breast cancer detection where the dataset is from https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset. 

The work on the two datasets has been separated into two different python files as well.

## Statistical Investigation  

Generally, when dealing with categorical variables, we use the chi-squared test and when we are dealing with numerical variables we work with t-tests. However, for our study, we decided to work with Wilcoxon and mannwhitneyu tests instead of t-tests when working with numerical variables. The t-tests relies on the distribution being normal; in our specific cases most of the distributions were not normal or it is not safe to assume they are. We used Wilcoxon, when the number of samples in both distributions were the same, and mannwhitneyu when the number of samples were not the same. 

## Results 
### Heart Attack Dataset

In the Heart attack project, we were working with both categorical and numerical variables. For the categorical variables, we ran a chi-squared test on each of the categorical variables with respect to the output variables. We wanted to see if the output variables were dependent on the variables which are categorical, if they are we kept the variables and if they were not we removed them. However from our results, we saw that none of the variables were independent of the output variable, which meant removing any of them was not justified. We also ran the mannwhitneyu on the numerical variables, and saw no difference in the outputs, it showed that all distributions were differenct for the output variables. 

Because there was no difference, we decided to test out the variance threshold and power transform method, and see how that would affect the results. We saw just by adding two lines of code to the dataset, we were able to increase the average accuracy by 6%, this in theory should also be able to improve the final models, since they will be fine tuned on the dataset which has less noise. 


### Breast Cancer Dataset

Even though this project was a classification project, all the variables were numerical. So I first ran a mannwhitneyu test, and removed the variables which were independent to the output variable. I also had the normal dataset do a standard scaler, and the other do a power transform (which is good when not all distributions are normal). Even in this project we saw that we were able to increase the mean accuracy by 1%, which was signficant even with the small number of samples. 

## Main Learnings

We can remove noise from the dataset by removing the variables which are indepedent to the output variables (when dealing with classification) using statistical significance testing.
We should use wilcoxon and mannwhitneyu tests when dealing with numerical variables, because it ensures that we don’t make strong assumptions about the distributions being normal.
We should use power transform instead of standard scaler, especially if the distributions are not normal. That way we allow the data points to behave more Gaussian; a standard scaler might not be able to adjust the skewness of those distributions to the same extent a power transform does.
