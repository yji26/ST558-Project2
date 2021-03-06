ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Introduction](#introduction)
  - [R Markdown Automation](#r-markdown-automation)
  - [Day of the Week Models](#day-of-the-week-models)
  - [Model Selection](#model-selection)
  - [Conclusions](#conclusions)

## Introduction

The purpose of this vignette is to compare the prediction accuracy of
two types of regression models on a provided data set. The data used is
an online news popularity data set for web articles published by
Mashable over a two-year period. Models fitted to this data set consist
of a multiple linear regression model and a nonlinear Random Forest
regression model. The data description and download links can be [found
here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#).

## R Markdown Automation

I use the following code to generate data and fitted models for each day
of the week. The `apply` function calls another R Markdown file
[`YunJi_Project2_DayOfWeekModel.Rmd`](https://github.com/yji26/ST558-Project2/blob/master/YunJi_Project2_DayOfWeekModel.Rmd)
which contains the code needed to import, explore, clean, filter, split,
and model the news popularity data for each day.

``` r
days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
output_file <- paste0(days, "Analysis.md")
params <- lapply(days, FUN = function(x){list(day = x)})
reports <- tibble(output_file, params)

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "YunJi_Project2_DayOfWeekModel.Rmd", 
               output_file = x[[1]], 
               params = x[[2]]
              )
      }
)
```

## Day of the Week Models

The model vignettes for each day of the week are linked below. Each
day’s vignette contains data exploration, along with model fitting on
the training data set and model evaluation for predicting the testing
data set.

  - [Monday Models](MondayAnalysis.html)

  - [Tuesday Models](TuesdayAnalysis.html)

  - [Wednesday Models](WednesdayAnalysis.html)

  - [Thursday Models](ThursdayAnalysis.html)

  - [Friday Models](FridayAnalysis.html)

  - [Saturday Models](SaturdayAnalysis.html)

  - [Sunday Models](SundayAnalysis.html)

## Model Selection

Comparing the pair of linear regression and Random Forest models for
each of the seven days of the week, I find that on the testing set
predictions, the root mean-square prediction errors between the two
models are fairly close. Sunday articles generated the most accurate
predictions, while Monday articles generated the least accurate. But
overall the inter-day differences are not large.

For each day’s data set, my Random Forest model does slightly better
than the linear regression model, however this comes at the cost of
interpretability. Linear regressions have coefficients and confidence
intervals that provide a reasonable range for weighing each predictor
variable. Random Forest allows for relative feature importance between
predictor variables, but due to being an ensemble method, Random
Forest’s feature importance does not tell you in a straightforward way
a feature’s relationship to the target variable or about the effects of
feature interactions on the outcome.

So in this case, choosing one model over the other as a predictor comes
down to the needs of the model consumer. If prediction accuracy is
valued and transparency is not critical, as is often the case in an
industry like marketing, we’re better off with the Random Forest model.
Conversely, if interpretability is more important, such as in more
regulated industries like insurance, a linear regression model that can
get you most of the way there *and* can be easily explained would be
more valuable.

## Conclusions

A critical aspect to good data science is the need to models are kept
up-to-date. In many instances models are trained to predict people’s
behavior, and often that behavior will change over time as a result of
new regulations, shifting consumer preferences or changing reward
incentives for various economic activities. The way to keep models
current is to hold back some of your data from training and use it to
evaluate the prediction accuracy of your fitted models. The evaluation
metric may differ depending on the nature of the predictions,
i.e. classification vs. regression, or prioritizing false
positives/negatives over a general misclassification rate. For this
project, I use root mean-square error (RMSE) between predicted and
actual values of the log of `shares`.

Comparing the difference in RMSE between the training and testing set
predictions, I find no consistent bias going from training to testing.
On some days of the week the RMSPE from the testing set predictions are
slightly above the training RMSE for a given model, and on other days
the opposite is true. This fact holds for both linear and nonlinear
models. The only large anomaly from my vignettes comes from the Tuesday
data set, where the RMSE of the linear regression model on the training
set is 2.45826 and the RMSPE of the same model on the testing set is
0.38327. This may be caused by an extreme outlier for one of the rows in
the training set, and a deeper dive could reveal additional insights
that may help improve the model.

In this project I think prediction accuracy should be valued more than
interpretability, and taking the aforementioned factors into account, I
would use a Random Forest regression model to fit a predictor for each
day of the week.
