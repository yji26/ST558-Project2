ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Data Set Information](#data-set-information)
  - [Filter for Tuesday Data](#filter-for-tuesday-data)
  - [Summary of Training Data](#summary-of-training-data)
  - [Modeling and Cross-Validation](#modeling-and-cross-validation)
  - [Model Test Performance](#model-test-performance)

## Data Set Information

The data set used in this project consists of 58 numerical predictor
variables along with 2 informational fields; the target for the data set
is the column `shares`, a measure of an article’s popularity.

Seven columns from the data set are indicator values for the day of week
that an article is published; these columns are named
`weekday_is_monday`, `weekday_is_tuesday`, etc. For each weekday, I
shall create a separate set of models to predict the `shares` value. The
follow code will load the data set, filter on a specific day of week,
then split the resulting data 70/30 into training and testing data sets.
The training set will be used for training the linear and non-linear
regression models, and the testing set for comparing which of the models
has a higher prediction accuracy.

``` r
newsData <- read_csv("./Data/OnlineNewsPopularity.csv")
```

Some columns can be removed to reduce the size of the data. Column `url`
merely contains the text URL for the article and is not needed. Column
`timedelta` represents the number of days between article publication
and data acquisition; although this can potentially be used to detect
the site’s popularity over time, the data dictionary describes this as
non-predictive data and therefore I will also exclude it from the
modeling data. Column `is_weekend` is made redundant when the data is
split by day of week, since for any given day the column value will be
either all zeroes or all ones. Columns `rate_positive_words` and
`rate_negative_words` represent the proportion of positive and negative
words among all non-neutral words, and always sum to 1; therefore, one
of these columns can be removed without any loss of information, and I
choose to remove `rate_negative_words`.

``` r
newsData <- newsData %>%
  select(!url & !timedelta & !is_weekend & !rate_negative_words)
```

## Filter for Tuesday Data

Next I filter for the day of week, then remove the `weekday_is_*`
columns.

``` r
dayOfWeek <- params$day
dayColumn <- paste0("weekday_is_", tolower(dayOfWeek))

newsDataFiltered <- newsData %>%
  filter((!!as.symbol(dayColumn)) == 1) %>%
  select(!starts_with("weekday_is_"))
```

Finally I split the resulting data into training and testing sets.

``` r
set.seed(seed)
train <- sample(1:nrow(newsDataFiltered), size = nrow(newsDataFiltered)*0.7)
test <- dplyr::setdiff(1:nrow(newsDataFiltered), train)
newsDataTrain <- newsDataFiltered[train, ]
newsDataTest <- newsDataFiltered[test, ]
```

## Summary of Training Data

The training data will not be the same from one day of week to another,
but certain observations hold across all.

First, we can examine the distribution of the target variable `shares`.

``` r
g <- ggplot(data = newsDataTrain, aes(x = shares))
g + geom_histogram(bins = 50)
```

![](TuesdayAnalysis_files/figure-gfm/shares%20histogram-1.png)<!-- -->

Values of `shares` appear to follow a power-law distribution where a few
very popular articles receive an outsized share of views. Therefore for
regression it may be better to transform the target variable using a
logarithmic function.

``` r
newsDataTrain <- newsDataTrain %>%
  mutate(shares = log10(shares))

newsDataTest <- newsDataTest %>%
  mutate(shares = log10(shares))

g <- ggplot(data = newsDataTrain, aes(x = shares))
g + geom_histogram(bins = 50)
```

![](TuesdayAnalysis_files/figure-gfm/target%20transformation-1.png)<!-- -->

The range of values for the predictor columns vary: some columns such as
`global_subjectivity` are proportions and are limited to values between
0 and 1, some indicator values like `data_channel_is_world` only have
values 0 or 1, and others like `n_tokens_content` are raw counts which
are natural numbers with no theoretical upper bound.

``` r
g <- ggplot(data = newsDataTrain, aes(x = global_subjectivity))
g + geom_histogram(bins = 10)
```

![](TuesdayAnalysis_files/figure-gfm/predictor%20histograms-1.png)<!-- -->

``` r
g <- ggplot(data = newsDataTrain, aes(x = data_channel_is_world))
g + geom_histogram(bins = 2)
```

![](TuesdayAnalysis_files/figure-gfm/predictor%20histograms-2.png)<!-- -->

``` r
g <- ggplot(data = newsDataTrain, aes(x = n_tokens_content))
g + geom_histogram(bins = 50)
```

![](TuesdayAnalysis_files/figure-gfm/predictor%20histograms-3.png)<!-- -->

Because of this, when selecting for models, it is advised to standardize
(that is, center and scale) predictor values prior to fitting the
models.

## Modeling and Cross-Validation

Two regression models are fitted to the training data: a multiple linear
regression model and a non-linear Random Forest regression model. To
tune these models, I perform three repeated 5-fold cross-validations
from the `caret` package. Cross-validation is used to limit overfitting
on the training data, as in each trial one of the folds is held out as a
validation set while the remaining folds are combined and used to train.
Besides centering and scaling, the linear regression model uses the
package default options, while for the Random Forest model, I specified
possible tree depths to be 3, 5 or 10, as I found that deep trees in
conjunction with repeated CV took up a lot of computation time on my PC.

``` r
set.seed(seed)
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

lm_fit <- train(shares ~ ., 
                data = newsDataTrain, 
                method = "lm",
                trControl = trctrl,
                preProcess = c("center", "scale")
)

rf_fit <- train(shares ~ ., 
                data = newsDataTrain, 
                method = "rf",
                trControl = trctrl,
                preProcess = c("center", "scale"),
                tuneGrid = expand.grid(mtry = c(3,5,10))
)
```

Comparing the performance of the models on the training set, we get the
following:

``` r
lm_fit$results
```

    ##   intercept     RMSE   Rsquared       MAE   RMSESD RsquaredSD    MAESD
    ## 1      TRUE 2.458257 0.08164837 0.3480221 4.387911 0.04261816 0.143339

``` r
rf_fit$results
```

    ##   mtry      RMSE  Rsquared       MAE     RMSESD RsquaredSD       MAESD
    ## 1    3 0.3740387 0.1252374 0.2793586 0.01040194 0.01514743 0.005522485
    ## 2    5 0.3744444 0.1223540 0.2799495 0.01004925 0.01548049 0.005536086
    ## 3   10 0.3751705 0.1193254 0.2804068 0.01051256 0.01550822 0.005807565

The model with higher `Rsquared` value and lower `RMSE` value is the
better performer on the training data (which model is better may vary
depending on the day of week used). However, the real test comes when
the fitted models are evaluated for prediction accuracy on the testing
data set.

## Model Test Performance

``` r
newsLmPred <- predict(lm_fit, newdata = newsDataTest)
lm_mspe <- mean((newsDataTest$shares - newsLmPred)^2)
lm_mspe
```

    ## [1] 0.1468922

``` r
newsRfPred <- predict(rf_fit, newdata = newsDataTest)
rf_mspe <- mean((newsDataTest$shares - newsRfPred)^2)
rf_mspe
```

    ## [1] 0.1443374

Here we compare the mean-square prediction error on the log-scaled
target variable `shares`: the multiple linear regression model has a
MSPE of 0.14689 and the Random Forest regression model has a MSPE of
0.14434. The model with lower MSPE is the better performer on the
testing data set.
