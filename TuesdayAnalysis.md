ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Data Set Information](#data-set-information)
  - [Filter for Tuesday Data](#filter-for-tuesday-data)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Modeling and Cross-Validation](#modeling-and-cross-validation)
  - [Model Test Performance](#model-test-performance)

## Data Set Information

The news article popularity data set used in this project consists of 58
numerical predictor variables along with 2 informational fields; the
target for the data set is the column `shares`, a measure of an
article’s popularity.

Seven columns from the data set are indicator values for the day of week
that an article is published; these columns are named
`weekday_is_monday`, `weekday_is_tuesday`, etc. For each weekday, I
created a pair of linear and nonlinear regression models to predict the
numerical value of `shares`. The code below loads the data set, filters
on a specific day of week according to the parameter variable
`params$day`, then splits the resulting data 70/30 into training and
testing data sets. The training set is used for training the two
regression models, and the testing set is used for comparing which of
the models has a higher prediction accuracy.

First I load the raw data from a stored file in my repository into a
data frame in R.

``` r
newsData <- read_csv("./Data/OnlineNewsPopularity.csv")
```

Some columns of the data frame may be removed to reduce the size of the
data. Column `url` merely contains the text URL for the article and is
not needed. Column `timedelta` represents the number of days between
article publication and data acquisition; although this can potentially
be used to detect the site’s popularity over time, the data dictionary
describes this as non-predictive data and therefore I also exclude it
from the modeling data. Column `is_weekend` is made redundant when the
data is split by day of week, since for any given day its column value
will be either all zeroes or all ones, and therefore not useful for
prediction. Columns `rate_positive_words` and `rate_negative_words`
represent the proportion of positive and negative words among all
non-neutral words, and always sum to 1; one of them may be removed
without any loss of information, and I choose to remove
`rate_negative_words`.

``` r
newsData <- newsData %>%
  select(!url & !timedelta & !is_weekend & !rate_negative_words)
```

## Filter for Tuesday Data

Next I filter rows for the day of week based on parameter value
`params$day`, then remove the `weekday_is_*` columns from the data
frame. This is the resulting data set right before being split into
training and testing sets.

``` r
dayOfWeek <- params$day
dayColumn <- paste0("weekday_is_", tolower(dayOfWeek))

newsDataFiltered <- newsData %>%
  filter((!!as.symbol(dayColumn)) == 1) %>%
  select(!starts_with("weekday_is_"))
```

## Exploratory Data Analysis

Inputs to the models require that all columns have numerical non-null
values. Make sure this is true.

``` r
summary(newsDataFiltered)
```

    ##  n_tokens_title  n_tokens_content n_unique_tokens    n_non_stop_words  
    ##  Min.   : 4.00   Min.   :   0.0   Min.   :  0.0000   Min.   :   0.000  
    ##  1st Qu.: 9.00   1st Qu.: 248.0   1st Qu.:  0.4732   1st Qu.:   1.000  
    ##  Median :10.00   Median : 398.5   Median :  0.5409   Median :   1.000  
    ##  Mean   :10.44   Mean   : 542.6   Mean   :  0.6257   Mean   :   1.111  
    ##  3rd Qu.:12.00   3rd Qu.: 690.0   3rd Qu.:  0.6093   3rd Qu.:   1.000  
    ##  Max.   :19.00   Max.   :7081.0   Max.   :701.0000   Max.   :1042.000  
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs      num_imgs      
    ##  Min.   :  0.0000         Min.   :  0.00   Min.   : 0.000   Min.   :  0.000  
    ##  1st Qu.:  0.6290         1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.:  1.000  
    ##  Median :  0.6909         Median :  7.00   Median : 3.000   Median :  1.000  
    ##  Mean   :  0.7609         Mean   : 10.63   Mean   : 3.303   Mean   :  4.479  
    ##  3rd Qu.:  0.7542         3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.:  4.000  
    ##  Max.   :650.0000         Max.   :304.00   Max.   :62.000   Max.   :100.000  
    ##    num_videos     average_token_length  num_keywords   
    ##  Min.   : 0.000   Min.   :0.000        Min.   : 1.000  
    ##  1st Qu.: 0.000   1st Qu.:4.476        1st Qu.: 6.000  
    ##  Median : 0.000   Median :4.660        Median : 7.000  
    ##  Mean   : 1.308   Mean   :4.543        Mean   : 7.186  
    ##  3rd Qu.: 1.000   3rd Qu.:4.850        3rd Qu.: 9.000  
    ##  Max.   :73.000   Max.   :7.975        Max.   :10.000  
    ##  data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   :0.0000            Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.:0.0000            1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median :0.0000            Median :0.0000                Median :0.0000     
    ##  Mean   :0.0452            Mean   :0.1739                Mean   :0.1599     
    ##  3rd Qu.:0.0000            3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :1.0000            Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000       
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000       
    ##  Median :0.00000        Median :0.0000       Median :0.0000       
    ##  Mean   :0.06198        Mean   :0.1995       Mean   :0.2092       
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000       
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000       
    ##    kw_min_min       kw_max_min       kw_avg_min        kw_min_max    
    ##  Min.   : -1.00   Min.   :     0   Min.   :   -1.0   Min.   :     0  
    ##  1st Qu.: -1.00   1st Qu.:   441   1st Qu.:  139.8   1st Qu.:     0  
    ##  Median : -1.00   Median :   657   Median :  233.1   Median :  1300  
    ##  Mean   : 24.66   Mean   :  1121   Mean   :  306.2   Mean   : 13632  
    ##  3rd Qu.:  4.00   3rd Qu.:  1000   3rd Qu.:  355.4   3rd Qu.:  8300  
    ##  Max.   :217.00   Max.   :139600   Max.   :15851.2   Max.   :843300  
    ##    kw_max_max       kw_avg_max       kw_min_avg       kw_max_avg    
    ##  Min.   : 17100   Min.   :  3460   Min.   :  -1.0   Min.   :  2019  
    ##  1st Qu.:843300   1st Qu.:173091   1st Qu.:   0.0   1st Qu.:  3533  
    ##  Median :843300   Median :243859   Median : 994.1   Median :  4289  
    ##  Mean   :756044   Mean   :262009   Mean   :1110.6   Mean   :  5593  
    ##  3rd Qu.:843300   3rd Qu.:334350   3rd Qu.:2060.1   3rd Qu.:  6013  
    ##  Max.   :843300   Max.   :843300   Max.   :3609.7   Max.   :178675  
    ##    kw_avg_avg      self_reference_min_shares self_reference_max_shares
    ##  Min.   :  713.9   Min.   :     0            Min.   :     0           
    ##  1st Qu.: 2366.0   1st Qu.:   641            1st Qu.:  1100           
    ##  Median : 2844.7   Median :  1200            Median :  2900           
    ##  Mean   : 3125.1   Mean   :  4028            Mean   : 10329           
    ##  3rd Qu.: 3570.7   3rd Qu.:  2700            3rd Qu.:  8000           
    ##  Max.   :29240.8   Max.   :690400            Max.   :843300           
    ##  self_reference_avg_sharess     LDA_00            LDA_01       
    ##  Min.   :     0.0           Min.   :0.00000   Min.   :0.00000  
    ##  1st Qu.:   990.9           1st Qu.:0.02506   1st Qu.:0.02501  
    ##  Median :  2266.7           Median :0.03339   Median :0.03334  
    ##  Mean   :  6420.7           Mean   :0.18385   Mean   :0.13616  
    ##  3rd Qu.:  5257.6           3rd Qu.:0.24465   3rd Qu.:0.13498  
    ##  Max.   :690400.0           Max.   :0.91998   Max.   :0.91994  
    ##      LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.0000     
    ##  1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.02858   1st Qu.:0.3953     
    ##  Median :0.04002   Median :0.04000   Median :0.05000   Median :0.4521     
    ##  Mean   :0.21666   Mean   :0.21985   Mean   :0.24335   Mean   :0.4416     
    ##  3rd Qu.:0.33037   3rd Qu.:0.35701   3rd Qu.:0.42968   3rd Qu.:0.5059     
    ##  Max.   :0.92000   Max.   :0.91997   Max.   :0.92719   Max.   :1.0000     
    ##  global_sentiment_polarity global_rate_positive_words
    ##  Min.   :-0.30881          Min.   :0.00000           
    ##  1st Qu.: 0.05843          1st Qu.:0.02852           
    ##  Median : 0.11960          Median :0.03915           
    ##  Mean   : 0.11971          Mean   :0.03961           
    ##  3rd Qu.: 0.17707          3rd Qu.:0.05000           
    ##  Max.   : 0.61923          Max.   :0.11458           
    ##  global_rate_negative_words rate_positive_words avg_positive_polarity
    ##  Min.   :0.000000           Min.   :0.0000      Min.   :0.0000       
    ##  1st Qu.:0.009346           1st Qu.:0.6000      1st Qu.:0.3045       
    ##  Median :0.015152           Median :0.7143      Median :0.3569       
    ##  Mean   :0.016346           Mean   :0.6864      Mean   :0.3508       
    ##  3rd Qu.:0.021390           3rd Qu.:0.8000      3rd Qu.:0.4077       
    ##  Max.   :0.135294           Max.   :1.0000      Max.   :0.8333       
    ##  min_positive_polarity max_positive_polarity avg_negative_polarity
    ##  Min.   :0.00000       Min.   :0.0000        Min.   :-1.0000      
    ##  1st Qu.:0.05000       1st Qu.:0.6000        1st Qu.:-0.3250      
    ##  Median :0.10000       Median :0.8000        Median :-0.2500      
    ##  Mean   :0.09464       Mean   :0.7531        Mean   :-0.2565      
    ##  3rd Qu.:0.10000       3rd Qu.:1.0000        3rd Qu.:-0.1833      
    ##  Max.   :0.70000       Max.   :1.0000        Max.   : 0.0000      
    ##  min_negative_polarity max_negative_polarity title_subjectivity
    ##  Min.   :-1.0000       Min.   :-1.0000       Min.   :0.0000    
    ##  1st Qu.:-0.7000       1st Qu.:-0.1250       1st Qu.:0.0000    
    ##  Median :-0.5000       Median :-0.1000       Median :0.1000    
    ##  Mean   :-0.5139       Mean   :-0.1076       Mean   :0.2797    
    ##  3rd Qu.:-0.3000       3rd Qu.:-0.0500       3rd Qu.:0.5000    
    ##  Max.   : 0.0000       Max.   : 0.0000       Max.   :1.0000    
    ##  title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :-1.00000         Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.: 0.00000         1st Qu.:0.1667         1st Qu.:0.0000              
    ##  Median : 0.00000         Median :0.5000         Median :0.0000              
    ##  Mean   : 0.07324         Mean   :0.3464         Mean   :0.1547              
    ##  3rd Qu.: 0.13636         3rd Qu.:0.5000         3rd Qu.:0.2500              
    ##  Max.   : 1.00000         Max.   :0.5000         Max.   :1.0000              
    ##      shares      
    ##  Min.   :    42  
    ##  1st Qu.:   897  
    ##  Median :  1300  
    ##  Mean   :  3202  
    ##  3rd Qu.:  2500  
    ##  Max.   :441000

Indeed, it can be verified that the data fit these criteria, so no
further data cleaning is required. The data is not the same from one day
of week to another, but certain observations hold across all.

First, let us examine the distribution of the target variable `shares`.

``` r
g <- ggplot(data = newsDataFiltered, aes(x = shares))
g + geom_histogram(bins = 50) +
  labs(title = paste0("Histogram of Shares for ", dayOfWeek),
       x = "Shares", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](TuesdayAnalysis_files/figure-gfm/shares%20histogram-1.png)<!-- -->

Values of `shares` appear to follow a power-law distribution, having a
long tail consisting of a few very popular articles that receive an
outsized share of views. Therefore for regression it may be better to
transform the target variable using a logarithmic function.

``` r
newsDataFiltered <- newsDataFiltered %>%
  mutate(shares = log10(shares))

g <- ggplot(data = newsDataFiltered, aes(x = shares))
g + geom_histogram(bins = 50) +
  labs(title = paste0("Histogram of Log-Adjusted Shares for ", dayOfWeek),
       x = "Log10 of Shares", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](TuesdayAnalysis_files/figure-gfm/target%20transformation-1.png)<!-- -->

The log-adjusted distribution shows that news articles with middling
scores are most frequent, and the frequency tapers off as the shares go
to the extremes on either side. With this distribution for the target
variable, a regression model (rather than classification model with two
categories) would be appropriate.

From the summary of the data frame, we observe that the range of values
for the predictor columns vary: some columns such as
`global_subjectivity` are proportions and are limited to continuous
values between 0 and 1, some indicator values like
`data_channel_is_world` have only integer values 0 or 1, and others like
`n_tokens_content` are raw counts which are natural numbers with no
theoretical upper bound.

``` r
g <- ggplot(data = newsDataFiltered, aes(x = global_subjectivity))
g + geom_histogram(bins = 10) +
  labs(title = paste0("Log-Adjusted Histogram of global_subjectivity for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](TuesdayAnalysis_files/figure-gfm/predictor%20histograms-1.png)<!-- -->

``` r
g <- ggplot(data = newsDataFiltered, aes(x = data_channel_is_world))
g + geom_histogram(bins = 2) +
  labs(title = paste0("Log-Adjusted Histogram of data_channel_is_world for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](TuesdayAnalysis_files/figure-gfm/predictor%20histograms-2.png)<!-- -->

``` r
g <- ggplot(data = newsDataFiltered, aes(x = n_tokens_content))
g + geom_histogram(bins = 50) +
  labs(title = paste0("Log-Adjusted Histogram of n_tokens_content for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](TuesdayAnalysis_files/figure-gfm/predictor%20histograms-3.png)<!-- -->

Because of this, when selecting for models, it is advised to standardize
(that is, center and scale) all predictor values prior to fitting the
models. With this many predictor variables it would be difficult to
tease out strong univariate relationships between a single predictor and
the target variable, regardless of other predictor values. Therefore it
is advisible to not exclude any predictor variable from the models. And
while there is a chance of overfitting, especially for the nonlinear
model, if tuning parameters and methodology are well-chosen, this risk
may be minimized.

As a last step before modeling, I split the news data into training and
testing sets in a 70-to-30 proportion.

``` r
set.seed(seed)
train <- sample(1:nrow(newsDataFiltered), size = nrow(newsDataFiltered)*0.7)
test <- dplyr::setdiff(1:nrow(newsDataFiltered), train)
newsDataTrain <- newsDataFiltered[train, ]
newsDataTest <- newsDataFiltered[test, ]
```

## Modeling and Cross-Validation

Two regression models are fitted to the training data: a multiple linear
regression model and a nonlinear Random Forest regression model. To tune
these models, I perform three repeated 5-fold cross-validations from the
`caret` package. Cross-validation is used to limit overfitting on the
training data, because in each CV one of the folds is held out as a
validation set while the remaining folds are combined and used to train.
Besides centering and scaling, the linear regression model uses the
package default options, while for the Random Forest model, I specified
tree depths to be 3, 5 or 10, because deeper trees can be prone to
overfitting and take up a lot of computation time.

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
                tuneGrid = expand.grid(mtry = c(3, 5, 10))
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
lm_rmspe <- sqrt(mean((newsDataTest$shares - newsLmPred)^2))
lm_rmspe
```

    ## [1] 0.3832652

``` r
newsRfPred <- predict(rf_fit, newdata = newsDataTest)
rf_rmspe <- sqrt(mean((newsDataTest$shares - newsRfPred)^2))
rf_rmspe
```

    ## [1] 0.3799177

Here we compare the root mean-square prediction error on the log-scaled
target variable `shares`: the multiple linear regression model has RMSPE
of 0.38327 and the Random Forest regression model has RMSPE of 0.37992.
The model with lower RMSPE is the better performer on the testing data
set.

To test whether the models could be overfit for the training data, we
compare the root mean-square prediction error against the root
mean-square error from the training data set. For the linear regression
model the training RMSE is 2.45826 and for the Random Forest model the
RMSE is 0.37404. For the models to not be overfit, each model’s training
RMSE and testing RMSPE values ought to be close to each other.
