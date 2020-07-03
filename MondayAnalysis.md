ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Data Set Information](#data-set-information)
  - [Filter for Monday Data](#filter-for-monday-data)
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

## Filter for Monday Data

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

    ##  n_tokens_title  n_tokens_content n_unique_tokens  n_non_stop_words
    ##  Min.   : 2.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.: 9.00   1st Qu.: 249.0   1st Qu.:0.4742   1st Qu.:1.0000  
    ##  Median :10.00   Median : 400.0   Median :0.5420   Median :1.0000  
    ##  Mean   :10.42   Mean   : 543.9   Mean   :0.5314   Mean   :0.9715  
    ##  3rd Qu.:12.00   3rd Qu.: 720.0   3rd Qu.:0.6087   3rd Qu.:1.0000  
    ##  Max.   :18.00   Max.   :7764.0   Max.   :1.0000   Max.   :1.0000  
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs      num_imgs     
    ##  Min.   :0.0000           Min.   :  0.00   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.:0.6289           1st Qu.:  4.00   1st Qu.: 1.000   1st Qu.: 1.000  
    ##  Median :0.6923           Median :  7.00   Median : 3.000   Median : 1.000  
    ##  Mean   :0.6743           Mean   : 10.74   Mean   : 3.398   Mean   : 4.447  
    ##  3rd Qu.:0.7551           3rd Qu.: 13.00   3rd Qu.: 4.000   3rd Qu.: 3.000  
    ##  Max.   :1.0000           Max.   :162.00   Max.   :51.000   Max.   :93.000  
    ##    num_videos     average_token_length  num_keywords   
    ##  Min.   : 0.000   Min.   :0.000        Min.   : 1.000  
    ##  1st Qu.: 0.000   1st Qu.:4.479        1st Qu.: 6.000  
    ##  Median : 0.000   Median :4.656        Median : 7.000  
    ##  Mean   : 1.336   Mean   :4.547        Mean   : 7.141  
    ##  3rd Qu.: 1.000   3rd Qu.:4.841        3rd Qu.: 9.000  
    ##  Max.   :75.000   Max.   :8.042        Max.   :10.000  
    ##  data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   :0.04834           Mean   :0.2039                Mean   :0.1731     
    ##  3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000       
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000       
    ##  Median :0.00000        Median :0.0000       Median :0.0000       
    ##  Mean   :0.05059        Mean   :0.1854       Mean   :0.2036       
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000       
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000       
    ##    kw_min_min       kw_max_min       kw_avg_min        kw_min_max    
    ##  Min.   : -1.00   Min.   :     0   Min.   :   -1.0   Min.   :     0  
    ##  1st Qu.: -1.00   1st Qu.:   440   1st Qu.:  136.0   1st Qu.:     0  
    ##  Median : -1.00   Median :   648   Median :  229.2   Median :  1300  
    ##  Mean   : 26.28   Mean   :  1240   Mean   :  318.8   Mean   : 11781  
    ##  3rd Qu.:  4.00   3rd Qu.:  1000   3rd Qu.:  354.1   3rd Qu.:  7100  
    ##  Max.   :318.00   Max.   :298400   Max.   :42827.9   Max.   :843300  
    ##    kw_max_max       kw_avg_max       kw_min_avg       kw_max_avg    
    ##  Min.   :     0   Min.   :     0   Min.   :  -1.0   Min.   :     0  
    ##  1st Qu.:843300   1st Qu.:173688   1st Qu.:   0.0   1st Qu.:  3531  
    ##  Median :843300   Median :244636   Median : 975.4   Median :  4273  
    ##  Mean   :748525   Mean   :258792   Mean   :1074.8   Mean   :  5580  
    ##  3rd Qu.:843300   3rd Qu.:332686   3rd Qu.:1978.7   3rd Qu.:  5939  
    ##  Max.   :843300   Max.   :843300   Max.   :3602.1   Max.   :298400  
    ##    kw_avg_avg    self_reference_min_shares self_reference_max_shares
    ##  Min.   :    0   Min.   :     0            Min.   :     0           
    ##  1st Qu.: 2361   1st Qu.:   671            1st Qu.:  1100           
    ##  Median : 2841   Median :  1200            Median :  2900           
    ##  Mean   : 3076   Mean   :  3901            Mean   : 10074           
    ##  3rd Qu.: 3540   3rd Qu.:  2600            3rd Qu.:  8000           
    ##  Max.   :43568   Max.   :690400            Max.   :843300           
    ##  self_reference_avg_sharess     LDA_00            LDA_01       
    ##  Min.   :     0             Min.   :0.01818   Min.   :0.01819  
    ##  1st Qu.:  1014             1st Qu.:0.02520   1st Qu.:0.02504  
    ##  Median :  2200             Median :0.03345   Median :0.03337  
    ##  Mean   :  6351             Mean   :0.18954   Mean   :0.15297  
    ##  3rd Qu.:  5200             3rd Qu.:0.25776   3rd Qu.:0.17249  
    ##  Max.   :690400             Max.   :0.91999   Max.   :0.91997  
    ##      LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.01819   Min.   :0.01819   Min.   :0.01818   Min.   :0.0000     
    ##  1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.02857   1st Qu.:0.3949     
    ##  Median :0.04000   Median :0.04000   Median :0.04005   Median :0.4516     
    ##  Mean   :0.20648   Mean   :0.21816   Mean   :0.23285   Mean   :0.4413     
    ##  3rd Qu.:0.31581   3rd Qu.:0.35101   3rd Qu.:0.40040   3rd Qu.:0.5051     
    ##  Max.   :0.92000   Max.   :0.92653   Max.   :0.92708   Max.   :1.0000     
    ##  global_sentiment_polarity global_rate_positive_words
    ##  Min.   :-0.38021          Min.   :0.00000           
    ##  1st Qu.: 0.05588          1st Qu.:0.02846           
    ##  Median : 0.11813          Median :0.03851           
    ##  Mean   : 0.11721          Mean   :0.03932           
    ##  3rd Qu.: 0.17503          3rd Qu.:0.04985           
    ##  Max.   : 0.57551          Max.   :0.13636           
    ##  global_rate_negative_words rate_positive_words avg_positive_polarity
    ##  Min.   :0.000000           Min.   :0.0000      Min.   :0.0000       
    ##  1st Qu.:0.009749           1st Qu.:0.6000      1st Qu.:0.3056       
    ##  Median :0.015447           Median :0.7083      Median :0.3584       
    ##  Mean   :0.016821           Mean   :0.6806      Mean   :0.3545       
    ##  3rd Qu.:0.021818           3rd Qu.:0.8000      3rd Qu.:0.4121       
    ##  Max.   :0.092160           Max.   :1.0000      Max.   :1.0000       
    ##  min_positive_polarity max_positive_polarity avg_negative_polarity
    ##  Min.   :0.00000       Min.   :0.0000        Min.   :-1.0000      
    ##  1st Qu.:0.05000       1st Qu.:0.6000        1st Qu.:-0.3292      
    ##  Median :0.10000       Median :0.8000        Median :-0.2531      
    ##  Mean   :0.09525       Mean   :0.7603        Mean   :-0.2594      
    ##  3rd Qu.:0.10000       3rd Qu.:1.0000        3rd Qu.:-0.1861      
    ##  Max.   :1.00000       Max.   :1.0000        Max.   : 0.0000      
    ##  min_negative_polarity max_negative_polarity title_subjectivity
    ##  Min.   :-1.000        Min.   :-1.0000       Min.   :0.0000    
    ##  1st Qu.:-0.700        1st Qu.:-0.1250       1st Qu.:0.0000    
    ##  Median :-0.500        Median :-0.1000       Median :0.1000    
    ##  Mean   :-0.521        Mean   :-0.1058       Mean   :0.2754    
    ##  3rd Qu.:-0.300        3rd Qu.:-0.0500       3rd Qu.:0.5000    
    ##  Max.   : 0.000        Max.   : 0.0000       Max.   :1.0000    
    ##  title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :-1.00000         Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.: 0.00000         1st Qu.:0.1500         1st Qu.:0.0000              
    ##  Median : 0.00000         Median :0.5000         Median :0.0000              
    ##  Mean   : 0.06549         Mean   :0.3408         Mean   :0.1509              
    ##  3rd Qu.: 0.13636         3rd Qu.:0.5000         3rd Qu.:0.2500              
    ##  Max.   : 1.00000         Max.   :0.5000         Max.   :1.0000              
    ##      shares      
    ##  Min.   :     1  
    ##  1st Qu.:   919  
    ##  Median :  1400  
    ##  Mean   :  3647  
    ##  3rd Qu.:  2700  
    ##  Max.   :690400

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

![](MondayAnalysis_files/figure-gfm/shares%20histogram-1.png)<!-- -->

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

![](MondayAnalysis_files/figure-gfm/target%20transformation-1.png)<!-- -->

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

![](MondayAnalysis_files/figure-gfm/predictor%20histograms-1.png)<!-- -->

``` r
g <- ggplot(data = newsDataFiltered, aes(x = data_channel_is_world))
g + geom_histogram(bins = 2) +
  labs(title = paste0("Log-Adjusted Histogram of data_channel_is_world for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](MondayAnalysis_files/figure-gfm/predictor%20histograms-2.png)<!-- -->

``` r
g <- ggplot(data = newsDataFiltered, aes(x = n_tokens_content))
g + geom_histogram(bins = 50) +
  labs(title = paste0("Log-Adjusted Histogram of n_tokens_content for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](MondayAnalysis_files/figure-gfm/predictor%20histograms-3.png)<!-- -->

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

    ##   intercept      RMSE Rsquared       MAE     RMSESD RsquaredSD       MAESD
    ## 1      TRUE 0.3959899 0.110715 0.2928533 0.01392034  0.0144339 0.005526694

``` r
rf_fit$results
```

    ##   mtry      RMSE  Rsquared       MAE      RMSESD RsquaredSD       MAESD
    ## 1    3 0.3884975 0.1448962 0.2893699 0.008642456 0.01239923 0.004042702
    ## 2    5 0.3883492 0.1438031 0.2893785 0.008582747 0.01125974 0.004226203
    ## 3   10 0.3886497 0.1419615 0.2894178 0.008602275 0.01169789 0.004130390

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

    ## [1] 0.3964595

``` r
newsRfPred <- predict(rf_fit, newdata = newsDataTest)
rf_rmspe <- sqrt(mean((newsDataTest$shares - newsRfPred)^2))
rf_rmspe
```

    ## [1] 0.3911171

Here we compare the root mean-square prediction error on the log-scaled
target variable `shares`: the multiple linear regression model has RMSPE
of 0.39646 and the Random Forest regression model has RMSPE of 0.39112.
The model with lower RMSPE is the better performer on the testing data
set.

To test whether the models could be overfit for the training data, we
compare the root mean-square prediction error against the root
mean-square error from the training data set. For the linear regression
model the training RMSE is 0.39599 and for the Random Forest model the
RMSE is 0.38835. For the models to not be overfit, each model’s training
RMSE and testing RMSPE values ought to be close to each other.
