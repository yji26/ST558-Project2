ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Data Set Information](#data-set-information)
  - [Filter for Sunday Data](#filter-for-sunday-data)
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

## Filter for Sunday Data

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
    ##  Min.   : 3.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.: 9.00   1st Qu.: 247.0   1st Qu.:0.4606   1st Qu.:1.0000  
    ##  Median :10.00   Median : 463.0   Median :0.5249   Median :1.0000  
    ##  Mean   :10.45   Mean   : 609.5   Mean   :0.5249   Mean   :0.9708  
    ##  3rd Qu.:12.00   3rd Qu.: 823.0   3rd Qu.:0.6054   3rd Qu.:1.0000  
    ##  Max.   :19.00   Max.   :8474.0   Max.   :1.0000   Max.   :1.0000  
    ##  n_non_stop_unique_tokens   num_hrefs      num_self_hrefs      num_imgs      
    ##  Min.   :0.0000           Min.   :  0.00   Min.   : 0.000   Min.   :  0.000  
    ##  1st Qu.:0.6125           1st Qu.:  5.00   1st Qu.: 1.000   1st Qu.:  1.000  
    ##  Median :0.6774           Median :  9.00   Median : 2.000   Median :  1.000  
    ##  Mean   :0.6632           Mean   : 12.72   Mean   : 3.597   Mean   :  5.866  
    ##  3rd Qu.:0.7512           3rd Qu.: 17.00   3rd Qu.: 4.000   3rd Qu.:  9.000  
    ##  Max.   :1.0000           Max.   :153.00   Max.   :40.000   Max.   :128.000  
    ##    num_videos    average_token_length  num_keywords   
    ##  Min.   : 0.00   Min.   :0.000        Min.   : 1.000  
    ##  1st Qu.: 0.00   1st Qu.:4.485        1st Qu.: 6.000  
    ##  Median : 0.00   Median :4.677        Median : 8.000  
    ##  Mean   : 1.03   Mean   :4.575        Mean   : 7.636  
    ##  3rd Qu.: 1.00   3rd Qu.:4.882        3rd Qu.: 9.000  
    ##  Max.   :74.00   Max.   :7.218        Max.   :10.000  
    ##  data_channel_is_lifestyle data_channel_is_entertainment data_channel_is_bus
    ##  Min.   :0.00000           Min.   :0.0000                Min.   :0.0000     
    ##  1st Qu.:0.00000           1st Qu.:0.0000                1st Qu.:0.0000     
    ##  Median :0.00000           Median :0.0000                Median :0.0000     
    ##  Mean   :0.07673           Mean   :0.1958                Mean   :0.1253     
    ##  3rd Qu.:0.00000           3rd Qu.:0.0000                3rd Qu.:0.0000     
    ##  Max.   :1.00000           Max.   :1.0000                Max.   :1.0000     
    ##  data_channel_is_socmed data_channel_is_tech data_channel_is_world
    ##  Min.   :0.00000        Min.   :0.0000       Min.   :0.0000       
    ##  1st Qu.:0.00000        1st Qu.:0.0000       1st Qu.:0.0000       
    ##  Median :0.00000        Median :0.0000       Median :0.0000       
    ##  Mean   :0.05005        Mean   :0.1447       Mean   :0.2072       
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000       3rd Qu.:0.0000       
    ##  Max.   :1.00000        Max.   :1.0000       Max.   :1.0000       
    ##    kw_min_min       kw_max_min      kw_avg_min        kw_min_max    
    ##  Min.   : -1.00   Min.   :    0   Min.   :   -1.0   Min.   :     0  
    ##  1st Qu.: -1.00   1st Qu.:  475   1st Qu.:  156.0   1st Qu.:     0  
    ##  Median : -1.00   Median :  690   Median :  241.3   Median :  1900  
    ##  Mean   : 27.93   Mean   : 1103   Mean   :  314.5   Mean   : 12426  
    ##  3rd Qu.:  4.00   3rd Qu.: 1100   3rd Qu.:  375.4   3rd Qu.:  8900  
    ##  Max.   :217.00   Max.   :81200   Max.   :27123.0   Max.   :843300  
    ##    kw_max_max       kw_avg_max       kw_min_avg     kw_max_avg    
    ##  Min.   : 37400   Min.   :  7328   Min.   :   0   Min.   :  2536  
    ##  1st Qu.:843300   1st Qu.:170694   1st Qu.:   0   1st Qu.:  3609  
    ##  Median :843300   Median :233600   Median :1200   Median :  4747  
    ##  Mean   :754254   Mean   :244689   Mean   :1228   Mean   :  5946  
    ##  3rd Qu.:843300   3rd Qu.:308200   3rd Qu.:2167   3rd Qu.:  6782  
    ##  Max.   :843300   Max.   :843300   Max.   :3600   Max.   :120100  
    ##    kw_avg_avg      self_reference_min_shares self_reference_max_shares
    ##  Min.   :  743.5   Min.   :     0            Min.   :     0           
    ##  1st Qu.: 2497.6   1st Qu.:   660            1st Qu.:  1100           
    ##  Median : 3044.0   Median :  1300            Median :  2700           
    ##  Mean   : 3280.9   Mean   :  4339            Mean   : 10122           
    ##  3rd Qu.: 3832.7   3rd Qu.:  2700            3rd Qu.:  7800           
    ##  Max.   :15336.1   Max.   :843300            Max.   :843300           
    ##  self_reference_avg_sharess     LDA_00            LDA_01       
    ##  Min.   :     0.0           Min.   :0.01824   Min.   :0.01820  
    ##  1st Qu.:   990.3           1st Qu.:0.02500   1st Qu.:0.02500  
    ##  Median :  2192.0           Median :0.03333   Median :0.03334  
    ##  Mean   :  6250.4           Mean   :0.16566   Mean   :0.15900  
    ##  3rd Qu.:  4971.0           3rd Qu.:0.19477   3rd Qu.:0.18272  
    ##  Max.   :843300.0           Max.   :0.92000   Max.   :0.92595  
    ##      LDA_02            LDA_03            LDA_04        global_subjectivity
    ##  Min.   :0.01818   Min.   :0.01818   Min.   :0.01945   Min.   :0.0000     
    ##  1st Qu.:0.02500   1st Qu.:0.02562   1st Qu.:0.02531   1st Qu.:0.3988     
    ##  Median :0.03335   Median :0.05000   Median :0.04000   Median :0.4605     
    ##  Mean   :0.20334   Mean   :0.26067   Mean   :0.21132   Mean   :0.4487     
    ##  3rd Qu.:0.31304   3rd Qu.:0.49577   3rd Qu.:0.32403   3rd Qu.:0.5148     
    ##  Max.   :0.92000   Max.   :0.91998   Max.   :0.92644   Max.   :0.9125     
    ##  global_sentiment_polarity global_rate_positive_words
    ##  Min.   :-0.37393          Min.   :0.00000           
    ##  1st Qu.: 0.05961          1st Qu.:0.02829           
    ##  Median : 0.12213          Median :0.03991           
    ##  Mean   : 0.12476          Mean   :0.04120           
    ##  3rd Qu.: 0.18740          3rd Qu.:0.05310           
    ##  Max.   : 0.65500          Max.   :0.15217           
    ##  global_rate_negative_words rate_positive_words avg_positive_polarity
    ##  Min.   :0.00000            Min.   :0.0000      Min.   :0.0000       
    ##  1st Qu.:0.01002            1st Qu.:0.6029      1st Qu.:0.3100       
    ##  Median :0.01592            Median :0.7059      Median :0.3648       
    ##  Mean   :0.01691            Mean   :0.6809      Mean   :0.3635       
    ##  3rd Qu.:0.02235            3rd Qu.:0.8000      3rd Qu.:0.4240       
    ##  Max.   :0.10112            Max.   :1.0000      Max.   :1.0000       
    ##  min_positive_polarity max_positive_polarity avg_negative_polarity
    ##  Min.   :0.00000       Min.   :0.0000        Min.   :-1.0000      
    ##  1st Qu.:0.05000       1st Qu.:0.6000        1st Qu.:-0.3336      
    ##  Median :0.10000       Median :0.8000        Median :-0.2625      
    ##  Mean   :0.09683       Mean   :0.7805        Mean   :-0.2686      
    ##  3rd Qu.:0.10000       3rd Qu.:1.0000        3rd Qu.:-0.1990      
    ##  Max.   :1.00000       Max.   :1.0000        Max.   : 0.0000      
    ##  min_negative_polarity max_negative_polarity title_subjectivity
    ##  Min.   :-1.0000       Min.   :-1.0000       Min.   :0.0000    
    ##  1st Qu.:-0.7500       1st Qu.:-0.1250       1st Qu.:0.0000    
    ##  Median :-0.5000       Median :-0.1000       Median :0.2500    
    ##  Mean   :-0.5465       Mean   :-0.1087       Mean   :0.3116    
    ##  3rd Qu.:-0.3333       3rd Qu.:-0.0500       3rd Qu.:0.5000    
    ##  Max.   : 0.0000       Max.   : 0.0000       Max.   :1.0000    
    ##  title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :-1.00000         Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.: 0.00000         1st Qu.:0.1333         1st Qu.:0.0000              
    ##  Median : 0.00000         Median :0.4000         Median :0.1000              
    ##  Mean   : 0.09006         Mean   :0.3227         Mean   :0.1842              
    ##  3rd Qu.: 0.25000         3rd Qu.:0.5000         3rd Qu.:0.3000              
    ##  Max.   : 1.00000         Max.   :0.5000         Max.   :1.0000              
    ##      shares     
    ##  Min.   :   89  
    ##  1st Qu.: 1200  
    ##  Median : 1900  
    ##  Mean   : 3747  
    ##  3rd Qu.: 3700  
    ##  Max.   :83300

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

![](SundayAnalysis_files/figure-gfm/shares%20histogram-1.png)<!-- -->

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

![](SundayAnalysis_files/figure-gfm/target%20transformation-1.png)<!-- -->

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

![](SundayAnalysis_files/figure-gfm/predictor%20histograms-1.png)<!-- -->

``` r
g <- ggplot(data = newsDataFiltered, aes(x = data_channel_is_world))
g + geom_histogram(bins = 2) +
  labs(title = paste0("Log-Adjusted Histogram of data_channel_is_world for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](SundayAnalysis_files/figure-gfm/predictor%20histograms-2.png)<!-- -->

``` r
g <- ggplot(data = newsDataFiltered, aes(x = n_tokens_content))
g + geom_histogram(bins = 50) +
  labs(title = paste0("Log-Adjusted Histogram of n_tokens_content for ", dayOfWeek),
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](SundayAnalysis_files/figure-gfm/predictor%20histograms-3.png)<!-- -->

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

    ##   intercept      RMSE   Rsquared      MAE      RMSESD RsquaredSD       MAESD
    ## 1      TRUE 0.3622223 0.08505897 0.274708 0.009252876 0.02325228 0.007523536

``` r
rf_fit$results
```

    ##   mtry      RMSE  Rsquared       MAE     RMSESD RsquaredSD       MAESD
    ## 1    3 0.3546061 0.1172863 0.2717835 0.01269956 0.02409393 0.006626575
    ## 2    5 0.3544731 0.1166019 0.2716618 0.01261920 0.02053312 0.006596367
    ## 3   10 0.3554614 0.1111321 0.2719446 0.01201701 0.01958974 0.006281097

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

    ## [1] 0.3479335

``` r
newsRfPred <- predict(rf_fit, newdata = newsDataTest)
rf_rmspe <- sqrt(mean((newsDataTest$shares - newsRfPred)^2))
rf_rmspe
```

    ## [1] 0.3414611

Here we compare the root mean-square prediction error on the log-scaled
target variable `shares`: the multiple linear regression model has RMSPE
of 0.34793 and the Random Forest regression model has RMSPE of 0.34146.
The model with lower RMSPE is the better performer on the testing data
set.

To test whether the models could be overfit for the training data, we
compare the root mean-square prediction error against the root
mean-square error from the training data set. For the linear regression
model the training RMSE is 0.36222 and for the Random Forest model the
RMSE is 0.35447. For the models to not be overfit, each model’s training
RMSE and testing RMSPE values ought to be close to each other.
