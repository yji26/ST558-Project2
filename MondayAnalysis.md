ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Data Set Information](#data-set-information)
  - [Filter for Monday Data](#filter-for-monday-data)

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

## Filter for Monday Data

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
