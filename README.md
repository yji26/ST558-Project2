ST 558 - Summer 2020 - Project 2
================
Yun Ji
7/3/2020

  - [Introduction](#introduction)
  - [Build Automation](#build-automation)
  - [Day of Week Models](#day-of-week-models)

## Introduction

The purpose of this vignette is to compare the prediction accuracy of
two types of regression models on a provided data set. The data used is
an online new popularity data set for web articles published by Mashable
over a two-year period. The data description and download links can be
[found
here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#).

## Build Automation

I use the following code to generate data and fitted models for each day
of the week.

``` r
days <- c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
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

## Day of Week Models

The Markdown files for each day of the week are linked below.

  - [Monday Models](MondayAnalysis.md)

  - [Tuesday Models](TuesdayAnalysis.md)

  - [Wednesday Models](WednesdayAnalysis.md)

  - [Thursday Models](ThursdayAnalysis.md)

  - [Friday Models](FridayAnalysis.md)

  - [Saturday Models](SaturdayAnalysis.md)

  - [Sunday Models](SundayAnalysis.md)
