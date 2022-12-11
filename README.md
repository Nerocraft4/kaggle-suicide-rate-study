# Kaggle Lab APC UAB 2022-23
### Name: Pau Blasco Roca
### Dataset: Suicide Rates Overview 1985 to 2016 (number 20)
### URL: [kaggle](https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016)

## Abstract:
The dataset consists in a comparison between socio-economic information and suicide rates per year, country, sex, age and other variables.
The dataset sources are the United Nations Development Program (UNDP), the World Bank, and the World Health Organization (WHO).

## Dataset main goals
With this dataset, we would like to understand if there is a connection between socio-economic variables and suicide rates, and how it could be also related to other factors, such as gender, age, gdp per capita... The main goal is to learn more about suicide risk factors, and make predictions based on that.

## Experiments / focus points
We started focusing on visualizing the data and the trends in it at a global / world level. Some of the visualization experiments we did included:
 - Plotting the average suicide rate per 100k in the whole world, against time (year).
 - Plotting a bar plot of the average suicide rate per age group, and seeing if there were any trends or noticeable differences.
 - Plotting a double bar plot, comparing trends in the average suicide rate per 100k against age groups and sex.

## Pre-processing and data cleaning
For this part, and after having superficially explored the data  before, we started by removing redundant and useless columns.
After that, the data was normalized using a min-max linear mapping (the maximum is a 1, the minimum a 0, and anything in between is mapped linearly).
This allowed us to visualize a correlation matrix and decide what features to focus on.

## Feature selection
We first use the pearson correlation matrix alongside a statistic test (t-test) to check if our values are meaningful enough and which ones are the most meaningful to the target.

## Models
Remember to also compare to external models (models created by the community)

## Demo
Explain how to demo

## Conclusion
Talk about our best model and compare it to state of the art / other external models

## Further ideas / investigation

## License
The project has been developed under an MIT license.
