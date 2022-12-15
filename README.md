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
 - We first use the Pearson correlation matrix alongside a statistic test (t-test) to check if our values are meaningful enough and which ones are the most meaningful to the target.
 - We then use a Lasso feature selection, with a range of parameters, to have another approach at our feature importance.
 - After that we try to discard features which might have low variance, with a Scikit Learn method.
 - Finally we try yet another approach with the RFE feature selection model (from Scikit Learn too).

## Models
#### Multi-Linear Regression
After using a MLR with Year, Sex, Age, GDP per capita and Population as independent variables, we've achieved an R^2 of around 0.54, which isn't great, but is strong enough for us to take the results into consideration. We've observed the following:
 - Almost no correlation with the Year
 - Positive correlation with Sex
 - Stronger positive correlation with Age
 - Slightly positive correlation with Population
 - Slight negative correlation with the GDP per capita
#### Ridge-Tikhonov Regularization
We've achieved similar results with the Ridge regularization, with slight differences in the last two coefficients. The correlation with the population seems to be less significant, while the negative correlation with the GDP increases in a 1/100th of a unit. Our R^2 is also of 0.54.
#### Lasso Regression
Concerning the Lasso, we first explored what alpha would fit the model best, only to find that alphas over 0.01 were already ruining our model's performance. We found the best alpha (around 0.002) which gave us an R^2 of around 0.50. Still poor results, which did not seem to increase with any model.
#### Country-isolated study
After observing the mediocre performance by our models, we decided to change the approach: we would study each country individually, taking into account a personalized model for each case. Doing this, and then taking the average efficiency across our models, we bumped up our performance to an R^2 of almost 0.76 (having a maximum of 0.82, and a minimum of 0.70). The STD was only 0.038, which seemed to be just about right.
We have observed some interesting tendencies in some countries:
 - The Russian Federation and Ukraine have suicide rates highly biased by sex, having the highest correlation coefficients for that feature compared to all the other countries.
  - Again, for the Russian Federation, the GDP seems to highly influence the suicide rate, with a decent negative tendency differing a lot from other countries.
  - The Population seems to influence each country in a different way. Poland, for example, has a decent positive influence by the Population variable (more population means higher suicide rate) but the contrary happens with France (less population means higher suicide rate).
  - Age seems to be a positive correlating feature for all countries. The Russian Federation, Ukraine, France and Germany have the highest tendencies, while the UK's coefficients fall short from that.

## Demo
There are several demo python snippets in the demo folder, where the user can find examples on how to use the author-created functions and tools (the ones I created). They can't be run directly, and will need to be run from the code folder (or change the import path, which would work too).

## Conclusion
In conclusion, the method that performed the best is, by far, the country-isolated Regression. After studying the individual correlations per country, we've seen that the socioeconomic factors are indeed related to the suicide rates, but differently for each country. This is why separating the dataset country-wise bumped up our results that much.

Comparing it to similar studies, it has also been [observed](https://www.kaggle.com/code/chingchunyeh/suicide-rates-overview-1985-to-2016) that indeed socioeconomic factors affect differently each country, and that our most significantly correlated variables (sex and age) are also the ones being observed by other users.
Sadly, no other user got into the same depth when it comes to feature selection or model evaluation, so there is no way to compare our R^2 score to the community obtained results. Even so, personally, our R^2 is good enough to render the results meaningful, and I think the approach taken is rigorous enough for the scope of the study.

## Further ideas / investigation
To further understand how the socioeconomic factors affect each country's suicide rate, it would be interesting to study each and every one of them individually along some historical data (economic crisis, pandemics, wars, etc). That would contextualize our data in a way that would allow us (even with low-decent R^2 values) to make stronger assumptions and learn more concrete insights about this topic.

## License
The project has been developed under an MIT license.
