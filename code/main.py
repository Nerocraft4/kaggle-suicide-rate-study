# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#The objective of our study is to analyze the relationships between suicides
#and other socioeconomic factors accross different countries in Europe (continent)
#To keep the study in a reasonable scope, we'll study the 10 most populated
#regions in Europe

#read_data
raw = pd.read_csv("../data/master.csv")

################################################################################
############################ CLEANING / TRIMMING ###############################
################################################################################
# we first trim the dataframe to only our countries of interest.
eur_list = ['Russian Federation','Germany','United Kingdom','France','Italy','Spain','Ukraine','Poland','Romania','Netherlands']
#this reduces the dataframe from 27820 rows to only 3452 rowss
raw = raw[raw['country'].isin(eur_list)]

#our dependent variable will be the nº of suicides. we'll now determine if any
#of the columns has any null, and we'll work with the most complete ones
s = np.sum(raw['suicides_no'].apply(lambda x : 0 if pd.isna(x) else 1))
t = np.sum(raw['population'].apply(lambda x : 0 if pd.isna(x) else 1))
u = np.sum(raw['suicides/100k pop'].apply(lambda x : 0 if pd.isna(x) else 1))
#we can see that all these columns have 100% of data.
print(len(raw['country']),s,t,u)
#we will work with the relative column (suicides/100k pop), which I believe
#will better represent the relationships with the other variables

#we should now clean up our data. we will start by removing redundant / unnecessary
#columns from the dataset
cs = list(raw.columns)
#we will remove the following columns:
# 4: 'suicides_no', as we already have that data in 'suicides/100k pop'
# 5: 'population', as we will be working with values relative to population, this is no longer needed
# 7: 'country-year', completely redundant with 'country' and 'year'
# 9: ' gdp_ for_year ($) ', as we will be working with values relative to population
# 11: 'generation', as we already have 'age' and 'year'.
raw = raw.drop(columns=[cs[4],cs[5],cs[7],cs[9],cs[11]])

#as we've also detected a high number of NaNs for the HDI column, we'll now check
#if it's missing evenly between countries or if some countries just don't have
#any data of this metric
checkHDI = raw[['country','HDI for year']]
checkHDI['HDI for year'] = checkHDI['HDI for year'].apply(lambda x : 0 if pd.isna(x) else 1)
checkHDI = checkHDI.groupby('country')['HDI for year'].agg(['sum','count'])
checkHDI['completeness'] = checkHDI['sum']/checkHDI['count']
#print(checkHDI.sort_values(by='completeness',ascending=False))
#we can see that the most complete is Poland with only a 37.5% of records full
#we'll be dropping this metric too.
clean = raw.drop(columns=['HDI for year'])

################################################################################
############################### NORMALIZATION ##################################
################################################################################
#we will now replace our string data with "class" values, and normalize the numerical data
print(clean.head(3))
#first, we make a deep copy of our dataframe, to keep the clean data untouched
wdf = clean.copy()

#with the year we can just subtract the minimum year from all of our entries and then normalize
mini_year = np.min(wdf['year'])
maxi_mod_year = np.max(wdf['year'])-mini_year
print(mini_year)
wdf['year'] = (wdf['year']-mini_year)/maxi_mod_year
print(wdf.head(100))

#regarding the sex, we can just map it to female -> 0, male -> 1
wdf['sex'] = wdf['sex'].apply(lambda x: 0 if x=='female' else 1)
print(wdf.head(100))

#for the age column, we'll map the groups as follows:
age_mapping = {'5-14':10,'15-24':20,'25-34':30,'35-54':45,'55-74':65,'75+':80}
#after that we'll just normalize dividing everything by 80
wdf['age'] = wdf['age'].apply(lambda x: age_mapping[x.split()[0]]/80)
print(wdf.head(100))

#with the suicide rate, we will just divide by the max
maxi_rate = np.max(wdf['suicides/100k pop'])
wdf['suicides/100k pop'] = wdf['suicides/100k pop']/maxi_rate
print(wdf.head(100))

#we will do the same with the gdp per capita
maxi_gdp = np.max(wdf['gdp_per_capita ($)'])
wdf['gdp_per_capita ($)'] = wdf['gdp_per_capita ($)']/maxi_gdp
print(wdf.head(100))

################################################################################
################################################################################
################################################################################
