# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

from ingest_data import ingest
from clean_data import where_selection, lin_normalize, class_normalize
from generate_features import corr_feature_selection, lasso_feature_selection

#The objective of our study is to analyze the relationships between suicides
#and other socioeconomic factors accross different countries in Europe (continent)
#To keep the study in a reasonable scope, we'll study the 10 most populated
#regions in Europe

#read_data
raw = ingest("../data/master.csv")

################################################################################
############################# DATA EXPLORATION #################################
################################################################################
# we can start by observing the evolution of the data along time
year_evo = raw.groupby('year')['suicides/100k pop'].mean().reset_index()
sns.set_theme()
plt.plot(year_evo['year'],year_evo['suicides/100k pop'])
plt.title("Suicide rate distribution over time, 1985-2016")
plt.show()

#we can also plot a histogram with the age group and the average rate of suicide
#we first need to make an adjustment to the '5-14' value, as it does not appear
#correctly in the barplot
raw['age'] = raw['age'].apply(lambda x: '05-14 years' if x=='5-14 years' else x)
age_group_hist = raw.groupby('age')['suicides/100k pop'].mean().reset_index()
sns.set_theme()
sns.barplot(data=age_group_hist,x='age',y='suicides/100k pop')
plt.title("Suicide rate distribution over age, 1985-2016")
plt.show()

#it might also be interesting to have this representation separated by sex
male_rates = raw[raw['sex']=='male'].groupby('age')['suicides/100k pop'].mean().reset_index()
fema_rates = raw[raw['sex']=='female'].groupby('age')['suicides/100k pop'].mean().reset_index()
male_rates = male_rates.rename(columns={'suicides/100k pop':'male suicides/100k pop'})
male_rates['male suicides/100k pop'] = male_rates['male suicides/100k pop']*-1.0
fema_rates = fema_rates.rename(columns={'suicides/100k pop':'female suicides/100k pop'})
joined_age_sex = male_rates.merge(right=fema_rates,on='age',how='left')
ages = ['75+ years','55-74 years','35-54 years','25-34 years','15-24 years','05-14 years']
ax1 = sns.barplot(x='male suicides/100k pop', y='age', data=joined_age_sex, order=ages, color="blue")
ax2 = sns.barplot(x='female suicides/100k pop', y='age', data=joined_age_sex, order=ages, color="green")
plt.title("Suicide rate distribution over age and sex, 1985-2016")
plt.xlabel("Male (suicides/100k pop) Female")
plt.grid()
plt.xticks(ticks=[-40,-30,-20,-10,0,10,20,30,40],
labels=['40','30','20','10','0','10','20','30','40'])
plt.show()

#we can also chart the correlation matrix
corr_matrix = raw.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Initial correlation matrix")
plt.show()
#since most of our data isn't yet mapped to numbers, we can't really visualize
#it in our correlation matrix. We'll have to wait a little bit more before
#noticing relationships between variables.

################################################################################
############################ CLEANING / TRIMMING ###############################
################################################################################
# we first trim the dataframe to only our countries of interest.
eur_list = ['Russian Federation','Germany','United Kingdom','France','Italy','Spain','Ukraine','Poland','Romania','Netherlands']
#this reduces the dataframe from 27820 rows to only 3452 rowss
raw = where_selection(df=raw,column='country',vlist=eur_list,include=True)

#our dependent variable will be the rate of suicides. we'll now determine if any
#of the columns has any null, and we'll work with the most complete ones
s = np.sum(raw['suicides_no'].apply(lambda x : 0 if pd.isna(x) else 1))
t = np.sum(raw['population'].apply(lambda x : 0 if pd.isna(x) else 1))
u = np.sum(raw['suicides/100k pop'].apply(lambda x : 0 if pd.isna(x) else 1))
#we can see that all these columns have 100% of data.
#we will work with the relative column (suicides/100k pop), which I believe
#will better represent the relationships with the other variables

#we should now clean up our data. we will start by removing redundant /
#unnecessary columns from the dataset
#we will remove the following columns:
# 7: 'country-year', completely redundant with 'country' and 'year'
# 11: 'generation', as we already have 'age' and 'year'.
raw = raw.drop(columns=['country-year','generation'])

################################################################################
############################### NORMALIZATION ##################################
################################################################################
#we will now replace our string data with "class" values, and normalize the
#numerical data
#first, we make a deep copy of our dataframe, to keep the clean data untouched
wdf = raw.copy()

#we will use the lin_normalize function to normalize the following columns:
#'year','suicides_no','population','suicides/100k pop','HDI for year',
#' gdp_for_year ($) ','gdp_per_capita ($)'
cols_to_norm = ['year','suicides_no','population','suicides/100k pop', \
                'HDI for year',' gdp_for_year ($) ','gdp_per_capita ($)']
#first, unfortunately, we have to reformat a column due it's comma formatting
wdf[' gdp_for_year ($) '] = wdf[' gdp_for_year ($) '].apply( \
                            lambda x: float(x.replace(",","")))
#now we can apply the linear normalization
wdf = lin_normalize(df=wdf,columns=cols_to_norm)

#regarding the sex, we can just use the class_normalize function
wdf = class_normalize(df=wdf,column='sex',mapping_order=['female','male'])

#we can do the same for the age groups
age_mapping = ['05-14 years','15-24 years','25-34 years','35-54 years',\
               '55-74 years','75+ years']
#after that we'll just normalize with the class_normalize function
wdf = class_normalize(df=wdf,column='age',mapping_order=age_mapping)
print(wdf)

################################################################################
############################## FEATURE SELECTION ###############################
################################################################################
# we can start by calculating the correlation matrix again
corr_matrix = wdf.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation matrix with normalized and filtered data")
plt.show()
#we can appreciate here that both sex and age are strongly positively correlated
#with higher suicide rates. we also observe a decent negative correlation
#between the gpd per capita and the rate of suicides, same with the HDI.

#let's now check what features are meaningful following a t-test
cfs = corr_feature_selection(df=wdf,corr=corr_matrix,target='suicides/100k pop',
                             threshold=0.005)
print(cfs)
#as we can see, all features appear to be meaningful according to the test.

#Let's now try a Lasso feature classification
X = wdf.drop(columns=['country','HDI for year'])
features = lasso_feature_selection(df=X,target='suicides/100k pop', \
                                   alph=np.arange(0.005,1,0.005))

#we can now try with other methods from scikitlearn
#let's start by removing low-variance features
selector = VarianceThreshold(threshold=(0.75*(1-0.75)))
#we'll remove our target variable, and also the country (as it would not make
#sense to convert it to a number).
X = wdf.drop(columns=['suicides/100k pop','country'])
selector.fit_transform(X)
print(X.columns)
#we still get the same columns. They all have a decent variance

#we can also try an RFE feature selection
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
X = wdf.drop(columns=['country','HDI for year','suicides/100k pop'])
y = wdf['suicides/100k pop']
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
selected = [X.columns[i] for i in range(len(X.columns)) if selector.support_[i]]
discard = [X.columns[i] for i in range(len(X.columns)) if not selector.support_[i]]
print("Selected",selected)
print("Discarded",discard)

#Let's now take a bit of a different approach on studying the HDI. We've
#seen that it contains nulls, but we still don't know how many. Is it even worth
#considering?
#check nans per country
conts = list(set(list(wdf['country'])))
nonan = []
for c in conts:
    parted = wdf[wdf['country']==c]['HDI for year']
    parted = parted.apply(lambda x: 1 if pd.notna(x) else 0)
    nonan.append(sum(parted)/len(parted))
tdf = pd.DataFrame([conts,nonan]).transpose()
tdf = tdf.rename(columns={0:'country',1:'no-null ratio'})
sns.barplot(data=tdf,x='country',y='no-null ratio')

#same for year.
years = set(raw['year'])
nonan = []
for c in years:
    parted = raw[raw['year']==c]['HDI for year']
    parted = parted.apply(lambda x: 1 if pd.notna(x) else 0)
    nonan.append(sum(parted)/len(parted))
print(years)
tdf = pd.DataFrame([years,nonan]).transpose()
tdf = tdf.rename(columns={0:'year',1:'no-null ratio'})
sns.lineplot(data=tdf,x='year',y='no-null ratio')

#We won't be considering the HDI for the next studies

################################################################################
############################### MODEL LEARNING #################################
################################################################################

wdf = wdf.drop(columns=['suicides_no','HDI for year',' gdp_for_year ($) '])
X = wdf.drop(columns = ['suicides/100k pop','country'])
Y = wdf['suicides/100k pop']

##### Multi linear model
#we first need to import the necessary functions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \
                                                    random_state=1)

model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print("MSE : ", mean_squared_error(y_test, predictions))
print("MAE : ", mean_absolute_error(y_test, predictions))
r_squared = model.score(X_test, y_test)
print("R2 : ", r_squared)

##### Ridge - Tikhonov regularization
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \
                                                    random_state=12)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
#alpha adjustment
alphas = np.arange(0.001,100,0.1)
rs = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    r_squared = model.score(X_test, y_test)
    rs.append(r_squared)
plt.plot(alphas,rs) #we'll get alpha=1 as it is good enough

##### Lasso Regression
from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
model = linear_model.Lasso(alpha=0.001)
model.fit(X_train, y_train)
print("R^2",model.score(X_test, y_test))
print(list(X_train.columns))
print(model.coef_)
print(model.intercept_)

#coefficient optimization
coefs = []
rsqua = []
alphs = np.arange(0.06,0.001,-0.0005)
for alpha in alphs:
    model = linear_model.Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    coefs.append(model.coef_)
    rsqua.append(model.score(X_test, y_test))
#transpose coefficients
coefs = [[row[i] for row in coefs] for i in range(len(coefs[0]))]

plt.plot(alphs,rsqua)
plt.show()

#plot coefficient evolution
plt.plot(alphs,coefs[0],label='year')
plt.plot(alphs,coefs[1],label='sex')
plt.plot(alphs,coefs[2],label='age')
plt.plot(alphs,coefs[3],label='population')
plt.plot(alphs,coefs[4],label='gpd per capita')
plt.legend()
plt.show()

#find the best alpha:
myi = np.argmax(rsqua)
print("best R^2:", rsqua[myi])
best_alpha = alphs[myi]
#transpose again
coefs = [[row[i] for row in coefs] for i in range(len(coefs[0]))]
best_coefs = coefs[myi]
print(best_coefs)
# R^2 is still low, around 0.50-0.55. How can we make this analysis better?

##### Country-isolated analysis
rsqua = []
conts = []
for country in list(set(list(wdf['country']))):
    Xi = wdf[wdf['country']==country]
    X = Xi.drop(columns = ['country','suicides/100k pop'])
    y = Xi['suicides/100k pop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \
                                                        random_state=12)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    r_squared = model.score(X_test, y_test)
    conts.append(country)
    rsqua.append(r_squared)
print(conts)
print(rsqua)
print("Average R^2:",np.mean(rsqua),"STD:",np.std(rsqua))
# New R^2 of 0.76! We now have much more conffidence in our model's predictions.
