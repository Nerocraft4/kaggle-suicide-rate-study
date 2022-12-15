import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

def corr_feature_selection(df,corr,target,threshold=0.05):
    """
    This function, given a dataframe, its correlation matrix, and the target
    variable, returns what features are more significant for that target using
    a t-test evaluation for the correlation coefficients.

    :param df: A pandas dataframe.
    :param corr: A pandas correlation matrix.
    :param target: The target feature of our dataset.
    :param threshold: p-value to compare the test to. Default at 0.05.
    :return: A dataframe with four columns: feature, r_coef, t-score, accept.
    The last one will be true only if the feature is meaningful (comparing it to
    the threshold).
    """
    lens = [sum(df[col].apply(lambda x: 0 if pd.isna(x) else 1)) \
            for col in corr.columns]
    corr['df'] = lens #degrees of freedom
    #if we don't do it in this order, Panda converts the dataframe to a Series
    ncorr = corr[[target,'df']]
    ncorr['t-score'] = ncorr[target]*np.sqrt(ncorr['df']-2) \
                      /np.sqrt(1-ncorr[target]*ncorr[target])
    #data from
    #https://faculty.washington.edu/heagerty/Books/Biostatistics/TABLES/t-Tables/
    t_values = pd.read_csv("../data/t_table.csv",sep=";")
    ts = []
    for i in range(len(ncorr['df'])):
        #find the proper df
        found = False
        j = 0
        l = len(t_values['df'])
        while not found:
            if ncorr['df'][i]-2 > t_values['df'][l-j-1]:
                dfree = t_values['df'][l-j-1]
                idxfree = l-j-1
                found = True
            j+=1
        #find the proper alpha
        found = False
        i = 0
        while not found:
            if threshold > float(t_values.columns[1:][i]):
                alpha = t_values.columns[1:][i]
                idxalpha = i
                found = True
            i+=1
        ts.append(t_values.iloc[idxfree][idxalpha])
    ncorr['ts'] = ts
    ncorr = ncorr.drop([target])
    ncorr['temp'] = abs(ncorr['t-score'])
    ncorr['accept'] = ncorr['temp']>ncorr['ts']
    ncorr = ncorr.sort_values(by='temp',ascending=False)
    ncorr = ncorr.drop(columns=['temp'])
    ncorr['df'] = ncorr['df']-1
    return ncorr

def lasso_feature_selection(df, target, alph):
    '''
    The Lasso feature selection, given a dataframe and a target column, returns
    the accepted features given a certain alpha coefficient.

    :param df: Pandas DataFrame with data in it.
    :param target: The target feature of our dataset.
    :param alph: The alpha "correction" coefficient that the Lasso will use.
    :return: A list with the accepted column names.
    '''
    X = df.drop(columns=target)
    y = df[target]
    features = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    pipeline = Pipeline([('scaler',StandardScaler()),('model',Lasso())])
    search = GridSearchCV(pipeline, {'model__alpha': alph}, \
                          cv = 3, scoring="neg_mean_squared_error",verbose=0)
    search.fit(X_train,y_train)
    print(search.best_params_)
    cf = np.abs(search.best_estimator_.named_steps['model'].coef_)
    accepted = np.array(features)[cf > 0]
    print("Accepted:",accepted)
    print("Discarded:",np.array(features)[cf == 0])
    return accepted
