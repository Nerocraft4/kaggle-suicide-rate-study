import pandas as pd
import numpy as np

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
    print(lens)
    corr['df'] = lens #degrees of freedom
    #if we don't do it in this order, Panda converts the dataframe to a Series
    corr = corr[[target,'df']]
    corr['t-score'] = corr[target]*np.sqrt(corr['df']-2) \
                      /np.sqrt(1-corr[target]*corr[target])
    print(corr)
    t_values = pd.read_csv("../data/t_table.csv",sep=";")
    ts = []
    for i in range(len(corr['df'])):
        #find the proper df
        found = False
        i = 0
        l = len(t_values['df'])
        while not found:
            if corr['df'][i]-2 > t_values['df'][l-i-1]:
                dfree = t_values['df'][l-i-1]
                idxfree = l-i-1
                found = True
            i+=1
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
    corr['ts'] = ts
    corr['accept'] = abs(corr['t-score'])>corr['ts']
    print(corr)
    #print(t_values)
