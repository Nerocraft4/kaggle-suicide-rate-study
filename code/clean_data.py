import pandas as pd
import numpy as np

def where_selection(df,column,vlist,include=True):
    """
    The where_selection function acts as a WHERE SQL clause. Given a pandas df,
    a column name and a list, it returns a trimmed dataframe with rows only
    containing values in that list. If include is set to false, the function
    will do the opposite (exclude the values).

    :param df: A pandas dataframe containing data.
    :param column: A column name (string).
    :param vlist: An array with strings/ values to match.
    :param include: A boolean to know whether to include or exclude the values.
    :return: New dataframe with the included / without the excluded values.
    """
    if column not in df.columns:
        print("Column",column,"does not exist in the dataframe")
        print("Returning the dataframe without any changes")
        return df
    if include:
        return(df[df[column].isin(vlist)])
    return(df[~df[column].isin(vlist)])

def lin_normalize(df,columns):
    """
    The lin_normalize function allows the user to normalize various columns from
    a pandas dataframe using a linear mapping.

    :param df: A pandas dataframe containing data.
    :param columns: A list with column names.
    :return: Returns the dataframe with the normalized columns.
    """
    if any(column not in df.columns for column in columns):
        print("Some columns are not in the dataframe")
        print("Returning the dataframe without any changes")
        return df
    for col in columns:
        mi = np.min(df[col])
        ma = np.max(df[col])-mi
        df[col] = (df[col]-mi)/ma
    return df

def class_normalize(df,column,mapping_order):
    """
    The class_normalize function allows the user to normalize a label column
    from a pandas dataframe using a linear mapping, given ordered labels.

    :param df: A pandas dataframe containing data.
    :param column: A string with the column name.
    :param mapping_order: A list with the order of the mapping, from 0 to 1.
    :return: Returns the dataframe with the normalized column.
    """
    if column not in df.columns:
        print("Column",column,"does not exist in the dataframe")
        print("Returning the dataframe without any changes")
        return df
    n_classes = len(mapping_order)
    if n_classes < 0:
        print("No classes to map. List is empty")
        return df
    mapping = {}
    if n_classes>1:
        for i in range(n_classes):
            mapping[mapping_order[i]]=i/(n_classes-1)
    else:
        mapping[mapping_order[0]]=1
    df[column] = [mapping[x] for x in df[column]]
    return df

def zgauss_normalize(df,columns):
    """
    The zgauss_normalize function allows the user to normalize various columns
    from a pandas dataframe using a gaussian / normal distribution mapping.

    :param df: A pandas dataframe containing data.
    :param columns: A list with column names.
    :return: Returns the dataframe with the normalized columns.
    """
    if any(column not in df.columns for column in columns):
        print("Some columns are not in the dataframe")
        print("Returning the dataframe without any changes")
        return df
    for col in columns:
        mu = np.mean(df[col])
        std = np.std(df[col])
        df[col] = (df[col]-mu)/std
    return df
