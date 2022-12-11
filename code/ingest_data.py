import pandas as pd

def ingest(path):
    """
    The ingest function allows the user to ingest data directly from a csv

    :param path: It's the path where the data is located
    :return: Returns a dataframe pandas object with the data
    """
    return(pd.read_csv(path))
