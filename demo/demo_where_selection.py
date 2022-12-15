from clean_data import where_selection
from sklearn.datasets import load_digits

#load dataset
df = load_diabetes(as_frame=True).data

#select only rows which have 'pixel0_0==0.0'
trimmed = where_selection(df=df, column='pixel_0_0', vlist=[0.0], \
                          include=True)
trimmed.head()

#select rows that don't have 'pixel_1_1' equal to any of these: 1.0, 2.0, 3.0
trimmed = where_selection(df=df, column='pixel_1_1', vlist=[1.0,2.0,3.0], \
                          include=False)
trimmed.head()
