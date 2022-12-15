from generate_features import corr_feature_selection
import pandas as pd

#dummy dataframe
df = pd.DataFrame([[1,2,3,4],[2,3,4,5],[0.1,10,-2,7],[5,4,3,2]])
df.head()

#calculate the corr matrix
corr = df.corr()

#call the function
results = corr_feature_selection(df=df,corr=corr,target=0,threshold=0.05)

print(results)
