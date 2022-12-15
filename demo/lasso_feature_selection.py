from generate_features import lasso_feature_selection
import pandas as pd
import numpy as np

#dummy dataframe
df = pd.DataFrame([[1,2,3,4],[2,3,4,5],[0.1,10,-2,7],[5,4,3,2],[1,3,3,4], \
                   [1,6,3,-4],[0,2,3,1],[2,2,-3,2],[0,0,-3,0],])
df.head()

#our target will be the first column, and the alpha np.arange(0.1,1,0.1)
accepted = lasso_feature_selection(df=df,target=0,alph=np.arange(0.1,1,0.1))

#as you can see (and the data being kind of random) it discards every column
