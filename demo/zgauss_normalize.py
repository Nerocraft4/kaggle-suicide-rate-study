from clean_data import zgauss_normalize
import pandas as pd

#dummy dataframe
df = pd.DataFrame([[3,4,5],[4,3,1],[9,0,10],[-1,2,-1]])
df = df.rename(columns={0:'zero',1:'one',2:'two'})
df.head()

#normalize only rows 'zero' and 'one'
df = zgauss_normalize(df=df,columns=['zero','one'])
df.head()
