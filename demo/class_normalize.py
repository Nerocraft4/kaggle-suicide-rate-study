from clean_data import class_normalize
import pandas as pd

#dummy dataframe
df = pd.DataFrame([['toddlers',2,4],['adults',0,-1],['elders',0,0]])
df = df.rename(columns={0:'agegroup',1:'score1',2:'score2'})
df.head()

#normalize column 'agegroup' in indicated order
df = class_normalize(df=df,column='agegroup', \
                     mapping_order=['toddlers','adults','elders'])
df.head()
