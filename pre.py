import pandas as pd
import numpy as np

class FeatureExtractor():
    def __init__(self,query_df,train_pre='ker.csv') -> None:
        self.jarr=query_df
        self.train_pre=pd.read_csv(train_pre)
        return None

    def transforms(self):
        t=self.jarr
        train_pre=self.train_pre
        train_pre = pd.merge(t,train_pre,on = ['item_id','shop_id'],how = 'left')
        train_pre.fillna(0,inplace = True)
        X_test = train_pre.drop('0', axis=1)
        return X_test