import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso

sales_item_category=pd.read_csv('item_categories.csv')
sales_item=pd.read_csv('items.csv')
sales_train=pd.read_csv('sales_train.csv', parse_dates=['date'], dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32','item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})
sales_test=pd.read_csv('test.csv')
sales_shops=pd.read_csv('shops.csv')


def datatype_downcast(df):
    float_col= [i for i in df if df[i].dtype == 'float64']
    int_col = [i for i in df if df[i].dtype in ['int64','int32']]
    df[float_col] = df[float_col].astype(np.float32)
    df[int_col] = df[int_col].astype(np.int16)
    return df

sales_train = datatype_downcast(sales_train)

sales_dataset = sales_train.copy()

sales_dataset[sales_dataset['item_cnt_day'] == 2169.0]

sales_monthly = sales_dataset.groupby(['date_block_num','shop_id','item_id'])['date','item_price','item_cnt_day'].agg({'date':['mean','max'],'item_price':'mean','item_cnt_day':'sum'})

days = []
months = []
years = []

for day in sales_dataset['date']:
    days.append(day.day)
for month in sales_dataset['date']:
    months.append(month.month)
for year in sales_dataset['date']:
    years.append(year.year)


sales_dataset['day'] = days
sales_dataset['month'] = months
sales_dataset['year'] = years

sales_dataset = sales_dataset[sales_dataset['item_price'] < 100000]
sales_dataset = sales_dataset[sales_dataset['item_cnt_day'] < 1200]

sales_median = sales_dataset[(sales_dataset.shop_id==32)&(sales_dataset.item_id==2973)&(sales_dataset.date_block_num==4)&(sales_dataset.item_price>0)].item_price.median()

sales_dataset["item_price"] = sales_dataset["item_price"].map(lambda x: sales_median if x<0 else x)

sales_dataset['item_cnt_day'] = sales_dataset['item_cnt_day'].map(lambda x:0 if x<0 else x)


test_item_list = [x for x in (np.unique(sales_test['item_id']))]
train_item_list = [x for x in (np.unique(sales_dataset['item_id']))]

missing_item_ids_ = [element for element in test_item_list if element not in train_item_list]

sales_shops['shop_name'] = sales_shops['shop_name'].map(lambda x: x.split('!')[1] if x.startswith('!') else x)
sales_shops['shop_name'] = sales_shops['shop_name'].map(lambda x: 'СергиевПосад ТЦ "7Я"' if x == 'Сергиев Посад ТЦ "7Я"' else x)
sales_shops['shop_city'] = sales_shops['shop_name'].map(lambda x: x.split(" ")[0])
# lets assign code to these city names too
sales_shops['city_code'] = sales_shops['shop_city'].factorize()[0]

for shop_id in sales_shops['shop_id'].unique():
    sales_shops.loc[shop_id,'num_of_product'] = sales_dataset[sales_dataset['shop_id']==shop_id]['item_id'].nunique()
    sales_shops.loc[shop_id,'min_price'] = sales_dataset[sales_dataset['shop_id']==shop_id]['item_price'].min()
    sales_shops.loc[shop_id,'max_price'] = sales_dataset[sales_dataset['shop_id']==shop_id]['item_price'].max()
    sales_shops.loc[shop_id,'mean_price'] = sales_dataset[sales_dataset['shop_id']==shop_id]['item_price'].mean()

category_list =[]

for cat_name in sales_item_category['item_category_name']:
    category_list.append(cat_name.split('-'))

sales_item_category['split'] = (category_list)
sales_item_category['category_type'] = sales_item_category['split'].map(lambda x: x[0])
sales_item_category['category_type_code'] = sales_item_category['category_type'].factorize()[0]

sales_item_category['sub_category_type'] = sales_item_category['split'].map(lambda x: x[1] if len(x)>1 else x[0])

sales_item_category['sub_category_type_code'] = sales_item_category['sub_category_type'].factorize()[0]

sales_item_category.drop('split',axis = 1 ,inplace = True)


sales_dataset = sales_dataset[sales_dataset['item_cnt_day']>0]


sales_dataset = sales_dataset[["month", "date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"]].groupby(['date_block_num',"shop_id", "item_id"]).agg({"item_price":"mean","item_cnt_day":"sum","month":"min"}).reset_index()

# merging item , shops and category
sales_dataset.rename(columns={"item_cnt_day":"item_cnt_month"},inplace=True)

sales_dataset = pd.merge(sales_dataset,sales_item,on='item_id',how='inner')

sales_dataset = pd.merge(sales_dataset,sales_shops,on='shop_id',how='inner')

sales_dataset = pd.merge(sales_dataset,sales_item_category,on='item_category_id',how='inner')


sales_dataset.drop(['item_name','shop_name','shop_city','item_category_name','city_code','category_type','sub_category_type','sub_category_type_code'],axis = 1,inplace=True)

sales_dataset = sales_dataset[sales_dataset['shop_id'].isin(sales_test['shop_id'].unique())]
sales_dataset = sales_dataset[sales_dataset['item_id'].isin(sales_test['item_id'].unique())]

sales_train_new = sales_dataset.copy()

sales_train_new = sales_train_new.pivot_table(index=['item_id','shop_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value = 0).reset_index()

sales_train_new = pd.merge(sales_test,sales_train_new,on = ['item_id','shop_id'],how = 'left')
sales_train_new.fillna(0,inplace = True)

X_train = sales_train_new.drop(33, axis=1)
y_train = sales_train_new[32]



# deleting the column so that it can predict the future sales data
# X_test = sales_train_new.drop(0, axis=1)
# y_test = sales_train_new[33]

lasso = Lasso()
# Instantiate the model
alpha = 0.0001
lasso = Lasso(alpha=alpha)

classifier=lasso

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))