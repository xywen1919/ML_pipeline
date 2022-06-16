# Import data 
# %%
# define an import function to select numerical-columns

def import_num_listings (dat_path):
  import pandas as pd
  df = pd.read_csv(dat_path)
  
  # convert price to 'float'
  df['price'] = [float(n.replace(',','')[1:]) for n in df['price']]
  
  # select numer cloumns only
  df2 = df.select_dtypes(include='number')  
  # set 'id' as index
  df2 = df2.set_index(['id'] )
  # drop other id columns:'scrape_id','host_id'
  df2 = df2.drop(['scrape_id','host_id'], axis=1)
  
  # drop columns having less than 70% non null values
  n = df2.shape[0]*0.7
  df2 = df2.dropna(axis=1, thresh=n)
  
  # replace null values with column-means
  column_mean = df2.mean()
  df2 = df2.fillna(column_mean)
  
  return df2

# %%
# import numerical listing data
path = './data/raw/listings_train.csv'
list_num = import_num_listings(path)

# inspect numerical data
print(list_num.shape)      # dimention
list_num.describe()        # summary statistics
list_num.isnull().sum()    # null values

#%%
# write to pickle
from gettext import install
import pickle
list_num.to_pickle("./data/raw/list_num.pkl")

#%%
# define function to select object-columns start with keyword

def import_str_listings (dat_path, keywrd):
  # import data
  import pandas as pd
  df = pd.read_csv(dat_path)
  
  # set index 'id'
  df = df.set_index('id')  
  # print(df.shape)
  
  # exclude numberic columns
  df2 = df.select_dtypes(include='object')
  # print(df2.shape)
  
  # select columns start with keyword 
  df2 = df2.filter(like=keywrd, axis=1)
  
  return df2

#%%
# select text columns with keywords
path = './data/raw/listings_train.csv'
keywords = ['host', 'neighbo', 'type', 'amenities', 'text']

import pandas as pd
list_obj = pd.DataFrame()

for word in keywords:
  df_obj = import_str_listings(path,word)
  print(f'{word}: {df_obj.shape}')
  list_obj = list_obj.join(df_obj, how='outer', lsuffix='_')

list_obj.shape

#%%
# write to pickle
list_obj.to_pickle("./data/raw/list_obj.pkl")


#%% import review data
import pandas as pd

# define an import function for review comments
def import_rev_comments (rev_path):
  # import raw data
  df = pd.read_csv(rev_path)
  df = df[['listing_id','comments']]
  df.shape
  
  # group by listing_id and concat comments
  df['comments'] = df['comments'].astype(str) # make sure all the comments are string
  df2 = df.groupby('listing_id')['comments'].apply(','.join)
  df2 = df2.to_frame()  
  
  return df2

# import comments grouped by listing id
rev_path = './data/raw/reviews_train.csv'
rev_df = import_rev_comments(rev_path)

# inspect data
print(rev_df.shape)
rev_df.head(3)

#%%
# inspect review data
rev_df.describe
# write to pickle
rev_df.to_pickle("./data/raw/comments.pkl")

# ==========================================================
# %%
# ! pip install scikit-learn

#%%
# ! pip install seaborn

# %%
# build linear regression model 
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# read-in data
list_num = pd.read_pickle("./data/raw/list_num.pkl")
list_num.info()

# identify features and response variables
X,y = list_num.drop('price', axis=1), np.array(list_num['price'])

# scaling the X
X = StandardScaler().fit_transform(X)
X.shape

#%%
# correlations 
X2 = pd.DataFrame(X)
y2 = pd.DataFrame(y)
cor = pd.concat([X2,y2],axis=1).corr()
print(cor)

# plot correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# plotting correlation heatmap
dataplot = sns.heatmap(cor, cmap="YlGnBu")
  
# displaying heatmap
mp.show()


#%%
# select correlated features  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# split train, test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

# save the variables for model prediction
with open ('./data/intermid/X_test_fs.pkl', 'wb') as file:
  pickle.dump(X_test_fs, file)
with open ('./data/intermid/y_test.pkl', 'wb') as file:
  pickle.dump(y_test, file)

# create linear regression object
linear = linear_model.LinearRegression()

# fit the model
linear.fit(X_train_fs, y_train)

# save the model
pkl_FileName = 'linear_model.pkl'
with open (pkl_FileName, 'wb')as file:
  pickle.dump(linear, file)


