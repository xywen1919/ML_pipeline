#%%
# Import data 
import fx

# numerical data in listing_train.csv
url = 'https://drive.google.com/file/d/17U0C2i5C6t36PSCkEXPqOY4mDeIGcV82'
list_num = fx.import_num_listings(url)
list_num.shape

# object data with selected keyword in listing_train.csv
url = 'https://drive.google.com/file/d/17U0C2i5C6t36PSCkEXPqOY4mDeIGcV82'
keywords = ['host', 'neighbo', 'type', 'amenities', 'text']

import pandas as pd
list_obj = pd.DataFrame()

for word in keywords:
  df_obj = fx.import_str_listings(url,word)
  print(f'{word}: {df_obj.shape}')
  list_obj = list_obj.join(df_obj, how='outer', lsuffix='_')

list_obj.shape

# import review comments
url = 'https://drive.google.com/file/d/1gvMGs7aBbEpddpyv09m9q0wQYizbtzbZ'
rev_df = fx.import_rev_comments(url)

rev_df.shape

# pickle the datasets
list_num.to_pickle("./data/raw/list_num.pkl")

list_obj.to_pickle("./data/raw/list_obj.pkl")

rev_df.to_pickle("./data/raw/comments.pkl")


# %%
# build linear regression model using numerical data from listing

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
import numpy as np
import pandas as pd

# read-in data
list_num = pd.read_pickle("./data/raw/list_num.pkl")
list_num.info()

# identify features and response variables
X,y = list_num.drop('price', axis=1), np.array(list_num['price'])

# scaling the X
X = StandardScaler().fit_transform(X)

# split train, test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# create linear regression object
linear = linear_model.LinearRegression()

# fit the model
linear.fit(X_train, y_train)

# save the model
pkl_FileName = 'linear_model.pkl'
with open (pkl_FileName, 'wb')as file:
  pickle.dump(linear, file)

# save the variables for model prediction
with open ('./data/intermid/X_test.pkl', 'wb') as file:
  pickle.dump(X_test, file)
with open ('./data/intermid/y_test.pkl', 'wb') as file:
  pickle.dump(y_test, file)
  
#%%
# preprocess review comments and object data from listing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stopWords = set(stopwords.words('english'))

# process comments (select noun and adj, acess mean sentiment for each id)
review_sentiments = fx.review_sentiment_fromPickle('./data/raw/comments.pkl')

# pickle the sentiment file
review_sentiments.to_pickle('./data/raw/review_sentiments.pkl')

# load the list-obj data
list_obj = pd.read_pickle("./data/raw/list_obj.pkl")

# tokenize selected the columns 
list_obj['neighborhood_overview'] = [fx.create_word_token(corpus) for corpus in list_obj['neighborhood_overview']]

list_obj['amenities'] = [fx.create_word_token(corpus) for corpus in list_obj['amenities']]

# pickle the file
list_obj.to_pickle('./data/intermid/list_obj.pkl')

#%%
# preprocess the target variable to catagorical dtype
# load the listing-num data
list_num = pd.read_pickle("./data/raw/list_num.pkl")

# plot the price histogram
import matplotlib.pyplot as plt
import pandas as pd
plt.hist(list_num['price'])

# binning the target variable --'price'
list_num['price_bin'] = pd.qcut(list_num['price'], q=10, precision=0)
list_num['price_bin'].value_counts()

# pickle the file
list_num.to_pickle('./data/intermid/list_num.pkl')

#%%
# combine all files together to -- data
# join listing dfs
dataListing = list_num.join(list_obj, how='outer')

review_sentiments = pd.read_pickle('./data/raw/review_sentiments.pkl')
review_sentiments.index = review_sentiments.index.rename('id')
review_sentiments.columns = ['review_senti']
data = dataListing.join(review_sentiments, how='outer')

# cleanse data in additional
data['n_amenities'] = [len(amen) for amen in data['amenities']]

from nltk import word_tokenize
data['host_location'] = [word_tokenize(str(loc))[0] for loc in data['host_location']]
data['neighbourhood'] = [word_tokenize(str(nei))[0] for nei in data['neighbourhood']]

data = data.drop(['host_url', 'host_name','host_since','host_about','host_response_time','host_thumbnail_url','host_picture_url','host_verifications', 'neighborhood_overview'], axis=1)

data.to_pickle('./data/intermid/data.pkl')

#%%
# build prediction model
# seperate X and y variables
X, y = data.drop(['price','price_bin', 'amenities'], axis=1), data['price_bin'].to_numpy()

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
y_label = LabelEncoder().fit_transform(y)

# proprocess X features to fit model training
numerical_features = X.select_dtypes(exclude="object").columns.to_list()
categorical_features = X.select_dtypes(include="object").columns.to_list()

X[categorical_features] = X[categorical_features].astype(str)

# process features depends on dtype
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

# transform and then concat
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_features),
    ('standard_scaler', numerical_preprocessor, numerical_features)])


# build model pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

# split train, test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_label, random_state=1000)


# train the model
_ = model.fit(X_train, y_train)

# save model and test-datasets for future use
pickle.dump(model, open('./models/model.pkl', 'wb'))

pickle.dump(X_test, open('./data/X0_test.pkl','wb'))
pickle.dump(y_test, open('./data/y0_test.pkl','wb'))