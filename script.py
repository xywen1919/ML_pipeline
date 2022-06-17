# Import data 
# %%
# define an import function to select numerical-columns

def import_num_listings (url):
  import pandas as pd
  import io
  import requests
  s = requests.get(url).content
  df = pd.read_csv(io.StringIO(s.decode('utf-8')))
  
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
# path = './data/raw/listings_train.csv'
url = 'https://drive.google.com/file/d/17U0C2i5C6t36PSCkEXPqOY4mDeIGcV82'
list_num = import_num_listings(url)

# inspect numerical data
print(list_num.shape)      # dimention
list_num.describe()        # summary statistics
list_num.isnull().sum()    # null values

#%%
# write to pickle
from base64 import decode
import pickle
from tkinter.font import names
import matplotlib
list_num.to_pickle("./data/raw/list_num.pkl")

#%%
# define function to select object-columns start with keyword

def import_str_listings (url, keywrd):
  # import data
  import pandas as pd
  import requests
  import io
  s = requests.get(url).content
  df = pd.read_csv(io.StringIO(s.decode('utf-8')))
  
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
# path = './data/raw/listings_train.csv'
url = 'https://drive.google.com/file/d/17U0C2i5C6t36PSCkEXPqOY4mDeIGcV82'
keywords = ['host', 'neighbo', 'type', 'amenities', 'text']

import pandas as pd
list_obj = pd.DataFrame()

for word in keywords:
  df_obj = import_str_listings(url,word)
  print(f'{word}: {df_obj.shape}')
  list_obj = list_obj.join(df_obj, how='outer', lsuffix='_')

list_obj.shape

#%%
# write to pickle
list_obj.to_pickle("./data/raw/list_obj.pkl")


#%% import review data
import pandas as pd

# define an import function for review comments
def import_rev_comments (url):
  import pandas as pd
  import requests
  import io
  
  # import raw data
  s = requests.get(url).content
  df = pd.read_csv(io.StringIO(decode('utf-8')))
  df = df[['listing_id','comments']]
  df.shape
  
  # group by listing_id and concat comments
  df['comments'] = df['comments'].astype(str) # make sure all the comments are string
  df2 = df.groupby('listing_id')['comments'].apply(','.join)
  df2 = df2.to_frame()  
  
  return df2

# import comments grouped by listing id
# rev_path = './data/raw/reviews_train.csv'
url = 'https://drive.google.com/file/d/1gvMGs7aBbEpddpyv09m9q0wQYizbtzbZ'
rev_df = import_rev_comments(url)

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
# build linear regression model 
import pickle
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
# split train, test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# save the variables for model prediction
with open ('./data/intermid/X_test.pkl', 'wb') as file:
  pickle.dump(X_test, file)
with open ('./data/intermid/y_test.pkl', 'wb') as file:
  pickle.dump(y_test, file)

# create linear regression object
linear = linear_model.LinearRegression()

# fit the model
linear.fit(X_train, y_train)

# save the model
pkl_FileName = 'linear_model.pkl'
with open (pkl_FileName, 'wb')as file:
  pickle.dump(linear, file)

# ========================================================

# %%
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stopWords = set(stopwords.words('english'))

def create_word_token (corpus):
  import nltk
  from nltk.tokenize import word_tokenize
  from nltk.corpus import stopwords
  
  # tokenize
  lines = str(corpus)
  words = word_tokenize(lines)
  
  # filtered by stopWords
  stopWords = set(stopwords.words('english'))
  wlst = [word for word in words if word not in stopWords]
  
  # text cleaning
  import re
  import string  
  w_cleaned = []
  for text in wlst:
    # Make word lowercase, remove word in square brackets, remove punctuation and remove words containing numbers
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\#\@', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    
    if len(text) != 0:
      w_cleaned.append(text)
  
  return w_cleaned
   
    
#%%
from textblob import TextBlob
from textblob import Word
import nltk
from nltk import word_tokenize 
from nltk.tag import pos_tag
nltk.download('wordnet')

# function to select noun and adj from corpus
def extrac_nadj (corpus):
  from textblob import TextBlob
  from textblob import Word
  import nltk
  from nltk import word_tokenize 
  from nltk.tag import pos_tag
  
  lines = corpus.lower()
  tokenized = nltk.word_tokenize(lines)
  
  # extract noun and adj words
  is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
  nadjList = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun_adj(pos)]
      
  return nadjList


# %%
# nltk.download() 

# %%
def checkSentiment(wList):
  from textblob import TextBlob
  from textblob import Word
  import statistics
  # wlst = [TextBlob(x).correct() for x in wList]   
  # wlst2 = [Word(word).lemmatize() for word in wlst]
    
  polarity = [TextBlob(wd).sentiment[0] for wd in wList]
  if len(polarity) != 0:
    return round(statistics.mean(polarity),3)
  return pd.NA


#%%
# read comments.pkl
import pickle
import pandas as pd
import fx

# define function to check sentiment for the review comments
def review_sentiment_fromPickle (path_to_pkl):
  import pickle
  import pandas as pd
  import fx 
  
  # read in pickle file
  rev_comments = pd.read_pickle(path_to_pkl)
    
  # extract noun and adj for each review comment and measure sentiment
  rev_sentiments = []
  for corpus in rev_comments['comments']:
    nadjList = extrac_nadj(corpus)
    senti_mean = checkSentiment(nadjList)
    rev_sentiments.append(senti_mean)
  
  rev_sentiments = pd.DataFrame(rev_sentiments)
  rev_sentiments.index = rev_comments.index
  return rev_sentiments

#%%
review_sentiments = review_sentiment_fromPickle('./data/raw/comments.pkl')


# %%
# save the review sentiment result
import pandas as pd
review_sentiments.to_pickle('./data/raw/review_sentiments.pkl')
review_sentiments.describe()
review_sentiments.head()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# load the listing-num data
list_num = pd.read_pickle("./data/raw/list_num.pkl")

# plot the price histogram
plt.hist(list_num['price'])

#%%
# binning the target variable --'price'
list_num['price_bin'] = pd.qcut(list_num['price'], q=10, precision=0)
list_num['price_bin'].value_counts()

#%%
list_num.info() # check if set id as idext
list_num.head(3)

list_num.to_pickle('./data/intermid/list_num.pkl')

#%%
import pandas as pd
import fx

# load the list-obj data
list_obj = pd.read_pickle("./data/raw/list_obj.pkl")

#%%
# tokenize some of the columns for further analysis
list_obj['neighborhood_overview'] = [create_word_token(corpus) for corpus in list_obj['neighborhood_overview']]

list_obj['amenities'] = [create_word_token(corpus) for corpus in list_obj['amenities']]

list_obj.info() 
# list_obj = list_obj.set_index('id')

#%%
list_obj.to_pickle('./data/intermid/list_obj.pkl')

#%%
# join listing dfs
dataListing = list_num.join(list_obj, how='outer')
# 
review_sentiments = pd.read_pickle('./data/raw/review_sentiments.pkl')
review_sentiments.index = review_sentiments.index.rename('id')
review_sentiments.columns = ['review_senti']
data = dataListing.join(review_sentiments, how='outer')
data.info()
#%%
# cleanse data in additional
data['n_amenities'] = [len(amen) for amen in data['amenities']]

from nltk import word_tokenize
data['host_location'] = [word_tokenize(str(loc))[0] for loc in data['host_location']]
data['neighbourhood'] = [word_tokenize(str(nei))[0] for nei in data['neighbourhood']]

data = data.drop(['host_url', 'host_name','host_since','host_about','host_response_time','host_thumbnail_url','host_picture_url','host_verifications', 'neighborhood_overview'], axis=1)

data.to_pickle('./data/intermid/data.pkl')

#%%

# seperate X and y variables
X, y = data.drop(['price','price_bin', 'amenities'], axis=1), data['price_bin'].to_numpy()

#%%
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




#%%
import pickle
# save model and test-datasets for future use
pickle.dump(model, open('./models/model.pkl', 'wb'))

pickle.dump(X_test, open('./data/X0_test.pkl','wb'))
pickle.dump(y_test, open('./data/y0_test.pkl','wb'))


# %%
