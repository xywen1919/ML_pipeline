# define an import function to select numerical-columns

from pandas import NA


def import_num_listings (url):  
  import pandas as pd
  import requests
  import io
  
  s=requests.get(url).content
  df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    
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


def import_str_listings (url, keywrd):
  # import data
  import pandas as pd
  import requests
  import io
  
  s=requests.get(url).content
  df=pd.read_csv(io.StringIO(s.decode('utf-8')))
  
  # set index 'id'
  df = df.set_index('id')  
  # print(df.shape)
  
  # exclude numberic columns
  df2 = df.select_dtypes(include='object')
  # print(df2.shape)
  
  # select columns start with keyword 
  df2 = df2.filter(like=keywrd, axis=1)
  
  return df2


# define an import function for review comments
def import_rev_comments (url):
  # import raw data
  import pandas as pd
  import requests
  import io
  
  s=requests.get(url).content
  df=pd.read_csv(io.StringIO(s.decode('utf-8')))  
  
  df = df[['listing_id','comments']]
  df.shape
  
  # group by listing_id and concat comments
  df['comments'] = df['comments'].astype(str) # make sure all the comments are string
  df2 = df.groupby('listing_id')['comments'].apply(','.join)
  df2 = df2.to_frame()  
  
  return df2


# select correlated features
def select_features(X_train, y_train, X_test):
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_regression
  
  # configure to select all features
  fs = SelectKBest(score_func=f_regression, k='all')
  
	# learn relationship from training data
  fs.fit(X_train, y_train)
	# transform train input data
  X_train = fs.transform(X_train)
	# transform test input data
  X_test = fs.transform(X_test)
  
  return X_train, X_test, fs


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


def checkSentiment(wList):
  from textblob import TextBlob
  import statistics
  import pandas as pd
    
  polarity = [TextBlob(wd).sentiment[0] for wd in wList]
  if len(polarity) != 0:
    return round(statistics.mean(polarity),3)
  return pd.NA


# define function to check sentiment for the review comments
def review_sentiment_fromPickle (path_to_pkl):
  import pickle
  import pandas as pd   
  
  # read in pickle file
  rev_comments = pd.read_pickle(path_to_pkl)
  rev_comments = rev_comments.set_index('listing_id')
  
  # extract noun and adj for each review comment and measure sentiment
  rev_sentiments = pd.DataFrame(index=rev_comments.index)
  for corpus in rev_comments['comments']:
    nadjList = extrac_nadj(corpus)
    senti_mean = checkSentiment(nadjList)
    rev_sentiments.append(senti_mean)
  
  return rev_sentiments


if __name__ == '__main__':
  review_sentiment_fromPickle()
  
# remove outliers 
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
  
  
# model pipeline
def test_model (data,model):
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  from sklearn.preprocessing import LabelEncoder
  from sklearn.compose import ColumnTransformer  
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import make_pipeline
  from sklearn.model_selection import cross_val_score
      
  # seperate X and y variables
  X, y = data.drop(['price','price_bin', 'amenities'], axis=1), data['price_bin'].to_numpy()

  # Label encode y
  y_label = LabelEncoder().fit_transform(y)
  
  # preprocess X features depends on dtype
  numerical_features = X.select_dtypes(exclude="object").columns.to_list()
  categorical_features = X.select_dtypes(include="object").columns.to_list()
  X[categorical_features] = X[categorical_features].astype(str)

  categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
  numerical_preprocessor = StandardScaler()
  
  # transform X features and then concat
  from sklearn.compose import ColumnTransformer
  preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_features),
    ('standard_scaler', numerical_preprocessor, numerical_features)])

  # buid model pipeline
  model = make_pipeline(preprocessor, model)

  # split train, test subsets
  X_train, X_test, y_train, y_test = train_test_split(X, y_label)

  # train the model
  _ = model.fit(X_train, y_train)
  print(model.score(X_train, y_train))
     
  # estimate model scale
  score = cross_val_score(model, X_test, y_test,cv=10)
  
  return score