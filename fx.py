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