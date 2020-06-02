
import numpy as np
import pandas as pd
pd.set_option('max_columns',None)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
plt.style.use('ggplot')
import datetime
import eli5
from scipy import stats
"""from scipy.sparse import hstack,csr_matrix
from wordcloud import WordCloud
from collections import Counter"""
from sklearn.model_selection import train_test_split,KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
from nltk.util import ngrams
import os
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

train = pd.read_csv(r'C:\Users\Pranay\TMDB\train.csv')
test = pd.read_csv(r'C:\Users\Pranay\TMDB\test.csv')

#Visualizizng Target Distribution
fig,ax = plt.subplots(figsize = (16,6))
plt.subplot(1,2,1)
sns.distplot(train['revenue'],kde=False);
plt.title("Distribution of Revenue")
plt.subplot(1,2,2)
sns.distplot(np.log1p(train['revenue']),kde=False);
plt.title("Distribution of log-transformed Revenue")
train['log_revenue'] = np.log1p(train['revenue'])

#Relationship between Film Revenue and Budget
fig,ax = plt.subplots(figsize = (16,6))
plt.subplot(1,2,1)
sns.scatterplot(x=train['budget'],y=train['revenue']);
plt.title("Revenue vs Budget")
plt.subplot(1,2,2)
sns.scatterplot(x=np.log1p(train['budget']),y=train['log_revenue']);
plt.title("Log_Revenue vs Log_Budget")
train['log_budeget'] = np.log1p(train['budget'])
test['log_budget'] = np.log1p(train['budget'])

#Does Homepage Affect Revenue
train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False,'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False,'has_homepage'] = 1
sns.catplot(x ='has_homepage',y ='revenue',data = train);
plt.title("Revenue for films with and without homepage")


#Distribution of languages in Film
language_data = train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]
fig,ax = plt.subplots(figsize=(16,6))
plt.subplot(1,2,1)
sns.boxplot(x='original_language',y='revenue',data=language_data);
plt.title("Mean revenue per Language")
plt.subplot(1,2,2)
sns.boxplot(x='original_language',y='log_revenue',data=language_data);
plt.title("Mean Log_Revenue per Language")
plt.show()

#Frequent Words in Films Titles and Descriptions

plt.figure(figsize = (12,12))
text = ' '.join(train['original_title'].values)
wordcloud = WordCloud(max_font_size = None,background_color='white',width=1200,height=1000).generate(text)
plt.imshow(wordcloud);
plt.title("Top Words Across Movie Titles")
plt.axis('off')
plt.show()

plt.figure(figsize = (10,10))
text = ' '.join(train['overview'].fillna('').values)
wordcloud = WordCloud(max_font_size = None,background_color='white',width=1200,height=1000).generate(text)
plt.imshow(wordcloud);
plt.title("Top Words Across Movie Overviews")
plt.axis('off')
plt.show()

vectorizer = TfidfVectorizer(
    sublinear_tf = True,
    analyzer = 'word',
    token_pattern = r'\w{1,}',
    ngram_range = [1,2],
    min_df = 5
)
overview_text = vectorizer.fit_transform(train['overview'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text,train['log_revenue'])
eli5.show_weights(linreg,vec = vectorizer,top=20,feature_filter = lambda x:x!='<BIAS>')

#Analayzing movie Release Dates
test.loc[test['release_date'].isnull()==False,'release_date'].head()

def fix_date(x):
  year = x.split('/')[2]
  if int(year)<=19:
    return x[:-2] + '20' + year
  else:
    return x[:-2] + '19'+year

test.loc[test['release_date'].isnull()==True]

test.loc[test['original_title'] == 'Jails, Hospitals & Hip-Hop','release_date'] = '05/01/00'

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

#Creating features based on release date
train['release_date'] = pd.to_datetime(train['release_date'])
test['release_date'] = pd.to_datetime(test['release_date'])

def process(df):
  date_parts = ['year','weekday','month','weekofyear','day','quarter']
  for part in date_parts:
    part_col = 'release_date'+'_'+part
    df[part_col] = getattr(df['release_date'].dt,part).astype(int)
  return df
train = process(train)
test = process(test)

#Using Plotly to Visualize the Number of films Per Year
d1 = train['release_date_year'].value_counts().sort_index()
d2 = test['release_date_year'].value_counts().sort_index()

data1 = [go.Scatter(x=d1.index,y=d1.values,name='Train'),go.Scatter(x=d2.index,y=d2.values,name='Test')]
layout1 = go.Layout(dict(title='Number of films per year'
        ,xaxis=dict(title='Year'),yaxis=dict(title='Count of films'),legend=dict(orientation='v')))
py.plot(dict(data=data1,layout=layout1))

#Number of films and total Revenue per year
d3 = train['release_date_year'].value_counts().sort_index()
d4 = train.groupby(train['release_date_year'])['revenue'].sum()

data2 = [go.Scatter(x=d3.index,y=d3.values,name='Film Count'),go.Scatter(x=d4.index,y=d4.values,name='Total Revenue',yaxis='y2')]
layout2 = go.Layout(dict(title='Number of films and Total Revenue per year',xaxis=dict(title='Year'),yaxis=dict(title='Count of films'),
                   yaxis2=dict(title='Total Revenue',overlaying='y',side='right'),legend=dict(orientation='v')))
py.plot(dict(data=data2,layout=layout2))

#Number of films and Mean Revenue per year
d5 = train['release_date_year'].value_counts().sort_index()
d6 = train.groupby(train['release_date_year'])['revenue'].mean()

data3 = [go.Scatter(x=d5.index,y=d5.values,name='Film Count'),go.Scatter(x=d6.index,y=d6.values,name='T Revenue',yaxis='y2')]
layout3 = go.Layout(dict(title='Number of films and Mean Revenue per year',xaxis=dict(title='Year'),yaxis=dict(title='Count of films'),
                   yaxis2=dict(title='Mean Revenue',overlaying='y',side='right'),legend=dict(orientation='v')))
py.plot(dict(data=data3,layout=layout3))

#Do Release days impact revenue
sns.catplot(x='release_date_weekday',y='revenue',data = train)
plt.title("Revenue on different days of week")
plt.show()

#Relationship between Runtime and Revenue
sns.distplot(train['runtime'].fillna(0) / 60,bins=40,kde = False)
plt.title("Distribution of length of films in hours")
plt.show()

sns.scatterplot(x=train['runtime'].fillna(0)/60,y = train['revenue'])
plt.title("Runtime vs Revenue")
plt.show()