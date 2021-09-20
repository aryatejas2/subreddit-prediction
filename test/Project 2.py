# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:00:57 2019

@author: Karshit
"""

#Importing Libraries
import requests
import pandas as pd
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim
import IPython

import warnings

#Getting data
rel_url = 'https://www.reddit.com/r/relationships/.json?sort=top&t=all'
rel_headers = {'User-agent': 'Project2'}
rel_res = requests.get(rel_url, headers=rel_headers)

funny_text = 'https://www.reddit.com/r/funny/.json?sort=top&t=all'
nba_url='https://www.reddit.com/r/nba/.json?sort=top&t=all'
nba_headers = {'User-agent': 'Project2'}
nba_res = requests.get(nba_url, headers=nba_headers)

rel_json = rel_res.json()
nba_json = nba_res.json()

pd.DataFrame(rel_json['data']['children']).head()
pd.DataFrame(nba_json['data']['children']).head()

#Taking posts from json data
rel_posts = []
rel_after = None
for rel in range(60):
    rel_res = requests.get(rel_url, headers=rel_headers)
    rel_json = rel_res.json()
    rel_posts.extend(rel_json['data']['children'])
    bb_after = rel_json['data']['after']
rel_posts=rel_posts[:1000]

nba_posts=[]
nba_after =None
   
for nba in range(60):
    nba_res = requests.get(nba_url, headers=nba_headers)
    nba_json = nba_res.json()
    nba_posts.extend(nba_json['data']['children'])
    nba_after = nba_json['data']['after']
nba_posts=nba_posts[:1000]

#Taking title and text of each post
rel_text = [x['data']['selftext'] for x in rel_posts]
rel_titles = [x['data']['title'] for x in rel_posts]

nba_text = [x['data']['selftext'] for x in nba_posts]
nba_titles = [x['data']['title'] for x in nba_posts]

#Prepoccessing Data
#Coverting data to lowercase
rel_text=[x.lower() for x in rel_text]
rel_titles=[x.lower() for x in rel_titles]
nba_text=[x.lower() for x in nba_text]
nba_titles=[x.lower() for x in nba_titles]
#Removing Punctuations
rel_text = [''.join(c for c in s if c not in string.punctuation) for s in rel_text]
rel_titles = [''.join(c for c in s if c not in string.punctuation) for s in rel_titles]
nba_text = [''.join(c for c in s if c not in string.punctuation) for s in nba_text]
nba_titles = [''.join(c for c in s if c not in string.punctuation) for s in nba_titles]

#Creating dataframes
rel_dataframe = pd.DataFrame({'text': rel_text, 'title': rel_titles})
rel_dataframe['target'] = 1

nba_dataframe = pd.DataFrame({'text': nba_text, 'title': nba_titles})
nba_dataframe['target'] = 0

dataframe = pd.concat([rel_dataframe, nba_dataframe])
Y=dataframe[['target']]

#Converting text to feature vectors
feature_vector = TfidfVectorizer()
feature_vector.fit(dataframe['title'])
print(feature_vector.vocabulary_)
# print(feature_vector.idf_)
feature_vector = feature_vector.transform(dataframe['title'])
# print(feature_vector.shape)
# print(feature_vector.toarray())
feature_vector = feature_vector.todense()
fv_dataframe=pd.DataFrame(feature_vector)

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(fv_dataframe, Y, random_state=2, test_size=0.2)
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
print(confusion_matrix(y_test, predictions))

#WordCloud
wordcloud = WordCloud  (stopwords= STOPWORDS, background_color= 'white',
                        width= 3000, height= 2000).generate(str(dataframe))

plt.savefig(wordcloud,'WordCloud1')
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()

#LDA Topic Modeling

# NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')

tokenized_data = []

for t in dataframe['title']:
    print(t)
    tokenized_text = word_tokenize(t)
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS]
    tokenized_data.append(cleaned_text)

dictionary = corpora.Dictionary(tokenized_data)

corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# print(corpus)

lda_model = models.LdaModel(corpus= corpus, id2word= dictionary)

for idx in range(10):
    print('Topic #%s:' %idx, lda_model.print_topic(idx,10))

print('\nPerplexity:', lda_model.log_perplexity(corpus))

coherence_model_lda = models.CoherenceModel(model=lda_model,texts=dataframe['title'],
                                            dictionary= dictionary, coherence='c_v')

# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score:', coherence_lda)

# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model,corpus,dictionary)
pyLDAvis.save_html(vis,'LDA.html')






