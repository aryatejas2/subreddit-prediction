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
import matplotlib.pyplot as plt

rel_url = 'https://www.reddit.com/r/relationships/.json?sort=top&t=all'
rel_headers = {'User-agent': 'Project2'}
rel_res = requests.get(rel_url, headers=rel_headers)

nba_url = 'https://www.reddit.com/r/nba/.json?sort=top&t=all'
nba_headers = {'User-agent': 'Project2'}
nba_res = requests.get(nba_url, headers=nba_headers)

rel_json = rel_res.json()
nba_json = nba_res.json()

pd.DataFrame(rel_json['data']['children']).head()
pd.DataFrame(nba_json['data']['children']).head()

rel_posts = []
rel_after = None
for rel in range(60):
    rel_res = requests.get(rel_url, headers=rel_headers)
    rel_json = rel_res.json()
    rel_posts.extend(rel_json['data']['children'])
    bb_after = rel_json['data']['after']
rel_posts = rel_posts[:1000]

nba_posts = []
nba_after = None

for nba in range(60):
    nba_res = requests.get(nba_url, headers=nba_headers)
    nba_json = nba_res.json()
    nba_posts.extend(nba_json['data']['children'])
    nba_after = nba_json['data']['after']
nba_posts = nba_posts[:1000]

rel_text = [x['data']['selftext'] for x in rel_posts]
rel_titles = [x['data']['title'] for x in rel_posts]

nba_text = [x['data']['selftext'] for x in nba_posts]
nba_titles = [x['data']['title'] for x in nba_posts]

rel_text=[x.lower() for x in rel_text]
rel_titles=[x.lower() for x in rel_titles]
nba_text=[x.lower() for x in nba_text]
nba_titles=[x.lower() for x in nba_titles]
#Removing Punctuations
rel_text = [''.join(c for c in s if c not in string.punctuation) for s in rel_text]
rel_titles = [''.join(c for c in s if c not in string.punctuation) for s in rel_titles]
nba_text = [''.join(c for c in s if c not in string.punctuation) for s in nba_text]
nba_titles = [''.join(c for c in s if c not in string.punctuation) for s in nba_titles]

rel_dataframe = pd.DataFrame({'text': rel_text, 'title': rel_titles})
rel_dataframe['target'] = 1

nba_dataframe = pd.DataFrame({'text': nba_text, 'title': nba_titles})
nba_dataframe['target'] = 0

dataframe = pd.concat([rel_dataframe, nba_dataframe])
Y = dataframe[['target']]

X = dataframe
Y = label_binarize(Y, classes= [0,1])

feature_vector = TfidfVectorizer()
feature_vector.fit(dataframe['title'])
# print(feature_vector.vocabulary_)
# print(feature_vector.idf_)
feature_vector = feature_vector.transform(dataframe['title'])
print(feature_vector.shape)
print(feature_vector.toarray())
feature_vector = feature_vector.todense()
fv_dataframe=pd.DataFrame(feature_vector)

X_train, X_test, y_train, y_test = train_test_split(fv_dataframe, Y, random_state=2, test_size=0.2)

n_classes = y_test.shape[1]

classifier=SVC(kernel='linear')

y_score = classifier.fit(X_train,y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
plt.plot(fpr[0],tpr[0])
# plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
