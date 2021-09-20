# -*- coding: utf-8 -*-
"""
@author: Tejas
@author: Karshit
"""

#Importing Libraries

import string

import matplotlib.pyplot as plt
import pandas as pd
import praw
import pyLDAvis.gensim
import seaborn as sn
from gensim import models, corpora
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from wordcloud import WordCloud, STOPWORDS

#%% For Grade B

#Extracting Data From Reddit

reddit = praw.Reddit(client_id='DSSGTfLAEa09zg', 
                     client_secret="XHFVNzZ0pubvb-Tzq5wdENWzTv4",
                     password='12345678', user_agent='Project2',
                     username='ksssh18')

posts1 = reddit.subreddit('cricket').hot(limit=600)
posts2 = reddit.subreddit('nba').hot(limit=600)
rel_text = []
rel_titles = []
nba_text = []
nba_titles = []
sci_text = []
sci_titles = []
for post in posts1:
    rel_titles.append(post.title)
    rel_text.append(post.selftext)
    
for post in posts2:
    nba_titles.append(post.title)
    nba_text.append(post.selftext)


#Prepoccessing Data

#Coverting data to lowercase
rel_text=[x.lower() for x in rel_text]
rel_titles=[x.lower() for x in rel_titles]
nba_text=[x.lower() for x in nba_text]
nba_titles=[x.lower() for x in nba_titles]

#Removing Punctuations
rel_text = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in rel_text]
rel_titles = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in rel_titles]
nba_text = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in nba_text]
nba_titles = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in nba_titles]

#Creating dataframes
rel_dataframe = pd.DataFrame({'text': rel_text, 'title': rel_titles})
rel_dataframe['target'] = 1

nba_dataframe = pd.DataFrame({'text': nba_text, 'title': nba_titles})
nba_dataframe['target'] = 0

dataframe = pd.concat([rel_dataframe, nba_dataframe])
y=dataframe[['target']]
y=y.reset_index()
y=y.drop(['index'], axis=1)

#Word Cloud
wc = WordCloud(stopwords= STOPWORDS, background_color= 'white',
                        width= 4000, height= 3000).generate(str(dataframe))
plt.imshow(wc)
plt.axis('off')
plt.show()

#LDA Topic Modeling
sp = stopwords.words('english')
tData = []
for x in dataframe['title']:
    tokens = word_tokenize(x)
    withoutSP = [t for t in tokens if t not in sp]
    tData.append(withoutSP)
lex = corpora.Dictionary(tData)
cor = [lex.doc2bow(text) for text in tData]
lda_model = models.LdaModel(corpus= cor, id2word= lex)
LDA1 = pyLDAvis.gensim.prepare(lda_model,cor,lex)
pyLDAvis.save_html(LDA1,'LDA1.html')

#Converting text to feature vectors
feature_vector = TfidfVectorizer()
feature_vector = feature_vector.fit_transform(dataframe['title'])
feature_vector = feature_vector.todense()
fv_dataframe=pd.DataFrame(feature_vector)

feature_vector2 = TfidfVectorizer()
feature_vector2 = feature_vector2.fit_transform(dataframe['text'])
feature_vector2 = feature_vector2.todense()
fv_dataframe2=pd.DataFrame(feature_vector2)

#Splitting data into Train(50%), Development(25%) and Test(25%) sets
X = pd.concat([fv_dataframe, fv_dataframe2], axis=1)
X=X.reset_index()
X=X.drop(['index'], axis=1)

X_train_rel=X[0:300]
X_dev_rel=X[300:450]
X_test_rel=X[450:600]

X_train_nba=X[600:900]
X_dev_nba=X[900:1050]
X_test_nba=X[1050:1200]

X_train=pd.concat([X_train_rel, X_train_nba])
X_dev=pd.concat([X_dev_rel, X_dev_nba])
X_test=pd.concat([X_test_rel, X_test_nba])

y_train_rel=y[0:300]
y_dev_rel=y[300:450]
y_test_rel=y[450:600]

y_train_nba=y[600:900]
y_dev_nba=y[900:1050]
y_test_nba=y[1050:1200]

y_train=pd.concat([y_train_rel, y_train_nba])
y_dev=pd.concat([y_dev_rel, y_dev_nba])
y_test=pd.concat([y_test_rel, y_test_nba])

#SVM Classifier
svm_classifier=SVC(kernel='linear', random_state=2, probability=True)
svm_score=svm_classifier.fit(X_train,y_train)
svm_predictions = svm_classifier.predict(X_test)

#Plotting Learning Curve for SVM
y_pred_svm = svm_classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_svm)
auc = metrics.roc_auc_score(y_test, y_pred_svm)
plt.plot(fpr,tpr)
plt.legend(loc=0)
#
#plt.show()
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                   ]
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(train_t, labels)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()
    
#
#from mlxtend.plotting import plot_learning_curves
#plot_learning_curves(X_train, y_train, X_test, y_test, svm_classifier)
#plt.show()

#Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svm_predictions)
svm_cm = pd.DataFrame(cm_svm, range(2), range(2))
sn.set(font_scale=1.7)
sn.heatmap(svm_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Classification Report for SVM
print(classification_report(y_test,svm_predictions))

#Random Forest CLassifier
rf_classifier=RandomForestClassifier(n_estimators=10, criterion='gini', 
                                     random_state=0)
rf_classifier.fit(X_train,y_train)
rf_predictions = rf_classifier.predict(X_test)

#Plotting Learning Curve for Random Forest
y_pred_rf = rf_classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_rf)
auc = metrics.roc_auc_score(y_test, y_pred_rf)
plt.plot(fpr,tpr)
plt.legend(loc=0)

#Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_predictions)
rf_cm = pd.DataFrame(cm_rf, range(2), range(2))
sn.set(font_scale=1.7)
sn.heatmap(rf_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Classification Report for SVM
print(classification_report(y_test,rf_predictions))

#%% For Grade A

#Importing the Keras libraries and packages

reddit = praw.Reddit(client_id='DSSGTfLAEa09zg', client_secret="XHFVNzZ0pubvb-Tzq5wdENWzTv4",
                         password='12345678', user_agent='Project2',
                     username='ksssh18')

posts1 = reddit.subreddit('cricket').hot(limit=600)
posts2 = reddit.subreddit('nba').hot(limit=600)
posts3 = reddit.subreddit('science').hot(limit=600)
rel_text = []
rel_titles = []
nba_text = []
nba_titles = []
sci_text = []
sci_titles = []
for post in posts1:
    rel_titles.append(post.title)
    rel_text.append(post.selftext)
    
for post in posts2:
    nba_titles.append(post.title)
    nba_text.append(post.selftext)
    
for post in posts3:
    sci_titles.append(post.title)
    sci_text.append(post.selftext)

#Prepoccessing Data

#Coverting data to lowercase
rel_text=[x.lower() for x in rel_text]
rel_titles=[x.lower() for x in rel_titles]
nba_text=[x.lower() for x in nba_text]
nba_titles=[x.lower() for x in nba_titles]
sci_text=[x.lower() for x in sci_text]
sci_titles=[x.lower() for x in sci_titles]

#Removing Punctuations
rel_text = [''.join(c for c in s if c not in string.punctuation) for s in rel_text]
rel_titles = [''.join(c for c in s if c not in string.punctuation) for s in rel_titles]
nba_text = [''.join(c for c in s if c not in string.punctuation) for s in nba_text]
nba_titles = [''.join(c for c in s if c not in string.punctuation) for s in nba_titles]
sci_text = [''.join(c for c in s if c not in string.punctuation) for s in sci_text]
sci_titles = [''.join(c for c in s if c not in string.punctuation) for s in sci_titles]

#Creating dataframes
rel_dataframe = pd.DataFrame({'text': rel_text, 'title': rel_titles})
rel_dataframe['target'] = 1

nba_dataframe = pd.DataFrame({'text': nba_text, 'title': nba_titles})
nba_dataframe['target'] = 0

sci_dataframe = pd.DataFrame({'text': sci_text, 'title': sci_titles})
sci_dataframe['target'] = 2

dataframe = pd.concat([rel_dataframe, nba_dataframe, sci_dataframe])
y=dataframe[['target']]
y=y.reset_index()
y=y.drop(['index'], axis=1)

#Word Cloud
wc = WordCloud(stopwords= STOPWORDS, background_color= 'white',
                        width= 4000, height= 3000).generate(str(dataframe))
plt.imshow(wc)
plt.axis('off')
plt.show()

#LDA Topic Modeling
sp = stopwords.words('english')
tData = []
for x in dataframe['title']:
    tokens = word_tokenize(x)
    withoutSP = [t for t in tokens if t not in sp]
    tData.append(withoutSP)
lex = corpora.Dictionary(tData)
cor = [lex.doc2bow(text) for text in tData]
lda_model = models.LdaModel(corpus= cor, id2word= lex)



LDA2 = pyLDAvis.gensim.prepare(lda_model,cor,lex)
pyLDAvis.save_html(LDA2,'LDA2.html')

#Converting text to feature vectors
feature_vector = TfidfVectorizer()
feature_vector = feature_vector.fit_transform(dataframe['title'])
feature_vector = feature_vector.todense()
fv_dataframe=pd.DataFrame(feature_vector)

feature_vector2 = TfidfVectorizer()
feature_vector2 = feature_vector2.fit_transform(dataframe['text'])
feature_vector2 = feature_vector2.todense()
fv_dataframe2=pd.DataFrame(feature_vector2)

#Splitting data into Train(50%), Development(25%) and Test(25%) sets
X = pd.concat([fv_dataframe, fv_dataframe2], axis=1)
X=X.reset_index()

X_train_rel=X[0:300]
X_dev_rel=X[300:450]
X_test_rel=X[450:600]

X_train_nba=X[600:900]
X_dev_nba=X[900:1050]
X_test_nba=X[1050:1200]

X_train_sci=X[1200:1500]
X_dev_sci=X[1500:1650]
X_test_sci=X[1650:1800]

X_train=pd.concat([X_train_rel, X_train_nba, X_train_sci])
X_dev=pd.concat([X_dev_rel, X_dev_nba, X_dev_sci])
X_test=pd.concat([X_test_rel, X_test_nba, X_test_sci])
X_train=X_train.drop(['index'], axis=1)
X_dev=X_dev.drop(['index'], axis=1)
X_test=X_test.drop(['index'], axis=1)

y_train_rel=y[0:300]
y_dev_rel=y[300:450]
y_test_rel=y[450:600]

y_train_nba=y[600:900]
y_dev_nba=y[900:1050]
y_test_nba=y[1050:1200]

y_train_sci=y[1200:1500]
y_dev_sci=y[1500:1650]
y_test_sci=y[1650:1800]

y_train=pd.concat([y_train_rel, y_train_nba, y_train_sci])
y_dev=pd.concat([y_dev_rel, y_dev_nba, y_dev_sci])
y_test=pd.concat([y_test_rel, y_test_nba, y_test_sci])

#SVM Classifier
svm_classifier2=SVC(kernel='linear', random_state=2)
svm_classifier2.fit(X_train,y_train)
svm_predictions2 = svm_classifier2.predict(X_test)

#Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svm_predictions2)
svm_cm = pd.DataFrame(cm_svm, range(3), range(3))
sn.set(font_scale=1.7)
sn.heatmap(svm_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Random Forest Classifier
rf_classifier2=RandomForestClassifier(n_estimators=10, criterion='gini', 
                                     random_state=0)
rf_classifier2.fit(X_train,y_train)
rf_predictions2 = rf_classifier2.predict(X_test)

#Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_predictions2)
rf_cm = pd.DataFrame(cm_rf, range(3), range(3))
sn.set(font_scale=1.7)
sn.heatmap(rf_cm, annot=True,annot_kws={"size": 15},fmt='g')

Xt=X_train.values
yt=y_train.values
Xtt=X_test.values
y_test2=y_test.values
Xd=X_dev.values
X_train1=Xt.reshape(X_train.shape[0], 1, X_train.shape[1])
X_dev1=Xd.reshape(X_dev.shape[0], 1, X_dev.shape[1])
y_train1=yt.reshape(y_train.shape[0], 1, y_train.shape[1])
X_test2=Xtt.reshape(X_test.shape[0], 1, X_test.shape[1])
#y_test22=y_test2.reshape(y_test.shape[0], 1, y_test.shape[1])

# Initialising the RNN
regressor = Sequential()
regressor.add(LSTM(units = 8, input_shape = (1,X_train.shape[1]),return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 8,return_sequences = False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1, activation='sigmoid'))
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
regressor.fit(X_train1, y_train, epochs = 40, batch_size = 32, validation_data=(X_dev1, y_dev))
score=regressor.evaluate(X_test2,y_test2)
lstm_predictions=regressor.predict(X_test2)
print(score)   
# print(confusion_matrix(y_test2,lstm_predictions.argmax(x=1)))

