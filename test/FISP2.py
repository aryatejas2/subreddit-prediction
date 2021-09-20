# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:00:57 2019

@author: Karshit
"""

#Importing Libraries
import praw
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim
import IPython
from wordcloud import WordCloud,STOPWORDS

#%% For Grade B

#Extracting Data From Reddit

reddit = praw.Reddit(client_id='DSSGTfLAEa09zg', 
                     client_secret="XHFVNzZ0pubvb-Tzq5wdENWzTv4",
                     password='12345678', user_agent='Project2',
                     username='ksssh18')

posts1 = reddit.subreddit('cricket').hot(limit=600)
posts2 = reddit.subreddit('nba').hot(limit=600)
subred1_text = []
subred1_titles = []
subred2_text = []
subred2_titles = []
subred3_text = []
subred3_titles = []
for post in posts1:
    subred1_titles.append(post.title)
    subred1_text.append(post.selftext)
    
for post in posts2:
    subred2_titles.append(post.title)
    subred2_text.append(post.selftext)


#Prepoccessing Data

#Coverting data to lowercase
subred1_text=[x.lower() for x in subred1_text]
subred1_titles=[x.lower() for x in subred1_titles]
subred2_text=[x.lower() for x in subred2_text]
subred2_titles=[x.lower() for x in subred2_titles]

#Removing Punctuations
subred1_text = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in subred1_text]
subred1_titles = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in subred1_titles]
subred2_text = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in subred2_text]
subred2_titles = [''.join(c for c in s if c not in string.punctuation) 
                                                        for s in subred2_titles]

#Creating dataframes
subred1_dataframe = pd.DataFrame({'text': subred1_text, 'title': subred1_titles})
subred1_dataframe['target'] = 1

subred2_dataframe = pd.DataFrame({'text': subred2_text, 'title': subred2_titles})
subred2_dataframe['target'] = 0

dataframe = pd.concat([subred1_dataframe, subred2_dataframe])
y=dataframe[['target']]
y=y.reset_index()
y=y.drop(['index'], axis=1)

#Word Cloud
wordcloud = WordCloud(stopwords= STOPWORDS, background_color= 'white',
                        width= 3000, height= 2000).generate(str(dataframe))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#LDA Topic Modeling
STOPWORDS = stopwords.words('english')
tokenized_data = []
for t in dataframe['title']:
    tokenized_text = word_tokenize(t)
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS]
    tokenized_data.append(cleaned_text)
dictionary = corpora.Dictionary(tokenized_data)
corpus = [dictionary.doc2bow(text) for text in tokenized_data]
lda_model = models.LdaModel(corpus= corpus, id2word= dictionary)
vis = pyLDAvis.gensim.prepare(lda_model,corpus,dictionary)
pyLDAvis.save_html(vis,'LDA.html')

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

X_train_subred1=X[0:300]
X_dev_subred1=X[300:450]
X_test_subred1=X[450:600]

X_train_subred2=X[600:900]
X_dev_subred2=X[900:1050]
X_test_subred2=X[1050:1200]

X_train=pd.concat([X_train_subred1, X_train_subred2])
X_dev=pd.concat([X_dev_subred1, X_dev_subred2])
X_test=pd.concat([X_test_subred1, X_test_subred2])

y_train_subred1=y[0:300]
y_dev_subred1=y[300:450]
y_test_subred1=y[450:600]

y_train_subred2=y[600:900]
y_dev_subred2=y[900:1050]
y_test_subred2=y[1050:1200]

y_train=pd.concat([y_train_subred1, y_train_subred2])
y_dev=pd.concat([y_dev_subred1, y_dev_subred2])
y_test=pd.concat([y_test_subred1, y_test_subred2])

#SVM Classifier
svm_classifier=SVC()
svm_score=svm_classifier.fit(X_train,y_train)
svm_predictions = svm_classifier.predict(X_test)

svm_best_param=[{'kernel':['linear','rbf','poly'],'C':[1,10,100],'gamma':[0.1,0.01,0.001]}]
svm_gs=GridSearchCV(svm_classifier,svm_best_param,cv=5)
svm_gs.fit(X_dev, y_dev)
print(svm_gs.best_params_)

svm_classifier=SVC(C=10,gamma=0.1,kernel='linear',probability=True, random_state=2)
svm_score=svm_classifier.fit(X_train,y_train)
svm_predictions = svm_classifier.predict(X_test)

#Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svm_predictions)
svm_cm = pd.DataFrame(cm_svm, range(2), range(2))
sn.set(font_scale=1.7)
sn.heatmap(svm_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Plotting Learning Curve for SVM
y_pred_svm = svm_classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_svm)
auc = metrics.roc_auc_score(y_test, y_pred_svm)
plt.plot(fpr,tpr)
plt.legend(loc=0)

#Classification Report for SVM
print(classification_report(y_test,svm_predictions))

#Learning Curve for SVM
index = np.random.permutation(pd.concat([X_train,X_dev]).index)
train_size, train_score, test_score = learning_curve(svm_classifier,
                                        pd.concat([X_train,X_dev]).reindex(index), pd.concat([y_train,y_dev]).reindex(index),cv=3, scoring='accuracy', train_sizes=[50,150,300,450,600])

train_mean = np.mean(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
plt.plot(train_size, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_size, test_mean, color="#111111", label="Validation score")
plt.title("Learning Curve",fontsize=16)
plt.xlabel("DataSet Size",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.legend(loc=4,fontsize=16)
plt.tight_layout()
plt.show()

#Random Forest CLassifier
rf_classifier=RandomForestClassifier(n_estimators=10, criterion='gini', 
                                     random_state=0)
rf_classifier.fit(X_train,y_train)
rf_predictions = rf_classifier.predict(X_test)

rf_best_param=[{'n_estimators':[10,16,24],'max_depth':[2,4],'criterion':['gini','entropy']}]
rf_gs=GridSearchCV(rf_classifier,rf_best_param,cv=5)
rf_gs.fit(X_dev, y_dev)
print(rf_gs.best_params_)

rf_classifier=RandomForestClassifier(n_estimators=10, criterion='entropy', 
                                     max_depth=4, random_state=2)
rf_classifier.fit(X_train,y_train)
rf_predictions = rf_classifier.predict(X_test)

#Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_predictions)
rf_cm = pd.DataFrame(cm_rf, range(2), range(2))
sn.set(font_scale=1.7)
sn.heatmap(rf_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Plotting Learning Curve for Random Forest
y_pred_rf = rf_classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_rf)
auc = metrics.roc_auc_score(y_test, y_pred_rf)
plt.plot(fpr,tpr)
plt.legend(loc=0)

#Classification Report for Random Forest
print(classification_report(y_test,rf_predictions))

#Classification Report for SVM
print(classification_report(y_test,svm_predictions))

#Learning Curve for Random Forest
index = np.random.permutation(pd.concat([X_train,X_dev]).index)
train_size, train_score, test_score = learning_curve(rf_classifier,
                                        pd.concat([X_train,X_dev]).reindex(index), pd.concat([y_train,y_dev]).reindex(index),cv=3, scoring='accuracy', train_sizes=[50,150,300,450,600])

train_mean = np.mean(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
plt.plot(train_size, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_size, test_mean, color="#111111", label="Validation score")
plt.title("Learning Curve",fontsize=16)
plt.xlabel("DataSet Size",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.legend(loc=4,fontsize=16)
plt.tight_layout()
plt.show()

#Reshaping data for LSTM
Xt=X_train.values
yt=y_train.values
Xtt=X_test.values
y_test2=y_test.values
Xd=X_dev.values
X_train1=Xt.reshape(X_train.shape[0], 1, X_train.shape[1])
X_dev1=Xd.reshape(X_dev.shape[0], 1, X_dev.shape[1])
y_train1=yt.reshape(y_train.shape[0], 1, y_train.shape[1])
X_test2=Xtt.reshape(X_test.shape[0], 1, X_test.shape[1])

#LSTM
lstm_classifier = Sequential()
lstm_classifier.add(LSTM(units = 16, input_shape = (1,X_train.shape[1]),return_sequences = True))
lstm_classifier.add(Dropout(0.2))
lstm_classifier.add(LSTM(units = 8,return_sequences = False))
lstm_classifier.add(Dropout(0.2))
lstm_classifier.add(Dense(units = 1, activation='sigmoid'))
lstm_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
lstm_classifier.fit(X_train1, y_train, epochs = 40, batch_size = 32, validation_data=(X_dev1, y_dev))
score=lstm_classifier.evaluate(X_test2,y_test2)
lstm_predictions=lstm_classifier.predict(X_test2)
print(score)   
print(confusion_matrix(y_test2,lstm_predictions.argmax(axis=1)))

#Confusion Matrix for LSTM
cm_lstm = confusion_matrix(y_test, lstm_predictions.argmax(axis=1))
lstm_cm = pd.DataFrame(cm_lstm, range(2), range(2))
sn.set(font_scale=1.7)
sn.heatmap(lstm_cm, annot=True,annot_kws={"size": 15},fmt='g')

#%% For Grade A

#Importing the Keras libraries and packages

reddit = praw.Reddit(client_id='DSSGTfLAEa09zg', client_secret="XHFVNzZ0pubvb-Tzq5wdENWzTv4",
                         password='12345678', user_agent='Project2',
                     username='ksssh18')

posts1 = reddit.subreddit('cricket').hot(limit=600)
posts2 = reddit.subreddit('nba').hot(limit=600)
posts3 = reddit.subreddit('science').hot(limit=600)
subred1_text = []
subred1_titles = []
subred2_text = []
subred2_titles = []
subred3_text = []
subred3_titles = []
for post in posts1:
    subred1_titles.append(post.title)
    subred1_text.append(post.selftext)
    
for post in posts2:
    subred2_titles.append(post.title)
    subred2_text.append(post.selftext)
    
for post in posts3:
    subred3_titles.append(post.title)
    subred3_text.append(post.selftext)

#Prepoccessing Data

#Coverting data to lowercase
subred1_text=[x.lower() for x in subred1_text]
subred1_titles=[x.lower() for x in subred1_titles]
subred2_text=[x.lower() for x in subred2_text]
subred2_titles=[x.lower() for x in subred2_titles]
subred3_text=[x.lower() for x in subred3_text]
subred3_titles=[x.lower() for x in subred3_titles]

#Removing Punctuations
subred1_text = [''.join(c for c in s if c not in string.punctuation) for s in subred1_text]
subred1_titles = [''.join(c for c in s if c not in string.punctuation) for s in subred1_titles]
subred2_text = [''.join(c for c in s if c not in string.punctuation) for s in subred2_text]
subred2_titles = [''.join(c for c in s if c not in string.punctuation) for s in subred2_titles]
subred3_text = [''.join(c for c in s if c not in string.punctuation) for s in subred3_text]
subred3_titles = [''.join(c for c in s if c not in string.punctuation) for s in subred3_titles]

#Creating dataframes
subred1_dataframe = pd.DataFrame({'text': subred1_text, 'title': subred1_titles})
subred1_dataframe['target'] = 1

subred2_dataframe = pd.DataFrame({'text': subred2_text, 'title': subred2_titles})
subred2_dataframe['target'] = 0

subred3_dataframe = pd.DataFrame({'text': subred3_text, 'title': subred3_titles})
subred3_dataframe['target'] = 2

dataframe = pd.concat([subred1_dataframe, subred2_dataframe, subred3_dataframe])
y=dataframe[['target']]
y=y.reset_index()
y=y.drop(['index'], axis=1)

#Word Cloud
wordcloud = WordCloud(stopwords= STOPWORDS, background_color= 'white',
                        width= 3000, height= 2000).generate(str(dataframe))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#LDA Topic Modeling
STOPWORDS = stopwords.words('english')
tokenized_data = []
for t in dataframe['title']:
    tokenized_text = word_tokenize(t)
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS]
    tokenized_data.append(cleaned_text)
dictionary = corpora.Dictionary(tokenized_data)
corpus = [dictionary.doc2bow(text) for text in tokenized_data]
lda_model = models.LdaModel(corpus= corpus, id2word= dictionary)
vis = pyLDAvis.gensim.prepare(lda_model,corpus,dictionary)
pyLDAvis.save_html(vis,'LDA.html')

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

X_train_subred1=X[0:300]
X_dev_subred1=X[300:450]
X_test_subred1=X[450:600]

X_train_subred2=X[600:900]
X_dev_subred2=X[900:1050]
X_test_subred2=X[1050:1200]

X_train_subred3=X[1200:1500]
X_dev_subred3=X[1500:1650]
X_test_subred3=X[1650:1800]

X_train=pd.concat([X_train_subred1, X_train_subred2, X_train_subred3])
X_dev=pd.concat([X_dev_subred1, X_dev_subred2, X_dev_subred3])
X_test=pd.concat([X_test_subred1, X_test_subred2, X_test_subred3])
X_train=X_train.drop(['index'], axis=1)
X_dev=X_dev.drop(['index'], axis=1)
X_test=X_test.drop(['index'], axis=1)

y_train_subred1=y[0:300]
y_dev_subred1=y[300:450]
y_test_subred1=y[450:600]

y_train_subred2=y[600:900]
y_dev_subred2=y[900:1050]
y_test_subred2=y[1050:1200]

y_train_subred3=y[1200:1500]
y_dev_subred3=y[1500:1650]
y_test_subred3=y[1650:1800]

y_train=pd.concat([y_train_subred1, y_train_subred2, y_train_subred3])
y_dev=pd.concat([y_dev_subred1, y_dev_subred2, y_dev_subred3])
y_test=pd.concat([y_test_subred1, y_test_subred2, y_test_subred3])

#SVM Classifier
svm_classifier2=SVC(C=10,gamma=0.1,kernel='linear',probability=True, random_state=2)
svm_classifier2.fit(X_train,y_train)
svm_predictions2 = svm_classifier2.predict(X_test)

#Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svm_predictions2)
svm_cm = pd.DataFrame(cm_svm, range(3), range(3))
sn.set(font_scale=1.7)
sn.heatmap(svm_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Learning Curve for SVM
index = np.random.permutation(pd.concat([X_train,X_dev]).index)
train_size, train_score, test_score = learning_curve(svm_classifier2,
                                        pd.concat([X_train,X_dev]).reindex(index), pd.concat([y_train,y_dev]).reindex(index),cv=3, scoring='accuracy', train_sizes=[50,150,300,450,600])

train_mean = np.mean(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
plt.plot(train_size, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_size, test_mean, color="#111111", label="Validation score")
plt.title("Learning Curve",fontsize=16)
plt.xlabel("DataSet Size",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.legend(loc=4,fontsize=16)
plt.tight_layout()
plt.show()

#Classification Report for SVM
print(classification_report(y_test,svm_predictions2))

#Random Forest CLassifier
rf_classifier2=RandomForestClassifier(n_estimators=10, criterion='gini', 
                                     random_state=2)
rf_classifier2.fit(X_train,y_train)
rf_predictions2 = rf_classifier2.predict(X_test)

#Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_predictions2)
rf_cm = pd.DataFrame(cm_rf, range(3), range(3))
sn.set(font_scale=1.7)
sn.heatmap(rf_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Classification Report for Random Forest
print(classification_report(y_test,rf_predictions2))

#Learning Curve for Random Forest
index = np.random.permutation(pd.concat([X_train,X_dev]).index)
train_size, train_score, test_score = learning_curve(rf_classifier2,
                                        pd.concat([X_train,X_dev]).reindex(index), pd.concat([y_train,y_dev]).reindex(index),cv=3, scoring='accuracy', train_sizes=[50,150,300,450,600])

train_mean = np.mean(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
plt.plot(train_size, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_size, test_mean, color="#111111", label="Validation score")
plt.title("Learning Curve",fontsize=16)
plt.xlabel("DataSet Size",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.legend(loc=4,fontsize=16)
plt.tight_layout()
plt.show()

#Artificial Neural Network
nn_classifier = Sequential()
nn_classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'subred1u', input_dim = 15091))
nn_classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'subred1u'))
nn_classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
nn_classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

parameters = {'batch_size': [16, 32],
              'epochs': [20, 40],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = nn_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Fitting the ANN to the Training set
nn_classifier.fit(X_train, y_train, batch_size = 16, epochs = 40)
y_pred_nn = nn_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_nn.argmax(axis=1))
print(cm)

#Confusion Matrix for Neural Newtwork
nn_cm = pd.DataFrame(cm, range(3), range(3))
sn.set(font_scale=1.7)
sn.heatmap(nn_cm, annot=True,annot_kws={"size": 15},fmt='g')

#Classification Report for Neural Network
print(classification_report(y_test,y_pred_nn.argmax(axis=1)))