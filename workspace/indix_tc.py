# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 15:21:14 2016

@author: ekardee
"""
import nltk
import string
import pandas as pd
import numpy as np
import operator
from collections import Counter
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction import text

tknzr = TweetTokenizer()
data = defaultdict(list)
tokenized_text=[]
text_after_stop_words=[]

def read_file(filename):
    df = pd.read_csv(filename ,sep="\t",low_memory = False)
    return df
    
def read_fileData(dataFormatter):
    for row in dataFormatter.itertuples():
        data[row[3]].append(row [1])
    text = data["425"]
    print(text)
    return data

def tokenizer(productTitles):
    for t in productTitles:
        tokenized_text.append(tknzr.tokenize(t))
    return tokenized_text
    
    
def remove_stopwords():
  punctuation = list(string.punctuation)
  sentiment_stop_words = ['rt','via','RT','\\x80', 'I','i',':)',';)',':/',':(']
  stop_words = text.ENGLISH_STOP_WORDS
  stop_words = stop_words.union(punctuation)
  stop_words = stop_words.union(sentiment_stop_words)
  print len(stop_words) 
  return stop_words  

def perform_stopwords(text_tokenized,stop_words):
    for terms in text_tokenized:
        text_after_stop_words.append([i for i in terms if i not in stop_words])
    return text_after_stop_words

def process(title_data_stop_words):
    featureContent =[]
    featureContents=[]
    for idx, value in enumerate(title_data_stop_words):
        featureContent =[]
        for term in value:
            featureContent.append(term.encode('ascii','ignore'))
        str = " ".join(featureContent)
        featureContents.append(filter(None, str))
    return featureContents
    
file_name="C:\\DEEPAK\\INDIX\\classification_train1\\classification_train.tsv"
dataFormatter=read_file(file_name)
fileData = read_fileData(dataFormatter)
title_data = fileData["425"]
text_tokenized = tokenizer(title_data)
#print(text_tokenized)
stop_words = remove_stopwords()
text_stop_words = perform_stopwords(text_tokenized,stop_words)
text_stop_words
corpus = process(text_stop_words)
print(corpus)
## saving the corpus as a unlabeled document ##
np.savetxt(r"C:\\DEEPAK\\INDIX\\classification_train1\\unlabeledDoc.txt",corpus, fmt='%s')






import pandas as pd
import numpy as np

from collections import defaultdict

filename =  "C:\\DEEPAK\\INDIX\\classification_train1\\classification_train.tsv"
df = pd.read_csv(filename ,sep="\t",low_memory = False)
data = defaultdict(list)

for row in df.itertuples():
    data[row[3]].append(row [1])

text = data["425"]
print(text)



import operator
from collections import Counter
tokenized_text = []
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
for t in text:
    tokenized_text.append(tknzr.tokenize(t))

tokenized_text


import nltk
import string
from sklearn.feature_extraction import text
punctuation = list(string.punctuation)
tweet_stop_words = ['rt','via','RT','\\x80', 'I','i',':)',';)',':/',':(']
stop_words = text.ENGLISH_STOP_WORDS
print len(stop_words)
punctuation
stop_words = stop_words.union(punctuation)
print len(stop_words)
stop_words = stop_words.union(tweet_stop_words)
stop_words
print len(stop_words) 


text_after_stop_words=[]
for terms in tokenized_text:
    text_after_stop_words.append([i for i in terms if i not in stop_words])

text_after_stop_words


featureContent =[]
featureContents=[]
for idx, value in enumerate(text_after_stop_words):
    featureContent =[]
    for term in value:
        featureContent.append(term.encode('ascii','ignore'))
    str = " ".join(featureContent)
    featureContents.append(filter(None, str))

np.savetxt(r"C:\\DEEPAK\\INDIX\\classification_train1\\testtest.txt",featureContents, fmt='%s')

############### Apply LDA to identify label #################


from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
n_features = 1000
n_samples = 2000
n_topics = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
topicList=[] 
def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
         #print(" ".join([feature_names[i]
          #              for i in topic.argsort()[:-n_top_words - 1:-1]]))
         topicList.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]).encode('utf-8'))
    print("Deepak printing **************")
    print(topicList)
    print()
# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=20, max_features=n_features,
                                stop_words='english')
t0 = time()
corpus = featureContents
tf = tf_vectorizer.fit_transform(corpus)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features,"
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tf)
#exit()
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
print_topics(lda, tf_feature_names, n_top_words)

###########################################################################

########### Naive Bayes Classfier ##########################
#############Classification on the blind data set################

from nltk.classify import PositiveNaiveBayesClassifier


def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)

data_425_sentences = topicList
various_sentences = [ 'The President did not comment',
                       'I lost the keys',
                       'The team won the game',
                       'Sara has two kids',
                       'The ball went off the court',
                       'They had the ball for the whole game',
                       'The show is over' ]

data_425_featuresets = list(map(features, data_425_sentences))
classifier = PositiveNaiveBayesClassifier.train(data_425_featuresets,
                                                 unlabeled_featuresets)
                                                 
classifier.classify(features('The cat is on the table'))                                                 
classifier.classify(features('sata cable'))

#############################################################

c_filename =  "C:\\DEEPAK\\INDIX\\classification_blind_set_corrected\\classification_blind_set_corrected.tsv"

c_df = pd.read_csv(c_filename ,sep="\t",low_memory = False)
c_data = defaultdict(list)


for c_row in c_df.itertuples():
    c_data[c_row[2]].append(c_row [1])

print(c_data.keys())
c_text = c_data[425]
print(c_text)



print featureContent

str = " ".join(featureContents)

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = featureContents
print corpus
vectorizer = TfidfVectorizer(min_df=100)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
vect_tweets=  dict(zip(vectorizer.get_feature_names(), idf))
print vect_tweets


np.savetxt(r"C:\\DEEPAK\\INDIX\\classification_train1\\testtest1.txt",str, fmt='%s')

 
text_file = open("C:\\DEEPAK\\INDIX\\classification_train1\\testtest.txt", "w")
text_file.write("%s" % data)
text_file.close()    
df5 = df.head(n=10)


 measurements = [
...     {'city': 'Dubai', 'temperature': 33.},
...     {'city': 'London', 'temperature': 12.},
...     {'city': 'San Fransisco', 'temperature': 18.},
... ]

print measurements

from sklearn.feature_extraction import DictVectorizer
 vec = DictVectorizer()

vec.fit_transform(measurements).toarray()

vec.get_feature_names()


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
print vectorizer                     

 corpus = [
...     'This is the first document.',
...     'This is the second second document.',
...     'And the third one.',
...     'Is this the first document?',
... ]

corpus = featureContents
 X = vectorizer.fit_transform(corpus)
 X 
 
analyze = vectorizer.build_analyzer()
print analyze

vectorizer.get_feature_names()

np.savetxt(r"C:\\DEEPAK\\INDIX\\classification_train1\\testtest1.txt",
vectorizer.get_feature_names(), fmt='%s')

X.toarray()

vectorizer.vocabulary_.get('document')

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=20, #max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(corpus)

## Generated the feature set. - Now apply naive Bayes classifier for Blind 
## data Set
import numpy as np
import lda
X = corpus
X
vocab = lda.datasets.load_reuters_vocab()
vocab
titles = lda.datasets.load_reuters_titles()
X.shape
X.sum()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(corpu)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
