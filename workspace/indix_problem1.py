# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:28:15 2016

@author: Deepak k
"""


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
    data = defaultdict(list)
    for row in dataFormatter.itertuples():
        data[row[3]].append(row[1])
    text = data["425"]
    print(text)
    return data

tokenized_text=[]
def tokenizer(productTitles):
    tokenized_text=[]
    for t in productTitles:
        tokenized_text.append(tknzr.tokenize(t))
    return tokenized_text
    
    
def remove_stopwords():
  punctuation = list(string.punctuation)
  sentiment_stop_words = ['rt','via','RT','\\x80', 'I','i',':)',';)',':/',':(']
  stop_words = text.ENGLISH_STOP_WORDS
  stop_words = stop_words.union(punctuation)
  stop_words = stop_words.union(sentiment_stop_words)
  print(len(stop_words))
  return stop_words  
  
tokenized_text=[]
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


############### Apply LDA to identify label #################


from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
n_features = 100
n_samples = 200
n_topics = 10
n_top_words = 20

topicList=[] 

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    

def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
         topicList.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]).encode('utf-8'))
    print(topicList)
    print()
    
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=20, max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf)
print("\n Generating Topics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
print_topics(lda, tf_feature_names, n_top_words)

#######################################################

########### POC On Naive Bayes Classfier ##########################

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
unlabeled_featuresets = list(map(features, various_sentences))

classifier = PositiveNaiveBayesClassifier.train(data_425_featuresets,
                                                 unlabeled_featuresets)
                                                 
classifier.classify(features('The cat is on the table'))                                                 
classifier.classify(features('sata cable'))

#############################################################


def c_read_fileData(dataFormatter):
    c_data = defaultdict(list)
    for row in dataFormatter.itertuples():
        c_data[row[2]].append(row[1])
    text = c_data[425]
    print(text)
    return c_data
c_filename =  "C:\\DEEPAK\\INDIX\\classification_blind_set_corrected\\classification_blind_set_corrected.tsv"
c_dataFormatter=read_file(c_filename)
c_fileData = c_read_fileData(c_dataFormatter)
c_title_data = c_fileData[425]
c_text_tokenized = tokenizer(c_title_data)
print(c_text_tokenized)
stop_words = remove_stopwords()
c_text_stop_words = perform_stopwords(c_text_tokenized,stop_words)
c_text_stop_words
corpus = process(c_text_stop_words)
print(corpus)

data_425_featuresets = list(map(features, corpus))
unlabeled_featuresets = list(map(features, various_sentences))
classifier = PositiveNaiveBayesClassifier.train(data_425_featuresets,
                                                 unlabeled_featuresets)
               
for c in corpus:
    classifier.classify(features( c))
    break
