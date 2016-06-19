# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:23:08 2016

@author: ekardee
"""
from nltk.classify.api import ClassifierI, MultiClassifierI
from nltk.classify.megam import config_megam, call_megam
from nltk.classify.weka import WekaClassifier, config_weka
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.positivenaivebayes import PositiveNaiveBayesClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify.rte_classify import rte_classifier, rte_features, RTEFeatureExtractor
from nltk.classify.util import accuracy, apply_features, log_likelihood
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.maxent import (MaxentClassifier, BinaryMaxentFeatureEncoding,
                                  TypedMaxentFeatureEncoding,
                                  ConditionalExponentialClassifier)


from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
names
random.shuffle(names)
len(names)
names[0:100]

def gender_features(word):
    return {'last_letter': word[-1]}

gender_features('Gary')

for (n , g) in names:
    print n 
    print g
    break

featuresets = [(gender_features(n), g) for (n, g) in names]
featuresets

len(featuresets)
train_set, test_set = featuresets[500:], featuresets[:500] 
train_set


from nltk import NaiveBayesClassifier

nb_classifier = NaiveBayesClassifier.train(train_set)

nb_classifier.classify(gender_features('Gary'))

nb_classifier.classify(gender_features('Grace'))

from nltk import classify
 
classify.accuracy(nb_classifier, test_set)

nb_classifier.show_most_informative_features(5)

















from nltk.classify import PositiveNaiveBayesClassifier


data_425_sentences=['sata cable ata serial angle right data red inch cables 18 startech combo device 18in internal ft power connector pack', 'esata cable sata external degree 90 ft latch type shielded locking black 6gbps 1m 180 feet uv blue 3ft cables', 'packard hewlett pavilion air adapter hi auto capacity slave dv7 cbi hdd cable right 7200rpm inch 500gb 4pin 15pin 12', 'sata cable ide ata serial drive adapter hard converter hdd data power pin usb new 100 disk raid 15 dual', 'sata startech cable com power adapter lp4 latching inch slimline splitter 6in 12 18 pyo lindy panel aleratec lsata 12in', 'sata cable sas sff new internal pcb hdd cbl toshiba mini 8087 lane 4x angled 10 multi 3ware siig 6g', 'sata usb adapter drive hard cable esata inch hdd disk blue ssd 24 7200rpm 25 1tb micro av 500gb 250gb', 'pin power sata male cable female 15 adapter molex 4pin extension 22 15pin dual splitter 12 inch 2x data bitfenix', 'port ii pci cable profile low ultra card express sata bracket 16 ribbon pc left adapter flat computer monoprice right', 'serial ata cable sata 7pin lite tripp pin signal dell hp 001 300 esata ft 18 controller product 600 straight']

final=[]
for dd in data_425_sentences:
    lll = dd.split()
    final.append(lll)

print(final)
data_425_sentences.split()

sports_sentences = [ 'The team dominated the game',
                      'They lost the ball',
                      'The game was intense',
                      'The goalkeeper catched the ball',
                     'The other team controlled the ball' ]


various_sentences = [ 'The President did not comment',
                       'I lost the keys',
                       'The team won the game',
                       'Sara has two kids',
                       'The ball went off the court',
                       'They had the ball for the whole game',
                       'The show is over' ]

def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)

positive_featuresets = list(map(features, sports_sentences))
positive_featuresets
unlabeled_featuresets = list(map(features, various_sentences))

data_425_featuresets = list(map(features, data_425_sentences))


classifier = PositiveNaiveBayesClassifier.train(data_425_featuresets,
                                                 unlabeled_featuresets)
                                                 
                                                 
classifier.classify(features('The cat is on the table'))                                                 


classifier.classify(features('sata ca'))
True

positive_featuresets = list(map(features, data_425_sentences))
classifier = PositiveNaiveBayesClassifier.train(positive_featuresets)                                             )
   
