# IS620 - Assignment 10
# Program: assignment10.py
# Student: Neil Acampa
# Date:    10/31/16
# Function:



# 1. Choose a corpus of interest and perform a classification

#    First choose the movie rating corpus
#    Select top most frequent words (2000) without stop words
#    Call document_features Train with Test with first 100 records

#    Second Pass 
#    Select top most frequent words (2000) without stop words
#    Call document_features Train on 90% and Test with 10%

#    Third Pass 
#    Select top most frequent words (2000) include stop words
#    Call document_features_all Train with Test with first 100 records

#    Forth Pass 
#    Select top most frequent words (2000) include stop words
#    Call document_features_all Train on 90% and Test with 10%


#    Fith Pass
#    Extract top 2000 most frequent words  
#    Create a feature function that counts the frequency of word(i) in document(j) 
#    and assignes feature with frequency
#    Use AFINN sentiment value (-5 to +5) and Harvard's Inquirybasic.xls with
#    a word list designated Positive/Negative which is converted to a +1 and -1 respectively
#    Each word feature is returned with a sentiment value


#    Did not do this yet
#    Evaluate the classifier: Show document's polarity verses overall document sentiment value
#    see if they match

#    Did not do this yet
#    Evaluate the classifier (# of times Word(i) in doc(j) with word sentiment rating (k))
#    Word(i) = "Amazing" occurs in Doc(j) = 5 times with Word Sentiment Rating(k) = 7 
#    Return features {Contains Word(i): True  Frequency: 5 and Word Sentiment Rating:  35
#    Currently shows something line Pos/Neg 11 to 1 - 11 times Positive
#    Try to use Frequency and Rating

#    For Fith read in AFINN-111.txt

from __future__ import absolute_import 
from __future__ import division
import re
import os 
import math
import decimal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import networkx as nx
import random
from urllib import urlopen
import nltk
nltk.download('gutenberg')
from nltk import word_tokenize
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('stopwords')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')


linelst=[]
lines  = ""
allwords          = []   # Contains all words
sentimentdict     = []   # Contains words from AFINN
sentimentdictcnt  = []   # Contains a sentiment value (-5 to +5) for sentiment word
sentiment         = []   # 2-D matrix sentiment word and value

 
sentimentdoc      = []   # Total document sentiment value using AFINN
sentimentdoccat   = []   # Document category
dindx             = 0    # Document index


masterdict        = []
masterdictcnt     = []
masterdictcntpos  = []
masterdictcntneg  = []
masterdictcat     = []
# Table Elements

fheadings      = [] 

fheadings.append("Words without stopwords                     ")
fheadings.append("Words without stopwords                90/10")
fheadings.append("Words with stopwords                        ")
fheadings.append("Words with stopwords                   90/10")
fheadings.append("Words without stopwords and Sentiment Value ")



rejectchars = [',','.','?','<','>','!','"','-','%','&','#','(',')','*',';'];
rcnt = len(rejectchars);


def remove_characters(word):
  """Replace special characters in the word"""
  
  for i in range(rcnt):
    rchar = rejectchars[i]
    if rchar in word:
      word = word.replace(rchar,"")

  return word


def remove_symbols(word):
  """Replace symbols in the word"""
  w = len(word)
  word = (ord(c) for c in word) 
  word = map(lambda x:x if x<123 or x>255 else " ", word)
  newword=""
  for c in range(w):
    if word[c] <> " ":
      newword += chr(word[c]);
  
  return newword


def find_word(word, sentimentdict):
  """Find and return index of word in sentiment dictionary"""

  masterlen = len(sentimentdict)
  find=0
  temp="x"
  try:
   temp = sentimentdict.index(word);
   return temp
  except ValueError:
   return temp



def document_features(document):
  docwords = set(document)
  features = {}
  for word in wordfeatures:
    features['contains(%s)' % word] = (word in docwords)
 
  return features


def document_features_all(document):
  docwords = set(document)
  features = {}
  for word in wordfeaturesall:
    features['contains(%s)' % word] = (word in docwords)
 
  return features


def document_features_sentiment(document):
  docwords = set(document)
  features = {}
  totalsentimentvalue = 0
  sl = len(sentimentdict)
  for word in wordfeaturesSent:
    tword = word.encode('ascii')
    findx = find_word(tword, sentimentdict)
    sentimentval = 0
    if (findx != "x"):
      sentimentval = int(sentimentdictcnt[findx]) + 0
      totalsentimentvalue = totalsentimentvalue + sentimentval
    
    
    
    features['contains(%s SV:%s)' % (word, sentimentval)] = (word in docwords)

 
  return features






filepath=""
temp    =""
tokens  = ""
valid   = 0
p       = 1
cwd = os.getcwd()
corpus     = "AFINN"
fullcorpus = "AFINN-111.txt"
currfilepath = str(cwd) + "\AFINN-111.txt"
print currfilepath
print ("Enter the Full File Path including the File")
print ("or Press return to use current File Path %s") % (currfilepath)
filepath = raw_input("Please enter the File Path now ")
valid = 0
if filepath == "":
   filepath = currfilepath

 
try:
       f = open(filepath,"r")
       try:
         valid=1
         x =0
         j=0
         for lines in f:
           lines = lines.rstrip()
           temp = lines.split("\t");
           sentimentdict.append(temp[0])
           sentimentdictcnt.append(temp[1])
                            
       finally:
            f.close()
         
except IOError:
       print ("File not Found - Program aborting")

if not(valid):
     exit()



cwd = os.getcwd()
corpus     = "inquirebasic.txt"
fullcorpus = "inquirebasic.txt"
currfilepath = str(cwd) + "\inquirebasic.csv"
print currfilepath
print ("Enter the Full File Path including the File")
print ("or Press return to use current File Path %s") % (currfilepath)
filepath = raw_input("Please enter the File Path now ")
valid = 0
if filepath == "":
   filepath = currfilepath

 
try:
       f = open(filepath,"r")
       try:
         valid=1
         x =0
         j=0
         for lines in f:
           lines = lines.rstrip()
           temp = lines.split(",");
           word = temp[0].lower()
           sentimentdict.append(word)
           if (temp[1] != ""):
             # give positive words + 1
             sentimentdictcnt.append(1)
           else:
             # give negative words -1 
             if (temp[2] != ""):
               sentimentdictcnt.append(-1)
             else:
               sentimentdictcnt.append(0)
                            
       finally:
            f.close()
         
except IOError:
       print ("File not Found - Program aborting")

if not(valid):
     exit()
 

sentimentcnt = len(sentimentdict)
for i in range(sentimentcnt):
  sentiment.append([sentimentdict[i], sentimentdictcnt[i]])


 
sl = len(sentimentdict)

print
print
print("Getting Movie Review words")
results = []
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
dl = len(documents)

print
print("Getting top 2000 most frequent movie review words subtracting Stopwords")
# Top most frequent words in movie reviews without stopwords
fd              = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w.lower() not in stopwords)
wordfeatures    = fd.keys()[:2000]

print
print("Getting top 2000 most frequent movie review words including Stopwords")
# Top most frequent words in movie reviews with stopwords
fdall           = nltk.FreqDist(w.lower() for w in movie_reviews.words())
wordfeaturesall = fdall.keys()[:2000]


print
print("Getting top 200 most frequent words subtracting Stopwords for sentiment features")
# Top most frequent words in movie reviews without stopwords for sentiment value
#fd              = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w.lower() not in stopwords)
wordfeaturesSent= fd.keys()[:500]



# Features Positive/Negative with stop words removed
print
print("Processing Features without stopwords")
print("Training from Document 100 to 2000, Testing on the first 100")
print
featuresets=[]
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set  = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)
results.append(accuracy)
print(classifier.show_most_informative_features(10))


# Features Positive/Negative with stop words removed
print
print("Processing Features without stop words")
print("Training on 90 percent, Testing on 10 percent")
print
# Train on 90%, Test on 10%
featuresets = []
featuresets = [(document_features(d), c) for (d,c) in documents]
fcnt = len(featuresets)
testlim  = int(fcnt*.10)
trainlim = testlim +1
train_set, test_set  = featuresets[trainlim:], featuresets[:testlim]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)
results.append(accuracy)
print(classifier.show_most_informative_features(10))



# Features Positive/Negative with stop words included
print
print("Processing Features with stop words included")
print("Training from Document 100 to 2000, Testing on the first 100")
print
featuresets = []
featuresets = [(document_features_all(d), c) for (d,c) in documents]
train_set, test_set  = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)
results.append(accuracy)
print(classifier.show_most_informative_features(10))

print
print("Processing Features with stop words included")
print("Training on 90 percent, Testing on 10 percent")
print
featuresets = []
featuresets = [(document_features_all(d), c) for (d,c) in documents]
fcnt = len(featuresets)
testlim  = int(fcnt*.10)
trainlim = testlim +1
train_set, test_set  = featuresets[trainlim:], featuresets[:testlim]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)
results.append(accuracy)
print(classifier.show_most_informative_features(10))



# Features Positive/Negative with stop words removed show word sentiment value
print
print("Processing Features without stopwords")
print("Training from Document 100 to 500, Testing on the first 100")
print("Display word with sentiment value")
print
featuresets=[]
featuresets = [(document_features_sentiment(d), c) for (d,c) in documents]
train_set, test_set  = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)
results.append(accuracy)
print(classifier.show_most_informative_features(10))



print
print
l = len(results)
print(l)
print
print ("%s\t%s\t%s") % ("Feature" , "Feature Desc                              ", "Accuracy")
indx = 0
for i in range(l):
  indx = indx + 1
  print("%d\t%s\t%.4f") % (indx, fheadings[i], results[i])
  



  
    
