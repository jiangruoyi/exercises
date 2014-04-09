""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
- How general is your classifier?
Answer: The generalization performance at least for given data sets is great. The test_deals.txt
has no ground truth, but on good_deals.txt (positive) and bad_deals.txt (negative), the avg accuracy is around 93% with standard
deviation 0.01. 
The model itself is not complicated, though I have tried a few. The most important part is feature extraction. 
In the code bellow, two type features: 1) generic NLP features; 2) Emperica features
1) generic NLP features: key words from good deals only (removing stop words and normalize words) feature, with 1 representing if a word 
reprenting the word occuring in the deal, and 0 otherwise;
2) Emperical features: based on my observation, the good deals are typically long and contains word like "save" "\d+%" "off".
So I include 3 emperical features: number of key words and if the deal contains "save", "off".  

Without emperical features, the classification performance is very bad (around 60%) since each deal is very short and no big difference between good and bad deals.
I tried liear svm, decision tree, logistic regression with Lasso penalty, naive Bayesian, among which lienar SVM performs the best.
- How did you test your classifier?
Answer: Randomly sample 1/6 of training data as test data, use 5/6 as training, build model and select parameters. Repat the process for 20 times.
Finally compute average accuracy and train the model on all the training data, and apply to test_deals.txt. However 
test data is unlabeled hence I only output the prediction result. 
By manually checking, line 42,50,53 mentioned coupon codes, and our prediction results are all 1! 
"""
import nltk
import re
import numpy as np
from sklearn import svm
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
if ('off' in stopwords):
    stopwords.remove('off')
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word
 
def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    word= word.lower()
    accepted = bool(2 <= len(word) <= 40 and word not in stopwords)
    return accepted
def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
def ml_features(document): 
    document_words = set(document) 
    features = [0]*len(word_features)
    for i in range(len(word_features)): #word in word_features:
        if  (word_features[i] in document_words):       
            features[i] = 1
        else:
            features[i] = 0        
    features.append(len(document))
    if 'save' in document:
        features.append(1)
    else:
        features.append(0) 
    if 'off' in document:
        features.append(1)
    else:
        features.append(0)   
    return features
def docs_extractor(file_name):
    docs=[];
    with open(file_name) as fp:
        for line in fp:
            if len(line) ==0:
                continue
            line=line.strip()        
            toks = re.split('[\s+|\.|-|\/]',line)
            #print toks
            terms = [normalise(w) for w in toks if acceptable_word(w)] 
            docs += [terms]
    fp.closed
    return docs
good_docs=docs_extractor('../data/good_deals.txt');
bad_docs=docs_extractor('../data/bad_deals.txt');
# extract key word features from good deals.txt
all_words = [word for wlist in good_docs for word in wlist]
# caculate key word distributions 
all_words_dist = nltk.FreqDist(w for w in all_words)
# Here we can select top key features, since deals are short and feature size is small, use them all.
word_features = all_words_dist.keys() 
# build data matrix 
featuresets = [ml_features(d) for d in good_docs] + [ml_features(d) for d in bad_docs]
# build label vector
labels = [1]*len(good_docs) + [-1] *len(bad_docs)
for i in range(len(labels)):
    featuresets[i].append(labels[i])
trainData =np.array(featuresets)
sizes = trainData.shape
X = trainData [:,0:sizes[1]-1]
y = trainData [:,sizes[1]-1]
avg_scores = list()
# model selection
Clist = [pow(2,2),pow(2,0),pow(2,-2),pow(2,-4),pow(2,-6),pow(2,-8)]
for c in Clist:
    scores = list()
    for i in range(20): #repeat 20 times, for each round, randomly sample 10 as testing and the rest for training
        np.random.shuffle(trainData); #shuffle the data
        X = trainData [:,0:sizes[1]-1]
        y = trainData [:,sizes[1]-1]
        X_train = np.concatenate((X[0:25],X[35:60]),axis=0)
        X_test = X[25:35]
        y_train=np.concatenate((y[0:25],y[35:60]),axis=0)
        y_test=y[25:35]
#        X_train = X[0:50]
#        X_test = X[50:60]
#        y_train=y[0:50]
#        y_test=y[50:60]
        svc = svm.SVC(C=c, kernel='linear')
        scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
        #print scores
#now load test data
    avg_scores.append(np.mean(scores)) 
print avg_scores
# select the best C
c= Clist[avg_scores.index(np.array(avg_scores).max())]
clf = svm.SVC(C=c, kernel='linear')
# train a model on all the labeled data given C
clf.fit(X, y)
# load test data
test_docs = docs_extractor('../data/test_deals.txt')
# extract data matrix for test data
testFeaturesets = [ml_features(d) for d in test_docs]
# make prediction, but test data is unlabeled hence only output the prediction result
yhat = clf.predict(testFeaturesets)
#By manually checking, line 42,50,53 mentioned coupon codes, and our prediction results are all 1! 
print yhat
np.savetxt("test_result.csv", yhat, delimiter="\n") 
