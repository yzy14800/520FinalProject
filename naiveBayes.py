# FileName: naiveBayes.py
# Author: Zhengyang Yuan
# Time Created: Nov. 30th, 2015
from collections import Counter
import math
import features
import numpy as np
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

class NaiveBayesClassifier():
    def __init__(self, k,f):
        self.k = k
        #feature extraction type
        #0->basic,1->advanced
        self.f=f

    def calPriorDistribution(self, labels):
        "calculate P(Y)"
        self.dist = Counter(labels)
        self.prior = {}
        for l in labels:
            self.prior[l] = float(self.dist[l]) / len(labels)
        return self.prior

    def calConditionalProbabilities(self, data, labels):
        "calculate P(F_i|Y)"
        occurrence = {}
        "first count the occurrence of all labels "
        for i in range(len(labels)):
            l = labels[i]
            feature = features.featuresExtract(data[i],self.f)
            if l not in occurrence:
                occurrence[l] = np.array(feature)
            else:
                occurrence[l] += np.array(feature)
        self.conds = {}
        "then estimate the conditional probabilities with Adaptive Smoothing"
        for l in labels:
            self.conds[l] = np.divide(occurrence[l] + self.k, float(self.dist[l] + self.k * 2))
        return self.conds

    def calLogJointProbabilities(self, datum):
        "calculate the log of joint probability"
        logJoint = {}
        feature = features.featuresExtract(datum,self.f)
        for l in self.dist.keys():
            "the log of P(f_i=1|Y=y)"
            logConds = np.log(self.conds[l])
            "the log of P(f_i=0|Y=y)"
            logCondsC = np.log(1 - self.conds[l])
            """
             feature is an Indicator array of which features equal to 1
             1-feature is an Indicator array of which features equal to 0
             sum of dot product between Indicator array and logP(f_i|Y=y) calculates total
            """
            logJoint[l] = np.sum(np.array(feature) * logConds, dtype=float)
            logJoint[l] += np.sum((1 - np.array(feature)) * logCondsC, dtype=float)
            "adding up the log of P(Y=y)"
            logJoint[l] += math.log(self.prior[l])
        return logJoint

    @timing
    def train(self, trainData, trainLabels):
        self.calPriorDistribution(trainLabels)
        self.calConditionalProbabilities(trainData, trainLabels)

    def classify(self, testData):
        guess = []
        self.posteriors = []
        for datum in testData:
            post = self.calLogJointProbabilities(datum)
            guess.append(np.argmax(post.values()))
            self.posteriors.append(post)
        return guess
