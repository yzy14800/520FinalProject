# FileName: neuralNetwork.py
# Author: Zhengyang Yuan
# Time Created: Dec. 5th, 2015
import features
import numpy as np
import scipy.optimize as opt
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    """
    derivative of sigmoid
    in this function y is already sigmoided
    """
    return y * (1.0 - y)


class NeuralNetworkClassifier():
    def __init__(self, inputNum, hiddenNum, outputNum, dataNum, l):
        """
        input: the number of input neurons (in this case features)
        hidden: the number of hidden neurons (should be tuned)
        output: the number of output neurons (the classifications of image)
        l: lambda
        """
        self.input = inputNum  # without bias node
        self.hidden = hiddenNum  # without bias node
        self.output = outputNum
        self.dataNum = dataNum
        self.l = l

        "allocate memory for activation matrix of 1s"
        self.inputActivation = np.ones((self.input + 1, dataNum))  # add bias node
        self.hiddenActivation = np.ones((self.hidden + 1, dataNum))  # add bias node
        self.outputActivation = np.ones((self.output, dataNum))

        "allocate memory for bias vector"
        self.bias = np.ones((1, dataNum))

        "allocate memory for change matrix of 0s"
        self.inputChange = np.zeros((self.hidden, self.input + 1))
        self.outputChange = np.zeros((self.output, self.hidden + 1))

        "calculate epsilon for randomization"
        self.hiddenEpsilon = np.sqrt(6.0 / (self.input + self.hidden))
        self.outputEpsilon = np.sqrt(6.0 / (self.input + self.output))

        "allocate memory for randomized weights"
        self.inputWeights = np.random.rand(self.hidden, self.input + 1) * 2 * self.hiddenEpsilon - self.hiddenEpsilon
        self.outputWeights = np.random.rand(self.output, self.hidden + 1) * 2 * self.outputEpsilon - self.outputEpsilon

    def setLambda(self, l):
        "update lambda"
        self.l = l

    def feedForward(self, thetaVec):
        "reshape thetaVec into two weights matrices"
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        "hidden activation"
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        "output activation"
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)

        "calculate J"
        costMatrix = self.outputTruth * np.log(self.outputActivation) + (1 - self.outputTruth) * np.log(
            1 - self.outputActivation)
        regulations = (np.sum(self.outputWeights[:, :-1] ** 2) + np.sum(self.inputWeights[:, :-1] ** 2)) * self.l / 2
        return (-costMatrix.sum() + regulations) / self.dataNum

    def backPropagate(self, thetaVec):
        "reshape thetaVec into two weights matrices"
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))

        "calculate lower case delta"
        outputError = self.outputActivation - self.outputTruth
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * dsigmoid(self.hiddenActivation[:-1:])

        "calculate upper case delta"
        self.outputChange = outputError.dot(self.hiddenActivation.T) / self.dataNum
        self.inputChange = hiddenError.dot(self.inputActivation.T) / self.dataNum

        "add regulations"
        self.outputChange[:, :-1].__add__(self.l * self.outputWeights[:, :-1])
        self.inputChange[:, :-1].__add__(self.l * self.inputWeights[:, :-1])

        return np.append(self.inputChange.ravel(), self.outputChange.ravel())

    @timing
    def train(self, trainData, trainLabels, iteration=100):
        "input activation"
        self.inputActivation[:-1, :] = trainData
        "output truth labels"
        self.outputTruth = self.genTruthMatrix(trainLabels)
        "propagate"
        thetaVec = np.append(self.inputWeights.ravel(), self.outputWeights.ravel())
        thetaVec = opt.fmin_cg(self.feedForward, thetaVec, fprime=self.backPropagate, maxiter=iteration)
        self.inputWeights = thetaVec[0:self.hidden * (self.input + 1)].reshape((self.hidden, self.input + 1))
        self.outputWeights = thetaVec[-self.output * (self.hidden + 1):].reshape((self.output, self.hidden + 1))


    def classify(self, feature):
        "input activation"
        "for classify in case the difference of size between trainData and testData "
        if feature.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.input + 1, feature.shape[1]))
            self.hiddenActivation = np.ones((self.hidden + 1, feature.shape[1]))
            self.outputActivation = np.ones((self.output + 1, feature.shape[1]))
        self.inputActivation[:-1, :] = feature

        "hidden activation"
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        "output activation"
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation>0.5).ravel()

    def genTruthMatrix(self, trainLabels):
        truth = np.zeros((self.output, self.dataNum))
        for i in range(self.dataNum):
            label = trainLabels[i]
            if self.output == 1:
                truth[:,i] = label
            else:
                truth[label, i] = 1
        return truth
