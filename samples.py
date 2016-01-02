# FileName: samples.py
# Author: Zhengyang Yuan
# Time Created: Nov. 30th, 2015

from naiveBayes import NaiveBayesClassifier
import numpy as np
from perceptron import PerceptronClassifier
from neuralNetwork import NeuralNetworkClassifier

class Datum:
    """
    the class Datum is pixels representing the digits or the face images
    """

    def __init__(self, data, width, height):
        self.width = width
        self.height = height
        if data == None:
            data = [[' ' for i in range(width)] for j in range(height)]
        self.pixels = mapToInteger(data)

    def getPixel(self, row, col):
        # return the value of a specific pixel at column and row
        return self.pixels[row][col]

    def getPixels(self):
        "return all the pixels in a matrix"
        return self.pixels

def loadImagesFile(filename, n, width, height):
    "load n images from specific file and return a list of object"
    images = []
    with open(filename, 'r') as f:
        for i in range(n):
            pixels = []
            for j in range(height):
                pixels.append(list(f.readline())[:-1])
            if len(pixels[0]) < width - 1:
                "touching the end of file"
                break
            images.append(Datum(pixels, width, height))
    return images


def loadLabelsFile(filename, n):
    "load n labels from specific file and return a list of integer"
    labels = []
    with open(filename, 'r') as f:
        for i in range(n):
            line = f.readline()
            if (line == ""):
                break
            labels.append(int(line.strip()))
    return labels


def mapToInteger(data):
    "recursively map each pixel in the datum to an integer"
    if (type(data) != type([])):
        return convertPixel(data)
    else:
        return map(mapToInteger, data)


def convertPixel(char):
    """
    convert a pixel to an integer
    0: no edge (blank)
    1: gray pixel (+) [used for digits only]
    2: edge [for face] or black pixel [for digit] (#)
    """
    if (char == ' '):
        return 0
    elif (char == '+'):
        return 1
    elif (char == '#'):
        return 2


def verify(classifier, guesses, testLabels):
    hit = 0
    for i in range(len(guesses)):
        predict = guesses[i]
        truth = testLabels[i]
        if predict == truth:
            hit += 1
    accuracy = float(hit) / len(guesses)
    if accuracy>0.6:
        print "==================================="
        print "Total %d examples" % len(guesses)
        print "Prediction hits %d" % hit
        print "Accuracy: %f" % accuracy
        return False
    else:
        return True



# def testing(num):
#     trainData = loadImagesFile("data/facedata/facedatatrain", num, 60, 70)
#     trainLabels = loadLabelsFile("data/facedata/facedatatrainlabels", num)
#     testData = loadImagesFile("data/facedata/facedatatest", 150, 60, 70)
#     testLabels = loadLabelsFile("data/facedata/facedatatestlabels", 150)
#     validData = loadImagesFile("data/facedata/facedatavalidation", 301, 60, 70)
#     validLabels = loadLabelsFile("data/facedata/facedatavalidationlabels", 301)
#
#     # trainData = loadImagesFile("data/digitdata/trainingimages", num, 28, 28)
#     # trainLabels = loadLabelsFile("data/digitdata/traininglabels", num)
#     # testData = loadImagesFile("data/digitdata/testimages", 1000, 28, 28)
#     # testLabels = loadLabelsFile("data/digitdata/testlabels", 1000)
#     # validData = loadImagesFile("data/digitdata/validationimages", 1000, 28, 28)
#     # validLabels = loadLabelsFile("data/digitdata/validationlabels", 1000)
#
#     nb = NaiveBayesClassifier(1,1)
#     nb.train(trainData, trainLabels)
#     print "==================================="
#     print "Test Data"
#     guess = nb.classify(testData)
#     verify(nb,guess,testLabels)
#     print "==================================="
#     print "Validation Data"
#     guess=nb.classify(validData)
#     verify(nb,guess,validLabels)
#
#     # perceptron=PerceptronClassifier(trainData, trainLabels,0)
#     # perceptron.train(trainData, trainLabels,10)
#     # print "==================================="
#     # print "Test Data"
#     # guess=perceptron.classify(testData)
#     # verify(perceptron, guess, testLabels)
#     # print "==================================="
#     # print "Validation Data"
#     # guess=perceptron.classify(validData)
#     # verify(perceptron,guess,validLabels)
#
#     # neural = NeuralNetworkClassifier(28 * 28, 50, 10, num, 3.5,0)
#     # neural.train(trainData, trainLabels, 100)
#     # print "Test Data"
#     # guess = neural.classify(testData)
#     # verify(neural, guess, testLabels)
#     # print "==================================="
#     # print "Validation Data"
#     # guess = neural.classify(validData)
#     # verify(neural, guess, validLabels)
#     # print "==================================="
#
#     # neural = NeuralNetworkClassifier(60 * 70, 500, 1, num, 10)
#     # neural.train(trainData, trainLabels, 100)
#     # print "Test Data"
#     # guess = neural.classify(testData)
#     # verify(neural, guess, testLabels)
#     # print "==================================="
#     # print "Validation Data"
#     # guess = neural.classify(validData)
#     # verify(neural, guess, validLabels)
#     # print "==================================="
#
#
# if __name__ == "__main__":
#         sampleDigit=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
#         sampleFace=[45,90,135,180,225,270,315,300,405,450]
#         sampleR=[0.03,0.1,0.1,0.3,0.3,0.3,3,10,10,10]
#         sample=sampleFace
#         # sample=sampleDigit
#         for i in range(len(sample)):
#             print str(10*(i+1))+"%% training data, %d" % sample[i]
#             testing(sample[i])
#             print "==================================="
#     # testing(5000)
