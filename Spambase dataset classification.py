from util import *


def readTheFile(filename):

    # Theta and any other parameters
    lengthOfAttritube = 57
    trainingDataset = []
    testDataset = []
    count = 1
    pianyi = 0
    theta = [0.0] * lengthOfAttritube
    Batch_learningRate = 0.001
    stochastic_learningRate = 0.01
    iterations = 100


    with open(filename) as infile:
        for line in infile:
            if((count + pianyi) % 10 == 0):
                testDataset.append(line)
                count += 1
            else:
                trainingDataset.append(line)
                count += 1

    # Z-score Format of both the training dataset and test dataset
    trainingDataset = Format_trainingdata(trainingDataset)
    testDataset = Format_testdata(testDataset)

    # four different gradient descent function
    log_sgd(trainingDataset, iterations, theta, stochastic_learningRate)
    log_bgd(trainingDataset, iterations, theta, Batch_learningRate)
    linear_sgd(trainingDataset, iterations, theta, stochastic_learningRate)
    linear_bgd(trainingDataset, iterations, theta, Batch_learningRate)
    # calculate the AUC value and then plot the ROC curve
    calculate_AUC_value_and_plot_Roc_Curve(testDataset, theta)

'''import the data that contains the spam'''
if __name__ == '__main__':
     readTheFile("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
