from util import *
from random import *

def readTheFile(filename):

    # Theta and any other parameters
    lengthOfAttritube = 57

    theta = [0.0] * lengthOfAttritube
    Batch_learningRate = 0.0001
    stochastic_learningRate = 0.01
    iterations = 1000
    K_fold_training_dataset = []
    K_fold_test_dataset = []
    trainingDataset = []
    testDataset = []

    # 10-Fold
    for i in range(1):
        count = 1
        temp_for_fold_test = []
        temp_for_fold_train = []
        with open(filename) as infile:
            for line in infile:
                if((count + i) % 10 == 0):
                    temp_for_fold_test.append(line)
                    count += 1
                else:
                    temp_for_fold_train.append(line)
                    count += 1
            # Z-score Format of both the training dataset and test dataset
        K_fold_training_dataset.append(temp_for_fold_train)
        K_fold_test_dataset.append(temp_for_fold_test)

    trainingDataset = (Format_trainingdata(temp_for_fold_train))
    testDataset = (Format_testdata(temp_for_fold_test))

    # Shuffle the dataset
    for Not_shuffle_train_dataset in K_fold_training_dataset:
        shuffle(Not_shuffle_train_dataset)
    for Not_shuffle_test_dataset in K_fold_training_dataset:
        shuffle(Not_shuffle_test_dataset)

    # four different gradient descent function
    #log_sgd(K_fold_training_dataset, trainingDataset, iterations, theta, stochastic_learningRate)
    #log_bgd(K_fold_training_dataset, trainingDataset, iterations, theta, Batch_learningRate)
    #linear_sgd(K_fold_training_dataset, trainingDataset, iterations, theta, stochastic_learningRate)
    linear_bgd(K_fold_training_dataset, trainingDataset, iterations, theta, Batch_learningRate)

    # calculate the AUC value and then plot the ROC curve
    calculate_AUC_value_and_plot_Roc_Curve(K_fold_test_dataset, testDataset, theta)

'''import the data that contains the spam'''
if __name__ == '__main__':
     readTheFile("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
