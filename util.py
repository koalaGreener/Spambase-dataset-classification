import numpy as np
import math
from sklearn.metrics import *
from util import *
import pandas as pd
from docutils.nodes import inline
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

lengthOfAttritube = 57

# Format the test dataset using Z-score
def Format_testdata(testDataset):

    # Sum of training dataset
    sumValueOfTraingingDataset = [0.0] * (lengthOfAttritube + 1)
    for trainingData in testDataset:
        count = 0
        for dev in trainingData.split(","):
            sumValueOfTraingingDataset[count] += float(dev)
            count += 1

    # Mean value of training dataset
    meanValueOfTrainingDataset = [0.0] * (lengthOfAttritube + 1)
    count = 0
    for tempData in sumValueOfTraingingDataset:
        meanValueOfTrainingDataset[count] = (tempData / len(testDataset))
        count += 1


    # Standard Value calculation
    stanardValueOfTrainingDataset = [0.0] * (lengthOfAttritube + 1)

    for eachData in testDataset:
        countForValue = 0
        for eachValue in eachData.split(","):
            stanardValueOfTrainingDataset[countForValue] += ( ( float(eachValue) - meanValueOfTrainingDataset[countForValue] ) ** 2 )
            countForValue += 1

    count = 0
    for value in stanardValueOfTrainingDataset:
        stanardValueOfTrainingDataset[count] = (value / len(testDataset)) ** 0.5
        count += 1

    # Z-score Format
    trainingDatasetInZScoreFormat = []

    for trainingData in testDataset:
        zScore = [0.0] * (lengthOfAttritube + 1)
        count = 0
        for dev2 in trainingData.split(","):
            if (count == lengthOfAttritube):
                zScore[count] = int(dev2)
            else:
                zScore[count] = (float(dev2) - meanValueOfTrainingDataset[count]) / stanardValueOfTrainingDataset[count]
            count += 1
        trainingDatasetInZScoreFormat.append(zScore)

    return trainingDatasetInZScoreFormat

# Format the training dataset using Z-score
def Format_trainingdata(trainingDataset):

    # Sum of training dataset
    sumValueOfTraingingDataset = [0.0] * (lengthOfAttritube + 1)
    for trainingData in trainingDataset:
        count = 0
        for dev in trainingData.split(","):
            sumValueOfTraingingDataset[count] += float(dev)
            count += 1

    # Mean value of training dataset
    meanValueOfTrainingDataset = [0.0] * (lengthOfAttritube + 1)
    count = 0
    for tempData in sumValueOfTraingingDataset:
        meanValueOfTrainingDataset[count] = (tempData / len(trainingDataset))
        count += 1


    # Standard Value calculation
    stanardValueOfTrainingDataset = [0.0] * (lengthOfAttritube + 1)

    for eachData in trainingDataset:
        countForValue = 0
        for eachValue in eachData.split(","):
            stanardValueOfTrainingDataset[countForValue] += ( ( float(eachValue) - meanValueOfTrainingDataset[countForValue] ) ** 2 )
            countForValue += 1

    count = 0
    for value in stanardValueOfTrainingDataset:
        stanardValueOfTrainingDataset[count] = (value / len(trainingDataset)) ** 0.5
        count += 1

    # Z-score Format
    trainingDatasetInZScoreFormat = []

    for trainingData in trainingDataset:
        zScore = [0.0] * (lengthOfAttritube + 1)
        count = 0
        for dev2 in trainingData.split(","):
            if (count == lengthOfAttritube):
                zScore[count] = int(dev2)
            else:
                zScore[count] = (float(dev2) - meanValueOfTrainingDataset[count]) / stanardValueOfTrainingDataset[count]
            count += 1
        trainingDatasetInZScoreFormat.append(zScore)

    return trainingDatasetInZScoreFormat

# sigmoid function
def sigmoid(x):
    #print(x)
    return 1 / (1 + math.exp(-x))


# calculate the MSE/cost when the model is linear
def cost_function_calculation_linear(K_fold_training_dataset, trainingDataset, theta):
        output = 0.0
        for k_fold_dataset in K_fold_training_dataset:
            for everydata in Format_trainingdata(k_fold_dataset):
                tempOutput = 0.0

                for i in range(0, lengthOfAttritube):
                    tempOutput += everydata[i] * theta[i]

                output += (tempOutput - everydata[lengthOfAttritube]) ** 2
        return (1.0/ (2 * len(trainingDataset)) ) * output


# calculate the MSE/cost when the model is logistic
def cost_function_calculation_logistic(K_fold_training_dataset, trainingDataset, theta):
        output = 0.0
        for k_fold_dataset in K_fold_training_dataset:
            for everydata in Format_trainingdata(k_fold_dataset):
                tempOutput = 0.0
                for i in range(0, lengthOfAttritube):
                    tempOutput += everydata[i] * theta[i]
                if abs(tempOutput) >= 10:
                    tempOutput = 10 * abs(tempOutput) / tempOutput
                output += ((-1.0 * everydata[lengthOfAttritube]) * np.log(sigmoid(tempOutput)) - ((1.0 - everydata[lengthOfAttritube]) * np.log(1.0 - sigmoid(tempOutput))))
        return (1.0/ (len(trainingDataset) * len(K_fold_training_dataset)) ) * output


# logistic regression, using SGD
def stochastic_gradient_descent_logistic (data_X_Y, thetaList, learningRate):
            hx_y = 0.0
            for countCycle1 in range(0, lengthOfAttritube):
                hx_y += data_X_Y[countCycle1] * thetaList[countCycle1]
            for countCycle2 in range(0, lengthOfAttritube):
                # in the logistic SGD, we need to add the sigmoid function in the hx_y
                thetaList[countCycle2] -= (learningRate * (sigmoid(hx_y) - data_X_Y[lengthOfAttritube]) * data_X_Y[countCycle2])

# linear regression, using SGD
def stochastic_gradient_descent_linear(data_X_Y, thetaList, learningRate):
            hx_y = 0.0
            for countCycle1 in range(0, lengthOfAttritube):
                hx_y += data_X_Y[countCycle1] * thetaList[countCycle1]
            for countCycle2 in range(0, lengthOfAttritube):
                thetaList[countCycle2] -= (learningRate * (hx_y - data_X_Y[lengthOfAttritube]) * data_X_Y[countCycle2])

# logistic regression, using BGD
def batch_gradient_descent_logistic (data_Full, thetaList, learningRate):
        # update the index one theta
        for index in range(0, lengthOfAttritube):
            hx_y = 0.0
            for data in data_Full:
                temp_hx_y = 0.0
                if abs(temp_hx_y) >= 10:
                    temp_hx_y = 10 * abs(temp_hx_y) / temp_hx_y
                for countBGD in range(0, lengthOfAttritube):
                    temp_hx_y += (data[countBGD] * thetaList[countBGD])
                hx_y -= ((sigmoid(temp_hx_y) - data[lengthOfAttritube]) * data[index])
            thetaList[index] += (1.0 * learningRate * hx_y)
            #print(thetaList[index])

# linear regression, using BGD
def batch_gradient_descent_linear (data_Full, thetaList, learningRate):
        # update the index one theta
        for index in range(0, lengthOfAttritube):
            hx_y = 0.0
            for data in data_Full:
                temp_hx_y = 0.0
                for countBGD in range(0, lengthOfAttritube):
                    temp_hx_y += (data[countBGD] * thetaList[countBGD])
                hx_y += (data[lengthOfAttritube] - temp_hx_y) * data[index]
            thetaList[index] += (1.0 * learningRate * hx_y)

# calculate the score of each test dataset to plot the roc curve and output the auc value
# I found that whether the return value contains sigmoid, the result is the same
def calculateTheScore(data, theta):
        sum = 0.0
        for i in range(lengthOfAttritube):
            sum += (data[i] * theta[i])
        return (sum)


# calculate the AUC value
def calculateTheAUC(TPR_FPR):
        temp = [0.0, 0.0]
        TPR_FPR.insert(0, temp)
        sum = 0.0
        for i in range(0, len(TPR_FPR) - 1):
            sum += ((TPR_FPR[i+1][1] - TPR_FPR[i][1]) * (TPR_FPR[i+1][0] + TPR_FPR[i][0]))
        return 1.0 / 2 * sum

# implemented the roc calculation, while I used the library one
def output_Roc_data(testDataset, theta, flag):

    # Calculate the TPR and FPR value
    threshold = 0.0
    TPR_FPR = []
    for everyTestData in testDataset:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        threshold = calculateTheScore(everyTestData, theta, flag)
        for everyTestData2 in testDataset:
            print(everyTestData2[lengthOfAttritube])
            compareOne = calculateTheScore(everyTestData2, theta, flag)
            if (compareOne >= threshold) and (everyTestData2[lengthOfAttritube] == 1):
                TP += 1
            if (compareOne < threshold) and (everyTestData2[lengthOfAttritube] == 0):
                TN += 1
            if (compareOne >= threshold) and (everyTestData2[lengthOfAttritube] == 0):
                FP += 1
            if (compareOne < threshold) and (everyTestData2[lengthOfAttritube] == 1):
                FN += 1
        # init a temp List that contains both TPR and FPR
        tempTPR_FPR = []
        tempTPR_FPR.append(1.0 * TP / (TP + FN))
        tempTPR_FPR.append(1.0 * FP / (FP + TN))
        # make sure the TPR_FPR are sorted in order, so that the calculateTheAUC function can calculate the AUC value
        j = 0
        while j < len(TPR_FPR) and tempTPR_FPR[0] >= TPR_FPR[j][0]:
            j += 1
        TPR_FPR.insert(j, tempTPR_FPR)
        #print(str(1.0 * TP / (TP + FN)) + "," + str(1.0 * FP / (FP + TN)))
    print(calculateTheAUC(TPR_FPR))

# use the library to calculate the auc score and plot the curve
def calculate_AUC_value_and_plot_Roc_Curve(K_fold_test_dataset, testDataset, theta):
    #Calculate the AUC value
    ytrue = []
    yprediect = []
    # ytrue is the data[lengthOfAttritube], the flag in the data
    # yprediect is the prediect value
    for everyTestData in (testDataset):
        ytrue.append(everyTestData[lengthOfAttritube])
    for everyTestData2 in (testDataset):
        yprediect.append(calculateTheScore(everyTestData2, theta))
    print(roc_auc_score(ytrue, yprediect))

    #plot the ROC Curve
    fpr ,tpr, thresold = roc_curve(ytrue,yprediect)
    plt.plot(fpr,tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()

# logistic regression trained via sgd
def log_sgd(K_fold_training_dataset, trainingDataset, iterations, theta, stochastic_learningRate):
        # stochastic_gradient_descent function for logistic
    times = 0
    for i in range (iterations): #Loop
        #if times == 10:
            #break
        for data in trainingDataset: # from 1 to m
            stochastic_gradient_descent_logistic(data, theta, stochastic_learningRate)
            times += 1
            print(str(times) + "," + str(cost_function_calculation_logistic(K_fold_training_dataset, trainingDataset, theta)))
            #if times == 10:
                #break

# logistic regression trained via bgd
def log_bgd(K_fold_training_dataset, trainingDataset, iterations, theta, Batch_learningRate):
    # batch_gradient_descent function for logistic
    times = 0
    for epoch in range (iterations): #Loop
        batch_gradient_descent_logistic(trainingDataset, theta, Batch_learningRate)
        times += 1
        print(str(times) + "," + str(cost_function_calculation_logistic(K_fold_training_dataset, trainingDataset, theta)))


# linear regression trained via sgd
def linear_sgd(K_fold_training_dataset, trainingDataset, iterations, theta, stochastic_learningRate):
        # stochastic_gradient_descent function for linear
    times = 0
    for i in range (iterations): #Loop
        #if times == 23553:
            #break
        for data in trainingDataset: # from 1 to m
            stochastic_gradient_descent_linear(data, theta, stochastic_learningRate)
            times += 1
            print(str(times) + "," + str(cost_function_calculation_linear(K_fold_training_dataset, trainingDataset, theta)))
            #if times == 23553:
                #break

# linear regression trained via bgd
def linear_bgd(K_fold_training_dataset, trainingDataset, iterations, theta, Batch_learningRate):
    # batch_gradient_descent function for linear
    times = 0
    for epoch in range (iterations): #Loop
        batch_gradient_descent_linear(trainingDataset, theta, Batch_learningRate)
        times += 1
        print(str(times) + "," + str(cost_function_calculation_linear(K_fold_training_dataset, trainingDataset, theta)))


