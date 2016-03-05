import numpy as np
import math
from decimal import *
from sklearn.metrics import *
from util import *
import pandas as pd
from docutils.nodes import inline
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns


def readTheFile(filename):

    lengthOfAttritube = 57
    trainingDataset = []
    testDataset = []
    count = 1
    pianyi = 0

    with open(filename) as infile:
        for line in infile:
            if((count + pianyi) % 10 == 0):
                testDataset.append(line)
                count += 1
            else:
                trainingDataset.append(line)
                count += 1

    # Theta and any other parameters
    #theta = [-0.054044384656347873, -0.11760661993826893, 0.098901346461049866, 0.47980044713725861, 0.37637206968370068, 0.18277993091410719, 1.2986326904057814, 0.29280365913362544, 0.19939228455658972, 0.053979666154919267, 0.010659065928094777, -0.15342753973507764, -0.0093617810060309045, 0.010187280218548702, 0.48296753106365642, 0.48703095710280381, 0.42774926433474886, 0.16717605278778255, 0.17604469693507141, 0.66933289160437093, 0.24266093573995492, 0.48934800241689663, 1.421093377590847, 0.3638201291350639, -1.092913918509526, -0.46168049224488128, -0.76957249516270743, 0.20329077536047499, -0.16891673893013184, -0.2087350945416005, -0.044709835641178176, 0.071522534301151602, -0.40162869664120493, 0.043171971393311026, -0.20972588555245172, 0.22794434874086705, -0.09102952052623825, -0.06203116867313458, -0.15661140966810466, -0.12658850982940972, -0.29376817135697708, -0.55692667880988744, -0.077146550098039723, -0.43127560178077723, -0.51968013470212215, -0.57957401372117967, -0.16049564319248852, -0.32271161845081331, -0.36515025779097887, -0.082670799740634571, -0.13446470569606708, 0.51724919754976761, 1.4363191515959477, 0.50113522719177683, 0.43063940749857571, 0.95274454829393096, 0.36247771470125456]
    theta = [0.0] * lengthOfAttritube
    Batch_learningRate = 0.001
    stochastic_learningRate = 0.01
    iterations = 100

    trainingDataset = Format_trainingdata(trainingDataset)
    testDataset = Format_testdata(testDataset)



    # stochastic_gradient_descent function for logistic
    times = 0
    for i in range (iterations): #Loop
        if times == 2750:
            break
        for data in trainingDataset: # from 1 to m
            stochastic_gradient_descent_logistic(data, theta, stochastic_learningRate)
            times += 1
            print(str(times) + "," + str(cost_function_calculation_logistic(trainingDataset, theta)))
            if times == 2750:
                break





    #Calculate the AUC value
    ytrue = []
    yprediect = []
    #ytrue直接抓[lengthOfAttritube] prediect用预测值
    for everyTestData in (testDataset):
        ytrue.append(everyTestData[lengthOfAttritube])
    for everyTestData2 in (testDataset):
        yprediect.append(calculateTheScore(everyTestData2, theta, False))
    print(roc_auc_score(ytrue, yprediect))

    #plot the ROC Curve
    fpr ,tpr, thresold = roc_curve(ytrue,yprediect)
    plt.plot(fpr,tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()


'''

    # stochastic_gradient_descent function for linear
    times = 0
    for i in range (iterations): #Loop
        if times == 23553:
            break
        for data in trainingDataset: # from 1 to m
            stochastic_gradient_descent_linear(data, theta, stochastic_learningRate)
            times += 1
            print(str(times) + "," + str(cost_function_calculation_linear((trainingDataset), theta)))
            if times == 23553:
                break

    # batch_gradient_descent function for linear
    times = 0
    for epoch in range (iterations): #Loop
        batch_gradient_descent_linear(trainingDataset, theta, Batch_learningRate)
        times += 1
        print(str(times) + "," + str(cost_function_calculation_linear(trainingDataset, theta)))






    # batch_gradient_descent function for logistic
    times = 0
    for epoch in range (iterations): #Loop
        batch_gradient_descent_logistic(trainingDataset, theta, Batch_learningRate)
        times += 1
        print(str(times) + "," + str(cost_function_calculation_logistic(trainingDataset, theta)))




'''


'''import the data that contains the spam'''
if __name__ == '__main__':
     readTheFile("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
