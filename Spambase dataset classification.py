
def readTheFile(filename):

    lengthOfAttritube = 57
    trainingDataset = []
    testDataset = []
    count = 1

    with open(filename) as infile:
        for line in infile:
            if(count % 10 == 0):
                testDataset.append(line)
                count += 1
            else:
                trainingDataset.append(line)
                count += 1

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
            if (count == 57):
                zScore[count] = int(dev2)
            else:
                zScore[count] = (float(dev2) - meanValueOfTrainingDataset[count]) / stanardValueOfTrainingDataset[count]
            count += 1
        trainingDatasetInZScoreFormat.append(zScore)

    # Theta and any other parameters
    theta = [0.0] * lengthOfAttritube

    # BGD 1 score = 0.130181147
    # BGD 0.1 score = 0.130181345
    # BGD 0.01 score = 0.130181551
    # AUC = 0.6155963484425435
    Batch_learningRate = 1

    # SGD 0.01 score = 0.161840911
    # SGD 0.001 score = 0.150537496
    # SGD 0.0001 score = 0.130540309
    stochastic_learningRate = 0.0001

    iterations = 6


    def cost_function_calculation(trainingDataset, theta):
        output = 0.0
        for everydata in trainingDataset:
            tempOutput = 0.0

            for i in range(0, lengthOfAttritube):
                tempOutput += everydata[i] * theta[i]

            output += (tempOutput - everydata[lengthOfAttritube]) ** 2
        return (1.0/ (2 * len(trainingDataset)) ) * output


    def stochastic_gradient_descent (data_X_Y, thetaList, learningRate):
            hx_y = 0.0
            for countCycle1 in range(0, lengthOfAttritube):
                hx_y += data_X_Y[countCycle1] * thetaList[countCycle1]
            for countCycle2 in range(0, lengthOfAttritube):
                thetaList[countCycle2] -= (learningRate * (hx_y - data_X_Y[lengthOfAttritube]) * data_X_Y[countCycle2])

    def batch_gradient_descent (data_Full, thetaList, learningRate):
        # update the index one theta
        for index in range(0, lengthOfAttritube):
            hx_y = 0.0
            for data in data_Full:
                temp_hx_y = 0.0
                for countBGD in range(0, lengthOfAttritube):
                    temp_hx_y += (data[countBGD] * thetaList[countBGD])
                hx_y += (data[lengthOfAttritube] - temp_hx_y) * data[index]
            thetaList[index] += (1.0 * learningRate * hx_y / len(data_Full))


    #print("\"epoch\"" + "," + "\"cost\"")


    def calculateTheScore(data, theta):
        sum = 0.0
        tempDataForCal = []
        for dev in data.split(","):
            tempDataForCal.append(float(dev))

        for i in range(lengthOfAttritube):
            sum += (tempDataForCal[i] * theta[i])
        return sum

    # calculate the AUC value
    def calculateTheAUC(TPR_FPR):
        temp = [0.0, 0.0]
        TPR_FPR.insert(0, temp)
        sum = 0.0
        for i in range(0, len(TPR_FPR) - 1):
            sum += ((TPR_FPR[i+1][1] - TPR_FPR[i][1]) * (TPR_FPR[i+1][0] + TPR_FPR[i][0]))
        return 1.0 / 2 * sum

    # batch_gradient_descent function
    times = 0
    for epoch in range (iterations): #Loop
        batch_gradient_descent(trainingDatasetInZScoreFormat, theta, Batch_learningRate)
        times += 1
        #print(str(times) + "," + str(cost_function_calculation(trainingDatasetInZScoreFormat, theta)))


    # Calculate the TPR and FPR value
    threshold = 0
    TPR_FPR = []
    for everyTestData in testDataset:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        threshold = calculateTheScore(everyTestData, theta)
        for everyTestData2 in testDataset:
            compareOne = calculateTheScore(everyTestData2, theta)
            tempDataForCount = []
            for dev in everyTestData2.split(","):
                tempDataForCount.append(float(dev))
            if (compareOne >= threshold) and (tempDataForCount[lengthOfAttritube] == 1):
                TP += 1
            if (compareOne < threshold) and (tempDataForCount[lengthOfAttritube] == 0):
                TN += 1
            if (compareOne >= threshold) and (tempDataForCount[lengthOfAttritube] == 0):
                FP += 1
            if (compareOne < threshold) and (tempDataForCount[lengthOfAttritube] == 1):
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

'''


    # stochastic_gradient_descent function
    times = 0
    for i in range (iterations): #Loop
        for data in trainingDatasetInZScoreFormat: # from 1 to m
            stochastic_gradient_descent(data, theta, stochastic_learningRate)
            times += 1
            print(str(times) + "," + str(cost_function_calculation(trainingDatasetInZScoreFormat, theta)))


'''




'''import the data that contains the spam'''
if __name__ == '__main__':
     readTheFile("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
