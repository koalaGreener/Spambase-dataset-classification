
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


    #Standard Value calculation
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
    #print(stanardValueOfTrainingDataset)


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
            #print(dev2, "/", zScore[count],"/", count)
            count += 1
        trainingDatasetInZScoreFormat.append(zScore)

    #print(trainingDatasetInZScoreFormat)

    # Theta and any other parameters
    theta = [0.0] * lengthOfAttritube
    Batch_learningRate = 0.01
    stochastic_learningRate = 0.000
    iterations = 10


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


    print("\"epoch\"" + "," + "\"cost\"")


    # stochastic_gradient_descent function
    times = 0
    for i in range (iterations): #Loop
        for data in trainingDatasetInZScoreFormat: # from 1 to m
            stochastic_gradient_descent(data, theta, stochastic_learningRate)
            times += 1
            print(str(times) + "," + str(cost_function_calculation(trainingDatasetInZScoreFormat, theta)))

'''
    # batch_gradient_descent function
    for epoch in range (iterations): #Loop
        batch_gradient_descent(trainingDatasetInZScoreFormat, theta, Batch_learningRate)
        print(epoch, cost_function_calculation(trainingDatasetInZScoreFormat, theta))
'''



'''import the data that contains the spam'''
if __name__ == '__main__':
     readTheFile("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
