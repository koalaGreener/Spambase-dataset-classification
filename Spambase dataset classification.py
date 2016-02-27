import sys

def readTheFile(filename):

    lengthOfAttritube = 58
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
    sumValueOfTraingingDataset = [0.0] * lengthOfAttritube
    for trainingData in trainingDataset:
        count = 0
        for dev in trainingData.split(","):
            sumValueOfTraingingDataset[count] += float(dev)
            count += 1

    # Mean value of training dataset
    meanValueOfTrainingDataset = [0.0] * lengthOfAttritube
    count = 0
    for tempData in sumValueOfTraingingDataset:
        meanValueOfTrainingDataset[count] = (tempData / len(trainingDataset))
        count += 1


    #Standard Value calculation
    stanardValueOfTrainingDataset = [0.0] * lengthOfAttritube

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
        zScore = [0.0] * lengthOfAttritube
        count = 0
        for dev2 in trainingData.split(","):
            if (count == 57):
                zScore[count] = int(dev2)
            else:
                zScore[count] = (float(dev2) - meanValueOfTrainingDataset[count]) / stanardValueOfTrainingDataset[count]
            #print(dev2, "/", zScore[count],"/", count)
            count += 1
        trainingDatasetInZScoreFormat.append(zScore)

    print (trainingDatasetInZScoreFormat)











'''主函数调用import的数据'''
if __name__ == '__main__':
     readTheFile("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
