import sys

def runAndPrint(filename):

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
    sumValueOfTraingingDataset = [0] * lengthOfAttritube
    for trainingData in trainingDataset:
        count = 0
        for dev in trainingData.split(","):
            index = count % 58
            sumValueOfTraingingDataset[index] += float(dev)
            count += 1

    # Mean value of training dataset
    meanValueOfTrainingDataset = [0.0] * lengthOfAttritube
    count = 0
    for tempData in sumValueOfTraingingDataset:
        meanValueOfTrainingDataset[count] = (tempData / len(trainingDataset))
        #print(meanValueOfTrainingDataset[count])
        count += 1



    #print((meanValueOfTrainingDataset[57]))

    #print(testData)
    #print("---")

















'''主函数调用import的数据'''
if __name__ == '__main__':
     runAndPrint("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
