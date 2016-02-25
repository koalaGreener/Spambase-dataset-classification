import sys

def runAndPrint(filename):

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
    tempDataset = [0] * 58
    for trainingData in trainingDataset:
        count = 0
        for dev in trainingData.split(","):
            index = count % 58
            tempDataset[index] += float(dev)
            count += 1
    print(tempDataset)


    #print(testData)
    #print("---")

















'''主函数调用import的数据'''
if __name__ == '__main__':
     runAndPrint("/Users/HUANGWEIJIE/Dropbox/Web economics/assignment/PartB/dataset/spambase.data.txt")
