

lengthOfAttritube = 57

def calculateTheScore(data, theta):
        sum = 0.0
        for i in range(lengthOfAttritube):
            sum += (data[i] * theta[i])
        return sum

