import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(
                classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

def img2vector(filename):
    returnVect = np.zeros((1,1024))

    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect

if __name__ == '__main__':
    group,labels = createDataSet()

    print('group', group)
    print('labels', labels)
    #test img2vector
    img2vector('digits/testDigits/0_1.txt')
    print(classify0([0,0],group,labels,3))
