import numpy as np

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

