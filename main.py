import numpy as np

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

if __name__ == '__main__':
    group,labels = createDataSet()

    print('group', group)
    print('labels', labels)

