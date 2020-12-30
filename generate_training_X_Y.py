import numpy as np
from sklearn.preprocessing import Imputer
import random
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Nthreshold = 200     #for whole dataset.
# Nthreshold = 1  # for testing.

trainingData_icd = np.load('trainingData_icd.npy')
trainingData_index = np.load('trainingData_index.npy')
y_index = np.load('y_index.npy')
icd = trainingData_icd[:,0]

np.set_printoptions(threshold=np.nan)

# ------- obtain the ICD-9 Code as string -------
icd_codes = np.array(
['140.9','141.9','142.9','143.9','144.9',
'145.9','147.9','148.9','150.9','151.9',
'153.9','154.1','155.0','157.9','159.9',
'161.9','162.9','170.0','170.9','171.9',
'172.9','173.3','173.4','173.9','174.9',
'180.9','182.0','183.0','184.0','184.4',
'185','187.4','187.7','188.9','189.0',
'189.1','191.0','191.6','191.7','191.9',
'192.0','192.1','192.2','192.3','193',
'196.0','198.3','198.5','199.1','202.8',
'203.00','210.0','210.2','213','213.9',
'214.1','214.8','214.9','214.9','215.9',
'216.0','216.1','216.2','216.3','216.4',
'216.5','216.6','216.7','216.9','217',
'218.9','219.9','220','222.1','222.4',
'225.0','225.1','225.2','225.3','225.4',
'226','228.00','228.01','232.9','233.1',
'237.70','239.3','239.6','239.7'])

def reIndex(oriArray):
    unique, counts = np.unique(oriArray, return_counts=True)
    ranks = np.asarray((unique, counts))
    x = ranks[0, :]
    index = np.argsort(x)
    sorted_index = np.searchsorted(x[index], oriArray)
    return sorted_index

# ------- Imputer2DArray(X) is dealing with Nan data in X ---------
def Imputer2DArray(X):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    tempArray = imp.transform(X)
 
    return tempArray

# ------- Cal the statistic info from icd -------
unique, counts = np.unique(icd, return_counts=True)
ranks = np.asarray((unique, counts)).T
print type(ranks)
print "ranks.shape: ", ranks.shape

print "----------------- icd_ranks ------------------"
# icd_ranks =ranks[np.argsort(-ranks[:,1])] # decreasing order
icd_ranks =ranks[np.argsort(ranks[:,1])]   #increasing order.
print icd_ranks
print icd_ranks.shape


print "----------------- Get the row with N >= Nthreshold ------------------"
icdsEngouthPoints = ranks[np.where(ranks[:,1] >= Nthreshold)]
print "# of disease > Nthreshold: ", icdsEngouthPoints.shape[0]



print "----------------- Get the icd for cancers (N >= Nthreshold) ------------------"

icdCancers = icdsEngouthPoints[np.isin(icdsEngouthPoints[:,0], icd_codes)]    #for whole dataset.


sameNumOfTrainingData = True
X = np.array([])
Y = np.array([])

def imputerMulti2DArray(rawXY, icdColIndex, NumOficd):
    uniqueIndex = np.unique(rawXY[:, icdColIndex])
    print uniqueIndex
    results = np.zeros([rawXY.shape[0], rawXY.shape[1] - NumOficd])
    print "results.shape: ", results.shape
    for i in uniqueIndex:
        indices = np.where(rawXY[:, icdColIndex] == i)
        tempSubArray = rawXY[indices]
        tempSubArray = tempSubArray[:, NumOficd:(tempSubArray.shape[1] + 1)]
        tempSubArray = tempSubArray.astype(np.float32)
        tempSubArray = Imputer2DArray(tempSubArray)

        try:
            results[indices] = tempSubArray
        except ValueError:
            print "Xudong Warning: The dataset contain all Nan column. " \
                  "So imputer delete those column.Try to increase the Nthreshold" \
                  " to solve this problem"
    return results



if sameNumOfTrainingData:
    print "-------- training array with each cancer has the same # of points --------"
    rawXY = trainingData_icd[np.isin(trainingData_icd[:, 0], icdCancers[:, 0])]
    print rawXY.shape
    print type(rawXY[0, 1])

    # NumofEachCancer = 85
    NumofEachCancer = Nthreshold


    resultArray = np.empty((0, rawXY.shape[1] + 1), dtype='string')  # generates an empty data point.
    i = 1
    for icdCode in icdCancers[:, 0]:
        temp = rawXY[np.isin(rawXY[:, 0], icdCode)]
        index = random.sample(range(0, temp.shape[0]), NumofEachCancer)
        tempWithIndex = np.append(np.ones((NumofEachCancer, 1)) * i, temp[index, :], axis=1)
        resultArray = np.append(resultArray, tempWithIndex, axis=0)
        i += 1

    print "resultArray.shape: ", resultArray.shape

    # print resultArray   #imputer need to deal with resultArray, which has the icd info.

    print "----------------- Get the training data X and Y ------------------"
    X = imputerMulti2DArray(resultArray, 1, 2)
    # print X.shape
    Y = resultArray[:, 0]
else:
    rawXYIndex = np.isin(trainingData_icd[:, 0], icdCancers[:, 0])
    rawXY = trainingData_icd[rawXYIndex]    #imputer need to deal with rawXY, which has the icd info.

    print rawXY.shape
    X = imputerMulti2DArray(rawXY, 0, 1)
    print X.shape
    Y = trainingData_index[rawXYIndex][:, 0]



# # # ----------- Dealing with the nan entry ----------------
X_without_nan = X.astype(np.float32)
# # X = np.nan_to_num(X)    # --> convert nan to 0
Y = Y.astype('int')     # Y = Y.astype(np.float32)


# #------------------- save to training_X.npy and training_Y.npy --------------------
np.save('training_X_with_nan.npy', X)
np.save('training_X_without_nan.npy', X_without_nan)
np.save('training_Y.npy', Y)



# ----------- write training data into excel to check ---------------
import xlsxwriter
workbook = xlsxwriter.Workbook('X.xlsx')
worksheet = workbook.add_worksheet()

row = 0
for col, data in enumerate(X.T):
     worksheet.write_column(row, col, data)
 workbook.close()
