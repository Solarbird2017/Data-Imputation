import pandas as pd
import datetime as dt
import numpy as np

# this is the dataset。
icd = pd.read_csv('datasetWithTitle/icd.csv')    # icd.csv has 7176 X 5, the 1st row is title.
patient = pd.read_csv('datasetWithTitle/patient.csv')
labresult = pd.read_csv('datasetWithTitle/lab.csv')
vitalsign = pd.read_csv('datasetWithTitle/vitalsign.csv')    # this is not important, so do not consider this info.
TestNumberThreshold = 3000

# ------- Define isTestNameDigital() ----------
def isTestNameDigital(testname):
    try:
        float(testname)
        return True
    except ValueError:
        return False
# ---------------------------------------------

# xudong: merge them in tables.
tables = [icd, patient, labresult, vitalsign]
for t in tables:
    print (list(t.columns.values), t.shape, type(t))

# Xudong: delete some columns in icd and labresult.
icd.drop(['ICDName','CodeSystem'],inplace=True,axis=1)
# icd.drop_duplicates(['PatientID', 'Date', 'ICD'],inplace = True) #Return Series with duplicate values removed.

labresult.drop(['LOINCNum','Unit'],inplace=True,axis=1)

# --- the original excel file contain duplicate rows, so need to drop_cuplicates ------
# labresult.drop_duplicates(['PatientID', 'ServiceDate', 'TestName', 'Value'],inplace = True) #Return Series with duplicate values removed.
labresult.drop_duplicates(['PatientID', 'ServiceDate', 'TestName'],inplace = True) #Return Series with duplicate values removed. ***** Very important to invoid having 4.34.2 (same patient, have same test at same day but different value)
labresult = labresult[ labresult['Value'].apply(isTestNameDigital)]    # drop the row if no date info.
# print labresult

vitalsign.drop(['HR', 'RR', 'FS', 'Temp', 'SPO2', 'PainLevel', 'Pulse'],inplace=True,axis=1)


for t in tables:
    print (list(t.columns.values), t.shape, type(t))

"""
['PatientID', 'Date', 'ICD'] (7175, 3)
['PatientID', 'PatientAge', 'PatientGender'] (583, 3)
['PatientID', 'ServiceDate', 'TestName', 'Value'] (105599, 4)
['PatientID', 'VisitDate', 'HT', 'WT', 'BP', 'RR', 'Temp', 'PainLevel'] (5095, 8)
"""
haveSlash = lambda x: '/' in x

def validate(date_text):
    # print type(date_text), date_text
    if not haveSlash(date_text):
        # print "0 -> "
        return False

    # temp = date_text
    try:
        # global i
        # print i
        # print 'try'
        dt.datetime.strptime(date_text, '%m/%d/%Y')
        # i += 1
        # print "-> 1"
        # return True
    except ValueError:

        dt.datetime.strptime(date_text, '%m/%d/%y')
        # print "2 -> "
        return True
    # except ValueError as b:
    except:
        # print "3 -> "
        return False
    else:
        # print "1 -> "
        return True

# def validate(date_text):
#     try:
#         dt.datetime.strptime(date_text, '%m/%d/%Y')
#         return True
#     except ValueError:
#         return False

icd = icd[ icd['Date'].apply(lambda x: validate(x))]    # drop the row if no date info.
labresult = labresult[ labresult['ServiceDate'].apply(lambda x: validate(x)) ]

# df.groupby("A").filter(lambda x: len(x) > 1)  # this is just a example for filter data.
# df = labresult.groupby(['TestName']).filter(lambda x: len(x) >= 3000)
df = labresult.groupby(['TestName']).filter(lambda x: len(x) >= TestNumberThreshold) # if # of testName is larger than 10, then have it.
#but some patient may not have it, even it is a popular testName.
# df.to_csv('xudong_output_df.csv')
table = pd.pivot_table(df, values='Value', index=['PatientID', 'ServiceDate'], columns=['TestName'], aggfunc=np.sum)
# table.drop(['BUN/CREATININE RATIO'],inplace=True,axis=1)


table.reset_index(level=0, inplace=True) # add PatientID back to columns
# table.reset_index(level=1, inplace=True) # add ServiceDate back to columns
table.reset_index(level=0, inplace=True) # add ServiceDate back to columns

vitalsign = vitalsign[ vitalsign['VisitDate'].apply(lambda x: validate(x)) ]
vitalsign = vitalsign[vitalsign['BP'].apply(lambda x: '/' in x)] # filter blood pressure
tmp = vitalsign['BP'].apply(lambda x: x.split('/'))
vitalsign['high'] = tmp.apply(lambda x: x[0])
vitalsign['low'] = tmp.apply(lambda x: x[1])
print (vitalsign.columns.values, vitalsign.shape)
vitalsign.drop(['BP'],inplace=True,axis=1)
print (vitalsign.columns.values, vitalsign.shape)


print ("merge by date......")
# time column are still in the type "object"
icd = icd.copy()
# table = table.copy()
# vitalsign = vitalsign.copy()
icd['Date'] = pd.to_datetime(icd['Date'])
table['ServiceDate'] = pd.to_datetime(table['ServiceDate'])
vitalsign['VisitDate'] = pd.to_datetime(vitalsign['VisitDate'])

res = icd
res = pd.merge(res, table, left_on=['Date', 'PatientID'], right_on=['ServiceDate', 'PatientID'], how='inner')
res.drop(['PatientID', 'ServiceDate', 'Date'],inplace=True,axis=1)

# res.drop(['ALBUMIN','PROTEIN,TOTAL'],inplace=True,axis=1)   # the value of test named 'ALBUMIN' contain wrong value as 4.224.66.
# res.drop(['PatientID', 'Date', 'VisitDate', 'ServiceDate'],inplace=True,axis=1)
np.save('y_icd.npy', res['ICD'].values[:])
np.save('trainingData_icd.npy', res)

def discretize(x, y):
    # find index of each y in list x
    # http://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices


icd_code = res['ICD'].unique()

#icd_code contains the different icd code from 0 to Nmax.
res['ICD'].values[:] = discretize(icd_code, res['ICD'].values[:])   # change the icd code to index in res.
np.save('y_index.npy', res['ICD'].values[:])
np.save('trainingData_index.npy', res)
res.to_csv('trainingData_index.csv')
# print res['ICD'].values[:]

res.drop(['ICD'],inplace=True,axis=1)
res.drop_duplicates()   #Return Series with duplicate values removed.
np.save('x.npy', res)
np.save('x_without_none.npy', res.dropna())
res.to_csv('xudong_output2.csv')
