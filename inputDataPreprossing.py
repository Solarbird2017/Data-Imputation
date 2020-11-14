import pandas as pd
import datetime as dt
import numpy as np

# TestNumberThreshold = 1
# xudong: read the following 4 csv file.
# icd = pd.read_csv('icd.csv')    # icd.csv has 7176 X 5, the 1st row is title.
# icd = pd.read_excel('SampleData/testing/icd_testing.xlsx')    # icd.csv has 7176 X 5, the 1st row is title. typs is DataFrame.

# icd = pd.read_csv('icd.csv')    # icd.csv has 7176 X 5, the 1st row is title.
# patient = pd.read_csv('SampleData/patient.csv')
# labresult = pd.read_csv('labresult.csv')
# vitalsign = pd.read_csv('vitalsign.csv')

# icd = pd.read_csv('SampleData/icd.csv')    # icd.csv has 7176 X 5, the 1st row is title.
# patient = pd.read_csv('SampleData/patient.csv')
# labresult = pd.read_csv('SampleData/labresult.csv')
# vitalsign = pd.read_csv('SampleData/vitalsign.csv')

# icd = pd.read_csv('testing/icd.csv')    # icd.csv has 7176 X 5, the 1st row is title. typs is DataFrame.
# patient = pd.read_csv('testing/patient_testing.csv')
# labresult = pd.read_csv('testing/labresult_testing.csv')
# vitalsign = pd.read_csv('testing/vitalsign_testing.csv')
# TestNumberThreshold = 1

# this is the data from dataset.zip.
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

# icd, patient labresult and vitalsign is <class 'pandas.core.frame.DataFrame'>
# print icd
print ("---------------------------- 1 ----------------------------------\n")

# print icd.head(30), '\n'  #print the first 30 rows.

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

# print icd.head(30)  #print the first 30 rows.

"""
['PatientID', 'Date', 'ICD'] (7175, 3)
['PatientID', 'PatientAge', 'PatientGender'] (583, 3)
['PatientID', 'ServiceDate', 'TestName', 'Value'] (105599, 4)
['PatientID', 'VisitDate', 'HT', 'WT', 'BP', 'RR', 'Temp', 'PainLevel'] (5095, 8)
"""
print ("---------------------------- 2 ----------------------------------\n")





# print "2.1:", icd['Date'], '\n'
# print icd['Date']

haveSlash = lambda x: '/' in x

# print haveSlash("//")
# print haveSlash("2232")
# print haveSlash("06/06/2016")

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

print ("2.2: ", icd.shape)
# print icd.head(30), '\n'
# a = lambda x: validate(x)
# b = lambda x: validate(x)
# print type(a)   # <type 'function'>
# print type(a('11/12/23'))   # <type 'bool'>
# print a('11/12/23') # return false.
# print b('11/12/2023')   # return true.
# print "type(icd) output -> ",type(icd)
# print "type(icd['PatientID']) output -> ",type(icd['PatientID'])
# print "type(icd['Date']) output -> ",type(icd['Date'])
# print "type(icd['ICD']) output -> ",type(icd['ICD'])
print ("-------------------------------\n")
icd = icd[ icd['Date'].apply(lambda x: validate(x))]    # drop the row if no date info.

print ("2.3: ", icd.shape)
# print icd.head(30)
# print "2.4 type(icd) output -> ",type(icd)
# print "2.5 type(icd['PatientID']) output -> ",type(icd['PatientID'])
# print "2.6 type(icd['Date']) output -> ",type(icd['Date'])
# print "2.7 type(icd['ICD']) output -> ",type(icd['ICD'])
#print icd.groupby(['ICD']).count().nlargest(50,'PatientID')
# print "icd info: ", icd.columns.values, icd.shape
# print "difference number: ", len(icd['PatientID'].unique())
# print icd.head(30)  #print the first 30 rows.
# print labresult
print ("---------------------------- 3 --------- deal with icd above-------------------------\n")

# print labresult.head(3)
print ("3.1: ", labresult.shape)

labresult = labresult[ labresult['ServiceDate'].apply(lambda x: validate(x)) ]
print ("3.2: ", labresult.shape)
print ("labresult info: ", labresult.columns.values, labresult.shape)
print ("different tests: ", len(labresult['TestName'].unique()))
# print labresult.groupby(['TestName']).count().nlargest(50,'PatientID')
print ("4.1 labresult.shape: ", labresult.shape)
print (labresult)
print ("-------------- no double values -------------- 4 -------- deal with lab result above --------------------------\n")




# http://stackoverflow.com/questions/29836836/how-do-i-filter-a-pandas-dataframe-based-on-value-counts
# df.groupby("A").filter(lambda x: len(x) > 1)  # this is just a example for filter data.
# df = labresult.groupby(['TestName']).filter(lambda x: len(x) >= 3000)
df = labresult.groupby(['TestName']).filter(lambda x: len(x) >= TestNumberThreshold) # if # of testName is larger than 10, then have it.
#but some patient may not have it, even it is a popular testName.

print ("4.2 df.shape: ", df.shape)
print ("4.3 ", df.columns.values, df.shape)   # >=3000's testing.
print ("4.4 type(df): ", type(df))
# print "4.5 \n", df   # 10 row ServiceDate.
# df.to_csv('xudong_output_df.csv')
print ("---------------------------- 5 --------- above: filter the less number testing => df -------------------------\n")

print ("5.1 type(icd): ", type(icd))
print ("5.2 type(tables): ", type(tables))

table = pd.pivot_table(df, values='Value', index=['PatientID', 'ServiceDate'], columns=['TestName'], aggfunc=np.sum)
# table.drop(['BUN/CREATININE RATIO'],inplace=True,axis=1)

print ("5.3 type(table): ", type(table))
# print table.head(5)    # 5 row ServiceDate. 1st level->patientID, 2nd level->ServiceDate.
print ("table info before reset_index: ", table.columns.values, table.shape)  # table.shape = num of row ServiceDate.
print ("----- has double values --------=====*** 6 ***=====--------------\n")
print (table) # has wrong data at here.
print ("---------------------------- 6 ---------- above: pivot_table on df => table ------------------------\n")


table.reset_index(level=0, inplace=True) # add PatientID back to columns
# table.reset_index(level=1, inplace=True) # add ServiceDate back to columns
table.reset_index(level=0, inplace=True) # add ServiceDate back to columns
print ("-------------=====*** 7 ***=====--------------\n")
print ("table info after reset_index: ", table.columns.values, table.shape)
# print table.head(10)    # 10 row ServiceDate.
print (table)
print ("---------------------------- 7 ------------ above: reset table => table ----------------------\n")


vitalsign = vitalsign[ vitalsign['VisitDate'].apply(lambda x: validate(x)) ]
print ("7.1: ", vitalsign.columns.values, vitalsign.shape)
print ("7.2: ", vitalsign['BP'], '\n')
vitalsign = vitalsign[vitalsign['BP'].apply(lambda x: '/' in x)] # filter blood pressure
print ("7.3: ", vitalsign.columns.values, vitalsign.shape)
print ("7.4: ", vitalsign['BP'], '\n')

# http://stackoverflow.com/questions/31737939/split-pandas-column-into-two
tmp = vitalsign['BP'].apply(lambda x: x.split('/'))
vitalsign['high'] = tmp.apply(lambda x: x[0])
vitalsign['low'] = tmp.apply(lambda x: x[1])
print (vitalsign.columns.values, vitalsign.shape)
vitalsign.drop(['BP'],inplace=True,axis=1)
print (vitalsign.columns.values, vitalsign.shape)
print ("------------------ 8 ---------- above: Change the BP to high and low in vitalsign ---------------\n")



print ("merge by date......")
# time column are still in the type "object"
print ("8.1: \n", icd.dtypes, icd.shape, '\n')
print ("8.2: \n", table.dtypes, '\n')
print ("8.3: \n", vitalsign.dtypes, '\n')

'''
PatientID     int64
Date         object
ICD          object
dtype:       object 
'''

# aa = pd.to_datetime("6/6/16")
# print aa
# print icd['Date']
print ("---------------- 8.1 ---------------", "\n")
# # icd['Date']
icd = icd.copy()
# table = table.copy()
# vitalsign = vitalsign.copy()
icd['Date'] = pd.to_datetime(icd['Date'])
table['ServiceDate'] = pd.to_datetime(table['ServiceDate'])
vitalsign['VisitDate'] = pd.to_datetime(vitalsign['VisitDate'])


print ("8.4: \n", icd.dtypes, icd.shape, '\n')
print ("8.5: \n", table.dtypes, '\n')
print ("8.6: \n", vitalsign.dtypes, '\n')

'''
PatientID             int64
Date         datetime64[ns]
ICD                  object
dtype:               object 
'''



# print "---------------- 8.2 --------------", "\n"
# print "8.7 icd: ", icd.columns.values,icd.shape,"\n"
# print "8.8: ", table.columns.values, table.shape, "\n"
# print "8.9: ", vitalsign.columns.values, vitalsign.shape, "\n"
# print "8.10: ", patient.columns.values, patient.shape, "\n"
# print "---------------- 8.3 --------------", "\n"

res = icd
# print "res = icd ", res.shape
# print "table.shape: ", table.shape
res = pd.merge(res, table, left_on=['Date', 'PatientID'], right_on=['ServiceDate', 'PatientID'], how='inner')
# print "8.11: ", res.columns.values, '\n'    # here: res comb table+icd, but dont include PatientID.
# print "res = icd + labresult", res.shape
# print "---------------- 8.4 --------------", "\n"
# print res
# print "---------------- 8.5 --------------", "\n"
# res = pd.merge(res, vitalsign, left_on=['Date', 'PatientID'], right_on=['VisitDate', 'PatientID'], how='inner')
# print "res = icd + labresult + vitalsign", res.shape
# print "---------------- 8.6 --------------", "\n"
# print res
# print "---------------- 8.7 --------------", "\n"
# res = pd.merge(res, patient, left_on=['PatientID'], right_on=['PatientID'], how='inner')
# print "res = icd + labresult + vitalsign + patient", res.shape
# print "---------------- 8.8 --------------", "\n"
# print res
# print "---------------- 8.9 --------------", "\n"
# print "---------------------------- 9 ----------------------------------\n"



# print res.head(1)

# http://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe
res.drop(['PatientID', 'ServiceDate', 'Date'],inplace=True,axis=1)

# res.drop(['ALBUMIN','PROTEIN,TOTAL'],inplace=True,axis=1)   # the value of test named 'ALBUMIN' contain wrong value as 4.224.66.
# res.drop(['PatientID', 'Date', 'VisitDate', 'ServiceDate'],inplace=True,axis=1)
# print res.columns.values, res.shape
# print res.head(5)
# print res['ICD'].value_counts()
np.save('y_icd.npy', res['ICD'].values[:])
np.save('trainingData_icd.npy', res)
res.to_csv('trainingData_icd.csv')
# print "---------------------------- 10 ----------------------------------\n"




def discretize(x, y):
    # find index of each y in list x
    # http://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    return indices


icd_code = res['ICD'].unique()
# print "10.1: ", icd_code
# print "10.2: \n", res['ICD']
# print "----------"

#icd_code contains the different icd code from 0 to Nmax.
res['ICD'].values[:] = discretize(icd_code, res['ICD'].values[:])   # change the icd code to index in res.
# print res['ICD'].value_counts()
# print "10.3: \n", res['ICD']    #change the icd code to index from 0 to Nmax, for training.
# print "------------- save y and whole ----------- 11 -------------------\n"
np.save('y_index.npy', res['ICD'].values[:])
np.save('trainingData_index.npy', res)
res.to_csv('trainingData_index.csv')
# print res['ICD'].values[:]



# change gender F/M to digital label as followings.
# gender = res['PatientGender'].unique()
# print gender
# res['PatientGender'].values[:] = discretize(gender, res['PatientGender'].values[:])


# print "---------------------------- 12 ----------------------------------\n"
#
#
# print res['ICD'].values[:]
# print "---------------------------- 13 ----------------------------------\n"
#
# print "14.1\n", res.columns.values, res.shape
# if we drop ICD, we see that there is no duplicate rows
# http://stackoverflow.com/questions/10791661/how-do-i-discretize-values-in-a-pandas-dataframe-and-convert-to-a-binary-matrix
res.drop(['ICD'],inplace=True,axis=1)
res.drop_duplicates()   #Return Series with duplicate values removed.
# print "14.2\n", res.columns.values, res.shape
#print res.head(10)
# print "---------------------------- 14 ----------------------------------\n"
# print res
# print "------------- 14.1 ------------"

# print rows containing "nan"
# http://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe
# print "# of rows with nan = ", res.shape[0] - res.dropna().shape[0]
# print "------- save x --------------------- 15 ----------------------------------\n"

# print res.shape[0]
# print number of "nan" in res
# http://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-the-column-in-panda-data-frame
# print "# of nan = "
# print res.isnull().sum()


np.save('x.npy', res)
np.save('x_without_none.npy', res.dropna())


print ("---------------------------- 16 ----------------------------------\n")
# print type(res)
# res.to_csv('xudong_output.csv', sep='\t')
res.to_csv('xudong_output2.csv')
print ("---------------------------- 17 ----------------------------------\n")
 # '''