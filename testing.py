import numpy as np
import pandas as pd
# import datetime as dt
from datetime import datetime as dt
from functools import reduce
from missingData import impute_em
from datetime import datetime as dt

X = np.load('training_X_with_nan.npy')
# Y = np.load('training_Y.npy')
# X = X[0:100,:]
print (X.shape)
# print (X)
print ("-------------- 1 -------------")

# X = X[:,~np.all(np.isnan(X), axis=0)] # delete the colomns with all nan.
# X = X[~np.all(np.isnan(X), axis=1),:] # delete the rows with all nan.
# print (X.shape)
print ("-------------- 2 -------------")


X = pd.DataFrame(data=X[0:,0:],index=X[0:,0],columns=X[0,0:])
print (X.shape)
X = X.dropna(thresh=len(X) * 0.6, axis=1)   #len(X) return number of rows in X. drop the column if a specific ratio of  data is missing.
print (X.shape)
print ("-------------- 2.1 -------------")


X = X.to_numpy()
# x = X[0:500, 0:6]
print (X.shape)
print ("-------------- 3 -------------")


# start = dt.now()
# result_imputed = impute_em(X)
# end = dt.now()
# print("Time Consumption: ")
# print(end - start)
# print ("-------------- 4 -------------")
# X = result_imputed['X_imputed']
# print(X)
#
#
# # # ----------- Dealing with the nan entry ----------------
#
# X = X.astype(np.float32)
# # X = np.nan_to_num(X)    # --> convert nan to 0
# np.save('training_X.npy', X)
#
# print ("-------------- 5 -------------")
