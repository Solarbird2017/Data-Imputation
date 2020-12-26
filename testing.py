import numpy as np
import pandas as pd
# import datetime as dt
from datetime import datetime as dt
from functools import reduce
from missingData import impute_em
from datetime import datetime as dt

X = np.load('training_X_with_nan.npy')

X = pd.DataFrame(data=X[0:,0:],index=X[0:,0],columns=X[0,0:])

X = X.dropna(thresh=len(X) * 0.6, axis=1)   #len(X) return number of rows in X. drop the column if a specific ratio of  data is missing.

X = X.to_numpy()
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
