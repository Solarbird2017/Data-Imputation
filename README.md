# Data-Imputation
Handling the missing data problems.
1. Impute the data by calculating the maximum likelihood iteratively.
2. Imputation accuracy is verified with different Machine Learning classifiers and Neural Networks.
3. Scikit-learn and Pytorch packages need to be installed before running the Python code.

We use the algorithm to handle the missing data problems in electronic health records (EHRs). Due to the privacy, we cannot release the original EHR dataset.

inputDataPreprocessing.py: preprocess the EHRs data and save them as *.npy files.

generate_training_X_Y.py: impute the data from the preprocessed *.npy files.

missingData.py: imputation module.

classifier.py: train and test the imputed data. 
