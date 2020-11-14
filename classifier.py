import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# from sklearn.lda import LDA
# from sklearn.qda import QDA
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime as dt

# X = np.load('training_X.npy')
X_with_nan = np.load('training_X_with_nan.npy')
X = np.load('training_X_without_nan.npy')
Y = np.load('training_Y.npy')


X = pd.DataFrame(data=X[0:,0:],index=X[0:,0],columns=X[0,0:])
print (X.shape)
X = X.dropna(thresh=len(X) * 0.8, axis=1)   #len(X) return number of rows in X. 0.6-> all feature.
print (X.shape)
print ("-------------- 2.1 -------------")


X = X.to_numpy()
# x = X[0:500, 0:6]
print (X.shape)
print ("-------------- 3 -------------")

print ("X.shape: ", X.shape)
# # Y = Y.astype('int')

# print (X)
# print (X_with_nan)


## convert your array into a dataframe
# df_X = pd.DataFrame (X)
# df_X_with_nan = pd.DataFrame (X_with_nan)

## save to xlsx file


# df_X.to_excel('X.xlsx', index=False)
# df_X_with_nan.to_excel('X_with_nan.xlsx', index=False)





# -------------- Feature Selection ---------------
# http://sklearn.lzjqsdd.com/modules/feature_selection.html

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

# -------------------- 3.3.2 -----------------------
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# # SelectFromModel(GradientBoostingClassifier()).fit_transform(X, Y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
#
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=14, random_state=0)
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)

# print '-------------------- 1.13.2 -----------------------'
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X = SelectKBest(chi2, k=15).fit_transform(X, Y)


# print '-------------------- 1.13.4.1 -----------------------'
# from sklearn.svm import LinearSVC
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
# # lsvc = SVC(gamma=0.27, penalty="l1", C=10).fit(X, Y)
# model = SelectFromModel(lsvc, prefit=True)
# X = model.transform(X)

# print ('-------------------- 1.13.4.3 Tree-based feature selection -----------------------')
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# clf = ExtraTreesClassifier()
# clf = clf.fit(X, Y)
# # print clf.feature_importances_
# model = SelectFromModel(clf, prefit=True)
# X = model.transform(X)
# print ("X.shape after feature selection: ", X.shape)




print ('--------  Start training ---------')

names = [
    # "1. Logistic Regression",
#          "2. Nearest Neighbors",
         "3. Linear SVM",
         "4. RBF SVM",
         "5. RBF SVM",
         # "6. Decision Tree",
         "7. Random Forest",
         # "8. AdaBoost",
         # "9. Naive Bayes",
         "10. Logistic Regression"]
classifiers = [
    # KNeighborsClassifier(4),
    SVC(kernel="linear", C=10),
    # SVC(gamma=0.27, C=1),
    # SVC(gamma=0.27, C=5),
    SVC(gamma=0.3, C=10, class_weight='balanced'),
    SVC(gamma=0.27, C=5),
    # SVC(gamma=0.5, C=20),
    # SVC(gamma=0.75, C=2500),
    # DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(max_depth=4, n_estimators=X.shape[1], max_features=X.shape[1]),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # LogisticRegression(),
    LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
]

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.05)

for name, clf in zip(names, classifiers):
    start = dt.now()
    clf.fit(X_train, y_train)
    # testingScore = clf.score(X_train, y_train)
    testingScore = clf.score(X_test, y_test)
    end = dt.now()
    print(name, 'testingScore: ', testingScore.mean(), " Time Consumption: ", end - start)




row_dim, col_dim = X_train.shape
print(row_dim)
print(col_dim)


N_hidden_elements = col_dim * 2

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(col_dim, N_hidden_elements)
        self.fc2 = nn.Linear(N_hidden_elements, N_hidden_elements)
        self.fc3 = nn.Linear(N_hidden_elements, col_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X


train_X = Variable(torch.Tensor(X_train).float())
test_X = Variable(torch.Tensor(X_test).float())
train_y = Variable(torch.Tensor(y_train).long())
test_y = Variable(torch.Tensor(y_test).long())

net = Net()

criterion = nn.CrossEntropyLoss()  # cross entropy loss

# optimizer = torch.optim.SGD(net.parameters(), lr=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
# opt = torch.optim.SGD(model.parameters(), lr=0.01)

start = dt.now()
for epoch in range(10000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print ('number of epoch', epoch, 'loss', loss.data)

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

print ('prediction accuracy', accuracy_score(test_y.data, predict_y.data))
end = dt.now()
print ("Time Consumption: ", end - start)







# from matplotlib import pyplot as plt
# from sklearn import svm
#
# def f_importances(coef, names):
#     imp = coef
#     imp,names = zip(*sorted(zip(imp,names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()
#
# features_names = ['input1', 'input2']

# svm = svm.SVC(kernel='linear')
# svm.fit(X_train, y_train)
# print svm.coef_
# # f_importances(svm.coef_, features_names)






# #------------------ GMM Classifier ---------------------#
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
#
#
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
#
# from sklearn import datasets
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.externals.six.moves import xrange
# from sklearn.mixture import GMM
#
#
# def make_ellipses(gmm, ax):
#     for n, color in enumerate('rgb'):
#         v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
#         u = w[0] / np.linalg.norm(w[0])
#         angle = np.arctan2(u[1], u[0])
#         angle = 180 * angle / np.pi  # convert to degrees
#         v *= 9
#         ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
#                                   180 + angle, color=color)
#         ell.set_clip_box(ax.bbox)
#         ell.set_alpha(0.5)
#         ax.add_artist(ell)
#
# # iris = datasets.load_iris()
# #
# # # Break up the dataset into non-overlapping training (75%) and testing
# # # (25%) sets.
# # skf = StratifiedKFold(iris.target, n_folds=4)
# # # Only take the first fold.
# # train_index, test_index = next(iter(skf))
#
#
# # X_train = iris.data[train_index]
# # y_train = iris.target[train_index]
# # X_test = iris.data[test_index]
# # y_test = iris.target[test_index]
#
# n_classes = len(np.unique(y_train))
#
# # Try GMMs using different types of covariances.
# classifiers = dict((covar_type, GMM(n_components=n_classes,
#                     covariance_type=covar_type, init_params='wc', n_iter=20))
#                    for covar_type in ['spherical', 'diag', 'tied', 'full'])
#
# n_classifiers = len(classifiers)
#
# plt.figure(figsize=(3 * n_classifiers / 2, 6))
# plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
#                     left=.01, right=.99)
#
#
# for index, (name, classifier) in enumerate(classifiers.items()):
#     # Since we have class labels for the training data, we can
#     # initialize the GMM parameters in a supervised manner.
#     classifier.means_ = np.array([X_train[y_train == i].mean(axis=0) for i in xrange(n_classes)])
#
#     # Train the other parameters using the EM algorithm.
#     classifier.fit(X_train)
#
#     h = plt.subplot(2, n_classifiers / 2, index + 1)
#     make_ellipses(classifier, h)
#
#     # for n, color in enumerate('rgb'):
#     #     data = iris.data[iris.target == n]
#     #     plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
#     #                 label=iris.target_names[n])
#     # Plot the test data with crosses
#     for n, color in enumerate('rgb'):
#         data = X_test[y_test == n]
#         plt.plot(data[:, 0], data[:, 1], 'x', color=color)
#
#     y_train_pred = classifier.predict(X_train)
#     train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#     plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
#              transform=h.transAxes)
#
#     y_test_pred = classifier.predict(X_test)
#     test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#     plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
#              transform=h.transAxes)
#
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(name)
#
# plt.legend(loc='lower right', prop=dict(size=12))
#
#
# plt.show()


