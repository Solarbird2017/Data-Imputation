import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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


# iris = load_iris()
iris = load_breast_cancer()
train_X, test_X, train_y, test_y = train_test_split(iris['data'], iris['target'], random_state=0)
# brCancer = load_breast_cancer()
# train_X, test_X, train_y, test_y = train_test_split(brCancer['data'], brCancer['target'], random_state=0)
row_dim, col_dim = train_X.shape
print(row_dim)
print(col_dim)


N_hidden_elements = col_dim * 3

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


# # load IRIS dataset
# dataset = pd.read_csv('dataset/iris.csv')
#
# # transform species to numerics
# dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
# dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
# dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2
#
# train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
#                                                     dataset.species.values, test_size=0.8)

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

net = Net()

criterion = nn.CrossEntropyLoss()  # cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print ('number of epoch', epoch, 'loss', loss.data)

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

print ('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

# print ('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
# print ('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
# print ('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
# print ('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))


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
    RandomForestClassifier(max_depth=10, n_estimators=train_X.shape[1], max_features=train_X.shape[1]),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # LogisticRegression(),
    LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
]
for name, clf in zip(names, classifiers):
    clf.fit(train_X, train_y)
    testingScore = clf.score(test_X, test_y)
    print (name, 'testingScore: ', testingScore.mean())
