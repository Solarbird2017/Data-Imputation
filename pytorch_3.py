import torch
from torch.autograd import Variable
from sklearn.datasets import load_iris
from keras.utils import to_categorical

# batch size
batch_size = 10

# loading iris data from sklearn
iris = load_iris()
x_data = iris.data
y_data = iris.target

# one hot encoding
y_data = to_categorical(y_data)

# numpy to pytorch variable
x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(4, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 3)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        out1 = self.relu(self.l1(x))
        drop1 = self.dropout(out1)
        out2 = self.relu(self.l2(drop1))
        drop2 = self.dropout(out2)
        y_pred = self.sigmoid(self.l3(drop2))
        return y_pred


model = Model().double()

# model summary
print(model)

# binary cross entropy loss
loss_fun = torch.nn.BCELoss(size_average=True)

# SGD optimizer
opt = torch.optim.SGD(model.parameters(), lr=0.01)

permutation = torch.randperm(x_data.size()[0])

# training
for epoch in range(100):
    print
    "Epoch: " + str(epoch)
    for i in range(0, x_data.size()[0], batch_size):
        print
        "batch: " + str(i)
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_data[indices], y_data[indices]
        y_pred_val = model(batch_x)
        loss = loss_fun(y_pred_val, batch_y)
        print(epoch, loss.data[0])

        opt.zero_grad()
        loss.backward()
        opt.step()