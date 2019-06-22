import torch
import torch.nn.functional as F
from torch import nn

from datasets import time_series_window_size
"""-------------------------------------------------------------------------"""
"""---------------------------------LSTMNet---------------------------------"""
"""-------------------------------------------------------------------------"""

class TimeSeriesTestNet(nn.Module) :
    def __init__(self, hidden_size, num_layers, dropout, batch_norm, num_features=3, window_size=time_series_window_size) :
        super(TimeSeriesTestNet, self).__init__()
        self.window_size = window_size
        self.num_features = num_features

        # sin
        self.h1 = hidden_size
        self.h2 = hidden_size

        self.num_layers = num_layers
        self.dropout = dropout
        self.p_dropout = 0.3
        self.batch_norm = batch_norm

        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=self.h1, batch_first=True)
        if self.num_layers == 2 :
            self.lstm2 = nn.LSTM(input_size=self.h1, hidden_size=self.h2, batch_first=True)
            self.fc1 = nn.Linear(self.h2, num_features)
        else:
            self.fc1 = nn.Linear(self.h1, num_features)


        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.h1 if self.num_layers == 1 else self.h2)
        

    def forward(self, x) :
        h_t1 = c_t1 = torch.zeros(1, x.size(0), self.h1).cuda() if torch.cuda.is_available() else torch.zeros(
                1,
                x.size(0),
                self.h1)

        # x = x.view(x.size(0), self.window_size, -1)
        h_t1, c_t1 = self.lstm1(x, (h_t1, c_t1))

        if self.num_layers == 2 :
            h_t2 = c_t2 = torch.zeros(1, x.size(0),
                                      self.h2).cuda() if torch.cuda.is_available() else torch.zeros(
                    1,
                    x.size(0),
                    self.h2)
            if self.batch_norm :
                h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
                output = self.fc1(self.bn(h_t2))
            else :
                h_t2, c_t2 = self.lstm2(F.dropout(h_t1, self.p_dropout, self.training) if self.dropout else h_t1,
                                        (h_t2, c_t2))
                output = self.fc1(F.dropout(h_t1, self.p_dropout, self.training) if self.dropout else h_t2)
        else :
            if self.batch_norm:
                output = self.fc1(self.bn(h_t1))
            else:
                output = self.fc1(h_t1)

        output = output[:, -1, :]  # .reshape(-1)  # extract the last value as the prediction
        return output

"""-------------------------------------------------------------------------"""
"""---------------------------------LSTMNet---------------------------------"""
"""-------------------------------------------------------------------------"""

class LSTMNet(nn.Module) :
    def __init__(self, num_features=3, window_size=50) :
        super(LSTMNet, self).__init__()
        self.window_size = window_size
        self.num_features = num_features

        self.h1 = 40
        self.h2 = 50
        self.h3 = 40
        self.h4 = 60
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=self.h1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.h1, hidden_size=self.h2, batch_first=True)
        self.fc1 = nn.Linear(self.h2, num_features)

    def forward(self, x) :
        h_t1 = c_t1 = torch.zeros(1, x.size(0), self.h1).cuda() if torch.cuda.is_available() else torch.zeros(1,
                                                                                                              x.size(0),
                                                                                                              self.h1)
        h_t2 = c_t2 = torch.zeros(1, x.size(0), self.h2).cuda() if torch.cuda.is_available() else torch.zeros(1,
                                                                                                              x.size(0),
                                                                                                              self.h2)
        h_t1, c_t1 = self.lstm1(x, (h_t1, c_t1))
        h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
        output = self.fc1(h_t2)
        output = output[:, -1, :]  # .reshape(-1)  # extract the last value as the prediction
        return output

"""-------------------------------------------------------------------------"""
"""---------------------------------JSBChoralesNet---------------------------"""
"""-------------------------------------------------------------------------"""

class JSBChoralesNet(nn.Module) :
    def __init__(self, window_size=2) :
        super(JSBChoralesNet, self).__init__()
        self.window_size = window_size
        self.input_size = 88

        self.h1 = 88
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.h1, batch_first=True)

    def forward(self, x) :
        h_t1 = c_t1 = torch.zeros(1, x.size(0), self.h1).cuda() if torch.cuda.is_available() else torch.zeros(1,
                                                                                                              x.size(0),
                                                                                                              self.h1)
        x = x.view(x.size(0), self.window_size, -1)
        h_t1, c_t1 = self.lstm1(x, (h_t1, c_t1))

        output = h_t1[:, -1, :]  # .reshape(-1)  # extract the last value as the prediction

        return torch.sigmoid(output)


"""-------------------------------------------------------------------------"""
"""----------------------------------MNIST----------------------------------"""
"""-------------------------------------------------------------------------"""

class MNISTNet(nn.Module) :
    def __init__(self) :
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.bn_c2 = nn.BatchNorm2d(20)
        self.bn_f1 = nn.BatchNorm1d(50)
        self.bn_f2 = nn.BatchNorm1d(10)

    def forward(self, x) :
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.bn_c2(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.bn_f1(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.bn_f2(self.fc2(x))
        return F.log_softmax(x, dim=-1)

"""-------------------------------------------------------------------------"""
"""------------------------SequentialMNIST----------------------------------"""
"""-------------------------------------------------------------------------"""

class SequentialMNIST(nn.Module) :
    def __init__(self) :
        super(SequentialMNIST, self).__init__()
        self.side_len = 28
        self.h1 = 100
        self.h2 = 50

        self.lstm1 = nn.LSTM(input_size=self.side_len, hidden_size=self.h1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.h1, hidden_size=self.h2, batch_first=True)
        self.fc = nn.Linear(self.h2, 10)

    def forward(self, x) :
        h_t1 = c_t1 = torch.zeros(1, x.size(0), self.h1).cuda() if torch.cuda.is_available() else torch.zeros(1,
                                                                                                              x.size(0),
                                                                                                              self.h1)
        h_t2 = c_t2 = torch.zeros(1, x.size(0), self.h2).cuda() if torch.cuda.is_available() else torch.zeros(1,
                                                                                                              x.size(0),
                                                                                                              self.h2)
        x = x.view(x.size(0), -1, self.side_len)
        h_t1, c_t1 = self.lstm1(x, (h_t1, c_t1))
        h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
        output = self.fc(h_t2)
        output = output[:, -1, :]  # .reshape(-1)  # extract the last value as the prediction

        return F.log_softmax(output, dim=-1)

"""-------------------------------------------------------------------------"""
"""--------------------------------CIFAR-10---------------------------------"""
"""-------------------------------------------------------------------------"""

class CIFAR10Net(nn.Module) :
    def __init__(self) :
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn_c2 = nn.BatchNorm2d(16)
        self.bn_f1 = nn.BatchNorm1d(120)
        self.bn_f2 = nn.BatchNorm1d(84)
        self.bn_f3 = nn.BatchNorm1d(10)

    def forward(self, x) :
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn_c2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn_f1(self.fc1(x)))
        x = F.relu(self.bn_f2(self.fc2(x)))
        x = self.bn_f3(self.fc3(x))
        return x
