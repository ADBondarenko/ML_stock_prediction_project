import torch
import torch.nn as nn
## LSTM
#Добавим линейный слой, для получения прогноза цены размерности 1.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)

        last_hidden_state = lstm_out[:, -1, :]

        output = self.fc(last_hidden_state)
        return output
    def predict(self, x):
        '''
        Метод для консистентности работы бека
        '''
        self.eval()
        x = torch.Tensor(x)
        with torch.no_grad():
            y_hat = self.forward(x)
        return y_hat
## MLP
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    def predict(self, x):
        '''
        Метод для консистентности работы бека
        '''
        self.eval()
        x = torch.Tensor(x)
        with torch.no_grad():
            y_hat = self.forward(x)
        return y_hat