import torch
import torch.nn as nn
# import numpy as np
# ## LSTM
# #Добавим линейный слой, для получения прогноза цены размерности 1.
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         lstm_out, (hn, cn) = self.lstm(x)
#         last_hidden_state = lstm_out[:, -1, :]
#         output = self.fc(last_hidden_state)
#         return output
#     def predict(self, x):
#         '''
#         Метод для консистентности работы бека
#         '''
#         self.eval()
#         if isinstance(x, pd.DataFrame):
#             x = torch.Tensor(x.to_numpy()).to('cpu')
        
#         if len(x.shape) == 2:
#             x = x.unsqueeze(0)
#         device = next(self.parameters()).device

#         x = x.to(device)
#         with torch.no_grad():
#             y_hat = self.forward(x)
#         # print(y_hat.size())
#         # if y_hat.shape[0] == 1 and len(y_hat.shape) > 1:
#         #     y_hat = y_hat.squeeze(0)
#         # print(y_hat, y_hat.size())
#         return y_hat.squeeze()
## MLP
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=True):
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
        x = torch.Tensor(x.to_numpy()).to('cpu')
        with torch.no_grad():
            y_hat = self.forward(x)
        return y_hat

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        x_seq = X[i:(i + seq_length)]
        y_seq = y[i + seq_length - 1]  # Take the target corresponding to the end of the sequence
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)