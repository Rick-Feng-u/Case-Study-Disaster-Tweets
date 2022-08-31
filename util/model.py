import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DisasterTweetsBidirectionalGRU(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(DisasterTweetsBidirectionalGRU, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, 2, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size * 2, 1)

    def forward(self, input):
        output = self.embedding(input)
        output, hidden_state = self.gru(output)
        output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        output = self.dropout1(output)
        output = F.relu(self.fc1(output))
        output = self.dropout2(output)
        output = self.fc2(output)
        output = torch.sigmoid(output)
        return output.reshape(-1)
