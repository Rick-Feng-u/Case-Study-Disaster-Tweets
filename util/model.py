import torch
import torch.nn as nn


class DisasterTweetsBidirectionalGRU(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(DisasterTweetsBidirectionalGRU, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
