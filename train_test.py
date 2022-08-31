from util import training_data_cleaning, testing_data_cleaning, vectorized_data_and_padding, \
    DisasterTweetsBidirectionalGRU
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

num_epochs = 2
batch_size = 64

if __name__ == "__main__":
    train_df = training_data_cleaning("data/train.csv")
    test_df = testing_data_cleaning("data/test.csv")
    x_train, y_train, x_test, vocab = vectorized_data_and_padding(train_df, test_df)

    GRU_classfication_model = DisasterTweetsBidirectionalGRU(len(vocab), 256)
    # print(x_train[1].dtype)
    print(x_train[0].size())

    output = GRU_classfication_model(x_train[1])
    print(output.size())
