from util import training_data_cleaning, testing_data_cleaning, vectorized_data_and_padding
import torch

num_epochs = 2
batch_size = 64

if __name__ == "__main__":
    train_df = training_data_cleaning("data/train.csv")
    test_df = testing_data_cleaning("data/test.csv")
    x_train, y_train, x_test = vectorized_data_and_padding(train_df, test_df)
