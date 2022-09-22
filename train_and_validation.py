from util import training_data_cleaning, testing_data_cleaning, vectorized_data_and_padding, \
     dataset, load_data

from script import DisasterTweetsBidirectionalGRU, train, evaluation

import torch
import torch.nn as nn
from sklearn import model_selection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 2
batch_size = 64

if __name__ == "__main__":
    train_df = training_data_cleaning("data/train.csv")
    test_df = testing_data_cleaning("data/test.csv")

    x_train, y_train, x_test, vocab = vectorized_data_and_padding(train_df, test_df)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x_train, y_train,
                                                                                    test_size=0.2, shuffle=True)

    input_size = len(vocab)
    hidden_size = 256
    learning_rate = 0.01

    train_dataset = dataset(X_train, Y_train)
    train_loader = load_data(train_dataset)

    validation_dataset = dataset(X_validation, Y_validation)
    validation_loader = load_data(validation_dataset)

    GRU_classfication_model = DisasterTweetsBidirectionalGRU(input_size, hidden_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(GRU_classfication_model.parameters(), lr=learning_rate)

    train(train_loader, num_epochs, GRU_classfication_model, criterion, optimizer)
    evaluation(validation_loader, GRU_classfication_model)
