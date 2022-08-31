from util import training_data_cleaning, testing_data_cleaning, vectorized_data_and_padding, \
    DisasterTweetsBidirectionalGRU, dataset, load_data

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

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x_train, y_train,
                                                                        test_size=0.2, shuffle=True)

    input_size = len(vocab)
    hidden_size = 256

    train_dataset = dataset(X_train, Y_train)
    train_loader = load_data(train_dataset)

    test_dataset = dataset(X_test, Y_test)
    test_loader = load_data(test_dataset)

    GRU_classfication_model = DisasterTweetsBidirectionalGRU(input_size, hidden_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(GRU_classfication_model.parameters(), lr=0.01)


