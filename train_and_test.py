from util import training_data_cleaning, testing_data_cleaning, vectorized_data_and_padding, \
    DisasterTweetsBidirectionalGRU, dataset, load_data, train, test_data_prediction

import torch
import torch.nn as nn

TRAINING_DATA_PATH = "data/train.csv"
TESTING_DATA_PATH = "data/test.csv"
SUBMISSION_CSV_OUTPUT_PATH = "data/submissions.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 2
batch_size = 64


if __name__ == "__main__":
    train_df = training_data_cleaning(TRAINING_DATA_PATH)
    test_df = testing_data_cleaning(TESTING_DATA_PATH)

    test_ids = test_df['id'].to_list()

    x_train, y_train, x_test, vocab = vectorized_data_and_padding(train_df, test_df)

    input_size = len(vocab)
    hidden_size = 256
    learning_rate = 0.01

    train_dataset = dataset(x_train, y_train)
    train_loader = load_data(train_dataset)

    GRU_classfication_model = DisasterTweetsBidirectionalGRU(input_size, hidden_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(GRU_classfication_model.parameters(), lr=learning_rate)

    train(train_loader, num_epochs, GRU_classfication_model, criterion, optimizer)
    test_data_prediction(SUBMISSION_CSV_OUTPUT_PATH, x_test, test_ids, GRU_classfication_model)

