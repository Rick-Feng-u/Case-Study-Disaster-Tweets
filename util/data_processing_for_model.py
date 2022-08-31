import pandas as pd
import numpy as np
from pandas import DataFrame

from util import only_text, tokenized_clean_list

import torch
from torchtext.vocab import build_vocab_from_iterator


def training_data_cleaning(raw_training_data_path) -> DataFrame:
    raw_train_df = pd.read_csv(raw_training_data_path)

    raw_train_df['clean_text'] = raw_train_df['text'].apply(lambda t: only_text(t))  # all raw text are now clean text
    raw_train_df['clean_text'] = raw_train_df['clean_text'].apply(lambda x: x.strip())  # combined text and keyowr5d
    # for additional input
    raw_train_df['combined_text'] = raw_train_df['keyword'].fillna('') + ' ' + raw_train_df['clean_text']  # append
    # keyword on to clean text
    raw_train_df['Tokenized_list'] = raw_train_df['clean_text'].apply(lambda t: tokenized_clean_list(t))  # clean
    # text is now tokenized

    cleaned_train_df = pd.DataFrame([raw_train_df.Tokenized_list, raw_train_df.target]).transpose()
    return cleaned_train_df


def testing_data_cleaning(raw_testing_data_path) -> DataFrame:
    raw_test_df = pd.read_csv(raw_testing_data_path)

    raw_test_df['clean_text'] = raw_test_df['text'].apply(lambda t: only_text(t))  # all raw text are now clean text
    raw_test_df['clean_text'] = raw_test_df['clean_text'].apply(lambda x: x.strip())  # combined text and keyowr5d
    # for additional input
    raw_test_df['combined_text'] = raw_test_df['keyword'].fillna('') + ' ' + raw_test_df['clean_text']  # append
    # keyword on to clean text
    raw_test_df['Tokenized_list'] = raw_test_df['clean_text'].apply(lambda t: tokenized_clean_list(t))  # clean
    # text is now tokenized

    cleaned_test_df = pd.DataFrame([raw_test_df.Tokenized_list]).transpose()
    return cleaned_test_df


def yield_tokens(data_iter) -> list:
    for l in data_iter:
        yield l


def vectorized_data_and_padding(cleaned_train_df, clean_test_df):
    X_train, Y_train, X_test = cleaned_train_df['Tokenized_list'].to_list(), cleaned_train_df['target'].to_list(), clean_test_df['Tokenized_list'].to_list()
    X = X_train + X_test
    longest_len = max(max(len(elem) for elem in X_train),max(len(elem) for elem in X_test))
    vocab = build_vocab_from_iterator(yield_tokens(X), specials=["<pad>"])
    text_pipeline = lambda t: vocab(t)
    X_train_ready, X_test_ready, Y_train_ready = [], [], torch.tensor(Y_train, dtype=torch.int64)
    for i in range(len(Y_train)):
        processed_text = torch.tensor(text_pipeline(X_train[i]), dtype=torch.int64)
        padding = torch.zeros(longest_len - len(processed_text), dtype=torch.int64)
        padded_text = torch.cat((processed_text, padding))
        X_train_ready.append(padded_text)

    for i in range(len(X_test)):
        processed_text = torch.tensor(text_pipeline(X_test[i]), dtype=torch.int64)
        padding = torch.zeros(longest_len - len(processed_text), dtype=torch.int64)
        padded_text = torch.cat((processed_text, padding))
        X_test_ready.append(padded_text)

    return X_train_ready, Y_train_ready, X_test_ready, vocab
