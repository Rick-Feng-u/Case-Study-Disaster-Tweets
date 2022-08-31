import pandas as pd
from pandas import DataFrame

from util import only_text, tokenized_clean_list


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


def vectorized_data_and_padding(cleaned_df):
    return
