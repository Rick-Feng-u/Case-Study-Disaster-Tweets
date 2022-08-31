import re
import string

from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem import *

from torchtext.data import get_tokenizer

stop_words = list(get_stop_words('en'))
nltk_words = list(stopwords.words('english'))
stop_words.extend(nltk_words)

stemmer = PorterStemmer()
tokenizer = get_tokenizer("basic_english")


def lowercase_words(text) -> string:
    return text.lower()


def no_url(text: string) -> string:
    return re.sub(r'https?://\S+|www\.\S+', "", text)


def no_num(text) -> string:
    return re.sub(r'\d+', '', text)


def no_emoji(text) -> string:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def no_punctuation(text) -> string:
    return text.translate(str.maketrans('', '', string.punctuation))


def only_text(text) -> string:
    text = lowercase_words(text)
    text = no_url(text)
    text = no_emoji(text)
    text = no_num(text)
    text = no_punctuation(text)
    return text


def tokenized(text) -> list:
    return tokenizer(text)


def stopword_removed(l) -> list:
    return [w for w in l if not w in stop_words]


def list_stemming(l) -> list:
    return [stemmer.stem(w) for w in l]


def tokenized_clean_list(text) -> list:
    clean_list = tokenized(text)
    clean_list = stopword_removed(clean_list)
    clean_list = list_stemming(clean_list)
    return clean_list
