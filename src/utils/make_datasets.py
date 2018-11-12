import pandas as pd
import pickle
from collections import namedtuple
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.utils.preprocessing import preprocess_text
import random
from nltk import word_tokenize, pos_tag
import argparse

def make_data_pickle(toy_dataset=False):
    seed = 42
    random.seed(seed)
    paths = []

    for f in ['pos', 'neg']:
        paths.extend(Path('.').glob('aclImdb/train/{}/*.txt'.format(f)))

    record = namedtuple('record', ['doc', 'sentiment'])
    all_records = []

    if toy_dataset:
        paths = random.sample(paths, 1000)

    for path in paths:
        raw = path.read_text()
        doc = preprocess_text(raw)
        sentiment = 1 if 'pos' in str(path) else 0
        all_records.append(record(doc, sentiment))

    data = pd.DataFrame.from_records(all_records, columns=record._fields)

    with Path('./data/imdb.pickle').resolve().open(mode='wb') as f:
        pickle.dump(data,f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create, preprocess and serialize data ")
    parser.add_argument('-t', '--toy_dataset', action="store_true", help="Use reduced dataset size for development and debugging")
    args = parser.parse_args()

    make_data_pickle(args.toy_dataset)
