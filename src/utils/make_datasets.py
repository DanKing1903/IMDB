import pandas as pd
import pickle
from collections import namedtuple
from sklearn.model_selection import train_test_split
from pathlib import Path
seed = 42

all_review_paths = []

for f in ['pos', 'neg']:
    all_review_paths.extend(Path('.').glob('aclImdb/*/{}/*.txt'.format(f)))

record = namedtuple('record', ['X', 'y', 'path'])
all_records = []

for path in all_review_paths:
    review = path.read_text()
    sentiment = 1 if 'pos' in str(path) else 0
    all_records.append(record(review, sentiment,path))

df = pd.DataFrame.from_records(all_records, columns=record._fields)

train, test = train_test_split(df, stratify=df.y, test_size=0.2, random_state=42)
train, val = train_test_split(train, stratify=train.y, test_size=1/8, random_state=42)

with Path('./data/imdb.pickle').resolve().open(mode='wb') as f:
    pickle.dump((train, val, test),f)
