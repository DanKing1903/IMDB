import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def encode(datasets, Tokenizer):
    maxlength = 250
    for i, data in enumerate(datasets):
        sequenced_data = Tokenizer.texts_to_sequences(data)
        padded_sequence = pad_sequences(sequenced_data, maxlength)
        datasets[i] = padded_sequence
    return datasets

def train_model():
    # set seeds for reproducability
    seed(1)
    set_random_seed(2)

    # Load data
    root = str(Path(__file__).resolve().parents[2])
    with Path(root+'/data/imdb.pickle').open('rb') as f:
        data = pickle.load(f)

    # Load embeddings
    embed_lookup = {}
    with Path(root+'/data/glove.6B.50d.txt').resolve().open() as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.array(values[1:])
            embed_lookup[word] = vec
    print("Loaded {} embeddings".format(len(embed_lookup)))

    # Split data into 70%/10%/20% training/validation/testing
    X_train, X_test, y_train, y_test = train_test_split(data.doc, data.sentiment, test_size=0.2, stratify=data.sentiment, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=1)

    # Fit tokenizer, get vocabulary and build embedding matrix
    tk = Tokenizer()
    tk.fit_on_texts(X_train)
    vocab_size = len(tk.word_index) + 1
    embed_matrix = np.zeros(shape=(vocab_size, 50))
    for word, i in tk.word_index.items():
        if word in embed_lookup:
            embed_matrix[i] = embed_lookup[word]

    # Tokenize, sequence and pad the data
    X_train, X_val, X_test = encode([X_train, X_val, X_test], tk)


    # Build and train the network
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50, weights=[embed_matrix], trainable=False))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    stopper = EarlyStopping()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=[X_val, y_val], callbacks=[stopper],batch_size=64, epochs=100)

    # Evaluate the network
    print("\nTest on {} samples".format(len(X_test)))
    y_pred = model.predict_classes(X_test)
    scores = {}

    scores['accuracy'] = accuracy_score(y_test, y_pred)
    scores['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    scores['f1_None'] = f1_score(y_test, y_pred, average=None)

    for score, value in scores.items():
        print("{}: {}".format(score, value))

    return history, y_test, y_pred

if __name__ == '__main__':
    model = train_model()
