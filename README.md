# IMDB
Sentiment analysis for movie reviews using deep learning

Instructions
============

1. Clone repository
2. Download dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract `aclImdb` folder into repository root folder
3. Create conda env from `requirements.txt`
4. Run `setup.py`
5. (Optional) run `src/utils/make_datasets.py`
6. (Optional) Download gLove embeddings from https://nlp.stanford.edu/data/glove.6B.zip and extract`glove.6B.50d.txt`into `data` folder

Notes
-----

Steps 5 and 6 only neccessary if you wish to first run the model from command line using `python src/models/LSTM.py`. If you simply wish to follow along with my jupyter notebook these steps will be done automatically.
