import os
import nltk
import numpy as np
import gensim.downloader as api

from joblib import load
from sklearn.svm import SVC
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class BullyTech:
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), 'svm_w2vec.joblib')):
        print("Loading models...", end='\r')

        self.vocab = set(words.words())  # load nltk dictionary
        self.svm: SVC = load(model_path)  # load svm
        self.sia = SentimentIntensityAnalyzer()  # load sentiment analyzer
        self.wv = api.load('word2vec-google-news-300')  # load word2vec

        print("Checking vocabulary...", end='\r')

        # make sure the required resources are downloaded
        nltk.download('punkt')
        nltk.download('words')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')

        print("✓ Done", end="\r")

    def _word_embedding(self, word: str) -> np.ndarray[np.float32]:
        try:
            return self.wv[word]
        except KeyError:
            return np.zeros(shape=(300,))
        
    def _sentiment_score(self, text: str) -> float:
        sentiment = self.sia.polarity_scores(text)
        return sentiment

    def predict(self, text: str):
        # clean text
        tokens = [token for token in word_tokenize(text.lower()) if token in self.vocab]

        # generate embedding
        sentiment = self._sentiment_score(text)
        embedding = np.mean([self._word_embedding(word) for word in tokens], axis=0)
        embedding = embedding + sentiment['compound']
        embedding = np.array([embedding])  # add new axis

        # predict class
        pred = self.svm.predict(embedding)[0]
        prob = self.svm.predict_proba(embedding)[0]

        return {
            'label': 'Bullying' if pred == 1 else 'Not Bullying',
            'probability': prob[int(pred)],
            'sentiment': sentiment
        }
        
        
        