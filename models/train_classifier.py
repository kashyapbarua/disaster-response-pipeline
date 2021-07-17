# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer
import pickle
import sys


def load_data(database_filepath):
    """
    Function that loads messages and categories from database using database_filepath as a filepath and sqlalchemy as library
    Returns two dataframes X and y
    """
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql('SELECT * FROM MessagesCategories', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    return X, y

def tokenize(text):
    """Tokenization function. Receives as input raw text which afterwards normalized, stop words removed, stemmed and lemmatized.
    Returns tokenized text"""

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")

    #tokenize
    tokens = word_tokenize(text)

    #stemming
    stemmed = [PorterStemmer().stem(w) for w in tokens]

    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]

    return words_lemmed


def build_model():

    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
        ])


    #model parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_split': [2, 3]
              }
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv
    #return pipeline


def save_model(model, model_filepath):
    """ Saving model's best_estimator_ using pickle
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #X, Y, category_names = load_data(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()