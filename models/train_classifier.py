# Importing libraries
import sys
import os
import pickle

import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine

# Import text libraries
import nltk
nltk.download(['wordnet', 'stopwords', 'punkt', 'averaged_perceptron_tagger'])
import re
from nltk.corpus import stopwords

#Import machine learning libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    '''
    Function to load the data from SQL database that was stored in the
    ETL Pipeline.

    INPUT:
    database_filepath --> the path to the SQL database file
    OUTPUT: 
    Returns X and y matrices for machine learning processing.
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)

    X = df['message']
    y = df.iloc[:, 4:]

    categories = list(y.columns.values)
    return X, y, categories


def tokenize(text):
    '''
    Function to tokenize and clean the data.
    Processes: 
    1. Replaces URLs with a placholder string value.
    2. Word-tokenizing the data.
    3. Lemmatizing the data. 
    4. Convert to lower case and strip white spaces.

    INPUT: 
    text --> the series containing the text values of messages column in df

    OUTPUT: 
    cleaned text
    '''

    # Replacing all the URLs with a placeholder
    url_code = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_code, text)

    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # word tokenizing
    tokens = nltk.word_tokenize(text)

    # Lemmatize to unify similar words
    lemmatizer = nltk.WordNetLemmatizer()
    cleaned_token_text = [lemmatizer.lemmatize(x).lower().strip() for x in tokens]
    return cleaned_token_text

# Custom transformer to extract the starting verb of a sentence
# to use in feature unions
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Function that build the model using a pipeline. 

    OUTPUT:
    Pipeline function that is defined for running the model.
    '''
    pipeline = Pipeline([

        ('feature', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('cvect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))

    ])

    parameters = {'classifier__estimator__n_estimators': [50,100]}
    model = GridSearchCV(estimator = pipeline, param_grid = parameters, scoring = 'accuracy', verbose = 2)

    return model  


def evaluate_model(model, X_test, y_test, target_names):
    '''
    Function to evaluate the model's (pipeline's) performance.

    INPUT:
    model --> the model function defined earlier.
    X_test --> the feature test matrix
    y_test --> the multioutput matrix of the target variables
    category_names --> names of each category to predict

    OUTPUT:
    nothing, just evaluating the model's performance.
    '''
    y_predicted = model.predict(X_test)
    accuracy_overall = (y_predicted == y_test).mean().mean()

    categories = list(y_test.columns.values)

    print('The overall Accuracy of the model is: {}%'.format(accuracy_overall*100))
    print(classification_report(y_test, y_predicted, target_names = categories))

def save_model(model, pickle_filepath):
    '''
    Save the pipeline model function.
    Storing the model as a pickle enables us to just load it later,
    and saves the work of recreating the model once again.

    INPUT:
    pipeline --> The model defined in the form of an sklearn Pipeline.
    pickle_filepath --> the path to the pickled model storage.
    '''
    pickle.dump(model, open(pickle_filepath, 'wb'))


def main():
    '''
    The main function of the train classifier.

    This function:
    1. Loads the SQL database data.
    2. Tokenizes the data (cleans it).
    3. Train the ML model on the training set.
    4. Evaluate the ML model on the test set.
    5. Save the model as a pickle file.
    '''

    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, categories = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model as pipeline...')
        model = build_model()
        
        print('Training model as pipeline...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(pickle_filepath))
        save_model(model, pickle_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()