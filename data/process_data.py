import sys
import pandas as pd
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the messages and categories data.

    INPUT: 
    messages_filepath --> path to the messages dataset
    categories_filepath --> path to the categories dataset

    OUTPUT:
    a merged df of messages and categories datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df
    

def clean_data(df):
    '''
    Function to clean the dataset to make it ready for modeling

    INPUT:
    df --> the merged messages and categories data

    OUTPUT:
    cleaned dataframe df
    '''
    categories = df['categories'].str.split(";", expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] =  categories[column].apply(lambda x: x[-1:])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    categories.drop('child_alone', 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join = 'inner', axis = 1)
    df = df.drop('categories', axis = 1)
    df = df.drop_duplicates()
    df = df[df['related'] != 2]
    

    return df


def save_data(df, database_filename):
    '''
    Function to save the Data into a SQL database.

    INPUT:
    df --> the final cleaned dataframe df
    database_filename --> the main path to the SQL database.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()