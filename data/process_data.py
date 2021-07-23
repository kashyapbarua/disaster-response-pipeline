# Import libraries to environment
import sqlite3
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

# Load the data files
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, left_on = 'id', right_on = 'id', how = 'inner', validate = 'many_to_many')

    # Create a dataframe of 36 individual category columns
    categories_split = df['categories'].str.split(';', expand = True)

    # Select the first row of the categories dataframe
    row = categories_split.iloc [0, :]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply (lambda x: x.split("-")[0])

    # Rename the columns of Categories
    categories_split.columns = category_colnames

    # Convert category values to just numbers
    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].str[-1]
        # convert column from string to numeric
        categories_split[column] = pd.to_numeric(categories_split[column], errors = 'coerce')

    # Drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_split], axis = 1, sort = False)

    return df

# Function to clean the dataset
def clean_data(df):
    #dropping duplicates
    df2 = df.drop_duplicates(subset = ['message'])

    # Dropping 'child_alone'
    df3 = df2.drop('child_alone', axis = 1)

    # Slicing to get a dataframe filtered out for related !=2
    df4 = df3[df3.related !=2]

    return df4

# Save the transformed dataset
def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df.to_sql('MessagesCategories', engine, index=False, if_exists = 'replace')

    pass

# Return progress messages
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
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