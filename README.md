# Disaster Response Pipeline Project

### Tools and Libraries required:
`numpy`
`pandas`
`matplotlib`
`json`
`plotly`
`nltk`
`flask`
`sklearn`
`sqlalchemy`
`sys`
`re`
`pickle`

This project also requires Python 3.x along with the above libraries installed as a pre-requisite

### Contents of the repository
* [process.py](https://github.com/kashyapbarua/disaster-response-pipeline/tree/main/data) This script extracts data from two csv files: Messages and Categories. The script cleans and merges both these datasets and stores it into an SQLite database
* [train_classifier.py](https://github.com/kashyapbarua/disaster-response-pipeline/tree/main/models) This script takes in the cleaned data from the SQLite database and uses the dataset to train a classification model, along with fine-tuning the algorithm with GridSearch approach. The final steps prints the model evaluation metrics for the user
* [run.py](https://github.com/kashyapbarua/disaster-response-pipeline/tree/main/app) This script runs the entire pipeline
* [disaster_messages.csv](https://github.com/kashyapbarua/disaster-response-pipeline/tree/main/data) The csv file contains real messages sent during disasters
* [disaster_categories.csv](https://github.com/kashyapbarua/disaster-response-pipeline/tree/main/data) The csv file contains the categories of each messages

### Instructions to run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
