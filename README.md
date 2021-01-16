# Disaster Response Pipelines

A project with [Udacity](https://www.udacity.com/)'s Data Scientist Nanodegree in collaboration with [Figure Eight](https://appen.com/) (Appen), to build an end-to-end Extract-Transform-Load (ETL) pipeline, and a Machine Learning (ML) pipeline for classifying disaster response messages into 36 original categories. A huge shoutout to both Udacity and Figure Eight for making this wonderful and unique project happen!

## Table of Contents

1. [Description and Overview](#description)
2. [Installations](#Installations)
3. [Data & Files](#data) <br>
   3.1. [ETL](#ETL) <br>
   3.2. [ML](#ML) <br>
   3.3. [Flask App](#Flask) <br>
4. [Modelling](#Modelling) <br>
   4.1. [Challenges](#Challenges) <br>
5. [Acknowledgements](#Acknowledgements)

<a name = "description"></a>

### Description & Overview

The idea is to analyze disaster messages data, build a model to classify these messages into 36 different original categories. The process uses Machine Learning and leverages Natural Language processing, to clean and ready the textual messages data, and to create a [Multi-Output Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) to classify the messages into 36 categories.

- The [ETL pipeline](data/process_data.py) loads in multiple files of data, merges, cleans and stores it in a SQL database.
- The [ML pipeline](models/train_classifier.py) then retrieves this database, and performs feature engineering, transformation of the features, and the actual estimation and evaluation of the classification model for the disaster data.
- The [Flask web app](app/run.py) uses HTML, python, JavaScript, CSS, to build a web app to deploy this model. This part of the project is modified from Udacity's given template.

### Installations

For the main model from scikit-learn:

    from sklearn.multioutput import MultiOutputClassifier

Use python 3.6/+.
Data Analysis/Manipulation: `pandas, numpy, matplotlib.pyplot`
Machine Learning and Natural Language Processing: `nltk, sklearn`
To save the model to never have to run again: `pickle`
To store the data in an SQL database: `sqlalchemy.create_engine`
Deployment of the app: `flask, plotly`

<a name="data"></a>

### Data & Files

This sections provides a brief summary of the files contained within the repository.

#### ETL

- The folder `data` contains individual `.csv` data of `messages` and `categories`.
- Contained within the same folder is `DisasterResponse.db ` database file - the one used for the Machine Learning model.
- `process_data.py` contains the entire ETL pipeline, loading `messages` and `categories`, cleaning the data and storing it in `DisasterResponse.db`.

#### ML

- `train_classifier.py` is the pipeline python script that executes the entire pipeline in the following steps: loading the database, tokenizing (cleaning) the feature vector and the target matrix, training a Grid Search model using the Multi Output Classifier, evaluating the model using Accuracy Score and F1 score for classification using a `classification_report`.
- The next step of the pipeline is to store the trained model in a pickle file - `classifier.pkl`.
- `classifier.pkl` is used in the flask app to call the model to fit new data.

#### Flask

- In the folder `app`, the subfolder `templates` contain two `.html` scripts, laying out the template for the flask app - `master.html` and `go.html`.
- The file `run.py` is used to run the app. This file calls the database file, and makes three visualizations, and calls the pickled model to basically classify new data on the app.

### Modelling

The ML Algorithm used to classify a vector of messages data into a matrix of 36 original categories is the [Multi-Target Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) from Scikit-Learn's `sklean.multioutput.MultiOutputClassifier`.
The algorithm's performance is boosted with the `AdaBoostClassifier` and optimized using `GridSearchCV` for the optimal hyperparameters. The Multi-Output Classifier's fits one classifier per target, and is extendable to decide which algorithm it runs on, for example, Random Forest Classifier, Decision Trees, Gradient Boosting Classifiers, etc, of which I have chosen AdaBoost, with a Decision Tree base estimator.

#### Results & Challenges

_The F1 Score_ - The model used in this project yielded an **F1-Score (or _F-Measure_) of 0.63**. The F1 Score is a weighted average statistic (or a Harmonic Mean) of the _Precision and Recall_. Simply put, precision is the number of correct classifications of for example, category _Earthquake_, divided by the total number of messages classified as Earthquake, correct or not. Recall, is the number of correct classifications of earthquake, divided by the total messages that belong to earthquake (How many of the messages that were classified Earthquake, we actually Earthquake).
The overall accuracy, calculated by just the average number of times a message was categorized correctly, was 94.72% Why is there a mismatch? And why is the overall accuracy a sub-optimal measure compared to the F1 Score?

![Categories by Number of Messages](DISASTER_OUTPUT/data_overview/graph_2_categories.png)
