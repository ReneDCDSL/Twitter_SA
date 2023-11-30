import numpy as np
import pandas as pd
from time import time
from pprint import pprint

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def ratio_positive(user, df):
    positive_tweets = df[df["user"] == user]['emotion'].sum()
    total_tweets = len(df[df["user"] == user])
    return np.round(positive_tweets / total_tweets * 100, 2)


def ratio_positive_all(user, df):
    positive_tweets = df[df["user"] == user]['emotion'].sum()
    total_tweets = len(df[df["user"] == user])
    return positive_tweets, total_tweets, np.round(
        positive_tweets / total_tweets * 100, 2)


def GridSearch_(X,
                y,
                parameters,
                model,
                scoring=['accuracy', 'f1', 'precision', 'recall'],
                refit="accuracy",
                n_jobs=-1):
    '''Perform the grid search analysis with selected parameters set, model and data. Returns the grid search object with the
    set of best parameters'''
    # Create a pipeline with model
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', model),
    ])
    # Create a grid search object with parameters to test
    grid_search = GridSearchCV(pipeline,
                               parameters,
                               scoring=scoring,
                               refit=refit,
                               n_jobs=n_jobs,
                               verbose=1)

    print("Performing grid search...")
    print()
    print("Data length: ", len(y))
    print("Pipeline:", ' '.join([str(_) for name, _ in pipeline.steps[:2]]),
          model)
    print()
    print("Parameters:")
    pprint(parameters)
    print()
    t0 = time()
    grid_search.fit(X, y)
    print("Duration: %0.1fs (n_jobs: %.f)" % ((time() - t0), n_jobs))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return grid_search


def make_results(model_name, size, model_object, metric='accuracy'):
    ''' Create dictionary that maps input metric to actual metric name in GridSearchCV'''

    metric_dict = {
        'precision': 'mean_test_precision',
        'recall': 'mean_test_recall',
        'f1': 'mean_test_f1',
        'accuracy': 'mean_test_accuracy',
    }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[
        cv_results[metric_dict[metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame(
        {
            'model': [model_name],
            'size': [size],
            'precision': [precision],
            'recall': [recall],
            'F1': [f1],
            'accuracy': [accuracy],
        }, )

    return table

def make_results_dl(model_name, y_test, y_pred):
    # Extract accuracy, precision, recall, and f1 score from that row
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create table of results
    table = pd.DataFrame(
        {
            'model': [model_name],
            'precision': [precision],
            'recall': [recall],
            'F1': [f1],
            'accuracy': [accuracy],
        }, )

    return table