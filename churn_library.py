#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Robert ABEL
Date Created: 16 Mar 2026

This module provides a comprehensive set of functions for processing, analyzing, and modeling
customer attrition (churn) data. It includes utilities for importing and cleaning data, performing
exploratory data analysis (EDA), engineering features, and encoding categorical variables. In
addition, the module offers robust tools for training, evaluating, and visualizing machine learning
models, specifically tailored to predict customer churn. The design emphasizes modularity and
clarity, making it suitable for both educational purposes and practical applications in data science
projects focused on customer retention and business analytics.
"""

import logging
import os
import sys
import warnings

from pathlib import Path
from typing import TYPE_CHECKING

import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import RocCurveDisplay, classification_report


if TYPE_CHECKING:
    from collections.abc import Iterable


# harmonize seaborn / matplotlib style across library
sns.set_theme()

# Qt Platform Abstraction: offscreen rendering for matplotlib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# get library-specific logger
logger = logging.getLogger(__name__)
"""Logger for Churn Library"""


def import_data(pth: str | os.PathLike) -> pd.DataFrame:
    """Import bank data from CSV file

    Transform Attrition_Flag if present into Churn column.

    Args:
        pth: path to the CSV file

    Returns:
        Pandas DataFrame containing the bank data

    Raises:
        FileNotFoundError: if the specified file does not exist
        KeyError: if Attrition_Flag column is missing
        OSError: general operating system errors, e.g. permission denied
    """
    logger.info('Importing data from %s...', pth)
    df = pd.read_csv(pth, index_col=0)
    logger.info(
        'Successfully imported %d rows with %d columns.',
        df.shape[0],
        df.shape[1])

    # Attrition_Flag is mandatory to compute Churn column, so raise error if
    # missing
    if 'Attrition_Flag' not in df.columns:
        raise KeyError('Attrition_Flag')

    # simply map categorical column to binary
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df: pd.DataFrame):
    """Perform Exploratory Data Analaysis (EDA) on the Bank Data

    Figures will be saved to the './images' directory.
    The following figures will be generated:
    - Churn distribution
    - Customer Age distribution
    - Marital Status distribution (relative)
    - Total Transaction Count distribution (relative, with Kernel Density Estimate)
    - Correlation Heatmap of all numeric features

    Args:
        df: Bank Data as a Pandas DataFrame

    Raises:
        FileExistsError: if a file named 'images' already exists
        KeyError: if any of the columns required for EDA are missing
                  (Churn, Customer_Age, Marital_Status, Total_Trans_Ct)
        OSError: general operating system errors, e.g. permission denied
    """
    # create output directory
    os.makedirs('./images', exist_ok=True)

    # print some info about columns
    cat_columns = df.select_dtypes(exclude='number').columns.tolist()
    num_columns = df.select_dtypes(include='number').columns.tolist()
    logger.info(
        'Found %d categorical columns: %s',
        len(cat_columns),
        ", ".join(cat_columns))
    logger.info(
        'Found %d numeric columns: %s',
        len(num_columns),
        ", ".join(num_columns))

    # simple histograms for churn and customer age (quantitative variables)
    for column in ('Churn', 'Customer_Age'):
        logger.info('Generating distribution plot for %s...', column)
        plt.figure(figsize=(8, 4))
        df[column].hist()
        plt.title(f'{column.replace("_", " ")} Distribution')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(
            f'./images/{column.lower().replace("_", "-")}-distribution.png', bbox_inches='tight')

    # histogram for marital status (relative; qualitative variable)
    plt.figure(figsize=(8, 4))
    df['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Density')
    plt.savefig(
        './images/marital-status-distribution.png',
        bbox_inches='tight')

    # histogram for total transaction count (relative; quantitative variable
    # w/ KDE)
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Transaction Count Distribution')
    plt.xlabel('Total Transaction Count')
    plt.ylabel('Density')
    plt.savefig(
        './images/total-transaction-count-distribution.png',
        bbox_inches='tight')

    # heatmap of correlation between quantitative variables
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig('./images/correlation-heatmap.png', bbox_inches='tight')

    # prevent figures from spilling into notebooks
    plt.close('all')


def encoder_helper(
        df: pd.DataFrame,
        categories: Iterable[str],
        response: str) -> pd.DataFrame:
    """Feature Encoding Helper Function

    Map categorical columns to mean of response column of original data when grouped by respective
    category first, e.g. result[category] = df.groupby(category).mean()[response].

    Args:
        df: Bank Data as a Pandas DataFrame
        categories: column names that contain categorical features
        response: response column name (e.g. 'Churn')

    Returns:
        Pandas DataFrame with new columns for encoded categorical features.

    Raises:
        KeyError: if any of the specified category or response columns are missing
    """
    result = pd.DataFrame()

    # compute response for each category
    for category in categories:
        grouped_response = df.groupby(category).mean(
            numeric_only=True)[response]
        result[f'{category}_{response}'] = [grouped_response.loc[val]
                                            for val in df[category]]

    logger.info(
        'Create %d additional quantitative features for %s per qualitative category.',
        len(categories),
        response)

    # return response frame w/o altering input data frame
    return result


def perform_feature_engineering(df: pd.DataFrame,
                                response: str) -> tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.Series,
                                                        pd.Series]:
    """Perform Feature Engineering

    Generate Features from every numeric column and transform categorical columns into numeric
    columns by computing the response mean per category and mapping that back to each row.

    Args:
        df: Bank Data as a Pandas DataFrame
        response: response column name (e.g. 'Churn')

    Returns:
        70/30 train/test split of bank data: (X_train, X_test, y_train, y_test)

    Raises:
        KeyError: if the specified response column is missing
    """
    logger.info('Performing Feature Engineering...')

    # exclude non-numeric columns and irrelevant numeric columns from features
    cat_columns = [
        v for v in df.select_dtypes(
            exclude='number').columns if v not in (
            'Attrition_Flag',)]
    num_columns = [
        v for v in df.select_dtypes(
            include='number').columns if v not in (
            'CLIENTNUM', response)]

    logger.info(
        'Found %d qualitative and %d quantitative features.',
        len(cat_columns),
        len(num_columns))

    # transform non-numeric columns into numeric response columns
    response_values = encoder_helper(df, cat_columns, response)

    # merge data
    X_data = df[num_columns].join(response_values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, df[response], test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_roc_cuve(X_test: pd.DataFrame,
                  y_test: pd.Series,
                  models: Iterable[tuple[str,
                                         RandomForestClassifier | LogisticRegression]]):
    """Plot Receiver Operating Characteristic (ROC) Curve for Logistic Regression and Random Forest
    Classifiers

    Args:
        X_test: X testing data
        y_test: true y values for test data
        models: iterable of models and their description as string

    Raises:
        FileExistsError: if a file named 'images' already exists
        OSError: general operating system errors, e.g. permission denied
    """
    # create output directory
    os.makedirs('./images', exist_ok=True)
    logger.info('Plotting ROC Curve...')

    # create legend from description and area under curve (AUC) for each model
    legend: list[str] = []

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    # iterate over each model and plot ROC curve
    for description, model in models:
        plot = RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax, curve_kwargs={'alpha': 0.8})
        legend.append(f'{description} (AUC = {plot.roc_auc:.2f})')
    # set legend
    plt.legend(legend)
    # RocCurveDisplay fixes aspect ratio to 1:1, but we  want 2:1
    ax.set_aspect('auto')
    # store result
    plt.title('Receiver Operating Characteristic')
    plt.savefig('./images/roc-curve.png', bbox_inches='tight')

    # prevent figures from spilling into notebooks
    plt.close('all')


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr: pd.Series,
                                y_train_preds_rf: pd.Series,
                                y_test_preds_lr: pd.Series,
                                y_test_preds_rf: pd.Series):
    """Store Classification Report for Training and Test Results as Images

    Images of classification results will be stored in './images' directory.

    Args:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    Raises:
        FileExistsError: if a file named 'images' already exists
        OSError: general operating system errors, e.g. permission denied
    """
    # create output directory
    os.makedirs('./images', exist_ok=True)
    logger.info('Creating Classification Report Images...')

    # organize report data in an iterable, so we only need to code it once
    report_data = (
        ('Random Forest', y_train_preds_rf, y_test_preds_rf),
        ('Logistic Regression', y_train_preds_lr, y_test_preds_lr),
    )

    for label, y_train_preds, y_test_preds in report_data:
        plt.figure(figsize=(4, 4))
        # manually create monospaced text in figure
        plt.text(
            0.01, 1.25, f'{label} Train', {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_test_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.6, f'{label} Test', {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, y_train_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        # save current figure
        plt.savefig(f'./images/{label.lower().replace(" ",
                                                      "-")}-classification-report.png',
                    bbox_inches='tight')

    # prevent figures from spilling into notebooks
    plt.close('all')


def feature_importance_plot(
        model: RandomForestClassifier,
        X_data: pd.DataFrame,
        output_pth: str | os.PathLike):
    """Store Feature Importance Plot as Image

    Feature Importances are retrieved from model and then plotted as a bar chart
    in descending order of importance.

    Args:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    Raises:
        FileExistsError: if a file with the same name as output_pth's parent already exists
        OSError: general operating system errors, e.g. permission denied
    """
    logger.info('Creating Feature Importance Plot...')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(12, 3))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # create output directory
    Path(output_pth).parent.mkdir(parents=True, exist_ok=True)
    # save output
    plt.savefig(output_pth, bbox_inches='tight')

    # prevent figures from spilling into notebooks
    plt.close('all')


def train_rfc(
        X_train: pd.DataFrame,
        y_train: pd.Series) -> RandomForestClassifier:
    """Train Random Forest Classifier

    Args:
        X_train: X training data
        y_train: y training data

    Returns:
        Best estimator of the trained Random Forest Classifier
    """
    logger.info('Training Random Forest Classifier...')

    # initialize RFC estimator
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy'],
    }

    # perform grid search over 120 parameter configurations above with 5-fold
    # cross validation
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # return the best estimator RFC
    return cv_rfc.best_estimator_


def train_lrc(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train Logistic Regression Classifier

    Args:
        X_train: X training data
        y_train: y training data

    Returns:
        Trained Logistic Regression Classifier
    """
    logger.info('Training Logistic Regression Classifier...')

    # initialize Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', l1_ratio=0.0, max_iter=3000)

    # fit to training data
    lrc.fit(X_train, y_train)

    # return the trained classifier
    return lrc


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        incl_orig_lrc: bool = True):
    """Train and Store Models

    Train Logistic Regression and Random Forest Classifiers on input train/test data.
    Images of classification results will be stored in './images' directory, whereas
    the best estimator of each trained model will be stored in './models' directory.

    The output images are:
    - ROC curve plots for all models (see `incl_orig_lrc` argument)
    - Classification Reports for Logistic Regression and Random Forest Classifiers
    - Feature importance plot for the Random Forest Classifier

    Args:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
        incl_orig_lrc: include original scikit-learn v0.22.x Logistic Regression Classifier in
                       output Receiver Operating Characteristic (ROC) Curve.

    Raises:
        FileExistsError: if a file named 'images' already exists
        OSError: general operating system errors, e.g. permission denied
    """
    logger.info('Training Classifiers...')

    # Unfortunately, due to a bug in early scikit-learn versions, logistic regression would ignore
    # the max_iter parameter and iterate until convergence was archieved, which is probably why this
    # cell originally ran 15 to 20 minutes in the first place.
    # Current versions of scikit-learn respect the max_iter parameter, leading to multiple warnings,
    # which we'll ignore at this point in order to avoid changing the original code too much.
    # As we'll see later, the LRC model is performing much better than
    # previously despite this.
    previous_filters = warnings.filters.copy()
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=FitFailedWarning)
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message='One or more of the test scores are non-finite')

    # train random forest classifier
    rfc = train_rfc(X_train, X_test, y_train, y_test)
    # train logistic regression classifier
    lrc = train_lrc(X_train, X_test, y_train, y_test)

    # restore previous warning filters
    warnings.filters = previous_filters

    # store models
    joblib.dump(rfc, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # generate list of models
    models = [('RFC sci-kit learn 1.8.x', rfc),
              ('LRC sci-kit learn 1.8.x', lrc)]

    # load old LRC model if requested
    if incl_orig_lrc:
        lrc_orig = joblib.load('./models/logistic_model.orig.pkl')
        models.append(('LRC sci-kit learn 0.22.x', lrc_orig))

    # generate ROC curve plots
    plot_roc_cuve(X_test, y_test, models)

    # generate classification predictions for each model
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # output classification report and feature importance plots
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(
        rfc, pd.concat(
            (X_train, X_test)), './images/random-forest-feature-importance.png')


def setup_logging():
    """Setup Logging using Basic Config

    Logs will be overwritten on every run in './logs/churn_library.log' with INFO level.
    Output format is 'logger name - log level - message'.

    Raises:
        FileExistsError: if a file named 'images' already exists
        OSError: general operating system errors, e.g. permission denied
    """
    # create output directory
    os.makedirs('./logs', exist_ok=True)

    # configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)-8s] %(message)s',
        datefmt='%d %b %Y %H:%M:%S',
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler('./logs/churn_library.log', mode='w'),
        ),
    )


def main() -> int:
    """Execute all parts of the library according to Sequence Diagram.

    Returns:
        0 if execution was successful, -1 otherwise
    """
    try:
        # setup logging
        setup_logging()
        # import data
        data = import_data('./data/bank_data.csv')
        # perform exploratory data analysis
        perform_eda(data)
        # get train/test data for model fitting
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data, 'Churn')
        # train models and store results
        train_models(X_train, X_test, y_train, y_test)
    except FileExistsError as exception:
        logger.error(
            'Could not create directory %s, because a file with the same name already exists.',
            exception.filename)
        return -1
    except FileNotFoundError as exception:
        logger.error('File not found: %s', exception.filename)
        return -1
    except KeyError as exception:
        logger.error(
            'Bank Data is missing mandatory \'%s\' column.',
            exception.args[0])
        return -1
    except OSError as exception:
        logger.error(
            'Operation System Error [%s]:\n%s',
            exception.errno,
            exception)
        return -1

    return 0


if __name__ == '__main__':
    sys.exit(main())
