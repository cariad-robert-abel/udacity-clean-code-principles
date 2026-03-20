#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Robert ABEL
Date Created: 19 Mar 2026
 
This module contains a comprehensive suite of unit tests for the customer attrition (churn) library
functions. It is designed to rigorously validate the correctness, robustness, and reliability of each
function within the churn library, covering a wide range of scenarios and edge cases. By
systematically testing data import, feature engineering, model training, and evaluation routines,
this module helps ensure that the churn prediction pipeline remains stable and maintainable as the
codebase evolves.
"""

import warnings

from dataclasses import dataclass
from unittest.mock import call, patch, Mock, ANY, DEFAULT
from typing import TYPE_CHECKING

import pandas as pd
import pytest

# import module-under-test
import churn_library as mut


if TYPE_CHECKING:
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class TrainingData:
    """Test Data for Training Models"""
    x_data: pd.DataFrame
    """Original Data w/o Feature Engineering"""
    x_train: pd.DataFrame
    """Training features"""
    x_test: pd.DataFrame
    """Testing features"""
    y_train: pd.Series
    """Training Response"""
    y_test: pd.Series
    """Testing Response"""
    categories: tuple[str, ...]
    """Categorical Column Names"""


@pytest.fixture(scope='module')
def trainingdata() -> TrainingData:
    """Load Training Data for pytest

    This fixture also tests `import_data` and `perform_feature_engineering` in a working
    configuration, i.e. expecting no exceptions etc.

    Returns:
        Training Data as 10% of actual data.
    """
    # import original data
    df = mut.import_data("./data/bank_data.csv")

    # use 10% for pytest to speed up testing
    x_data = df.sample(frac=0.1, random_state=42, ignore_index=True)

    # compute categorical columns
    cat_columns = [
        v for v in df.select_dtypes(
            exclude='number').columns if v not in (
            'Attrition_Flag',)]

    # 70/30 train/test split
    x_train, x_test, y_train, y_test = mut.perform_feature_engineering(
        x_data, 'Churn')
    return TrainingData(x_data, x_train, x_test, y_train, y_test, cat_columns)


@dataclass(frozen=True)
class TrainedModels:
    """Trained Models for Testing"""
    lrc: 'LogisticRegression'
    """Trained Logistic Regression Classifier"""
    rfc: 'RandomForestClassifier'
    """Trained Random Forest Classifier"""


@pytest.fixture(scope='module')
def trainedmodels(trainingdata: TrainingData) -> TrainedModels:
    """Train LRC and RFC models once using Training Data

    This fixture also tests `train_lrc` and `train_rfc` functions.

    Args:
        trainingdata: training data (fixture)
    """
    # ignore all warnings, because that's not the point of testing
    previous_filters = warnings.filters.copy()
    warnings.simplefilter('ignore')
    lrc = mut.train_lrc(trainingdata.x_train, trainingdata.y_train)
    rfc = mut.train_rfc(trainingdata.x_train, trainingdata.y_train)
    # restore warnings
    warnings.filters = previous_filters
    return TrainedModels(lrc, rfc)


def test_import_data_filenotfound():
    """Test `import_data` with a non-existent file path"""
    with pytest.raises(FileNotFoundError):
        mut.import_data('./this/path/does/not/exist.csv')


def test_import_data_wo_attrition_column(tmp_path: 'Path'):
    """Test `import_data` with a file missing the 'Attrition_Flag' column

    Args:
        tmp_path: temporary directory (fixture)
    """
    # read original CSV
    df = pd.read_csv('./data/bank_data.csv', index_col=0)
    # drop attrition flag
    df.drop(columns=['Attrition_Flag'], inplace=True)
    # write to temp directory
    df.to_csv(tmp_path / 'bank_data_missing_attrition.csv')

    # expect conversion to Churn column to fail
    with pytest.raises(KeyError):
        mut.import_data(tmp_path / 'bank_data_missing_attrition.csv')


def test_perform_feature_engineering_wo_response(trainingdata: TrainingData):
    """Test `perform_feature_engineering` with a missing response column

    Args:
        trainingdata: training data (fixture)
    """
    with pytest.raises(KeyError):
        mut.perform_feature_engineering(
            trainingdata.x_data, response='Missing-Column')


def test_encoder_helper_single_column(trainingdata: TrainingData):
    """Test `encoder_helper` with a single column

    Multiple columns are implicitly tested in trainingdata fixture already.

    Args:
        trainingdata: training data (fixture)
    """
    # encode a single column
    first_cat = trainingdata.categories[0]
    result = mut.encoder_helper(trainingdata.x_data, (first_cat,), 'Churn')

    # check result columns
    assert f'{first_cat}_Churn' in result.columns, \
        f'Expected encoded column {first_cat}_Churn not found in output DataFrame.'
    assert 1 == len(result.columns), \
        f'Expected only 1 column in output DataFrame, but found {len(result.columns)}.'


def test_encoder_helper_missing_column(trainingdata: TrainingData):
    """Test `encoder_helper` with missing category/response columns

    Args:
        trainingdata: training data (fixture)
    """
    # test missing response
    first_cat = trainingdata.categories[0]
    with pytest.raises(KeyError):
        mut.encoder_helper(trainingdata.x_data, (first_cat,), 'Missing-Column')
    # test missing column
    with pytest.raises(KeyError):
        mut.encoder_helper(
            trainingdata.x_data,
            (first_cat,
             'Missing-Column'),
            'Churn')


def test_perform_eda(
        trainingdata: TrainingData,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `perform_eda` with valid input data

    Args:
        trainingdata: training data (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # perform EDA
    mut.perform_eda(trainingdata.x_data)

    # check output directory
    outdir = tmp_path / 'images'

    assert outdir.exists(), \
        'Expected <cwd>/images directory to be created.'

    assert 0 < len(tuple(outdir.glob('*.png'))), \
        'Expected at lest one EDA output picture.'


def test_perform_eda_missing_column(
        trainingdata: TrainingData,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `perform_eda` with at least one missing column

    Args:
        trainingdata: training data (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # perform EDA
    with pytest.raises(KeyError):
        mut.perform_eda(trainingdata.x_data.drop('Churn'))


def test_perform_eda_fileexistserror(
        trainingdata: TrainingData,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `perform_eda` when output directory cannot be created

    Args:
        trainingdata: training data (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # create images file, so output directory cannot be created
    (tmp_path / 'images').touch()

    # perform EDA
    with pytest.raises(FileExistsError):
        mut.perform_eda(trainingdata.x_data)


def test_plot_roc_curve(
        trainingdata: TrainingData,
        trainedmodels: TrainedModels,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `plot_roc_cuve` with valid input data

    Args:
        trainingdata: training data (fixture)
        trainedmodels: trained models (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # plot curve
    models = (
        ('Logistic Regression', trainedmodels.lrc),
        ('Random Forest Classifier', trainedmodels.rfc)
    )
    mut.plot_roc_cuve(trainingdata.x_test, trainingdata.y_test, models)

    # check output directory
    outdir = tmp_path / 'images'

    assert outdir.exists(), \
        'Expected <cwd>/images directory to be created.'

    assert 0 < len(tuple(outdir.glob('*.png'))), \
        'Expected at lest one ROC curve output picture.'


def test_plot_roc_curve_fileexistserror(
        trainingdata: TrainingData,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `plot_roc_cuve` when output directory cannot be created

    Args:
        trainingdata: training data (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # create images file, so output directory cannot be created
    (tmp_path / 'images').touch()

    # plot curve
    with pytest.raises(FileExistsError):
        mut.plot_roc_cuve(trainingdata.x_test, trainingdata.y_test, ())


def test_classification_report_image(
        trainingdata: TrainingData,
        trainedmodels: TrainedModels,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `classification_report_image` with valid input data

    Args:
        trainingdata: training data (fixture)
        trainedmodels: trained models (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # create predictions based on trained models
    kwargs: dict[str, pd.Series] = {}
    for label, model in (('lr', trainedmodels.lrc), ('rf', trainedmodels.rfc)):
        kwargs[f'y_train_preds_{label}'] = model.predict(trainingdata.x_train)
        kwargs[f'y_test_preds_{label}'] = model.predict(trainingdata.x_test)

    # plot curve
    mut.classification_report_image(
        trainingdata.y_train,
        trainingdata.y_test,
        **kwargs)

    # check output directory
    outdir = tmp_path / 'images'

    assert outdir.exists(), \
        'Expected <cwd>/images directory to be created.'

    assert 2 == len(tuple(outdir.glob('*.png'))), \
        'Expected exactly two classification report output pictures.'


def test_classification_report_image_fileexistserror(
        trainingdata: TrainingData,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `classification_report_image` when output directory cannot be created

    Args:
        trainingdata: training data (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # create images file, so output directory cannot be created
    (tmp_path / 'images').touch()

    # plot curve
    with pytest.raises(FileExistsError):
        mut.classification_report_image(
            trainingdata.y_train,
            trainingdata.y_test,
            trainingdata.y_train,
            trainingdata.y_train,
            trainingdata.y_test,
            trainingdata.y_test)


def test_feature_importance_plot(
        trainingdata: TrainingData,
        trainedmodels: TrainedModels,
        tmp_path: 'Path'):
    """Test `classification_report_image` with valid input data

    Args:
        trainingdata: training data (fixture)
        trainedmodels: trained models (fixture)
        tmp_path: temporary directory (fixture)
    """
    # check output file
    outfile = tmp_path / 'directory' / 'subdirectory' / 'feature_importance.png'

    # reconstruct data w/ feature engineering
    x_data = pd.concat((trainingdata.x_train, trainingdata.x_test))

    # plot feature importances
    mut.feature_importance_plot(trainedmodels.rfc, x_data, outfile)

    assert outfile.exists(), \
        'Expected feature importance plot to be created.'


def test_feature_importance_plot_fileexistserror(
        trainingdata: TrainingData,
        trainedmodels: TrainedModels,
        tmp_path: 'Path'):
    """Test `classification_report_image` with valid input data

    Args:
        trainingdata: training data (fixture)
        trainedmodels: trained models (fixture)
        tmp_path: temporary directory (fixture)
    """
    # check output file
    outfile = tmp_path / 'directory' / 'subdirectory' / 'feature_importance.png'

    # create directory file, so output directory cannot be created
    (tmp_path / 'directory').touch()

    # reconstruct data w/ feature engineering
    x_data = pd.concat((trainingdata.x_train, trainingdata.x_test))

    # plot feature importances
    with pytest.raises(FileExistsError):
        mut.feature_importance_plot(trainedmodels.rfc, x_data, outfile)

    # create sub-directory file, so output directory cannot be created
    (tmp_path / 'directory').rename(tmp_path / 'directory2')
    (tmp_path / 'directory').mkdir()
    (tmp_path / 'directory' / 'subdirectory').touch()

    with pytest.raises(FileExistsError):
        mut.feature_importance_plot(trainedmodels.rfc, x_data, outfile)


def test_train_models(
        trainingdata: TrainingData,
        trainedmodels: TrainedModels,
        tmp_path: 'Path',
        monkeypatch: pytest.MonkeyPatch):
    """Test `train_models` with valid input data

    We ignore the original LRC, because it's not part of the original course work
    and I don't think we'll benefit from copying the model around.

    Args:
        trainingdata: training data (fixture)
        trainedmodels: trained models (fixture)
        tmp_path: temporary directory (fixture)
        monkeypatch: pytest MonkeyPatch fixture
    """
    # change into temp directory using monkeypath
    monkeypatch.chdir(tmp_path)

    # create models output directory (assumed to be present due to repo
    # structure)
    (tmp_path / 'models').mkdir()

    with patch.multiple(mut,
                        train_lrc=DEFAULT,
                        train_rfc=DEFAULT,
                        plot_roc_cuve=DEFAULT,
                        classification_report_image=DEFAULT,
                        feature_importance_plot=DEFAULT) as mocks:
        # return mock trained models instead of re-training
        mocks['train_lrc'].return_value = trainedmodels.lrc
        mocks['train_rfc'].return_value = trainedmodels.rfc

        # train models
        mut.train_models(
            trainingdata.x_train,
            trainingdata.x_test,
            trainingdata.y_train,
            trainingdata.y_test,
            incl_orig_lrc=False)

        # models should have been trained exactly once
        assert 1 == mocks['train_lrc'].call_count, \
            'Expected train_lrc to be called exactly once.'
        assert 1 == mocks['train_rfc'].call_count, \
            'Expected train_rfc to be called exactly once.'

        # output plots also need to be called exactly once
        assert 1 == mocks['plot_roc_cuve'].call_count, \
            'Expected plot_roc_cuve to be called exactly once.'
        assert 1 == mocks['classification_report_image'].call_count, \
            'Expected classification_report_image to be called exactly once.'
        assert 1 == mocks['feature_importance_plot'].call_count, \
            'Expected feature_importance_plot to be called exactly once.'


def test_main(trainingdata: TrainingData):
    """Test `main` functioncall order

    Args:
        trainingdata: training data (fixture)
    """
    # mock all relevant functions
    with patch.multiple(mut,
                        import_data=DEFAULT,
                        perform_eda=DEFAULT,
                        perform_feature_engineering=DEFAULT,
                        train_models=DEFAULT) as mocks:

        # make `import_data`, `perform_feature_engineering` return trainingdata
        mocks['import_data'].return_value = trainingdata.x_data
        mocks['perform_feature_engineering'].return_value = (
            trainingdata.x_train,
            trainingdata.x_test,
            trainingdata.y_train,
            trainingdata.y_test)

        # set up expected calling order (must be list!)
        expected_calls = [
            call.import_data(ANY),
            call.perform_eda(
                trainingdata.x_data),
            call.perform_feature_engineering(
                trainingdata.x_data,
                'Churn'),
            call.train_models(
                trainingdata.x_train,
                trainingdata.x_test,
                trainingdata.y_train,
                trainingdata.y_test),
        ]

        # create top-level mock to record calling order
        main = Mock()
        main.attach_mock(mocks['import_data'], 'import_data')
        main.attach_mock(mocks['perform_eda'], 'perform_eda')
        main.attach_mock(
            mocks['perform_feature_engineering'],
            'perform_feature_engineering')
        main.attach_mock(mocks['train_models'], 'train_models')

        # execute main function
        rc = mut.main()

        # make sure return code is 0
        assert 0 == rc, \
            'main failed with rc={rc}.'
        # make sure calls were in order with proper arguments
        assert main.mock_calls == expected_calls, \
            'main did not call functions in expected order/with expected arguments.'
