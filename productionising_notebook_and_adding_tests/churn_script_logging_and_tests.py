'''
This module contains functions for testing churn_library.py

Author: Leon
Created 16/10/23
'''
import logging
import os

import pandas as pd
import pytest
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering
# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library_testing.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    logging.info("Testing import_data: SUCCESS")


@pytest.fixture(scope="module")
def sample_df():
    '''
    generates sample df for other tests
    '''
    df = import_data("./data/bank_data.csv")
    return df


def test_perform_eda(sample_df):
    '''
    test perform eda function
    '''
    try:
        assert perform_eda(sample_df, 'testing') == 0
        non_df = [1, 2, 3]
        assert perform_eda(non_df, 'testing') == -1
    except AssertionError as err:
        logging.error("Testing perform_eda: function fails to exit correctly")
        raise err

    try:
        assert os.path.exists('testing/churn_histogram.png')
        assert os.path.exists('testing/customer_age_histogram.png')
        assert os.path.exists('testing/percent_marital_status_bar.png')
        assert os.path.exists('testing/total_trans_ct_histogram.png')
        assert os.path.exists('testing/feature_heatmap.png')
    except AssertionError as err:
        logging.error("Testing perform_eda: eda files not generated correctly")
        raise err
    logging.info("Testing perform_eda: SUCCESS")


def test_encoder_helper():
    '''
    test encoder_help function
    '''
    test_df = pd.DataFrame({
        'Category_A': ['a', 'b', 'a', 'c'],
        'Category_B': ['x', 'y', 'y', 'x'],
        'Churn': [1, 0, 0, 1]
    })

    # Test type checking
    try:
        result = encoder_helper("Not a DataFrame", ['Category_A'], 'Churn')
        assert result == -1, "Failed type check for df"

        result = encoder_helper(test_df, "Not a list", 'Churn')
        assert result == -1, "Failed type check for category_lst"

        result = encoder_helper(test_df, ['Category_A'], 123)
        assert result == -1, "Failed type check for response"

    except AssertionError as err:
        logging.error("Type check failed: %e",err)
        raise err

    # Test functionality
    try:
        result_df = encoder_helper(
            test_df, ['Category_A', 'Category_B'], 'Churn')
        expected_columns = set(
            ['Category_A',
             'Category_B',
             'Churn',
             'Category_A_Churn',
             'Category_B_Churn'])
        assert set(result_df.columns) == expected_columns,f"Expected columns {expected_columns}, got {set(result_df.columns)}"

    except AssertionError as err:
        logging.error("Functionality test failed: %e",err)
        raise err

    logging.info("Testing encoder_helper: SUCCESS")


@pytest.fixture(scope="module")
def sample_encoded_df(sample_df):
    '''
    generate encode df for use in other tests
    '''
    df_in = sample_df
    cols_to_encode = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    encoded_df = encoder_helper(df_in, cols_to_encode)
    return encoded_df


def test_perform_feature_engineering(sample_encoded_df):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        sample_encoded_df)
    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: rows or columns in X_train or X_test = 0")
        raise err

    try:
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: rows or columns in y_train or y_test = 0")
        raise err

    logging.info("Testing perform_feature_engineering: SUCCESS")


def test_train_save_models():
    '''
    test train_models
    '''
    try:
        assert os.path.exists(os.path.join('models', 'rfc_model.pkl'))
    except AssertionError as err:
        logging.error(
            "Testing train_save_models: rfc model not found in models folder")
        raise err

    try:
        assert os.path.exists(os.path.join('models', 'logistic_model.pkl'))
    except AssertionError as err:
        logging.error(
            "Testing train_save_models: logistic regression model not found in models folder")
        raise err

    logging.info("Testing train_save_models: SUCCESS")


def test_analyse_performance():
    '''
    test analyse_performance
    '''
    try:
        assert os.path.exists('images/random_forest_report.png')
    except AssertionError:
        logging.error(
            "Testing analyse_performance: random_forest_report.png not found in images folder")

    try:
        assert os.path.exists('images/logistic_regression_report.png')
    except AssertionError as err:
        logging.error(
            "Testing analyse_performance: logistic_regression_report.png not found in images folder")
        raise err

    try:
        assert os.path.exists('images/test_set_roc_curves.png')
    except AssertionError as err:
        logging.error(
            "Testing analyse_performance: test_set_roc_curves.png not found in images folder")
        raise err

    logging.info("Testing analyse_performance: SUCCESS")


def test_feature_importance_plot():
    '''
    test feature_importance_plot
    '''

    try:
        assert os.path.exists('images/feature_importance_plot.png') 
    except AssertionError as err:
        logging.error("Testing feature_importance_plot:\
                       feature_importance_plot.png not found in images folder")
        raise err

    logging.info("Testing feature_importance_plot: SUCCESS")


if __name__ == "__main__":
    test_encoded_df = sample_encoded_df
