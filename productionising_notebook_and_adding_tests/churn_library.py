# library doc string
"""
This module contains functions for: 
- processing a data frame,
- performing EDA,
- training and saving random forest classification and logistic regression models,
- analysisng model performance,
- finding feature importance

Author: Leon
Created 16/10/23
"""


# import libraries
import os
import logging
import joblib

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    try:
        assert os.path.exists(pth)
    except AssertionError:
        logging.error("ERROR: file path %f not found", pth)
        return -1
    df = pd.read_csv(pth)
    logging.info("INFO: file path exists and df created")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df, eda_output_dir='./images'):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        assert isinstance(df, pd.core.frame.DataFrame)
    except AssertionError:
        logging.error("ERROR: input df is not of type dataframe,\
                      instead is type {}".format(type(df)))
        return -1

    logging.info("INFO: df has shape {}".format(df.shape))
    print('Displaying null values per column:')
    print(df.isnull().sum())

    print('Displaying df descibe:')
    print(df.describe())

    print('Generating churn histogram:')
    df['Churn'].hist()
    plt.savefig(os.path.join(eda_output_dir, 'churn_histogram.png'))
    plt.close()
    logging.info("INFO: churn histogram saved")

    print('Generating customer age histogram:')
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(eda_output_dir, 'customer_age_histogram.png'))
    plt.close()
    logging.info("INFO: customer age histogram saved")

    print('generating percentage marital status barchart:')
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(eda_output_dir, 'percent_marital_status_bar.png'))
    plt.close()
    logging.info("INFO: percentage marital status barchart saved")

    print('generating total trans histogram:')
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(eda_output_dir, 'total_trans_ct_histogram.png'))
    plt.close()
    logging.info("INFO: total trans histogram saved")

    print('generating feature heatmap:')
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(eda_output_dir, 'feature_heatmap.png'))
    plt.close()
    logging.info("INFO: feature heatmap saved")

    logging.info("SUCCESS: all EDA steps completed")
    return 0


def encoder_helper(df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name 
                [optional argument that could be used for naming variables or index y column]
                
    output:
            df: pandas dataframe with new columns for
    '''
    try:
        assert isinstance(df, pd.core.frame.DataFrame)
    except AssertionError:
        logging.error("ERROR: input df is not of type dataframe,\
                       instead is type %i",type(df))
        return -1

    try:
        assert isinstance(category_lst, list)
    except AssertionError:
        logging.error("ERROR: input category_lst is not of type list,\
                       instead is type %i",type(category_lst))
        return -1

    if response is not None:
        try:
            assert isinstance(response, str)
        except AssertionError:
            logging.error("ERROR: input category_lst is not of type list,\
                       instead is type %i",type(response))
            return -1

    for col in category_lst:
        new_col_name = col + '_Churn'
        df[new_col_name] = df[col].map(df.groupby(col).mean()['Churn'])

    logging.info(
        "SUCCESS: categorical columns encoded,returning new df with shape {}".format(df.shape))
    return df


def perform_feature_engineering(df_encoded, response=None):
    '''
    input:
              df: pandas dataframe with categorical columns encoded
              response: string of response name 
                  [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    try:
        assert isinstance(df_encoded, pd.core.frame.DataFrame)
    except AssertionError:
        logging.error("ERROR: input df is not of type dataframe,\
              instead is type %i",type(df_encoded))
        return -1

    if response is not None:
        try:
            assert isinstance(response, str)
        except AssertionError:
            logging.error("ERROR: input category_lst is not of type list,\
                       instead is type %i",type(response))
            return -1

    x_cols = df_encoded[['Customer_Age',
                         'Dependent_count',
                         'Months_on_book',
                         'Total_Relationship_Count',
                         'Months_Inactive_12_mon',
                         'Contacts_Count_12_mon',
                         'Credit_Limit',
                         'Total_Revolving_Bal',
                         'Avg_Open_To_Buy',
                         'Total_Amt_Chng_Q4_Q1',
                         'Total_Trans_Amt',
                         'Total_Trans_Ct',
                         'Total_Ct_Chng_Q4_Q1',
                         'Avg_Utilization_Ratio',
                         'Gender_Churn',
                         'Education_Level_Churn',
                         'Marital_Status_Churn',
                         'Income_Category_Churn',
                         'Card_Category_Churn']]

    y = df_encoded['Churn']

    x_train, x_test, y_train, y_test = train_test_split(
        x_cols, y, test_size=0.3, random_state=42)

    logging.info(f"SUCCESS: train_test_splits created with shapes: \n\
    X_train: {x_train.shape} \n\
    X_test: {x_test.shape} \n\
    y_train: {y_train.shape} \n\
    y_test: {y_test.shape} \n")
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    rf_class_report_train = classification_report(y_train, y_train_preds_rf)
    rf_class_report_test = classification_report(y_test, y_test_preds_rf)
    lr_class_report_train = classification_report(y_train, y_train_preds_lr)
    lr_class_report_test = classification_report(y_test, y_test_preds_lr)

    # save rf report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01,
             0.05,
             str(rf_class_report_train),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01,
             0.7,
             str(rf_class_report_test),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/random_forest_report.png')
    plt.close()
    logging.info("SUCCESS: saved random forest classification report")

    # save lr report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01,
             0.05,
             str(lr_class_report_train),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01,
             0.7,
             str(lr_class_report_test),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/logistic_regression_report.png')
    plt.close()
    logging.info("SUCCESS: saved logistic regression report")

    return 0


def feature_importance_plot(model, x_data=None, output_pth='./images'):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), feature_names, rotation=90)

    save_path = os.path.join(output_pth, 'feature_importance_plot.png')
    plt.savefig(save_path)
    plt.close()
    logging.info("SUCCESS: saved feature importance plot")

    return 0


def train_save_models(x_train, y_train, model_dir='./models'):
    '''
    train models and store model objects in .pkl format
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        assert os.path.exists(model_dir)
    except FileNotFoundError:
        logging.error("ERROR: model directory %f not found",model_dir)

    # train and save randomforestclassifier
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [5, 10],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)
    logging.info("INFO: starting training: random forest classifier")
    print("starting training: random forest classifier")
    cv_rfc.fit(x_train, y_train)
    # rfc.fit(x_train, y_train)
    logging.info("SUCCESS: finished training: random forest classifier")
    print("finished training: random forest classifier")
    rfc_save_path = os.path.join(model_dir, 'rfc_model.pkl')
    # cv_rfc = rfc
    joblib.dump(cv_rfc.best_estimator_, rfc_save_path)
    # joblib.dump(rfc, rfc_save_path)
    logging.info("SUCCESS: saved random forest classifer model at {}".format(rfc_save_path))

    # logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    logging.info("INFO: starting training: logistic regression")
    print('starting training log regression')
    lrc.fit(x_train, y_train)
    logging.info("SUCCESS: finished training: logistic regression")
    print('finished training logistic regression')
    lr_save_path = os.path.join(model_dir, 'logistic_model.pkl')
    joblib.dump(lrc, lr_save_path)
    logging.info("SUCCESS: saved logistic regression model at {}".format(lr_save_path))

    return 0


def analyse_performance(model_dir, x_train, x_test, y_train, y_test):
    """
    Analyzes the performance of pre-trained Random Forest Classifier and Logistic Regression models

    This function loads the pre-trained models from the specified directory, makes predictions on the
    training and test sets, and then generates and saves classification reports and ROC curves.

    Parameters:
    -----------
    model_dir : str
        The directory where the pre-trained models are stored.
    X_train : pd.DataFrame or np.ndarray
        Features for the training set.
    X_test : pd.DataFrame or np.ndarray
        Features for the test set.
    y_train : pd.Series or np.ndarray
        Labels for the training set.
    y_test : pd.Series or np.ndarray
        Labels for the test set.

    Returns:
    --------
    int
        Returns 0 if the function runs successfully.

    Raises:
    -------
    FileNotFoundError
        If the specified model directory does not exist.
    """

    try:
        assert os.path.exists(model_dir)
    except FileNotFoundError:
        logging.error("ERROR: model directory %f not found",model_dir)
        return -1

    # load models
    try:
        rfc_model = joblib.load(os.path.join(model_dir, 'rfc_model.pkl'))
        lr_model = joblib.load(os.path.join(model_dir, 'logistic_model.pkl'))
    except:
        logging.error("ERROR: models could not be loaded")
        return -1              
    # get predictions random forest
    logging.info("INFO: getting predictions: random forest classifier")
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)
    logging.info("SUCCESS: finished getting predictions: random forest classifier")

    # get predictions logistic reression
    logging.info("INFO: getting predictions: logistic regression")
    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)
    logging.info("SUCCESS: finished getting predictions: logistic regression")

    # make and save classification reports
    logging.info("INFO: generating classification reports")
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # make and save roc_curves
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/test_set_roc_curves.png')
    plt.close()
    logging.info("SUCCESS: finished saving ROC curves")

    return 0


if __name__ == '__main__':
    bank_df = import_data(r"./data/bank_data.csv")
    print('loaded df')
    perform_eda(bank_df)
    print('completed eda')
    encoded_df = encoder_helper(bank_df,
                                ['Gender',
                                 'Education_Level',
                                 'Marital_Status',
                                 'Income_Category',
                                 'Card_Category'])
    print('encoded categorical columns')
    current_x_train, current_x_test, current_y_train, current_y_test = perform_feature_engineering(encoded_df)
    print('performed feature engineering')
    train_save_models(current_x_train, current_y_train, './models')
    print('trained and saved models')
    analyse_performance('./models', current_x_train, current_x_test, current_y_train, current_y_test)
    print('analysed model performance')
    test_rfc_model = joblib.load(os.path.join('./models', 'rfc_model.pkl'))
    feature_importance_plot(test_rfc_model, current_x_train)
    print('generated feature importance')
    print('all steps finished')
