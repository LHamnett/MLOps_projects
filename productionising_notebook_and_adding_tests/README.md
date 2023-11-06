# Converting Data Science notebook to production python files

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
Author: Leon
Created 16/10/23

## Project Description
This project involves converting a Data Science notebook for predicting customer churn into a aet of python functions and modules. 
Logging is included to troubleshoot the functions and a test suite has been configured to test each function is performing correctly,
and that the end-to-end process works as expected.

## Files and data description
Folder structure:\
├── churn_library.py\
├── churn_notebook.ipynb\
├── churn_script_logging_and_tests.py\
├── conftest.py\
├── data\
├── Guide.ipynb\
├── images\
├── logs\
├── models\
├── __pycache__\
├── pytest.ini\
├── README.md\
├── requirements_py3.6.txt\
├── requirements_py3.8.txt\
└── testing\

churn_library.py: This is the main library containing all the functions used in the project such as data importing, EDA, feature engineering, and model training.

churn_notebook.ipynb: Jupyter Notebook containing the exploratory data analysis and model training steps in an interactive format.

churn_script_logging_and_tests.py: This script includes logging and unit tests for the functions defined in churn_library.py.

conftest.py: Configuration file for pytest.

pytest.ini: Configuration file for pytest specifying the paths and settings for the testing framework.

requirements_py3.6.txt: List of Python packages required for the project running on Python 3.6.

requirements_py3.8.txt: List of Python packages required for the project running on Python 3.8.

Guide.ipynb: An additional notebook to guide users through the project.

README.md: This file, explaining the project structure and how to run the scripts.

Directories
data: Directory to hold the dataset(s) used in this project.

images: Directory to store images, usually generated from the data analysis.

logs: Directory where all log files are stored.

models: Directory to save trained machine learning models.

testing: (If applicable) Directory containing additional test scripts and testing resources.


## Running Files
**python churn_library.py** 

When the file is run, the following steps are executed:

### Initialization:

1. **Import Libraries**: Various libraries such as Pandas, Scikit-learn, Matplotlib, and others are imported.
2. **Logging Setup**: Logging is configured to log messages to `./logs/churn_library.log`.

### Main Execution:

1. **Import Data**: 
   - Calls the `import_data()` function to read a CSV file located at `./data/bank_data.csv`.
   - Adds a 'Churn' column based on the 'Attrition_Flag' column.
   - The resulting DataFrame (`df_test`) is printed for its shape.

2. **Encode Categorical Variables**: 
   - Calls `encoder_helper()` to encode the categorical columns in the DataFrame.
   - Encoded DataFrame (`encoded_df`) is returned.

3. **Feature Engineering**: 
   - Calls `perform_feature_engineering()` on the encoded DataFrame.
   - Splits the data into training and testing sets (`test_x_train`, `test_x_test`, `test_y_train`, `test_y_test`).

4. **Model Performance Analysis**: 
   - Calls `analyse_performance()` to evaluate the performance of pre-trained models.
   - Loads pre-trained Random Forest and Logistic Regression models.
   - Makes predictions on the training and test sets.
   - Generates and saves classification reports and ROC curves.

5. **Feature Importance**: 
   - Loads the pre-trained Random Forest model.
   - Calls `feature_importance_plot()` to generate a feature importance plot.

### Logging:

Throughout the code, logging is used to capture information, warnings, and errors. These logs are saved in `./logs/churn_library.log`.

### Exception Handling:

The code also includes exception handling to catch errors related to file paths, data types, and other issues, logging them accordingly.

### Note:

The code includes some commented-out lines for steps like EDA and model training, indicating that these steps are optional or could be uncommented for a full workflow.

**churn_script_logging_and_tests.py**

When the file is run, the following steps are executed:

### Initialization:

1. **Import Libraries**: Libraries such as `pandas`, `pytest`, and `logging` are imported. Also, functions from `churn_library.py` are imported.
2. **Logging Setup**: Logging is configured to log messages to `./logs/churn_library_testing.log`.

### Test Functions:

1. **`test_import()`**: 
    - Calls `import_data()` function to test data import.
    - Checks if the returned DataFrame has rows and columns.
    - Logs the success or failure of this test.

2. **`sample_df()`** (Pytest Fixture): 
    - Generates a sample DataFrame by calling `import_data()`.
    - This DataFrame is used in other tests.

3. **`test_perform_eda()`**: 
    - Tests the `perform_eda()` function.
    - Checks if the function returns the expected values and generates the expected files.
    - Logs the success or failure of this test.

4. **`test_encoder_helper()`**: 
    - Tests the `encoder_helper()` function.
    - Checks if the function properly encodes categorical variables and handles type errors.
    - Logs the success or failure of this test.

5. **`sample_encoded_df()`** (Pytest Fixture): 
    - Generates an encoded DataFrame by calling `encoder_helper()`.
    - This DataFrame is used in other tests.

6. **`test_perform_feature_engineering()`**: 
    - Tests the `perform_feature_engineering()` function.
    - Checks if the function properly splits the data into training and test sets.
    - Logs the success or failure of this test.

7. **`test_train_save_models()`**: 
    - Tests if the trained models exist in the specified directory.
    - Logs the success or failure of this test.

8. **`test_analyse_performance()`**: 
    - Tests the `analyse_performance()` function.
    - Checks if the function generates and saves the expected files.
    - Logs the success or failure of this test.

9. **`test_feature_importance_plot()`**: 
    - Tests the `feature_importance_plot()` function.
    - Checks if the function generates and saves the expected files.
    - Logs the success or failure of this test.

### Logging:

Throughout the code, logging is used to capture information about the success or failure of each test. These logs are saved in `./logs/churn_library_testing.log`.

### Main Execution:

- The script finally calls `sample_encoded_df()` to generate an encoded DataFrame when run as a standalone script.

### Note:

The code uses Pytest fixtures to reduce redundancy in tests. These fixtures (`sample_df` and `sample_encoded_df`) generate DataFrames that are reused in multiple test functions.
