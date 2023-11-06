import logging
def pytest_configure(config):
    logging.basicConfig(filename='./logs/churn_library_testing.log', level=logging.INFO)

