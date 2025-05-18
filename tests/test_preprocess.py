import pytest
import pandas as pd
import numpy as np
import os
from src.preprocess import load_data, preprocess_data

@pytest.fixture(scope="module")
def raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the relative path to the dataset
    data_path = os.path.join(base_dir, '..', 'dataset', 'ObesityDataSet.csv')
    return load_data(data_path)

@pytest.fixture(scope="module")
def processed_data(raw_data):
    return preprocess_data(raw_data)

def test_no_empty_or_null_values(processed_data):
    assert processed_data.isnull().sum().sum() == 0

def test_no_duplicate_rows(processed_data):
    assert processed_data.duplicated().sum() == 0

def test_categorical_columns_encoded(processed_data):
    assert pd.api.types.is_integer_dtype(processed_data['Gender'])
    assert pd.api.types.is_integer_dtype(processed_data['family_history_with_overweight'])
