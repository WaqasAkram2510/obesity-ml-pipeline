import pytest
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, preprocess_data

@pytest.fixture(scope="module")
def processed_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the relative path to the dataset
    data_path = os.path.join(base_dir, '..', 'dataset', 'ObesityDataSet.csv')
    return preprocess_data(load_data(data_path))


def test_model_accuracy_threshold(processed_data):
    X = processed_data.drop('NObeyesdad', axis=1)
    y = processed_data['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.80

def test_model_file_exists():
    assert os.path.isfile('model/random_forest.pkl')

def test_saved_model_loads_correctly():
    model = joblib.load('model/random_forest.pkl')
    assert hasattr(model, 'predict')
