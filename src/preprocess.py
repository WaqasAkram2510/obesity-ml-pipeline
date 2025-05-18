import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    df = df.replace(r'^\s*$', pd.NA, regex=True)
 
    df = df.dropna()

    df = df.drop_duplicates()

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df
