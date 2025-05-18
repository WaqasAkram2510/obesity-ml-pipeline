# src/train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess_data

df = load_data('dataset/ObesityDataSet.csv')
df = preprocess_data(df)

# print(df.head())
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

joblib.dump(model, 'model/random_forest.pkl')
