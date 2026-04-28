import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import DiabetesClassifier

df = pd.read_csv('data/diabetes_dataset_with_notes.csv')
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DiabetesClassifier()

model.fit(X_train, y_train)

print(f'F1 score: {model.f1(X_test, y_test)}')
print(f'precision score: {model.precision(X_test, y_test)}')
print(f'Recall score: {model.recall(X_test, y_test)}')
print(f'ROC AUC score: {model.roc_auc(X_test, y_test)}')
