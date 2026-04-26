import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import Normalizer, StandardScaler
import seaborn as sns
from src.logistic_regression import MyLogisticRegression
import cupy as cp

from src.pipeline import DiabetesClassifier

df = pd.read_csv('data/diabetes_dataset_with_notes.csv')
X = df.drop('diabetes', axis=1)
# X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
# print(X.std(axis=0))
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_train, y_train, X_test, y_test = X_train.to_numpy()[..., 1:], y_train.to_numpy(), X_test.to_numpy()[...,1:], y_test.to_numpy()
print(X_train.shape)
print(y_train.shape)
scaler = StandardScaler()
# print(scaled_X_train.std(axis=0))
#
model = DiabetesClassifier()
#
# # Подбор гиперпараметра theshold:
#
#
model.fit(X_train, y_train)
# plt.show()

    # print(f'BAE score: {balanced_accuracy_score(y_test, prediction)}')
print(f'F1 score: {model.f1(X_test, y_test)}')
print(f'precision score: {model.precision(X_test, y_test)}')
print(f'Recall score: {model.recall(X_test, y_test)}')
print(f'ROC AUC score: {model.roc_auc(X_test, y_test)}')

# sns.lineplot(x=range(len(f1)), y=f1)
# plt.show()
# sns.lineplot(x=range(len(recall)), y=recall)
# plt.show()
# print(f'Max f1 score: {max(f1)} with threshold: {np.argmax(f1)}')
# print(f'Max Recall score: {max(recall)} with thr: {np.argmax(recall)}')
# print(f'Max ROC AUC score: {max(roc_auc)} with threshold: {np.argmax(roc_auc)}')
# Вычисление ROC AUC
# params = {'threshold': np.arange(0.4, 0.5, 0.01), 'l2': [0, 0.0001, 0.001, 0.01, 0.1],
#           'learning_rate': [0.01, 0.1, 0.5, 1, 10], }
# gs = GridSearchCV(model, params)
# gs.fit(X_train, y_train)
# print("Лучшие параметры:", gs.best_params_)
# print("Лучший ROC AUC:", gs.best_score_)

# Best bas 0.6948454068029521 with l1=0.0001 l2= 0.001 lr=1 thershold=0.69
    # skmodel = LogisticRegression(solver='newton-cholesky', penalty='l2')
    # skmodel.fit(X_train, y_train)
    # prediction = skmodel.predict(X_test)
    # print(f'BAE score: {balanced_accuracy_score(y_test, prediction)}')
    # print(f'F1 score: {f1_score(y_test, prediction)}')
    # print(f'ROC AUC score: {roc_auc_score(y_test, prediction)}')
    # print(f'precision score: {precision_score(y_test, prediction)}')
    # print(f'Recall score: {recall_score(y_test, prediction)}')
# # model.weight_to_csv()
