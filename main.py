import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import Normalizer
import seaborn as sns
from src.logistic_regression import MyLogisticRegression
import cupy as cp


df = pd.read_csv('data/normal_data.csv')
X = df.drop('diabetes', axis=1)
# X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
print(X.std(axis=0))
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, y_train, X_test, y_test = X_train.to_numpy()[..., 1:], y_train.to_numpy(), X_test.to_numpy()[...,1:], y_test.to_numpy()
print(X_train.shape)
print(y_train.shape)
scaler = Normalizer()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
# print(scaled_X_train.std(axis=0))
#
model = MyLogisticRegression(mode='gpu')
#
# # Подбор гиперпараметра theshold:
#
#
losses = model.fit(scaled_X_train, y_train,  learning_rate=0.2)
if isinstance(losses, cp.ndarray):
    losses = cp.asnumpy(losses)
elif hasattr(losses, 'get'):
    losses = losses.get()
f1 = []
roc_auc = []
# sns.lineplot(x=range(len(losses)), y=losses)
# plt.show()
for threshold in np.arange(0, 1, 0.01):
    prediction = model.predict(scaled_X_test, threshold)
    if isinstance(prediction, cp.ndarray):
        prediction = cp.asnumpy(prediction)
    elif hasattr(prediction, 'get'):
        prediction = prediction.get()
    f1.append(f1_score(y_test, prediction))
    roc_auc.append(roc_auc_score(y_test, prediction))
    print(f'threshold: {threshold}')
    print(f'BAE score: {balanced_accuracy_score(y_test, prediction)}')
    print(f'F1 score: {f1_score(y_test, prediction)}')
    print(f'ROC AUC score: {roc_auc_score(y_test, prediction)}')
    print(f'precision score: {precision_score(y_test, prediction)}')
    print(f'Recall score: {recall_score(y_test, prediction)}')
sns.lineplot(x=range(len(f1)), y=f1)
plt.show()
sns.lineplot(x=range(len(roc_auc)), y=roc_auc)
plt.show()
print(f'Max f1 score: {max(f1)} with threshold: {np.argmax(f1)}')
print(f'Max ROC AUC score: {max(roc_auc)} with threshold: {np.argmax(roc_auc)}')

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
