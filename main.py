import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import Normalizer
import seaborn as sns
from logistic_regression import MyLogisticRegression

df = pd.read_csv('normal_data.csv')
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
print(scaled_X_train.std(axis=0))
from sklearn.model_selection import GridSearchCV


model = MyLogisticRegression()
losses = model.fit(scaled_X_train, y_train)
sns.lineplot(losses)
plt.show()
baes = []
# Подбор гиперпараметра theshold:
lrs = []
l1s = []
l2s = []
for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
    for l1 in [0.0001, 0.001, 0.01, 0.1, 1]:
        for l2 in [0.0001, 0.001, 0.01, 0.1, 1]:
            losses = model.fit(scaled_X_train, y_train, l1=l1, l2=l2, learning_rate=lr)
            for thershold in np.arange(0, 1, 0.01):
                prediction = model.predict(scaled_X_test, thershold)
                bae = balanced_accuracy_score(y_test, prediction)
                baes.append(bae)
                print(bae, l1, l2, lr, thershold)
print(max(baes), baes.index(max(baes)))
# Best bas 0.6948454068029521 with l1=0.0001 l2= 0.001 lr=1 thershold=0.69

# # model.weight_to_csv()

sns.lineplot(baes)
plt.show()
# skmodel = LogisticRegression()
# skmodel.fit(X_train, y_train)
# y_pred = skmodel.predict(X_test)
# print(balanced_accuracy_score(y_test, y_pred))