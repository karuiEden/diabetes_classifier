import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from models.logistic_regression import MyLogisticRegression


class DiabetesClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, mode='cpu', l2=0, learning_rate=0.1, tol=1e-5, threshold=0.44):
        self.ohe = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        self.idx = None
        self.l2 = l2
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.tol = tol
        self.mode = mode
        self.model = MyLogisticRegression(mode=mode, l2=l2, learning_rate=learning_rate, tol=tol)

    def clean_data(self, x):
        cat_col = ['clinical_notes', 'year']
        col_exist = [col for col in cat_col if col in x.columns]
        if col_exist:
            x.drop(columns=cat_col, inplace=True)
        idx = x[(x.gender == 'Other')].index
        self.idx = idx
        x.drop(idx, inplace=True)
        x.gender = x.gender.apply(lambda x: 0 if x == "Male" else 1)
        return x

    def fit(self, x, y):
        x = self.clean_data(x.copy())
        num_features = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
        x[num_features] = self.scaler.fit_transform(x[num_features])
        cat_features = ['location', 'smoking_history']
        x_cat = self.ohe.fit_transform(x[cat_features])
        ohe_cols = self.ohe.get_feature_names_out(cat_features)
        x_cat = pd.DataFrame(x_cat, columns=ohe_cols, index=x.index)
        x.drop(columns=cat_features, inplace=True)
        x = x.join(x_cat)
        x = x.to_numpy()[..., 1:]
        y.drop(self.idx, inplace=True)
        y = y.to_numpy()
        losses = self.model.fit(x, y)
        return losses

    def predict(self, x):
        x = self.clean_data(x.copy())
        cat_features = ['location', 'smoking_history']
        x_cat = self.ohe.transform(x[cat_features])
        ohe_cols = self.ohe.get_feature_names_out(cat_features)
        x_cat = pd.DataFrame(x_cat, columns=ohe_cols, index=x.index)
        x.drop(columns=cat_features, inplace=True)
        x = x.join(x_cat)
        num_features = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
        x[num_features] = self.scaler.transform(x[num_features])
        x = x.to_numpy()[..., 1:]
        return self.model.predict(x, threshold=self.threshold)

    def predict_prob(self, x):
        x = self.clean_data(x.copy())
        cat_features = ['location', 'smoking_history']
        x_cat = self.ohe.transform(x[cat_features])
        ohe_cols = self.ohe.get_feature_names_out(cat_features)
        x_cat = pd.DataFrame(x_cat, columns=ohe_cols, index=x.index)
        x.drop(columns=cat_features, inplace=True)
        x = x.join(x_cat)
        num_features = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
        x[num_features] = self.scaler.transform(x[num_features])
        x = x.to_numpy()[..., 1:]
        return self.model.predict_prob(x)

    def score(self, x, y_test, sample_weight=None):
        prediction = self.predict(x)
        if hasattr(prediction, 'get'):
            prediction = prediction.get()
        y = y_test.drop(self.idx)
        y = y.to_numpy()
        return (prediction == y).mean()

    def f1(self, x, y_test):
        prediction = self.predict(x)
        if hasattr(prediction, 'get'):
            prediction = prediction.get()
        y = y_test.drop(self.idx)
        y = y.to_numpy()
        return f1_score(y, prediction)

    def roc_auc(self, x, y_test):
        prediction = self.predict_prob(x)
        if hasattr(prediction, 'get'):
            prediction = prediction.get()
        y = y_test.drop(self.idx)
        y = y.to_numpy()
        return roc_auc_score(y, prediction)

    def recall(self, x, y_test):
        prediction = self.predict(x)
        if hasattr(prediction, 'get'):
            prediction = prediction.get()
        y = y_test.drop(self.idx)
        y = y.to_numpy()
        return recall_score(y, prediction)

    def precision(self, x, y_test):
        prediction = self.predict(x)
        if hasattr(prediction, 'get'):
            prediction = prediction.get()
        y = y_test.drop(self.idx)
        y = y.to_numpy()
        return precision_score(y, prediction)

    def save_weight(self):
        weights = self.model.get_weights()
        if hasattr(weights, 'get'):
            weights = weights.get()
        np.save('weights.npy', weights)

    def load_weight(self):
        weights = np.load('weights.npy')
        self.model.w = self.model.mv.asarray(weights)