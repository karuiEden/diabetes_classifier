import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(X):
    cat_col = ['clinical_notes', 'year']
    col_exist = [col for col in cat_col if col in X.columns]
    if col_exist:
        X.drop(columns=cat_col, inplace=True)
    return X


class Preprocessor:
    def __init__(self, num_features: list[str], cat_features: list[str]):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        self.scaler = StandardScaler()
        if num_features == [] or cat_features == []:
            raise ValueError("Features must be filled")
        self.num_features = num_features
        self.cat_features = cat_features

    def fit_transform(self, X: pd.DataFrame):
        X = clean_data(X)
        X[self.num_features] = self.scaler.fit_transform(X[self.num_features])
        x_cat = self.ohe.fit_transform(X[self.cat_features])
        ohe_cols = self.ohe.get_feature_names_out(self.cat_features)
        x_cat = pd.DataFrame(x_cat, columns=ohe_cols, index=X.index)
        X.drop(columns=self.cat_features, inplace=True)
        X = X.join(x_cat)
        return X

    def transform(self, X: pd.DataFrame):
        X = clean_data(X)
        X[self.num_features] = self.scaler.transform(X[self.num_features])
        x_cat = self.ohe.transform(X[self.cat_features])
        ohe_cols = self.ohe.get_feature_names_out(self.cat_features)
        x_cat = pd.DataFrame(x_cat, columns=ohe_cols, index=X.index)
        X.drop(columns=self.cat_features, inplace=True)
        X = X.join(x_cat)
        return X