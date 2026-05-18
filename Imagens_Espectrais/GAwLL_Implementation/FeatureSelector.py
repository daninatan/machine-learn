
class FeatureSelector():
    def __init__(self, selected_columns):
        self.selected_columns = selected_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_columns]
