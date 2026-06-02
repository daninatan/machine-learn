import pandas as pd
from sklearn.model_selection import train_test_split

class DatabaseConfig:

    def __init__(self, csv_name: str):
        self.base = pd.read_csv(csv_name, decimal=',')
        self.base.drop(["Data", "Code"], axis=1, inplace=True)
        self.base_soja = self.base

    def configDatabase(self):
        base_soja = self.base[self.base["Planta"] == "Soy"]
        self.base_soja = base_soja
        base_Y = base_soja["Contagio"]
        base_X  = self.base_soja.drop(["Planta", "Contagio"], axis = 1, inplace=False)
        

        X_train, X_test, y_train, y_test = train_test_split(base_X, base_Y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test, base_X
    
    def getLen(self):
        return len(self.base_soja.columns)
    
    def getDatabase(self):
        return self.base_soja
