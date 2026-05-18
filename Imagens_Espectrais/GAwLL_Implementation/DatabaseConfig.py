import pandas as pd
from sklearn.model_selection import train_test_split

class DatabaseConfig:

    def __init__(self, csv_name: str):
        self.base_abc = pd.read_csv(csv_name)

    variaveis = [ 
        "UF",
        "NH.TMED < 10 [sum]",
        "ND.TMIN < 10 [sum]",
        "Altitude Média [classe]",
        "Semeadura [data]",
        "Colheita [data]",
        "Genótipos Propósito",
        "Tipo Safra",
        "Solo Textura",
        "Talhão",
        "Solo Grupo",
        "Amplitude Térmica [soma]",
        "Temperatura Ar Mínima [soma]",
        "Temperatura Ar Máxima [soma]",
        "Fotoperiodo [soma]",
        "Genótipos",
        "Precipitação [soma]",
        "Graus Dias [soma]",
        "Temperatura Ar [soma]",
        "Umidade Relativa Máxima [soma]",
        "Umidade Relativa Mínima [soma]",
        "Umidade Relativa [soma]",
        "Radiação Solar [soma]",
        "Evapotranspiração Potencial PM [soma]",
        "Evapotranspiração Potencial PM2 [soma]",
        "Temperatura Orvalho [soma]",
        "Temperatura Ar Noturna [ºC]",
        "Temperatura Molhamento Foliar [soma]",
        "Altitude Máxima [m]",
        "Altitude Média [m]",
        "Altitude Mínima [m]",
        "NH.UR > 90 [sum]",
        "Ciclo Semeadura-Colheita [dias]",
        "Fungicida Planejado [aplicações]",
        "Herbicida Planejado [aplicações]",
        "Inseticida Planejado [aplicações]",
        "ND.UR < 60 [sum]",
        "ND.UR > 80 [sum]",
        "ND.UR > 90 [sum]",
        "NH.UR < 60 [sum]",
        "Talhão Latitude [grau decimal]",
        "Talhão Longitude [grau decimal]",
        "Colheita [decendio]",
        "ND.RAD < 10 [sum]",
        "ND.PREC > 5 [sum]",
        "NH.RAD > 800 [sum]",
        "Semeadura [decendio]",
        "ND.PREC = 0.0 [sum]",
        "ND.TMAX > 30 [sum]",
        "ND.PREC > 0.2 [sum]",
        "ND.RAD > 20 [sum]",
        "NH.RAD > 1000 [sum]",
        "NH.RAD > 600 [sum]",
    ]

    variaveis_categoricas = [
        "UF",
        "Altitude Média [classe]",
        "Genótipos Propósito",
        "Tipo Safra",
        "Solo Textura",
        "Talhão",
        "Solo Grupo",
        "Genótipos"
    ]

    variaveis_numericas = [
        "NH.TMED < 10 [sum]",
        "ND.TMIN < 10 [sum]",
        "Amplitude Térmica [soma]",
        "Temperatura Ar Mínima [soma]",
        "Temperatura Ar Máxima [soma]",
        "Fotoperiodo [soma]",
        "Precipitação [soma]",
        "Graus Dias [soma]",
        "Temperatura Ar [soma]",
        "Umidade Relativa Máxima [soma]",
        "Umidade Relativa Mínima [soma]",
        "Umidade Relativa [soma]",
        "Radiação Solar [soma]",
        "Evapotranspiração Potencial PM [soma]",
        "Evapotranspiração Potencial PM2 [soma]",
        "Temperatura Orvalho [soma]",
        "Temperatura Ar Noturna [ºC]",
        "Temperatura Molhamento Foliar [soma]",
        "NH.UR > 90 [sum]",
        "ND.UR < 60 [sum]",
        "ND.UR > 80 [sum]",
        "ND.UR > 90 [sum]",
        "NH.UR < 60 [sum]",
        "Talhão Latitude [grau decimal]",
        "Talhão Longitude [grau decimal]",
        "ND.RAD < 10 [sum]",
        "ND.PREC > 5 [sum]",
        "NH.RAD > 800 [sum]",
        "ND.PREC = 0.0 [sum]",
        "ND.TMAX > 30 [sum]",
        "ND.PREC > 0.2 [sum]",
        "ND.RAD > 20 [sum]",
        "NH.RAD > 1000 [sum]",
        "NH.RAD > 600 [sum]",
        "Semeadura [data]_ano",
        "Semeadura [data]_mes",
        "Semeadura [data]_dia",
        "Colheita [data]_ano",
        "Colheita [data]_mes",
        "Colheita [data]_dia",
        "Colheita [decendio]",
        "Semeadura [decendio]",
        "Inseticida Planejado [aplicações]",
        "Fungicida Planejado [aplicações]",
        "Herbicida Planejado [aplicações]",
        "Altitude Máxima [m]",
        "Altitude Média [m]",
        "Altitude Mínima [m]",
        "Ciclo Semeadura-Colheita [dias]",
    ]

    def configDatabase(self):
        base_abc_X  = self.base_abc[self.variaveis].copy()
        base_abc_Y = self.base_abc["Produtividade Realizada [kg/ha]"]

        #transformando as datas
        base_abc_X["Semeadura [data]"] = pd.to_datetime(base_abc_X["Semeadura [data]"], errors='coerce')
        base_abc_X["Colheita [data]"] = pd.to_datetime(base_abc_X["Colheita [data]"], errors='coerce')

        for col in ["Semeadura [data]", "Colheita [data]"]:
            base_abc_X[f"{col}_ano"] = base_abc_X[col].dt.year
            base_abc_X[f"{col}_mes"] = base_abc_X[col].dt.month
            base_abc_X[f"{col}_dia"] = base_abc_X[col].dt.day

        base_abc_X = base_abc_X.drop(columns=["Semeadura [data]", "Colheita [data]"])
        
        #transforma valores que estavam sendo interpretados como str em valores numericos
        for col in self.variaveis_numericas:
            if base_abc_X[col].dtype == "object":
                base_abc_X[col] = (
                    base_abc_X[col].str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )

        X_train, X_test, y_train, y_test = train_test_split(base_abc_X, base_abc_Y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test, base_abc_X
    
    def getLen(self):
        return len(self.base_abc.columns)
    
    def getDatabase(self):
        return self.base_abc
    
    def getNumericas(self):
        return self.variaveis_numericas
    
    def getCategoricas(self):
        return self.variaveis_categoricas