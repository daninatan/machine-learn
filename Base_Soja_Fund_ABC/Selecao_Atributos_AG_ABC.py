import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns):
        self.selected_columns = selected_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_columns]


def chromosome_to_columns(chromosome, all_columns):
    return [
        col for col, bit in zip(all_columns, chromosome)
        if bit == 1
    ]


def build_preprocessor(selected_columns):

    num_cols = [c for c in variaveis_numericas if c in selected_columns]
    cat_cols = [c for c in variaveis_categoricas if c in selected_columns]

    transformers = []

    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean'))
            ]),
            num_cols
        ))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]),
            cat_cols
        ))

    return ColumnTransformer(transformers)

base_abc = pd.read_csv("Base_ABC_Flor-Colheita.csv")
#variaveis utilizadas

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

base_abc_X  = base_abc[variaveis].copy()
base_abc_Y = base_abc["Produtividade Realizada [kg/ha]"]

#transformando as datas
base_abc_X["Semeadura [data]"] = pd.to_datetime(base_abc_X["Semeadura [data]"], errors='coerce')
base_abc_X["Colheita [data]"] = pd.to_datetime(base_abc_X["Colheita [data]"], errors='coerce')

for col in ["Semeadura [data]", "Colheita [data]"]:
    base_abc_X[f"{col}_ano"] = base_abc_X[col].dt.year
    base_abc_X[f"{col}_mes"] = base_abc_X[col].dt.month
    base_abc_X[f"{col}_dia"] = base_abc_X[col].dt.day

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

#transforma valores que estavam sendo interpretados como str em valores numericos
for col in variaveis_numericas:
    if base_abc_X[col].dtype == "object":
        base_abc_X[col] = (
            base_abc_X[col].str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

X_train, X_test, y_train, y_test = train_test_split(base_abc_X, base_abc_Y, test_size=0.3, random_state=42)

class Individual:
    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = []
        else:
            self.chromosome = chromosome.copy()

def initialize_population(n, n_population, population):
    for i in range(n_population):
        bits = []
        for j in range(n):
            number = random.randint(0, 1)
            bits.append(number)
        individual = Individual(bits)
        population.append(individual)

def calculate_fitness(population, estimator):
    if estimator == "mlp":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            # evita cromossomo vazio
            if len(selected_cols) == 0:
                individual.fitness = -1e9
                continue

            preprocessador = build_preprocessor(selected_cols)

            mlp_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', MLPRegressor())
            ])

            mlp_pipeline.fit(X_train, y_train)
            individual.fitness = mlp_pipeline.score(X_test, y_test)

    elif estimator == "knn":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            # evita cromossomo vazio
            if len(selected_cols) == 0:
                individual.fitness = -1e9
                continue

            preprocessador = build_preprocessor(selected_cols)

            knn_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', KNeighborsRegressor())
            ])

            knn_pipeline.fit(X_train, y_train)
            individual.fitness = knn_pipeline.score(X_test, y_test)
    elif estimator == "dt":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            # evita cromossomo vazio
            if len(selected_cols) == 0:
                individual.fitness = -1e9
                continue

            preprocessador = build_preprocessor(selected_cols)

            dt_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', DecisionTreeRegressor())
            ])

            dt_pipeline.fit(X_train, y_train)
            individual.fitness = dt_pipeline.score(X_test, y_test)

    elif estimator == "rf":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            # evita cromossomo vazio
            if len(selected_cols) == 0:
                individual.fitness = -1e9
                continue

            preprocessador = build_preprocessor(selected_cols)

            rf_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', RandomForestRegressor())
            ])

            rf_pipeline.fit(X_train, y_train)
            individual.fitness = rf_pipeline.score(X_test, y_test)

def generate_offspring(p1, p2, crossing_probability):
    o1 = Individual()
    o2 = Individual()
    crossing = random.randint(0, 100)
    if crossing < crossing_probability:
        crossing_point = random.randint(0, len(p1.chromosome) - 2)
        for i in range(0, crossing_point + 1):
            o1.chromosome.append(p1.chromosome[i])
            o2.chromosome.append(p2.chromosome[i])
        for i in range(crossing_point + 1, len(p1.chromosome)):
            o1.chromosome.append(p2.chromosome[i])
            o2.chromosome.append(p1.chromosome[i])
    else:
        o1 = Individual(p1.chromosome)
        o2 = Individual(p2.chromosome)
    return o1,o2

def apply_probability(population, s):
    size = len(population)
    for i in range(size):
        population[i].prob = ((2 - s) / size) + ((2 * i * (s - 1)) / (size * (size - 1)))

def apply_mutation(f, mutation_probability):

    for i in range(0, len(f.chromosome)):
        mutation = random.randint(0, 100)
        if(mutation <= mutation_probability):
            f.chromosome[i] = f.chromosome[i] ^ 1

def select_parent(population):
    r = random.random()  
    cumulative = 0.0

    for individual in population:
        cumulative += individual.prob
        if r <= cumulative:
            return individual

population = []
new_population = []
n = len(base_abc_X.columns)
n_population = 2
crossing_probability = 80
elitism_number = 20
mutation_probability = 2
s = 1.7
max_generations = 10000
generations = 0

initialize_population(n, n_population, population)
calculate_fitness(population, "mlp")
for individual in population:
    print(individual.fitness)