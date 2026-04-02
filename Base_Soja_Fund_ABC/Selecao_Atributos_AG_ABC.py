import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector():
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

#weighted sum between classification score and number of attributes
def calculate_fitness(population, estimator, fsw, faw):
    if estimator == "mlp":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            preprocessador = build_preprocessor(selected_cols)

            mlp_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', MLPRegressor(
                            hidden_layer_sizes=(32,),
                            max_iter=150,
                            early_stopping=True,
                            ))
            ])

            scores = cross_val_score(
                mlp_pipeline,
                X_train,
                y_train,
                n_jobs = -1
            )

            score = scores.mean()
            individual.fitness = fsw * (score) + faw * ((individual.chromosome.count(0) / len(individual.chromosome)))

    elif estimator == "knn":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            preprocessador = build_preprocessor(selected_cols)

            knn_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('scaler', StandardScaler(with_mean=False)),
                ('regressor', KNeighborsRegressor(n_jobs=1))
            ])

            scores = cross_val_score(
                knn_pipeline,
                X_train,
                y_train,
                cv=5,
                n_jobs = -1
            )

            score = scores.mean()
            individual.fitness = fsw * (score) + faw * ((individual.chromosome.count(0) / len(individual.chromosome)))

    elif estimator == "dt":
        for individual in population:

            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            preprocessador = build_preprocessor(selected_cols)

            dt_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('regressor', DecisionTreeRegressor())
            ])

            scores = cross_val_score(
                dt_pipeline,
                X_train,
                y_train,
                cv=5,
                n_jobs = -1
            )

            score = scores.mean()
            individual.fitness = fsw * (score) + faw * ((individual.chromosome.count(0) / len(individual.chromosome)))

    elif estimator == "rf":
        for individual in population:
            selected_cols = chromosome_to_columns(
                individual.chromosome,
                base_abc_X.columns
            )

            preprocessador = build_preprocessor(selected_cols)
            rf_pipeline = Pipeline(steps=[
                ('selector', FeatureSelector(selected_cols)),
                ('preprocessamento', preprocessador),
                ('regressor', RandomForestRegressor(
                                n_estimators=30,
                                max_depth=10,
                                min_samples_leaf=5,
                                n_jobs=1,
                                random_state=42
                            ))
            ])

            scores = cross_val_score(
                rf_pipeline,
                X_train,
                y_train,
                cv=3,
                n_jobs = -1
            )

            score = scores.mean()
            individual.fitness = fsw * (score) + faw * ((individual.chromosome.count(0) / len(individual.chromosome)))

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
        
def print_selected_attributes(individual, X_train):
    individual = np.asarray(individual).ravel().astype(bool)
    nomes_variaveis = X_train.columns[individual == 1]

    assert individual.ndim == 1
    assert len(individual) == X_train.shape[1]

    print("\n\nAtributos: ", len(nomes_variaveis), "\n\n")
    for nome in nomes_variaveis:
        print(nome)

#================MLP====================
def mlp_ag(n, n_population, mlp_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw):
    initialize_population(n, n_population, mlp_population)
    calculate_fitness(mlp_population, "mlp", fsw, faw)
    mlp_population = sorted(mlp_population, key=lambda p: p.fitness)
    apply_probability(mlp_population, s)

    while generations < max_generations:
            new_population = []
            
            for i in range(elitism_number):
                new_population.append(
                    Individual(mlp_population[-1 - i].chromosome)
                )


            for i in range(int((n_population - elitism_number) / 2)):
                p1 = select_parent(mlp_population)
                p2 = select_parent(mlp_population)
                f1, f2 = generate_offspring(p1, p2, crossing_probability)
                apply_mutation(f1, mutation_probability)
                apply_mutation(f2, mutation_probability)
                new_population.append(f1)
                new_population.append(f2)
            mlp_population = new_population
            calculate_fitness(mlp_population, "mlp", fsw, faw)
            mlp_population = sorted(mlp_population, key=lambda p: p.fitness)
            apply_probability(mlp_population, s)
            generations += 1
            
    mlp_best = mlp_population[len(mlp_population) - 1]
    mlp_selected_cols = chromosome_to_columns(
                    mlp_best.chromosome,
                    base_abc_X.columns
                )

    mlp_preprocessador = build_preprocessor(mlp_selected_cols)

    mlp_best__pipeline = Pipeline(steps=[
                    ('selector', FeatureSelector(mlp_selected_cols)),
                    ('preprocessamento', mlp_preprocessador),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('regressor', MLPRegressor())
                ])

    mlp_best__pipeline.fit(X_train, y_train)

    mlp_best_pipeline_score = mlp_best__pipeline.score(X_test, y_test)

    print("\nMLP Attributes: \n")
    print_selected_attributes(mlp_best.chromosome, X_train)

    print("\nScore: ", mlp_best_pipeline_score)

#===================KNN====================
def knn_ag(n, n_population, knn_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw):
    initialize_population(n, n_population, knn_population)
    calculate_fitness(knn_population, "knn", fsw, faw)
    knn_population = sorted(knn_population, key=lambda p: p.fitness)
    apply_probability(knn_population, s)

    while generations < max_generations:
            print("Generation: ", generations + 1)
            new_population = []

            #print(knn_population[len(knn_population) - 1].chromosome)
            
            for i in range(elitism_number):
                new_population.append(
                    Individual(knn_population[-1 - i].chromosome)
                )


            for i in range(int((n_population - elitism_number) / 2)):
                p1 = select_parent(knn_population)
                p2 = select_parent(knn_population)
                f1, f2 = generate_offspring(p1, p2, crossing_probability)
                apply_mutation(f1, mutation_probability)
                apply_mutation(f2, mutation_probability)
                new_population.append(f1)
                new_population.append(f2)
            knn_population = new_population
            calculate_fitness(knn_population, "knn", fsw, faw)
            knn_population = sorted(knn_population, key=lambda p: p.fitness)
            apply_probability(knn_population, s)
            generations += 1
            
    knn_best = knn_population[len(knn_population) - 1]
    knn_selected_cols = chromosome_to_columns(
                knn_best.chromosome,
                base_abc_X.columns
            )

    knn_preprocessador = build_preprocessor(knn_selected_cols)

    knn_best__pipeline = Pipeline(steps=[
                    ('selector', FeatureSelector(knn_selected_cols)),
                    ('preprocessamento', knn_preprocessador),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('regressor', KNeighborsRegressor(n_jobs= -1))
                ])

    knn_best__pipeline.fit(X_train, y_train)

    knn_best_pipeline_score = knn_best__pipeline.score(X_test, y_test)

    print("\nKNN Attributes: \n")
    print_selected_attributes(knn_best.chromosome, X_train)

    print("\nScore: ", knn_best_pipeline_score, "\n\n")

#=============DT=================
def dt_ag(n, n_population, dt_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw):
    initialize_population(n, n_population, dt_population)
    calculate_fitness(dt_population, "dt", fsw, faw)
    dt_population = sorted(dt_population, key=lambda p: p.fitness)
    apply_probability(dt_population, s)

    while generations < max_generations:
            print("Generation: ", generations + 1)
            new_population = []

            #print(dt_population[len(knn_population) - 1].chromosome)
            
            for i in range(elitism_number):
                new_population.append(
                    Individual(dt_population[-1 - i].chromosome)
                )


            for i in range(int((n_population - elitism_number) / 2)):
                p1 = select_parent(dt_population)
                p2 = select_parent(dt_population)
                f1, f2 = generate_offspring(p1, p2, crossing_probability)
                apply_mutation(f1, mutation_probability)
                apply_mutation(f2, mutation_probability)
                new_population.append(f1)
                new_population.append(f2)
            dt_population = new_population
            calculate_fitness(dt_population, "dt", fsw, faw)
            dt_population = sorted(dt_population, key=lambda p: p.fitness)
            apply_probability(dt_population, s)
            generations += 1
            
    dt_best = dt_population[len(dt_population) - 1]
    dt_selected_cols = chromosome_to_columns(
                dt_best.chromosome,
                base_abc_X.columns
            )

    dt_preprocessador = build_preprocessor(dt_selected_cols)

    dt_best__pipeline = Pipeline(steps=[
                    ('selector', FeatureSelector(dt_selected_cols)),
                    ('preprocessamento', dt_preprocessador),
                    ('regressor', DecisionTreeRegressor())
                ])

    dt_best__pipeline.fit(X_train, y_train)

    dt_best_pipeline_score = dt_best__pipeline.score(X_test, y_test)

    print("\nDT Attributes: \n")
    print_selected_attributes(dt_best.chromosome, X_train)

    print("\nScore: ", dt_best_pipeline_score)

#===================RF====================
def rf_ag(n, n_population, rf_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw):
    initialize_population(n, n_population, rf_population)
    calculate_fitness(rf_population, "rf", fsw, faw)
    rf_population = sorted(rf_population, key=lambda p: p.fitness)
    apply_probability(rf_population, s)

    while generations < max_generations:
            print("Generation: ", generations + 1)
            new_population = []

            #print(knn_population[len(knn_population) - 1].chromosome)
            
            for i in range(elitism_number):
                new_population.append(
                    Individual(rf_population[-1 - i].chromosome)
                )


            for i in range(int((n_population - elitism_number) / 2)):
                p1 = select_parent(rf_population)
                p2 = select_parent(rf_population)
                f1, f2 = generate_offspring(p1, p2, crossing_probability)
                apply_mutation(f1, mutation_probability)
                apply_mutation(f2, mutation_probability)
                new_population.append(f1)
                new_population.append(f2)
            rf_population = new_population
            calculate_fitness(rf_population, "rf", fsw, faw)
            rf_population = sorted(rf_population, key=lambda p: p.fitness)
            apply_probability(rf_population, s)
            generations += 1
            
    rf_best = rf_population[len(rf_population) - 1]
    rf_selected_cols = chromosome_to_columns(
                rf_best.chromosome,
                base_abc_X.columns
            )

    rf_preprocessador = build_preprocessor(rf_selected_cols)

    rf_best__pipeline = Pipeline(steps=[
                    ('selector', FeatureSelector(rf_selected_cols)),
                    ('preprocessamento', rf_preprocessador),
                    ('regressor', RandomForestRegressor(n_jobs= -1))
                ])

    rf_best__pipeline.fit(X_train, y_train)

    rf_best_pipeline_score = rf_best__pipeline.score(X_test, y_test)

    print("\nRF Attributes: \n")
    print_selected_attributes(rf_best.chromosome, X_train)

    print("\nScore: ", rf_best_pipeline_score)



mlp_population = []
knn_population = []
dt_population = []
rf_population = []
new_population = []
n = len(base_abc_X.columns)
n_population = 100
crossing_probability = 80
elitism_number = 10
mutation_probability = 3
s = 1.7
max_generations = 100
generations = 0

#fitness score weight and fitness attribute weight
fsw = 0.9
faw = 0.1

#dt_ag(n, n_population, dt_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw)
#rf_ag(n, n_population, rf_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw)
mlp_ag(n, n_population, mlp_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw)
#knn_ag(n, n_population, knn_population, new_population, generations, max_generations, elitism_number, crossing_probability, mutation_probability, s, fsw, faw)
