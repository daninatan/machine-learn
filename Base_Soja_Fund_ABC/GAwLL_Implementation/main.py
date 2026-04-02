from FeatureSelector import FeatureSelector
from DatabaseConfig import DatabaseConfig
from EVig import eVIG
from Individual import Individual
from PopulationOperators import PopulationOperators
from Model import Model

dbConfig = DatabaseConfig("/home/daniel/dev/ic/Base_Soja_Fund_ABC/Base_ABC_Flor-Colheita.csv")
X_train, X_test, y_train, y_test, base_abc_X = dbConfig.configDatabase()

#mlp_population = []
#knn_population = []
#dt_population = []
#rf_population = []
population = []
new_population = []
n = len(X_train.columns)
n_population = 100
crossing_probability = 80
elitism_number = 10
mutation_probability = 3
s = 1.7
max_generations = 5
generations = 0
model = "knn"

#fitness score weight and fitness attribute weight
fsw = 0.9
faw = 0.1

config_ga = {
    "n": n,
    "n_population": n_population,
    "generations": generations,
    "max_generations": max_generations,
    "elitism_number": elitism_number,
    "crossing_prob": crossing_probability,
    "mutation_prob": mutation_probability,
    "s": s,
    "fsw": fsw,
    "faw": faw
}

data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "base_abc_X": base_abc_X
}

model = Model(
    dbConfig=dbConfig,
    population=population,
    model=model,
    config=config_ga,
    data=data
)

best, score, pipeline = model.run_ag()
