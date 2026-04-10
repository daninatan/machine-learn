from DatabaseConfig import DatabaseConfig
from Model import Model
from multiprocessing import Process, Manager
from interface import ModelSelectorUI
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from EVig import eVIG
from Importance import Importance

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_model(model_name, config_ga, data, dbConfig, return_dict):

    population = []
    evig = eVIG(len(data["X_train"].columns))
    importance = Importance(len(data["X_train"].columns))

    model = Model(
        dbConfig=dbConfig,
        population=population,
        model=model_name,
        config=config_ga,
        data=data,
        evig=evig,
        importance=importance,
    )

    best, score, pipeline, importance, evig = model.run_ag()
    return_dict[model_name] = (best.chromosome, score, importance, evig)


def print_selected_attributes(individual, X_train):
    individual = np.asarray(individual).ravel().astype(bool)
    nomes_variaveis = X_train.columns[individual == 1]

    print("\nAtributos:", len(nomes_variaveis), "\n")
    for nome in nomes_variaveis:
        print(nome)

def print_importance_sorted(importance_values, feature_names):
    print("\n=== IMPORTÂNCIA (ORDENADA) ===\n")

    pairs = list(zip(feature_names, importance_values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, value in pairs:
        print(f"{name}: {value:.6f}")

def print_top_interactions(interaction_matrix, feature_names, top_n=10):
    print(f"\n=== TOP {top_n} INTERAÇÕES ===\n")

    interactions = []
    n = len(feature_names)

    for i in range(n):
        for j in range(i + 1, n):
            value = interaction_matrix[i][j]
            interactions.append((feature_names[i], feature_names[j], value))

    interactions.sort(key=lambda x: abs(x[2]), reverse=True)

    for f1, f2, value in interactions[:top_n]:
        print(f"{f1} <-> {f2}: {value:.6f}")




#ANTES DE RODAR CHECAR TODAS AS VARIAVEIS TANTO DA MAIN QUANTO DO MODEL.PY, ALGUMAS SOFRERAM MUDANÇA PARA TESTE
if __name__ == "__main__":
    # ================= INTERFACE =================
    ui = ModelSelectorUI()
    models = ui.run()

    if not models:
        print("Nenhum modelo selecionado.")
        exit()

    # ================= DATABASE =================
    dbConfig = DatabaseConfig("/home/daniel/dev/ic/Base_Soja_Fund_ABC/Base_ABC_Flor-Colheita.csv")
    X_train, X_test, y_train, y_test, base_abc_X = dbConfig.configDatabase()

    # ================= CONFIG GA =================

    config_ga = {
        "n": len(X_train.columns),
        "n_population": 50,
        "generations": 0,
        "max_generations": 50,
        "elitism_number": 10,
        "crossing_prob": 80,
        "mutation_prob": 3,
        "s": 1.7,
        "fsw": 0.9,
        "faw": 0.1
    }

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "base_abc_X": base_abc_X
    }

    # ================= MULTIPROCESS =================
    manager = Manager()
    return_dict = manager.dict()
    processes = []

    for m in models:
        p = Process(
            target=run_model,
            args=(m, config_ga, data, dbConfig, return_dict)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # ================= RESULTADOS =================
    print("\n======= RESULTADOS =======")

    for m in models:
        if m not in return_dict:
            print(f"{m} falhou.")
            continue

        best, score, importance, evig = return_dict[m]
        max_importance, min_importance = importance.export_importance_vectors()
        max_interaction, min_interaction = evig.export_interaction_matrix()

        print(f"\n{m.upper()}:\n")
        print_importance_sorted(max_importance, list(X_train.columns))
        print_top_interactions(max_interaction, list(X_train.columns), 20)
        print_selected_attributes(best, X_train)
        print("\nScore:", score, "\n\n")