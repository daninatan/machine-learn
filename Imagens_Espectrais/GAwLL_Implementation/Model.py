from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from FeatureSelector import FeatureSelector
from Individual import Individual
import numpy as np
from LinkageLearning import LinkageLearning

from PopulationOperators import PopulationOperators

class Model:
    def __init__(self, dbConfig, population, model, config, data, evig, importance):
        self.dbConfig = dbConfig
        self.population = population
        self.model = model
        self.__dict__.update(config)
        self.__dict__.update(data)
        self.evig = evig
        self.importance = importance
        self.linkage = LinkageLearning(
            evig=self.evig,
            importance=self.importance,
            eval_perf_batch=self.evaluate_batch
        )

        self.population_op = PopulationOperators()
        
    def chromosome_to_columns(self, chromosome, all_columns):
        return [
            col for col, bit in zip(all_columns, chromosome)
            if bit == 1
        ]

    def evaluate_individual(self, individual : Individual):
        selected_cols = self.chromosome_to_columns(
                individual.chromosome,
                self.X_train.columns
        )
        
        PIPELINES = {
            "mlp": (self.create_mlp_pipeline, 3),
            "knn": (self.create_knn_pipeline, 3),
            "dt": (self.create_dt_pipeline, 3),
            "rf": (self.create_rf_pipeline, 3),
        }

        create_pipeline, cv = PIPELINES[self.model]

        pipeline = create_pipeline(selected_cols)

        scores = cross_val_score(
            pipeline,
            self.X_train,
            self.y_train,
            scoring='accuracy',
            cv=cv,
            n_jobs = 1
        )

        score = scores.mean()
        fitness = self.fsw * (score) + self.faw * ((individual.chromosome.count(0) / len(individual.chromosome)))
        return fitness
    
    def evaluate_batch(self, chromosomes):
        fitness = []
        for chrom in chromosomes:
            ind = Individual(chrom)
            fitness.append(self.evaluate_individual(ind))
        return fitness
            

    def calculate_fitness(self):
        
        for individual in self.population:   
            individual.fitness = self.evaluate_individual(individual)

    def create_mlp_pipeline(self, selected_cols):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=150,
                early_stopping=True,
            ))
        ])
    
    def create_knn_pipeline(self, selected_cols):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', KNeighborsClassifier(n_jobs=1))
        ])
    
    def create_dt_pipeline(self, selected_cols):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('classifier', DecisionTreeClassifier())
        ])
    
    def create_rf_pipeline(self, selected_cols):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('classifier', RandomForestClassifier(
                n_estimators=30,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=42
            ))
        ])
    

    def run_ag(self):
    # ================= INICIALIZAÇÃO =================
        self.population_op.initialize_population(
            self.n,
            self.n_population,
            self.population
        )

        self.calculate_fitness()
        self.population = sorted(self.population, key=lambda p: p.fitness)
        self.population_op.apply_probability(self.population, self.s)

        num_crossover = int(self.n_population * (self.crossing_prob / 100))
        # ================= LOOP EVOLUTIVO =================
        while self.generations < self.max_generations:
            print("(", self.model, ")", " Generation:", self.generations + 1)

            new_population = []

            # 🔹 Elitismo
            for i in range(self.elitism_number):
                new_population.append(
                    Individual(self.population[-1 - i].chromosome.copy())
                )
            
            while len(new_population) < self.n_population:

                

                if len(new_population) < num_crossover:

                    # 🔹 Reprodução
                    p1 = self.population_op.select_parent(self.population)
                    p2 = self.population_op.select_parent(self.population)

                    f1, f2 = self.population_op.generate_offspring(
                        p1, p2
                    )

                    self.population_op.apply_mutation(f1, self.mutation_prob)
                    self.population_op.apply_mutation(f2, self.mutation_prob)

                    new_population.append(f1)
                    new_population.append(f2)
                elif len(new_population) < self.n_population - 2:
                    parent1 = self.population_op.select_parent(self.population)
                    chroms = self.linkage.mutation_ll(parent1)
                    new_population.extend([Individual(c) for c in chroms])
                else:
                    parent1 = self.population_op.select_parent(self.population)
                    child = Individual(parent1.chromosome.copy())
                    self.population_op.apply_mutation(child, self.mutation_prob)
                    new_population.append(child)
                    

            self.generations += 1    
            self.population = new_population
            self.calculate_fitness()
            self.population = sorted(self.population, key=lambda p: p.fitness)
            self.population_op.apply_probability(self.population, self.s)

            

        # ================= MELHOR INDIVÍDUO =================
        best = self.population[-1]

        selected_cols = self.chromosome_to_columns(
            best.chromosome,
            self.X_train.columns
        )

        PIPELINES = {
            "mlp": self.create_mlp_pipeline,
            "knn": self.create_knn_pipeline,
            "dt": self.create_dt_pipeline,
            "rf": self.create_rf_pipeline,
        }

        pipeline = PIPELINES[self.model](selected_cols)

        # ================= TREINO FINAL =================
        pipeline.fit(self.X_train, self.y_train)
        score = pipeline.score(self.X_test, self.y_test)

        return best, score, self.importance, self.evig