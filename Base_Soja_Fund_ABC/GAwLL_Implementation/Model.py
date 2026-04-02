from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from FeatureSelector import FeatureSelector
from Individual import Individual

from PopulationOperators import PopulationOperators

class Model:
    def __init__(self, dbConfig, population, model, config, data):
        self.dbConfig = dbConfig
        self.population = population
        self.model = model
        self.__dict__.update(config)
        self.__dict__.update(data)

        self.population_op = PopulationOperators()

    def start(self):
        self.population_op.initialize_population(self.n, self.n_population, self.population)
        
    def chromosome_to_columns(self, chromosome, all_columns):
        return [
            col for col, bit in zip(all_columns, chromosome)
            if bit == 1
        ]
    
    def build_preprocessor(self, selected_columns):

        num_cols = [c for c in self.dbConfig.getNumericas() if c in selected_columns]
        cat_cols = [c for c in self.dbConfig.getCategoricas() if c in selected_columns]

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

    def calculate_fitness(self):
        PIPELINES = {
            "mlp": (self.create_mlp_pipeline, 5),
            "knn": (self.create_knn_pipeline, 5),
            "dt": (self.create_dt_pipeline, 5),
            "rf": (self.create_rf_pipeline, 3),
}
        create_pipeline, cv = PIPELINES[self.model]

        for individual in self.population:

            selected_cols = self.chromosome_to_columns(
                individual.chromosome,
                self.base_abc_X.columns
            )

            preprocessador = self.build_preprocessor(selected_cols)

            pipeline = create_pipeline(selected_cols, preprocessador)

            scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=5,
                n_jobs = -1
            )

            score = scores.mean()
            individual.fitness = self.fsw * (score) + self.faw * ((individual.chromosome.count(0) / len(individual.chromosome)))


    def create_mlp_pipeline(self, selected_cols, preprocessador):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('preprocessamento', preprocessador),
            ('scaler', StandardScaler(with_mean=False)),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=(32,),
                max_iter=150,
                early_stopping=True,
            ))
        ])
    
    def create_knn_pipeline(self, selected_cols, preprocessador):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('preprocessamento', preprocessador),
            ('scaler', StandardScaler(with_mean=False)),
            ('regressor', KNeighborsRegressor(n_jobs=1))
        ])
    
    def create_dt_pipeline(self, selected_cols, preprocessador):
        return Pipeline([
            ('selector', FeatureSelector(selected_cols)),
            ('preprocessamento', preprocessador),
            ('regressor', DecisionTreeRegressor())
        ])
    
    def create_rf_pipeline(self, selected_cols, preprocessador):
        return Pipeline([
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

        # ================= LOOP EVOLUTIVO =================
        while self.generations < self.max_generations:
            print("Generation:", self.generations + 1)

            new_population = []

            # 🔹 Elitismo
            for i in range(self.elitism_number):
                new_population.append(
                    Individual(self.population[-1 - i].chromosome)
                )

            # 🔹 Reprodução
            for _ in range(int((self.n_population - self.elitism_number) / 2)):
                p1 = self.population_op.select_parent(self.population)
                p2 = self.population_op.select_parent(self.population)

                f1, f2 = self.population_op.generate_offspring(
                    p1, p2, self.crossing_prob
                )

                self.population_op.apply_mutation(f1, self.mutation_prob)
                self.population_op.apply_mutation(f2, self.mutation_prob)

                new_population.append(f1)
                new_population.append(f2)

            self.population = new_population

            self.calculate_fitness()
            self.population = sorted(self.population, key=lambda p: p.fitness)
            self.population_op.apply_probability(self.population, self.s)

            self.generations += 1

        # ================= MELHOR INDIVÍDUO =================
        best = self.population[-1]

        selected_cols = self.chromosome_to_columns(
            best.chromosome,
            self.dbConfig.getDatabase().columns
        )

        preprocessador = self.build_preprocessor(selected_cols)

        PIPELINES = {
            "mlp": self.create_mlp_pipeline,
            "knn": self.create_knn_pipeline,
            "dt": self.create_dt_pipeline,
            "rf": self.create_rf_pipeline,
        }

        pipeline = PIPELINES[self.model](selected_cols, preprocessador)

        # ================= TREINO FINAL =================
        pipeline.fit(self.X_train, self.y_train)
        score = pipeline.score(self.X_test, self.y_test)

        # ================= RESULTADOS =================
        print(f"\n{self.model.upper()} Attributes:\n")
        #print_selected_attributes(best.chromosome, self.X_train)

        print("\nScore:", score, "\n")

        return best, score, pipeline