from Individual import Individual
import random

class PopulationOperators:

        def initialize_population(self, n, n_population, population):
            for i in range(n_population):
                bits = []
                for j in range(n):
                    number = random.randint(0, 1)
                    bits.append(number)
                individual = Individual(bits)
                population.append(individual)

        def generate_offspring(self, p1, p2, crossing_probability):
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
        
        def apply_probability(self, population, s):
            size = len(population)
            for i in range(size):
                population[i].prob = ((2 - s) / size) + ((2 * i * (s - 1)) / (size * (size - 1)))

        def apply_mutation(self, f, mutation_probability):
            for i in range(0, len(f.chromosome)):
                mutation = random.randint(0, 100)
                if(mutation <= mutation_probability):
                    f.chromosome[i] = f.chromosome[i] ^ 1

        def select_parent(self, population):
            r = random.random()  
            cumulative = 0.0

            for individual in population:
                cumulative += individual.prob
                if r <= cumulative:
                    return individual