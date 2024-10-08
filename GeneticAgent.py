import random
from RewardSystem import RewardSystem


class GeneticAgent():
    """ Class that represents a genetic algorithm agent. """
    def __init__(self, population_size=10, mutation_rate=0.1, crossover_rate=0.5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 1
        self.population = self.initialize_population()
        self.current_weights_index = 0
        self.current_weights = self.population[self.current_weights_index][0]
        self.rewardSystem = RewardSystem()
        self.total_isolation_score = 0
        self.item_game = 0
        self.total_item_fitness = 0

    def initialize_population(self):
        """ initialize the first population with random values. """
        population = []
        for i in range(self.population_size):
            weights = {
                'aggregate_height': random.uniform(0, 1),
                'complete_lines': random.uniform(0, 5),
                'holes': random.uniform(0, 2),
                'bumpiness': random.uniform(0, 1),
                'highest_point': random.uniform(0, 1),
                'neighbours': random.uniform(0, 2)
            }
            population.append([weights, 0])
        return population

    def choose_best_final_state(self, possible_final_states):
        """ Chooses the best final state based on predicted values or randomly (epsilon-greedy). """
        best_state = None
        best_score = float('-inf')
        for state in possible_final_states:
            score = self.rewardSystem.calculate_reward(state, weights=self.current_weights)
            if score > best_score or best_state is None:
                best_score = score
                best_state = state
        return best_state

    def calculate_fitness(self, score):
        """ calculate the reward for the final grid state. """
        return score

    def evolve_population(self):
        """ Evolve the population by selecting the best individuals and creating new ones. """
        self.generation += 1
        ranked_population = sorted(self.population, key=lambda x: x[1], reverse=True)
        next_population = ranked_population[:3]
        ranked_population = ranked_population[:7]
        while len(next_population) < self.population_size:
            parent1 = self.selection(ranked_population)[0]
            parent2 = self.selection(ranked_population)[0]
            child = self.crossover(parent1, parent2) if random.random() < self.crossover_rate else parent1
            child = self.mutate(child)
            next_population.append([child, 0])
        self.population = next_population

    def selection(self, ranked_population):
        """ Select the best individuals from the population using tournament selection. """
        tournament_size = 3
        selected = random.sample(ranked_population, tournament_size)
        return max(selected, key=lambda x: x[1])

    def crossover(self, parent1, parent2):
        """ Crossover two parents to create a child. """
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def mutate(self, weights):
        """ Mutate the weights of the child. """
        for weight in weights:
            if random.random() < self.mutation_rate:
                weights[weight] *= random.uniform(0.8, 1.2)
        return weights

    def train(self, score):
        """ Train the agent with the final grid state. """
        if self.item_game == 2:
            avg_fitness = (self.calculate_fitness(score) + self.total_item_fitness) / 3
            self.population[self.current_weights_index][1] = avg_fitness
            self.total_item_fitness = 0
            self.item_game = 0
            if self.current_weights_index < len(self.population) - 1:
                self.current_weights_index += 1
            else:
                self.evolve_population()
                self.current_weights_index = 0
            self.current_weights = self.population[self.current_weights_index][0]
        else:
            self.total_item_fitness += self.calculate_fitness(score)
            self.item_game += 1
