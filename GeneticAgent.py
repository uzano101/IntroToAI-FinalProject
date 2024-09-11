import random
from RewardSystem import RewardSystem


class GeneticAgent():

    # constructor
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 1
        self.population = self.initialize_population()
        self.current_weights_index = 0
        self.current_weights = self.population[self.current_weights_index][0]
        self.rewardSystem = RewardSystem()
        self.total_isolation_score = 0

    def initialize_population(self):
        """
        initialize the first population with random values.
        :return: the first population
        """
        # TODO: change base on our reward, and the size we want to give it to.
        population = []
        for i in range(self.population_size):
            weights = {
                'aggregate_height': random.uniform(0, 1),
                'complete_lines': random.uniform(0, 5),
                'holes': random.uniform(0, 2),
                'bumpiness': random.uniform(0, 1),
                'highest_point': random.uniform(0, 1)
            }

            population.append([weights, 0])
        return population

    def choose_best_final_state(self, possible_final_states):
        # For each possible state, calculate its fitness based on the current weights
        best_state = None
        best_score = float('-inf')

        for state in possible_final_states:
            score = self.rewardSystem.calculate_reward(state, weights=self.current_weights)
            if score > best_score or best_state is None:
                best_score = score
                best_state = state

        return best_state

    def calculate_fitness(self, score, cleared_lines, level):
        """
        calculate the reward for the final grid state.
        :param score: the final score of the game.
        :param cleared_lines: number of lines cleared in the game.
        :param level: the level achieved in the game.
        :return: the reward for the game.
        """
        # Include the accumulated isolation score in the fitness calculation
        return score

    def evolve_population(self):
        self.generation += 1
        # Sort population based on fitness
        ranked_population = sorted(self.population, key=lambda x: x[1], reverse=True)

        # Select parents for reproduction, select the best two of them.
        next_population = ranked_population[:4]

        # Generate new population through crossover and mutation
        while len(next_population) < self.population_size:
            parent1 = self.selection(ranked_population)[0]
            parent2 = self.selection(ranked_population)[0]

            # Crossover to create a child
            child = self.crossover(parent1, parent2) if random.random() < self.crossover_rate else parent1

            # Apply mutation to the child
            child = self.mutate(child)

            next_population.append([child, 0])

        self.population = next_population

    def selection(self, ranked_population):
        # Tournament selection
        tournament_size = 3
        selected = random.sample(ranked_population, tournament_size)
        return max(selected, key=lambda x: x[1])

    def crossover(self, parent1, parent2):
        # Combine two sets of weights to produce a new set (crossover)
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def mutate(self, weights):
        for weight in weights:
            if random.random() < self.mutation_rate:
                # Apply small changes to make the mutation.
                weights[weight] *= random.uniform(0.8, 1.2)
        return weights

    def update_agent(self, state, reward, next_state, done):
        pass

    def train(self, score, cleared_lines, level):
        # TODO: think of a better way, no need to implement here, add score in the reward function.

        # Calculate fitness using the isolation score
        self.population[self.current_weights_index][1] = self.calculate_fitness(score, cleared_lines, level)

        if self.current_weights_index < len(self.population) - 1:
            self.current_weights_index += 1
        else:
            self.evolve_population()
            self.current_weights_index = 0

        self.current_weights = self.population[self.current_weights_index][0]
        self.total_isolation_score = 0  # Reset total isolation score for the next game

