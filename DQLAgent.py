import numpy as np
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# from BaseAgent import BaseAgent
from RewardSystem import RewardSystem


class DQLAgent():
    def __init__(self, state_size=209, num_final_states=1):
        self.state_size = state_size
        self.output = num_final_states
        self.gamma = 0.95
        self.Qvalue = deque(maxlen=10000)
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model(num_final_states)
        self.reward_system = RewardSystem()

    def build_model(self, output_size):
        model = Sequential()

        model.add(Dense(32, activation='relu'))
        for i in range(1, 3):
            model.add(Dense(32, activation='relu'))

        model.add(Dense(1, activation='linear'))  # Output size matches the number of final states

        model.compile(loss='mse', optimizer='adam')
        return model

    def update_agent(self, state, next_state, done, score):
        reward = self.reward_system.calculate_reward(next_state.grid)
        self.Qvalue.append((state, next_state, reward, done))

    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state, verbose=0)[0]

    def calculate_fitness(self, state, cleared_lines=None):
        """
        calculate the reward for the final grid state.
        :param grid: the last "picture" of the grid when the agent lost.
        :return: the reward for the grid.
        """
        return self.reward_system.calculate_reward(state.grid, cleared_lines)

    def choose_best_final_state(self, current_state, possible_final_states):
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(possible_final_states))
        else:
            for state in possible_final_states:
                # Get grid features only
                states_vector = self.convarte_state_to_vector(state)

                # Ensure the input is of shape (1, 4) for the neural network
                states_vector = states_vector.reshape(1, -1)

                value = self.predict_value(states_vector)

                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=128, epochs=1):
        '''Trains the agent using experience replay with batch updates'''

        # Ensure the batch size does not exceed the memory size
        if batch_size > len(self.Qvalue):
            print('WARNING: batch size is bigger than memory size. The agent will not be trained.')
            return

        # Proceed only if there is enough memory to start training
        if len(self.Qvalue) < batch_size:
            return

        # Sample a batch of experiences from memory
        batch = random.sample(self.Qvalue, batch_size)

        # Extract next states for batch prediction to optimize performance
        next_states = np.array([self.convarte_state_to_vector(x[1]) for x in batch])

        # Ensure next_states has the correct number of elements
        next_states = next_states.reshape(batch_size, 4)  # Shape should match (batch_size, 4)

        # Predict the Q-values for the next states in batch
        next_qs = self.model.predict(next_states)

        # Prepare the input (x) and target (y) for training
        x = []
        y = []

        # Build the x and y structures for batch fitting
        for i, (state, next_state, reward, done) in enumerate(batch):
            # Convert the current state to vector format
            state_vector = self.convarte_state_to_vector(state).reshape(1, 4)

            # Predict the current Q-values using the model
            target_qs = self.model.predict(state_vector)[0]

            # Calculate the target Q-value using the Bellman equation
            if not done:
                # Update Q-value based on future reward
                new_q = reward + self.gamma * np.max(next_qs[i])
            else:
                # If done, the future reward is 0
                new_q = reward

            # Update the Q-value for the action taken
            target_qs[0] = new_q  # Assuming single output Q-value

            # Append the input state and the updated target Q-values
            x.append(state_vector.flatten())
            y.append(target_qs)

        # Convert x and y to numpy arrays for batch training
        x = np.array(x)
        y = np.array(y)

        # Fit the model using the input states and updated target Q-values
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

        # Update epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def encode_grid(self, grid):
        """
        Converts the Tetris grid into a binary matrix. Each cell is represented as 0 if it's empty and 1 if it's occupied.
        This representation simplifies the understanding of the grid's occupancy state for the neural network.

        Returns:
        - A flattened 1D numpy array representing the current state of the grid.
        """
        return np.array([[1 if cell != 0 else 0 for cell in row] for row in grid]).flatten()

    def encode_tetrimino(self, tetrimino_grid, x_pos, y_pos, shape, grid_width, grid_height):
        """
        Encodes the Tetrimino's grid and position into a normalized form suitable for neural network input.

        Parameters:
        - tetrimino_grid: 2D list representing the Tetrimino's shape on the grid
        - position: Tuple (x, y) representing the Tetrimino's position on the game grid
        - grid_width: Width of the game grid
        - grid_height: Height of the game grid

        Returns:
        - A numpy array containing the normalized position and flattened grid shape of the Tetrimino.
        """
        normalized_x = x_pos / grid_width
        normalized_y = y_pos / grid_height

        # Flatten the Tetrimino grid
        flat_tetrimino = np.array(tetrimino_grid).flatten()
        if len(flat_tetrimino) < 6:
            flat_tetrimino = np.concatenate((flat_tetrimino, np.zeros(6 - len(flat_tetrimino))))

        # Combine normalized position and flattened Tetrimino grid into a single array
        encoded_tetrimino = np.concatenate(([normalized_x, normalized_y, shape], flat_tetrimino))

        return encoded_tetrimino

    def convarte_state_to_vector(self, state, include_grid_features=False):
        """
        Computes and returns only the grid features as a state vector.

        Parameters:
        - state: The state to be converted.
        - include_grid_features: Parameter is retained for compatibility but not used.

        Returns:
        - A numpy array containing only the grid features.
        """
        grid = state.grid

        # Calculate grid features
        grid_features = [
            self.reward_system.calculate_holes(grid),  # 1 feature
            self.reward_system.calculate_bumpiness(grid),  # 1 feature
            self.reward_system.calculate_aggregate_height(grid),  # 1 feature
            self.reward_system.calculate_highest_point(grid)  # 1 feature
        ]

        # Return only the grid features
        return np.array(grid_features)  # Total size: 4



