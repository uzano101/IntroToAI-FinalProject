import numpy as np
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from BaseAgentRL import BaseAgentRL


class DQLAgent(BaseAgentRL):
    def __init__(self, state_size=209, num_final_states=1):
        super().__init__(state_size, num_final_states)
        self.gamma = 0.95
        self.Qvalue = deque(maxlen=10000)
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.model = self.build_model(num_final_states)

    def build_model(self, output_size):
        model = Sequential([
            Dense(150, activation='relu'),
            Dense(120, activation='relu'),
            Dense(output_size, activation='linear')  # Output size matches the number of final states
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    def update_agent(self, state, next_state,reward , done):
        self.Qvalue.append((state, next_state, reward, done))

    def choose_best_final_state(self, current_state, possible_final_states):
        if np.random.rand() <= self.epsilon:
            return possible_final_states[np.random.randint(len(possible_final_states))]
        else:
            # Convert current state to vector
            state_vector = self.convarte_state_to_vector(current_state)
            # Append each possible final state to the current state for prediction
            inputs = np.array(
                [np.concatenate([state_vector, self.convarte_state_to_vector(final_state)]) for final_state in
                 possible_final_states])
            # Predict the values of all possible final states
            state_values = self.model.predict(inputs)
            # Select the best final state based on predicted values
            best_index = np.argmax(state_values)
            return possible_final_states[best_index]

    def train(self, batch_size=20):
        minibatch = random.sample(self.Qvalue, min(len(self.Qvalue), batch_size))
        for state, final_state, reward, done in minibatch:
            input_vector = np.concatenate(
                [self.convarte_state_to_vector(state), self.convarte_state_to_vector(final_state)])
            target = reward if done else reward + self.gamma * np.max(
                self.model.predict(input_vector.reshape(1, -1))[0])
            target_f = self.model.predict(input_vector.reshape(1, -1))
            target_f[0][0] = target  # Only one output, the value of the final state
            self.model.fit(input_vector.reshape(1, -1), target_f, epochs=1, verbose=0)
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

    def convarte_state_to_vector(self, state):
        """
        Combines the grid state, current tetrimino state, and next tetrimino state into a single state vector.
        This comprehensive state vector is then used by the DQL agent to make decisions.

        Returns:
        - A concatenated numpy array of the grid state, current tetrimino state, and next tetrimino state.
        """
        grid = state.grid
        tetrimino = state.current_tetrimino
        tetrimino_x_pos = tetrimino["x"]
        tetrimino_y_pos = tetrimino["y"]
        tetrimino_shape = ord(tetrimino["shape"])
        grid_state = self.encode_grid(grid)
        current_tetrimino_state = self.encode_tetrimino(tetrimino["matrix"], tetrimino_x_pos, tetrimino_y_pos,
                                                        tetrimino_shape, len(grid), len(grid[0]))
        return np.concatenate([grid_state, current_tetrimino_state])