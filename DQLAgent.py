import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import BaseAgentRL

GRID_WIDTH = 10  # Number of columns
GRID_HEIGHT = 20  # Number of rows

class DQLAgent(BaseAgentRL):
    def __init__(self, state_size, action_size):
        """
        Initializes the DQLAgent with the necessary parameters and settings for neural network training.
        Creates a neural network model that predicts the best action based on the game state.

        Parameters:
        - state_size: The size of the input state vector.
        - action_size: The number of possible actions the agent can take.
        """
        super().__init__(state_size, action_size)
        self.Qvalue = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        """
        Builds a neural network using Keras to approximate the Q-value function.
        The network outputs a prediction of the expected rewards for each possible action given a state.

        Returns:
        - The compiled neural network model.
        """
        model = Sequential([
            Dense(150, input_dim=self.state_size, activation='relu'),
            Dense(120, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def choose_action(self, state):
        """
        Selects the best action to take based on the current state, using an epsilon-greedy policy.
        With probability epsilon, a random action is chosen (exploration), and with probability 1-epsilon,
        the action with the highest predicted reward is chosen (exploitation).

        Parameters:
        - state: The current state vector of the game.

        Returns:
        - The index of the action to take.
        """
        # TODO: check possible actions.
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def update_agent(self, state, action, reward, next_state, done):
        self.Qvalue.append((state, action, reward, next_state, done))
        self.train()

    def train(self, batch_size=32):
        """
        Trains the neural network on a batch of experiences sampled from the memory.
        Updates the network to better predict the Q-values using the Bellman equation.

        Parameters:
        - batch_size: The number of experiences to sample from the memory.
        """
        minibatch = random.sample(self.Qvalue, min(len(self.Qvalue), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.max(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def convert_state_to_vector(self,state):

        def encode_grid(self, grid):
            """
            Converts the Tetris grid into a binary matrix. Each cell is represented as 0 if it's empty and 1 if it's occupied.
            This representation simplifies the understanding of the grid's occupancy state for the neural network.

            Returns:
            - A flattened 1D numpy array representing the current state of the grid.
            """
            return np.array([[1 if cell != 0 else 0 for cell in row] for row in grid]).flatten()

        def encode_tetrimino(tetrimino_grid, x_pos, y_pos,rotation, grid_width, grid_height):
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

            # Combine normalized position and flattened Tetrimino grid into a single array
            encoded_tetrimino = np.concatenate(([normalized_x, normalized_y, rotation], flat_tetrimino))

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
            x_pos = state.tetrimino_x
            y_pos = state.tetrimino_y
            rotation = state.tetrimino_rotation
            grid_state = self.encode_grid(grid)
            current_tetrimino_state = self.encode_tetrimino(tetrimino["matrix"],x_pos,y_pos,rotation,len(grid),len(grid[0]))
            next_tetrimino_state = self.encode_tetrimino(self.game.next_tetrimino)
            return np.concatenate([grid_state, current_tetrimino_state, next_tetrimino_state])
