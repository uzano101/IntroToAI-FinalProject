import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import BaseAgentRL


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
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def update_agent(self, state, action, reward, next_state, done):
        self.Qvalue.append((state, action, reward, next_state, done))
        self.train()
        self.epsilon *= self.epsilon_decay

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
