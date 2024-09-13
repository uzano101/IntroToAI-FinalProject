import numpy as np
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from RewardSystem import RewardSystem


class DQLAgent:
    def __init__(self, state_size=209, num_final_states=1):
        self.generation = 0
        self.state_size = state_size
        self.output = num_final_states
        self.gamma = 0.95
        self.Qvalue = deque(maxlen=10000)
        self.epsilon = 1
        self.epsilon_min = 0
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.reward_system = RewardSystem()

    def build_model(self):
        """
            Builds and compiles the neural network model.
            Returns:
                Compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Dense(32, activation='relu'))
        for i in range(1, 3):
            model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_agent(self, state, next_state, done):
        """
            Updates agent's experience (state, next_state, reward, done).
        """
        reward = self.reward_system.calculate_reward(state)
        self.Qvalue.append((state, next_state, reward, done))

    def predict_value(self, state):
        """
           Predicts the score for a certain state
        """
        return self.model.predict(state, verbose=0)[0]

    def choose_best_final_state(self, possible_final_states):
        """
            Chooses the best final state based on predicted values or randomly (epsilon-greedy).
        """
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(possible_final_states))
        else:
            for state in possible_final_states:
                states_vector = self.convarte_state_to_vector(state)

                # Ensure the input is of shape (1, 5) for prediction
                states_vector = states_vector.reshape(1, -1)

                value = self.predict_value(states_vector)

                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=108, epochs=1):
        """
            Trains the agent using experience replay with batch updates
        """

        # Ensure the batch size does not exceed the memory size and Proceed only if there is enough memory to start training.
        if batch_size > len(self.Qvalue) or len(self.Qvalue) < batch_size:
            return

        batch = random.sample(self.Qvalue, batch_size)
        x, y = self.build_training_batch(batch)

        self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

        # Update epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def build_training_batch(self, batch):
        """
        Builds the training batch data (input states and target Q-values).
        """
        next_states = np.array([self.convarte_state_to_vector(x[1]) for x in batch]).reshape(len(batch), 5)
        next_qs = self.model.predict(next_states)

        x, y = [], []
        for i, (state, next_state, reward, done) in enumerate(batch):
            state_vector = self.convarte_state_to_vector(state).reshape(1, 5)
            target_qs = self.model.predict(state_vector)[0]

            # Bellman equation to update Q-values
            target_qs[0] = reward if done else reward + self.gamma * np.max(next_qs[i])

            x.append(state_vector.flatten())
            y.append(target_qs)

        return x, y

    def convarte_state_to_vector(self, state):
        """
        Computes and returns only the grid features as a state vector.
        """
        grid = state.grid

        grid_features = [
            self.reward_system.calculate_holes(grid),
            self.reward_system.calculate_bumpiness(grid),
            self.reward_system.calculate_aggregate_height(grid),
            self.reward_system.calculate_highest_point(grid),
            self.reward_system.calculate_clear_lines(grid)
        ]

        return np.array(grid_features)
