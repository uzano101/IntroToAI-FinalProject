from abc import ABC, abstractmethod


class BaseAgentRL(ABC):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def update_agent(self, state, reward, next_state, done):
        pass

    @abstractmethod
    def choose_best_final_state(self, current_state, possible_final_states):
        pass

    @abstractmethod
    def train(self,batch_size=10):
        pass
