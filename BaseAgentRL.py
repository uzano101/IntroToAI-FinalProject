from abc import ABC, abstractmethod


class BaseAgentRL(ABC):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def update_agent(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass
