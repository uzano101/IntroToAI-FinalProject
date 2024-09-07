from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def update_agent(self, state, reward, next_state, done):
        pass

    @abstractmethod
    def choose_best_final_state(self, current_state, possible_final_states):
        pass

    @abstractmethod
    def train(self,batch_size=10):
        pass
