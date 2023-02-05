from collections import deque
import random
import numpy as np
import tensorflow as tf


"""
Example from: https://www.mlq.ai/deep-reinforcement-learning-for-trading-with-tensorflow-2-0/
"""


class Agent:

    def __init__(
            self,
            state_size: int,
            model: tf.keras.Sequential,
            action_space: int = 3,
            memory_size: int = 2000,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            epsilon_final: float = 0.1,
            epsilon_decay: float = 0.995,
            model_name: str = 'Horus',
    ):

        self.state_size = state_size
        self.action_space = action_space
        self.memory = []
        self.inventory = {'last_trade': None, 'last_price': None}
        self.model_name = model_name

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_history = [epsilon]

        self.model = model
        self.model_trained = False

    def use_policy_for_trading(self, state: np.ndarray) -> int:

        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            # argmax returns the position of the arg with max value, i.e. greedy-action
            np.argmax(self.model.predict(state.values.reshape(1, -1), verbose=0))

    def train_batch(self, batch_size: int):

        # create a batch. We need to use a for loop because a deque obj cannot be slicing
        # we assume it's a list of tuple with (state, action, reward, next_state, done)
        batch = []
        for i in range(len(self.memory) - batch_size, len(self.memory)):
            batch.append(self.memory[i])

        count = 0
        # it seems to train a model for each state-transition
        for state, action, reward, next_state, done, best_action in batch:
            # if not self.model_trained:
            #     x = [j[0] for j in batch]
            #     y = [j[2] for j in batch]
            #     self.model.fit(x, y)

            if count == 0:
                print('-- start batch')
            # print(f"batch-step: {count}")
            count += 1

            reward = reward
            if not done:
                # to understand
                reward = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])

            target = self.model.predict(state.reshape(1, -1), verbose=0)
            target[0][action] = reward  # to understand

            self.model.fit(state.reshape(1, -1), target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
            self.epsilon_history.append(self.epsilon)

    def reset_inventory(self) -> None:
        self.inventory = {'last_trade': None, 'last_price': None}

    def update_inventory(self, last_trade: str, last_price: float) -> None:
        self.inventory['last_trade'] = last_trade
        self.inventory['last_price'] = last_price
