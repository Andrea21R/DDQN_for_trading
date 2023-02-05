import pandas as pd
from tqdm import tqdm
from examples.DQN_trading import DataSource, Agent


class Simulator:

    def __init__(self, data_source: DataSource, agent: Agent, n_episodes: int, batch_size: int):
        self.data_source = data_source
        self.agent = agent

        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.trading_history = self._get_trading_history_df()

    def _get_trading_history_df(self):
        columns = [f'episode{n}' for n in range(1, self.n_episodes + 1)]
        index = range(self.data_source.train_len)
        return pd.DataFrame(columns=columns, index=index)

    def _run_episode(self, verbose: bool = False, final_verbose: bool = True):
        total_profit = 0
        self.agent.reset_inventory()  # reset to 0 the inventory at each episode

        count = 0
        for t in range(self.data_source.train_len - 1):

            if count == 0:
                print('- start episode')
            # print(f'- episode-time_step: {count}')
            count += 1

            current_state = self.data_source.get_train_state(t)
            action = self.agent.use_policy_for_trading(current_state)

            next_state = self.data_source.get_train_state(t + 1)
            reward = 0

            last_trade = self.agent.inventory['last_trade']
            last_price = self.agent.inventory['last_price']
            current_p = self.data_source.get_train_price(t)

            # =========================================================== BUY SIDE =====================================
            if action == 1:

                if last_price != 'long':
                    if  last_trade == 'short':
                        reward = last_price - current_p
                        total_profit += reward
                        self.agent.reset_inventory()
                        if verbose:
                            print(f"Agent CLOSE SHORT position, Profit: ${reward} | Tot_profit: ${total_profit}")

                    else:  # empty inventory
                        self.agent.update_inventory(last_trade='long', last_price=current_p)
                        if verbose:
                            print(f"Agent OPEN LONG (1 share) at ${current_p}")

            # =========================================================== SELL SIDE ====================================
            elif action == 2:

                if last_trade != 'short':
                    if last_trade == 'long':
                        reward = current_p - last_price
                        total_profit += reward
                        self.agent.reset_inventory()
                        if verbose:
                            print(f"Agent CLOSE LONG position, Profit: $ {reward} | Tot_profit: ${total_profit}")

                    else:
                        self.agent.update_inventory(last_trade='short', last_price=current_p)
                        if verbose:
                            print(f"Agent OPEN SHORT (1 share) at ${current_p}")


            if t == self.data_source.train_len - 2:
                done = True
            else:
                done = False

            # -----
            next_p = self.data_source.get_train_price(t + 1)
            if next_p > current_p * 1.01:
                best_action = 1
            elif next_p < current_p * 0.99:
                best_action = -1
            else:
                best_action = 0
            # -----

            # fill the batch buffer
            self.agent.memory.append(
                (current_state.values, action, reward, next_state.values, done, best_action)
            )

            if done and final_verbose:
                print(f'---> FINAL PROFIT for episode was: ${total_profit}')

            # use memory replay
            if len(self.agent.memory) > self.batch_size:
                # print('penso si blocchi')
                self.agent.train_batch(batch_size=self.batch_size)
                self.agent.memory = []
                # credo si blocchi qua
                # print(len(self.agent.memory), len(self.agent.memory) > self.batch_size)

    def train_agent(self, verbose: bool = True, episode_verbose: bool = False):

        for episode in range(1, self.n_episodes + 1):
            if verbose:
                print(f"============================= EPISODE-{episode} =================================")
                self._run_episode(verbose=episode_verbose, final_verbose=verbose)
