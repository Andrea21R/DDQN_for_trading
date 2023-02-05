from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
import gym
from gym.envs.registration import register

from examples.DDQN_trading import DDQNAgent, TradingEnvironment


# ------------------------------------------------------ Settings
np.random.seed(42)
tf.random.set_seed(42)
sns.set_style('whitegrid')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

results_path = Path('results', 'trading_bot')
if not results_path.exists():
    results_path.mkdir(parents=True)

# ----------------------------------------------------- Helper Function
def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

# ----------------------------------------------------- Setup Gym Environment
steps_per_episode = 60 * 24
gym.register(
    id='trading-v0',
    entry_point='trading_environment:TradingEnvironment',
    max_episode_steps=steps_per_episode
)
trading_cost_bps = 1e-3
time_cost_bps = 1e-4
print(f'Trading costs: {trading_cost_bps:.2%} | Time costs: {time_cost_bps:.2%}')
trading_environment = gym.make(
    'trading-v0',
    ticker='EURUSD2022_1m',
    steps_per_episode=steps_per_episode,
    trading_cost_bps=trading_cost_bps,
    time_cost_bps=time_cost_bps,
    start_end=("2022-08-01", "2022-09-01")
)
trading_environment.seed(42)

# ----------------------------------------------------- Get Environment Parameters
state_dim = trading_environment.observation_space.shape[0]
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps

# ----------------------------------------------------- Define Hyper-parameters
gamma = .99  # discount factor
tau = 100  # target network update frequency

# ----------------------------------------------------- ANN-architecture Parameters
architecture = (256, 256)  # units per layer
learning_rate = 0.0001  # learning rate
l2_reg = 1e-6  # L2 regularization

# ----------------------------------------------------- Experience-Replay Parameters
replay_capacity = int(1e6)
batch_size = 4096

# ----------------------------------------------------- Epsilon-Greedy policy
epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_steps = 250
epsilon_exponential_decay = .99

# ======================================================================================================================
# ----------------------------------------------------- Create a DDQN-Agent --------------------------------------------
# ======================================================================================================================
tf.keras.backend.clear_session()
ddqn = DDQNAgent(
    state_dim=state_dim,
    num_actions=num_actions,
    learning_rate=learning_rate,
    gamma=gamma,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    epsilon_decay_steps=epsilon_decay_steps,
    epsilon_exponential_decay=epsilon_exponential_decay,
    replay_capacity=replay_capacity,
    architecture=architecture,
    l2_reg=l2_reg,
    tau=tau,
    batch_size=batch_size
)

# --------------------------------------------------- print DDQN-Online-ANN architecture
print(ddqn.online_network.summary())


# -------------------------------------------------- Run-experiment
# ------------- Set parameters
total_steps = 0
max_episodes = 100

# ------------- Initialize variables
episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

# ------------ Visualization
def track_results(
        episode,
        nav_ma_100,
        nav_ma_10,
        market_nav_100,
        market_nav_10,
        win_ratio,
        total,
        epsilon
):
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)

    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(
        episode,
        format_time(total),
        nav_ma_100 - 1,
        nav_ma_10 - 1,
        market_nav_100 - 1,
        market_nav_10 - 1,
        win_ratio,
        epsilon)
    )

# ============= SET ENVIRONMENT
# trading_environment = TradingEnvironment(
#     steps_per_episode=steps_per_episode,
#     trading_cost_bps=trading_cost_bps,
#     time_cost_bps=time_cost_bps,
#     ticker="EURUSD2022_1m",
#     start_end=("2022-08-01", "2022-09-01")
# )

# ---------------------------------------------------- TRAIN AGENT
start = time()
results = []
for episode in range(1, max_episodes + 1):
    this_state = trading_environment.reset()  # reset to 0 the environment due to new episode was started
    # iterate over the episode's steps
    for episode_step in range(max_episode_steps):
        # to understand if this_state is a tuple or a list of tuple (i.e. vectorized or step by step). I think the second one
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))  # get an action
        next_state, reward, done, _ = trading_environment.step(action)  # given the action get S', R(t+1) and done

        ddqn.memorize_transition(s=this_state, a=action, r=reward, s_prime=next_state, not_done=0.0 if done else 1.0)

        # if we have to train ANN, do the experience replay approach to update ANNs models
        if ddqn.train:
            ddqn.experience_replay()
        if done:
            break
        this_state = next_state  # update current state with the next one

    # get DataFrame with sequence of actions, returns and nav values
    result = trading_environment.env.simulator.result()

    # get results of last step
    final = result.iloc[-1]

    # apply return (net of cost) of last action to last starting nav
    nav = final.nav * (1 + final.strategy_return)
    navs.append(nav)

    # market nav
    market_nav = final.market_nav
    market_navs.append(market_nav)

    # track difference between agent an market NAV results
    diff = nav - market_nav
    diffs.append(diff)

    # every 10 episode, print the temporary-results
    if episode % 10 == 0:
        track_results(
            episode,
            # show mov. average results for 100 (10) periods
            np.mean(navs[-100:]),
            np.mean(navs[-10:]),
            np.mean(market_navs[-100:]),
            np.mean(market_navs[-10:]),
            # share of agent wins, defined as higher ending nav
            np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
            time() - start,
            ddqn.epsilon
        )
    # to understand
    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        print(result.tail())
        break

trading_environment.close()


# Alla fine dovrò prendere la Target-ANN ed utilizzare quella per fare trading, in congiunta con TradingEnvironment?
