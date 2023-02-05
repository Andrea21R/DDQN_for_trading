import os
import pandas as pd

def load_data():
    file_path = os.getcwd() + r"\datasets"
    return pd.read_parquet(file_path + r"\aapl_ohlc.parquet")['close'].rename('AAPL')


def load_features():
    file_path = os.getcwd() + r"\datasets"
    return pd.read_parquet(file_path + r"\features_df.parquet")


if __name__ == "__main__":

    import warnings
    warnings.simplefilter("ignore", FutureWarning)

    from examples.DQN_trading import Agent, DataSource, Simulator
    from examples.DQN_trading.nn import ANN

    data = load_data()
    fe = load_features()
    train_size = 0.7

    n_episodes = 100
    batch_size = 400
    action_space = 3
    gamma=0.95
    epsilon=1.0
    epsilon_final=0.01
    epsilon_decay=0.995
    model = ANN.build_model(name='classic', input_dim=fe.shape[1], output_dim=action_space, activation_func='relu')

    data_source = DataSource(data=data, features=fe, train_size=train_size)
    agent = Agent(
        state_size=fe.shape[1],
        model=model,
        action_space=action_space,  # buy, flat, sell
        memory_size=100,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_final=epsilon_final,
        epsilon_decay=epsilon_decay,
        model_name='Horus'
    )
    simulator = Simulator(data_source=data_source, agent=agent, n_episodes=n_episodes, batch_size=batch_size)
    simulator.train_agent(verbose=True, episode_verbose=False)
