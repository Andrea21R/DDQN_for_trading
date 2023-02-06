config = {
    "ticker": 'EURUSD2022_1h',
    "start_end": None,

    "trading_cost_bps": 1e-4,
    "time_cost_bps": 1e-5,

    "n_episodes": 1000,
    "steps_per_episode": 24 * 5,

    "gamma": .99,  # discount factor
    "tau": 100,  # target network update frequency
    "architecture": (256, 256),  # neurons per layer
    "learning_rate": 0.0001,  # learning rate
    "l2_reg": 1e-6,  # L2 regularization
    "replay_capacity": int(1e6),
    "batch_size": 5000,

    "epsilon_start": 1.0,
    "epsilon_end": .01,
    "epsilon_decay_steps": 250,
    "epsilon_exponential_decay": .99
}
