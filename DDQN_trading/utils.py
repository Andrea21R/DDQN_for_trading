import numpy as np


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


def track_results(
        episode: int,
        nav_ma_100: float,
        nav_ma_10: float,
        market_nav_100: float,
        market_nav_10: float,
        win_ratio: float,
        total: float,
        epsilon: float
):
    # time_ma = np.mean([episode_time[-100:]])
    # T = np.sum(episode_time)

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
