from features_engineering import Features

GEN_PARS = {
    "SMA(20)": {
        'func': Features.Overlap.sma,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 20}
    },
    "MACD(12,26,9)": {
        'func': Features.Momentum.macd,
        'tgt_cols': ['close'],
        'kwargs': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 19}
    },
    "RSI(14)": {
        'func': Features.Overlap.rsi,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 14}
    },
    "BB(20)": {
        'func': Features.Overlap.bollinger,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 20, 'ndevup': 3, 'ndevdn': 3}
    },
    "VOL_EMA(20)": {
        'func': Features.Vola.vola_ema,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 20, 'on_rets': False}
    },
    "ROC(2)": {
        'func': Features.Momentum.roc,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 2}
    },
    "ROC(5)": {
        'func': Features.Momentum.roc,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 5}
    },
    "ROC(10)": {
        'func': Features.Momentum.roc,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 10}
    },
    "ROC(21)": {
        'func': Features.Momentum.roc,
        'tgt_cols': ['close'],
        'kwargs': {'timeperiod': 21}
    },
    "RETS": {
        'func': Features.Others.returns,
        'tgt_cols': ['close'],
        'kwargs': {'lags': [1, 2, 5 ,10]}
    },
    "SIGMA_EVENT|3std_50obs": {
        'func': Features.Dummy.extreme_events,
        'tgt_cols': ['close'],
        'kwargs': {'std_threshold': 3, 'std_window': 20, 'diff': False}
    }
}
