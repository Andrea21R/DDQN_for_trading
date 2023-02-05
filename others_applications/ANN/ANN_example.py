import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def get_close(ticker: str, start: str, end: str) -> pd.Series:
    data = yf.Ticker(ticker).history(start=start, end=end)['Close']
    data.name = 'close'
    return data

def preprocess_data(s: pd.Series, n_lags: int) -> tuple:
    rets = s.pct_change()

    y = rets.mask(rets > 0, 1).mask(rets < 0, 0).rename('y')
    x = pd.concat([rets.shift(n).rename(f'lag{n}') for n in range(n_lags)], axis=1)

    yx = pd.concat([y, x], axis=1).dropna()

    return yx['y'], yx[yx.columns.drop('y')]

def split_train_test(data, train_size: float) -> tuple:
    train_end = round(train_size * len(data))
    train_idx = data.index[:train_end]
    test_idx = data.index[train_end:]
    train = data.iloc[:train_end].to_numpy()
    test = data.iloc[train_end:].to_numpy()
    return train, test, train_idx, test_idx

def build_ann_model(n_input: int, n_hidden_layer: int, n_output: int, activation_func: str):

    model = tf.keras.models.Sequential()

    perceptrons = 32
    model.add(tf.keras.layers.Dense(input_dim=n_input, units=32, activation=activation_func))
    for _ in range(n_hidden_layer):
        perceptrons *= 2
        model.add(tf.keras.layers.Dense(units=perceptrons, activation=activation_func))
    model.add(tf.keras.layers.Dense(units=n_output, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
    return model

def predict_classes(model, data, threshold=0.5):
    predictions = model.predict(data, verbose=False)
    classes = np.where(predictions > threshold, 1, 0)
    return classes

def get_confusion_matrix(
        y_true: pd.Series,
        y_pred: pd.Series,
) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize='true').ravel()

    matrix = pd.DataFrame(index=['Positive', 'Negative'], columns=['Positive', 'Negative'])
    matrix.columns.name = "Actual-Values"
    matrix.index.name = "Predicted-Values"

    matrix.loc['Positive', 'Positive'] = tp
    matrix.loc['Positive', 'Negative'] = fp
    matrix.loc['Negative', 'Positive'] = fn
    matrix.loc['Negative', 'Negative'] = tn

    return matrix

def backtest_trading_strategy(y_pred: pd.Series, rets: pd.Series) -> pd.Series:
    return y_pred.replace(0, -1).shift(1).mul(rets)

def get_comp_cum_rets(rets: pd.Series) -> pd.Series:
    return (1 + rets).cumprod() - 1


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import warnings
    warnings.simplefilter("ignore", FutureWarning)

    ticker = "BTC-USD"
    start  = "2005-01-01"
    end    = "2022-01-01"

    n_input = 10
    n_hidden_layer = 2
    n_output = 1
    activation_func = 'relu'

    data = get_close(ticker, start, end)
    y, x = preprocess_data(data, n_lags=n_input)
    y_train, y_test, y_train_idx, y_test_idx  = split_train_test(y, train_size=0.8)
    x_train, x_test, x_train_idx, x_test_idx = split_train_test(x, train_size=0.8)
    model = build_ann_model(n_input, n_hidden_layer, n_output, activation_func)

    model.fit(x_train, y_train)
    predictions = pd.Series(predict_classes(model, x_test, threshold=0.5).squeeze(), index=y_test_idx)
    confusion_matrix_ = get_confusion_matrix(y_test, predictions)
    print(confusion_matrix_)

    rets = data.pct_change().loc[y_test_idx]
    pnl = backtest_trading_strategy(predictions, rets)
    cum_pnl = get_comp_cum_rets(pnl)
    cum_pnl.plot(title=f'{ticker} trading_system using ANN', grid=True, ylabel='cumulative PNL (comp.)')
    plt.show()
