import math
import pandas as pd
import yfinance as yf


def sigmoid(x):
    """
    Scale number to a range from 0 to 1
    """
    return 1 / (1 + math.exp(-x))


def get_data(ticker: str, start: str, end: str, freq: str) -> pd.Series:

    data = yf.Ticker(ticker).\
        history(start=start, end=end, interval=freq).\
        rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})

    if not data.empty:
        return data[['open', 'high', 'low', 'close']]
    else:
        raise Exception("There's something wrong with data downloading. Please, check the input pars!")
