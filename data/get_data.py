import yfinance as yf
import pandas as pd
import os


def get_data(ticker: str, period: str, freq: str) -> pd.DataFrame:
    return yf.Ticker(ticker).\
        history(period=period, interval=freq).\
        rename(columns={k: k.lower() for k in ('Open', 'High', 'Low', 'Close', 'Volume')}).\
        loc[:, ['open', 'high', 'low', 'close', 'volume']].\
        tz_convert(None)


def store_data(df: pd.DataFrame, dirname: str, filename: str) -> None:
    df.to_parquet(
        os.path.join(dirname, filename + ".parquet")
    )


if __name__ == "__main__":

    ticker = 'AAPL'
    period = '10y'
    freq = '1d'

    dirname = os.path.join(os.getcwd(), "dataset")
    file_name = f'{ticker}_{period}_{freq}'

    data = get_data(ticker, period, freq)

    store_data(data, dirname, file_name)
