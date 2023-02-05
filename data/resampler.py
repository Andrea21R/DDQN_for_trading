import pandas as pd
from typing import *


class Sampler:

    @staticmethod
    def _get_close(bars: pd.Series) -> float:
        return bars.iloc[-1]

    @staticmethod
    def _get_open(bars: pd.Series) -> float:
        return bars.iloc[0]

    @classmethod
    def aggregate_bar(cls, bars: pd.DataFrame) -> Union[pd.Series, None]:

        if not bars.empty:
            return bars.aggregate(
                {
                    'open': cls._get_open,
                    'high': max,
                    'low': min,
                    'close': cls._get_close,
                }
            )
        else:
            return None

    @classmethod
    def resample(cls, bars: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
        return bars.resample(rule=freq).apply(cls.aggregate_bar).dropna()


if __name__ == "__main__":

    import os

    dir_path      = os.getcwd() + rf"\dataset"
    file_name     = "\EURUSD2022_1m.parquet"
    new_file_name = "\EURUSD2022_1h.parquet"
    new_freq      = "1H"

    data = pd.read_parquet(dir_path + file_name)
    data_resampled = Sampler.resample(bars=data, freq=new_freq)

    data_resampled.to_parquet(dir_path + new_file_name)
