import pandas as pd


class Sampler:

    @staticmethod
    def _get_close(bars: pd.Series) -> float:
        return bars.iloc[-1]

    @staticmethod
    def _get_open(bars: pd.Series) -> float:
        return bars.iloc[0]

    @classmethod
    def aggregate_bar(cls, bars: pd.DataFrame) -> pd.Series:
        return bars.aggregate(
            {
                'open': cls._get_open,
                'high': max,
                'low': min,
                'close': cls._get_close,
            }
        )

    @classmethod
    def resample(cls, bars: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
        return bars.resample(rule=freq).apply(cls.aggregate_bar)


if __name__ == "__main__":

    import os

    file_name = "EURUSD_2022_1m.parquet"
    new_freq = "1H"

    data = pd.read_parquet(os.getcwd() + rf"\dataset\{file_name}")
    data_resampled = Sampler.resample(bars=data, freq=new_freq)


