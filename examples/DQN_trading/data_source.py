import pandas as pd
import numpy as np

from typing import *


class DataSource:

    def __init__(self, data: pd.Series, features: pd.DataFrame, train_size: float):
        self.data = data
        self.features = features
        self.train_size = train_size
        self.train_len = int(np.ceil(len(data) * train_size))

        self.train_data, self.test_data = self._split_train_test(data, train_size)
        self.train_fe, self.test_fe = self._split_train_test(features, train_size)

    @staticmethod
    def _split_train_test(dataset: Union[pd.Series, pd.DataFrame], train_size: float):
        # NORMALIZE?
        train_end = int(np.ceil(len(dataset) * train_size))
        return dataset.iloc[:train_end], dataset.iloc[train_end:]

    def get_train_state(self, t: int) -> pd.Series:
        if t > self.train_len - 1:
            raise Exception("Index requested exceeds the train length")
        else:
            return self.train_fe.iloc[t]

    def get_train_price(self, t: int) -> float:
        if t > self.train_len - 1:
            raise Exception("Index requested exceeds the train length")
        else:
            return self.train_data.iloc[t]
