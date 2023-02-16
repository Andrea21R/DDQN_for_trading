import os
import pandas as pd
from features_engineering import generate_features_dataframe
from others_applications.DQN_trading import get_data
from features_engineering.constants.parametrization_1 import GEN_PARS

import warnings
warnings.simplefilter("ignore", FutureWarning)


# data = get_data('AAPL', start="2020-01-01", end="2022-01-01", freq='1D')
data_dir_path = os.path.join(os.path.dirname(os.getcwd()), "data", "dataset")
file_name = "AAPL_10y_1d.parquet"
file_path = os.path.join(data_dir_path, file_name)

data = pd.read_parquet(file_path)
fe_df = generate_features_dataframe(data, GEN_PARS)
print('Features created')

fe_df.to_parquet(file_path.replace(file_name, "features_" + file_name))
print(f'Features stored at <{file_path.replace(file_name, "features_" + file_name)}>')
