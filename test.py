import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import missingno as mno

# from darts import TimeSeries, concatenate
# from darts.dataprocessing.transformers import Scaler
# from darts.models import TransformerModel
# from darts.metrics import mape, rmse
# from darts.utils.timeseries_generation import datetime_attribute_timeseries
# from darts.utils.likelihood_models import QuantileRegression
from pandas import read_csv
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
pd.set_option("display.precision", 2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

# load
df0 = pd.read_csv("./ETDataset-main/ETT-small/ETTh1.csv")
df0.info()
# datetime
df0["date"] = pd.to_datetime(df0["date"], utc=True, infer_datetime_format=True)

# any duplicate time periods?
print("count of duplicates:", df0.duplicated(subset=["date"], keep="first").sum())
df0.set_index("date", inplace=True)

# any non-numeric types?
print("non-numeric columns:", list(df0.dtypes[df0.dtypes == "object"].index))


# any missing values?
def gaps(df0):
    if df0.isnull().values.any():
        print("MISSING values:\n")
        mno.matrix(df0)
    else:
        print("no missing values\n")


gaps(df0)
print(df0.describe())
plt.figure(100, figsize=(20, 7))
sns.lineplot(x="date", y="OT", data=df0, palette="coolwarm")
# plt.show()

# convert int and float64 columns to float32
intcols = list(df0.dtypes[df0.dtypes == np.int64].index)
df0[intcols] = df0[intcols].applymap(np.float32)

f64cols = list(df0.dtypes[df0.dtypes == np.float64].index)
df0[f64cols] = df0[f64cols].applymap(np.float32)

f32cols = list(df0.dtypes[df0.dtypes == np.float32].index)
# df0.info()

# investigate the outliers in the pressure column
# print(df0["HULL"].nlargest(10))
