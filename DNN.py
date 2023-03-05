import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz
import os
import sys
import missingno as mno
from datetime import timezone, datetime
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.utils import torch
from pandas import read_csv
from datetime import datetime
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

LOAD = False      # True = load previously saved model from disk?  False = (re)train the model
SAVE = "\_TForm_model10e.pth.tar"   # file name to save the model under

EPOCHS = 0
INLEN = 512          # input size
FEAT = 32           # d_model = number of expected features in the inputs, up to 512
HEADS = 4           # default 8
ENCODE = 4          # encoder layers
DECODE = 4          # decoder layers
DIM_FF = 128        # dimensions of the feedforward network, default 2048
BATCH = 32          # batch size
ACTF = "relu"       # activation function, relu (default) or gelu
SCHLEARN = None     # a PyTorch learning rate scheduler; None = constant rate
LEARN = 1e-3        # learning rate
VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
DROPOUT = 0.1       # dropout rate
N_FC = 1            # output size

RAND = 42           # random seed
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = 3          # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.9         # train/test %

FIGSIZE = (9, 6)


qL1, qL2 = 0.01, 0.10        # percentiles of predictions: lower bounds
qU1, qU2 = 1-qL1, 1-qL2,     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'

mpath = os.path.abspath(os.getcwd()) + SAVE     # path and file name to save the model

# load
df0 = pd.read_csv("ETTh1.csv")
df0.info()
# datetime
df0["date"] = pd.to_datetime(df0["date"], utc=True, infer_datetime_format=True)

# any duplicate time periods?
print("count of duplicates:",df0.duplicated(subset=["date"], keep="first").sum())
df0.set_index("date", inplace=True)

# any non-numeric types?
print("non-numeric columns:",list(df0.dtypes[df0.dtypes == "object"].index))

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
sns.lineplot(x = "date", y = "OT", data = df0, palette="coolwarm")
#plt.show()

# convert int and float64 columns to float32
intcols = list(df0.dtypes[df0.dtypes == np.int64].index)
df0[intcols] = df0[intcols].applymap(np.float32)

f64cols = list(df0.dtypes[df0.dtypes == np.float64].index)
df0[f64cols] = df0[f64cols].applymap(np.float32)

f32cols = list(df0.dtypes[df0.dtypes == np.float32].index)
# limit the dataframe's date range
df2 = df0[df0.index >= "2017-01-01 00:00:00+00:00"]
print(df2.iloc[[0,-1]])

# check correlations of features with price
df_corr = df2.corr(method="pearson")
print(df_corr.shape)
print("correlation with OT:")
df_corrP = pd.DataFrame(df_corr["OT"].sort_values(ascending=False))
print(df_corrP)
# correlation matrix, limited to highly correlated features
df3 = df2[df_corrP.index]
df3.info()
idx = df3.corr().sort_values("OT", ascending=False).index
df3_sorted = df3.loc[:, idx]  # sort dataframe columns by their correlation with Appliances

plt.figure(figsize = (15,15))
sns.set(font_scale=0.75)
ax = sns.heatmap(df3_sorted.corr().round(3),
            annot=True,
            square=True,
            linewidths=.75, cmap="coolwarm",
            fmt = ".2f",
            annot_kws = {"size": 11})
ax.xaxis.tick_bottom()
plt.title("correlation matrix")
plt.show()

# additional datetime columns: feature engineering
df3["month"] = df3.index.month

df3["wday"] = df3.index.dayofweek
dict_days = {0:"1_Mon", 1:"2_Tue", 2:"3_Wed", 3:"4_Thu", 4:"5_Fri", 5:"6_Sat", 6:"7_Sun"}
df3["weekday"] = df3["wday"].apply(lambda x: dict_days[x])

df3["hour"] = df3.index.hour

df3 = df3.astype({"hour":float, "wday":float, "month": float})
df3.iloc[[0, -1]]
# dataframe with OT and features only
df4 = df3.copy()


df4.drop(["weekday", "month", "wday", "hour"], inplace=True, axis=1)
print("Df4 information/////////////////")
# print(df4.head())
# print(df4.info())
# create time series object for target variable
ts_P = TimeSeries.from_series(df4["OT"])
print("The values of ts_P")
print(type(ts_P[0]))

# check attributes of the time series
print("components:", ts_P.components)
print("duration:",ts_P.duration)
print("frequency:",ts_P.freq)
print("frequency:",ts_P.freq_str)
print("has date time index? (or else, it must have an integer index):",ts_P.has_datetime_index)
print("deterministic:",ts_P.is_deterministic)
print("univariate:",ts_P.is_univariate)

# create time series object for the feature columns
df_covF = df4.loc[:, df4.columns != "OT"]

print("df_covf information/////////////////")

ts_covF = TimeSeries.from_dataframe(df_covF)
# check attributes of the time series
print("components (columns) of feature time series:", ts_covF.components)
print("duration:",ts_covF.duration)
print("frequency:",ts_covF.freq)
print("frequency:",ts_covF.freq_str)
print("has date time index? (or else, it must have an integer index):",ts_covF.has_datetime_index)
print("deterministic:",ts_covF.is_deterministic)
print("univariate:",ts_covF.is_univariate)

# train/test split and scaling of target variable
ts_train, ts_test = ts_P.split_after(SPLIT)
print("training start:", ts_train.start_time())
print("training end:", ts_train.end_time())
print("training duration:",ts_train.duration)
print("test start:", ts_test.start_time())
print("test end:", ts_test.end_time())
print("test duration:", ts_test.duration)

scalerP = Scaler()
scalerP.fit_transform(ts_train)
ts_ttrain = scalerP.transform(ts_train)
ts_ttest = scalerP.transform(ts_test)
ts_t = scalerP.transform(ts_P)

# make sure data are of type float
ts_t = ts_t.astype(np.float32)
ts_ttrain = ts_ttrain.astype(np.float32)
ts_ttest = ts_ttest.astype(np.float32)

print("First and last row of scaled OT time series:")
pd.options.display.float_format = '{:,.2f}'.format
print(ts_t.pd_dataframe().iloc[[0,-1]])

# train/test split and scaling of feature covariates
covF_train, covF_test = ts_covF.split_after(SPLIT)

scalerF = Scaler()
scalerF.fit_transform(covF_train)
covF_ttrain = scalerF.transform(covF_train)
covF_ttest = scalerF.transform(covF_test)
covF_t = scalerF.transform(ts_covF)

# make sure data are of type float
covF_ttrain = ts_ttrain.astype(np.float32)
covF_ttest = ts_ttest.astype(np.float32)

pd.options.display.float_format = '{:.2f}'.format
print("First and last row of scaled feature covariates:")
print(covF_t.pd_dataframe().iloc[[0,-1]])

# feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays
covT = datetime_attribute_timeseries(ts_P.time_index, attribute="hour", until=pd.Timestamp("2018-07-01 11:00:00"), one_hot=False)
covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="day_of_week", one_hot=False))
covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="month", one_hot=False))
covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="year", one_hot=False))

covT = covT.add_holidays(country_code="ES")
covT = covT.astype(np.float32)


# train/test split
covT_train, covT_test = covT.split_after(SPLIT)


# rescale the covariates: fitting on the training set
scalerT = Scaler()
scalerT.fit(covT_train)
covT_ttrain = scalerT.transform(covT_train)
covT_ttest = scalerT.transform(covT_test)
covT_t = scalerT.transform(covT)

covT_t = covT_t.astype(np.float32)


pd.options.display.float_format = '{:.0f}'.format
print("first and last row of unscaled time covariates://")
print(covT.pd_dataframe().iloc[[0,-1]])

model = TransformerModel(
                    input_chunk_length = INLEN,
                    output_chunk_length = N_FC,
                    batch_size = BATCH,
                    n_epochs = EPOCHS,
                    model_name = "Transformer_OT",
                    nr_epochs_val_period = VALWAIT,
                    d_model = FEAT,
                    nhead = HEADS,
                    num_encoder_layers = ENCODE,
                    num_decoder_layers = DECODE,
                    dim_feedforward = DIM_FF,
                    dropout = DROPOUT,
                    activation = ACTF,
                    random_state=RAND,
                    likelihood=QuantileRegression(quantiles=QUANTILES),
                    optimizer_kwargs={'lr': LEARN},
                    add_encoders={"cyclic": {"future": ["hour", "dayofweek", "month"]}},
                    save_checkpoints=True,
                    force_reset=True
                    )
# training: load a saved model or (re)train
if LOAD:
    print("have loaded a previously saved model from disk:" + mpath)
    model = TransformerModel.load(mpath)                            # load previously model from disk
else:
    model.fit(  ts_ttrain,
                past_covariates=covF_t,
                verbose=True)
    print("have saved the model after training:", mpath)
    model.save(mpath)
#
# # testing: generate predictions
ts_tpred = model.predict(n=len(ts_ttest),
                        num_samples=N_SAMPLES,
                        n_jobs=N_JOBS,
                        verbose=True)
#retrieve forecast series for chosen quantiles,
#inverse-transform each series,
#insert them as columns in a new dataframe dfY
q50_RMSE = np.inf
q50_MAPE = np.inf
ts_q50 = None
pd.options.display.float_format = '{:,.2f}'.format

dfY = pd.DataFrame()
dfY["Actual"] = TimeSeries.pd_series(ts_test)
print(dfY["Actual"])
#
# # helper function: get forecast values for selected quantile q and insert them in dataframe dfY
dfY
def predq (ts_t,q):
    ts_tq = ts_t.quantile_timeseries(q)
    ts_q = scalerP.inverse_transform(ts_tq)
    s = TimeSeries.pd_series(ts_q)
    header = "Q" + format(int(q * 100), "02d")
    dfY[header] = s
    if q == 0.5:
        ts_q50 = ts_q
        q50_RMSE = rmse(ts_q50, ts_test)
        q50_MAPE = mape(ts_q50, ts_test)
        print("RMSE:", f'{q50_RMSE:.2f}')
        print("MAPE:", f'{q50_MAPE:.2f}')

#
# # call helper function predQ, once for every quantile
_ = [predq(ts_tpred, q) for q in QUANTILES]

# move Q50 column to the left of the Actual column
col = dfY.pop("Q50")
dfY.insert(1, col.name, col)
dfY.iloc[np.r_[0:2, -2:0]]