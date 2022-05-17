#%%
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_parquet(r"C:\Users\20191577\Downloads\data_processed.parquet")
data = pd.read_parquet(r"C:\Users\20191577\Downloads\dataset.parquet")
data = data.iloc[1,:]
# of_interest = ['TIMESERIES_Druck_links[N/mm2]',
# 'TIMESERIES_Druck_rechts[N/mm2]',
# 'TIMESERIES_Kraft_links[N]',
# 'TIMESERIES_Kraft_rechts[N]',
# 'TIMESERIES_Thermocouple_WS[°C]',
# 'TIMESERIES_Timestamp[s]',
# 'TIMESERIES_Waermestromsensor[µV]',
# 'TIMESERIES_Weg_links[mm]',
# 'TIMESERIES_Weg_rechts[mm]',]

# data = data[of_interest]
# data = data.apply(pd.Series).T




# plt.plot(data.iloc[:,7])
# %%




# other dataset

data = pd.read_parquet(r"C:\Users\20191577\Downloads\dataset.parquet")
