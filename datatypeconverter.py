#%%
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_parquet(r"C:\Users\20191577\Downloads\data_processed.parquet")
data = data.iloc[1,:]
of_interest = ['TIMESERIES_Druck_links[N/mm2]',
'TIMESERIES_Druck_rechts[N/mm2]',
'TIMESERIES_Kraft_links[N]',
'TIMESERIES_Kraft_rechts[N]',
'TIMESERIES_Thermocouple_WS[°C]',
'TIMESERIES_Timestamp[s]',
'TIMESERIES_Waermestromsensor[µV]',
'TIMESERIES_Weg_links[mm]',
'TIMESERIES_Weg_rechts[mm]',]

data = data[of_interest]
data = data.apply(pd.Series).T
data.columns = ['print_left', 'pressure_right', 'force_left' ,'force_right' ,'Thermocouple_WS' ,'timestamp' ,'heat flow' 'sensor' ,'way_left' ,'way_right']
data = data.drop(columns=['timestamp'])
daterange = pd.date_range(start=pd.Timestamp('10:10:15'), periods=data.shape[0], freq='1s')
daterange = daterange.strftime("%Y-%m-%d %H:%M:%S")
data.insert(0,'TIME', daterange)
data.to_csv('data/plastic_welding.csv', index=False)
data


# plt.plot(data.iloc[:,7])
# %%
data