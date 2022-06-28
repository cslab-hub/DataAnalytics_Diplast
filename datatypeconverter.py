#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
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
import pandas as pd 
data = pd.read_parquet(r"C:\Users\20191577\Downloads\dataset.parquet")

#%%
df = data.iloc[:,:]
# df['TCN_ActualProcessPower'].explode()
df = df.loc[:,~df.columns.str.startswith('TCN')]
df = df.loc[:,~df.columns.str.startswith('MET')]
df = df.loc[:,~df.columns.str.startswith('TCE')]
df = df.loc[:,~df.columns.str.startswith('SIM')]
# df = df.loc[:,~df.columns.str.startswith('LBL')]
df = df.loc[:,~df.columns.str.startswith('SET')]
df = df.loc[:,~df.columns.str.startswith('IR')] 
# df = df.loc[:,~df.columns.str.startswith('CV')]
df = df.loc[:,~df.columns.str.startswith('DOS')]
df = df.loc[:,~df.columns.str.startswith('DRY')]
df = df.loc[:,~df.columns.str.startswith('DXP')]
# df = df.loc[:,~df.columns.str.startswith('E77')]
df = df.loc[:,~df.columns.str.startswith('ENV')]
# df = df.loc[:,~df.columns.str.startswith('SCA')]

df = df.loc[:,~df.columns.str.startswith('CV_Image')]
df = df.loc[:,~df.columns.str.startswith('CV_Diameter')]
df = df.loc[:,~df.columns.str.startswith('CV_Height')]
df = df.loc[:,~df.columns.str.startswith('CV_Width')]
df = df.loc[:,~df.columns.str.startswith('LBL_Underfilled')]
df = df.loc[:,~df.columns.str.startswith('LBL_StreaksLevel')]
df = df.loc[:,~df.columns.str.startswith('LBL_Sprue')]
df = df.loc[:,~df.columns.str.startswith('LBL_NOK')]
df = df.loc[:,~df.columns.str.startswith('LBL_Old')]



print(df.shape)
print(df.columns)
df=df.dropna(axis=1)
print(df.shape)
#%%

df.to_csv('data/injection_molding_single_values.csv', index=False)
#%%
df
# %%

#! Time series version
import pandas as pd 
import numpy as np 
data = pd.read_parquet(r"C:\Users\20191577\Downloads\dataset.parquet")

#%%
df = data.iloc[:,:]

df = df[['DXP_Inj1PrsAct','DXP_MldCavPrs1Act','CV_Warpage','LBL_SinkMarks','SCA_PartWeight']]


# %%
def turn_into_equal_long(desired_variable):
    list_thing = []
    for i in df[desired_variable].to_list():
        list_thing.append(list(i))

    max_num = 0
    for i,j in enumerate(list_thing):
        if len(j) > max_num:
            max_num = len(j)
    print(max_num)



    sents_padded = [sent + [0]*(max_num - len(sent)) for sent in list_thing]
    return np.array(sents_padded)

turn_into_equal_long('DXP_MldCavPrs1Act').shape
#%%
turn_into_equal_long('DXP_Inj1PrsAct').shape

# %%
# %%
new = np.dstack([turn_into_equal_long('DXP_Inj1PrsAct'),turn_into_equal_long('DXP_MldCavPrs1Act')])
# %%
plt.plot(new[0,:,1])

#%%

# DO NOT RUN!!!!
# np.save('data/injection_molding_train.npy', new)
# %%
test = df[['CV_Warpage','LBL_SinkMarks','SCA_PartWeight']]
# test.to_csv('data/injection_molding_test.csv',index=False)
test.shape
