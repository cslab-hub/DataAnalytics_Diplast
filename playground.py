# %%
import os 
import stumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def data_loader():
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                print(filename)
                data = pd.read_csv(os.path.join(roots,filename))
                return data

data = data_loader()

#%%
#%%
def indi_matrix(data,mp_length=3):
    data = data.astype(float)
    mp_list = []
    mps = {}  # Store the 1-dimensional matrix profiles
    motifs_idx = {}  # Store the index locations for each pair of 1-dimensional motifs (i.e., the index location of two smallest matrix profile values within each dimension)
    for dim_name in data.columns:
        mps[dim_name] = stumpy.stump(data[dim_name], mp_length)
        mp_list.append(mps[dim_name])
        motif_distance = np.round(mps[dim_name][:, 0].min(), 1)
        motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]
        print(f"The motif pair matrix profile value in {dim_name} is {motif_distance}")
    fig, axs = plt.subplots(len(mps), sharex=True, gridspec_kw={'hspace': 0})
    fig.set_size_inches(24.5, 16.5)
    for i, dim_name in enumerate(list(mps.keys())):
        #axs[i].set_ylabel(dim_name, fontsize='20')
        axs[i].plot(data[dim_name])
        # axs[i].set_xlabel('Time', fontsize ='20')
        for idx in motifs_idx[dim_name]:
            axs[i].plot(data[dim_name].iloc[idx:idx+mp_length], c='red', linewidth=4)
            axs[i].axvline(x=idx, linestyle="dashed", c='black')
    # plt.show()
    mp_list = np.array(mp_list)
    return(fig)

mp_list = indi_matrix(data)

# %%


def mp_individual_plot(data, mp_length=3):
    print('hello')
    if isinstance(data, pd.DataFrame):
        mps = {}
        motifs_idx = {}
        fig, axs = plt.subplots(len(data.columns) * 2, sharex=True, gridspec_kw={'hspace': 0})
        fig.set_size_inches(24.5, 16.5)
        for dim_name in data.columns:
            mps[dim_name] = stumpy.stump(data[dim_name], seq_len)
            motif_distance = np.round(mps[dim_name][:, 0].min(), 1)
            motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]
            print(motifs_idx)
            

        for i, dim_name in enumerate(list(mps.keys())):
            mps_values = mps.get(dim_name)
            axs[i].plot(data[dim_name])
            axs[i].set_xlabel('Time', fontsize ='20')
            axs[i + len(data.columns)].plot(mps_values[:,0], c='orange')
            for idx in motifs_idx[dim_name]:
                axs[i].plot(data[dim_name].iloc[idx:idx+seq_len], c='red', linewidth=4)
                axs[i].axvline(x=idx, linestyle="dashed", c='black')
                axs[i + len(data.columns)].axvline(x=idx, linestyle="dashed", c='black')
    else:
        mps = stumpy.stump(data, seq_len)
        print(mps)
        motifs_idx = np.argsort(mps[0])
        fig, axs = plt.subplots(1 * 2, sharex=True, gridspec_kw={'hspace': 0})
        fig.set_size_inches(24.5, 16.5)
        axs[0].plot(data)
        axs[0].set_xlabel('Time', fontsize ='20')
        axs[1].plot(mps[:,0], c='orange')
        print(motifs_idx)


        axs[0].plot(data.iloc[motifs_idx:motifs_idx+seq_len], c='red', linewidth=4)
        

    print('hello')
    #plt.show()
    return(fig)
