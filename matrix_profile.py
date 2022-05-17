import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import pathlib

import os 
import stumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def data_loader():
    found_files = []
    found_files2 = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')


def return_matrix_profile():
    

    st.title('Matrix Profile')
    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
    if option == "Select a Dataset":
        st.stop()
    dataset =  pd.read_csv(option)


    def motif_mp_individual_plot(data, mp_length=3):
        data = data.astype(float)
        if isinstance(data, pd.DataFrame):
            mps = {}
            motifs_idx = {}
            fig, axs = plt.subplots(len(data.columns) * 2, sharex=True, gridspec_kw={'hspace': 0})
            fig.set_size_inches(24.5, 16.5)
            for dim_name in data.columns:
                mps[dim_name] = stumpy.stump(data[dim_name], mp_length)
                motif_distance = np.round(mps[dim_name][:, 0].min(), 1)
                motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]
                print(motifs_idx)

            for i, dim_name in enumerate(list(mps.keys())):
                mps_values = mps.get(dim_name)
                axs[i].plot(data[dim_name])
                axs[i].set_xlabel('Time', fontsize ='20')
                axs[i + len(data.columns)].plot(mps_values[:,0], c='orange')
                for idx in motifs_idx[dim_name]:
                    axs[i].plot(data[dim_name].iloc[idx:idx+mp_length], c='red', linewidth=4)
                    axs[i].axvline(x=idx, linestyle="dashed", c='black')
                    axs[i + len(data.columns)].axvline(x=idx, linestyle="dashed", c='black')
        else:
            mps = stumpy.stump(data, mp_length)
            # print(mps)
            motifs_idx = np.argsort(mps[:,0])[:2]
            print(motifs_idx)
            fig, axs = plt.subplots(1 * 2, sharex=True, gridspec_kw={'hspace': 0})
            fig.set_size_inches(24.5, 16.5)
            axs[0].plot(data)
            axs[0].set_xlabel('Time', fontsize ='20')
            axs[1].plot(mps[:,0], c='orange')
            for idx in motifs_idx:
                axs[0].plot(data[idx:idx+mp_length], c='red', linewidth=4)
                axs[0].axvline(x=idx, linestyle="dashed", c='black')
                axs[0 + 1].axvline(x=idx, linestyle="dashed", c='black')
        # print('hello')
        # plt.show()
        return(fig)


    # length = st.slider(label='choose length',min_value=10, max_value=50, value=10, step=10, key=2)
    length = st.slider(label='choose length for Motif discovery',min_value=0, max_value=len(dataset), value=0, step=int(len(dataset)/10), key=2)
    
    
    if length > 2:
        
        mp_list = motif_mp_individual_plot(dataset, mp_length=length)
        print('a calculation is done')
        st.pyplot(mp_list)
    else:
        st.write('choose a number')
        

    # data = pd.read_csv(option)

    def mp_individual_plot_anomaly(data, mp_length):
        data = data.astype(float)

        if isinstance(data, pd.DataFrame):
            mps = {}
            discord_idx = {}
            fig, axs = plt.subplots(len(data.columns) * 2, sharex=True, gridspec_kw={'hspace': 0})
            fig.set_size_inches(24.5, 16.5)
            for dim_name in data.columns:
                print(dim_name)
                mps[dim_name] = stumpy.stump(data[dim_name], mp_length)
                discord_idx[dim_name] = np.argsort(mps[dim_name][:,0])[-2:]
                print(discord_idx)

            for i, dim_name in enumerate(list(mps.keys())):
                print(i)
                # print(i)
                # print(dim_name)
                mps_values = mps.get(dim_name)
                axs[i].plot(data[dim_name])
                axs[i].set_xlabel('Time', fontsize ='20')
                axs[i + len(data.columns)].plot(mps_values[:,0], c='orange')
                for idx in discord_idx[dim_name]:
                    axs[i].plot(data[dim_name].iloc[idx:idx+mp_length], c='red', linewidth=4)
                    axs[i].axvline(x=idx, linestyle="dashed", c='black')
                    axs[i + len(data.columns)].axvline(x=idx, linestyle="dashed", c='black')
        else:
            mps = stumpy.stump(data, mp_length)
            print(mps)
            discord_idx = np.argsort(mps[:,0])[-2:]
            fig, axs = plt.subplots(1 * 2, sharex=True, gridspec_kw={'hspace': 0})
            fig.set_size_inches(24.5, 16.5)
            axs[0].plot(data)
            axs[0].set_xlabel('Time', fontsize ='20')
            axs[1].plot(mps[:,0], c='orange')
            for idx in discord_idx:
                axs[0].plot(data[idx:idx+mp_length], c='red', linewidth=4)
                axs[0].axvline(x=idx, linestyle="dashed", c='black')
                axs[0 + 1].axvline(x=idx, linestyle="dashed", c='black')
        return fig, discord_idx
    
    
    length2 = st.slider(label='choose length for Anomaly Detection',min_value=0, max_value=len(dataset), value=0, step=int(len(dataset)/10), key=3)
    

    if length2 > 2:
        
        mp_list, found_indexes = mp_individual_plot_anomaly(dataset, mp_length=length2)
        print('a calculation is done')
        st.write(f'value indexes are: {found_indexes}')
        st.pyplot(mp_list)
    else:
        st.write('choose a number')