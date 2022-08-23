import streamlit as st
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt
import sys 
import csv


##% DATALOADER
def data_loader():
    found_files = []
    cwd = os.getcwd()    
    cwd = cwd+'/data/preprocessed'
    print(cwd)
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')


def plot_columns(dataset,x):
    df = dataset.groupby('Sensor').agg({'TimeStamp': ['min', 'max']})
    df = df.droplevel(0, axis=1).reset_index()
    df.columns = ['Task','start','end'] 
    df = df[df.Task.str.contains(x)]

    df.start=pd.to_datetime(df.start)
    df.end=pd.to_datetime(df.end)
    df['duration']=df.end-df.start
    #convert duration to number and add one
    df.duration=df.duration.apply(lambda x: x.days+1)
    #sort in ascending order of start date
    df=df.sort_values(by='start', ascending=True)



    p_start=df.start.min()
    p_end=df.end.max()
    p_duration=(p_end-p_start).days+1

    #xticks
    x_ticks=[i for i in range(p_duration+1)]
    #xtick labels starts with project start date |formatted
    x_labels=[(p_start+dt.timedelta(days=i)).strftime('%d-%b') 
              for i in x_ticks]

    df['rel_start']=df.start.apply(lambda x: (x-p_start).days)

    fig, axs = plt.subplots()
    axs.plot(figsize=(20,10))
    #plot barh chart
    axs.barh(y=df.Task, left=df.rel_start, width=df.duration)
    #Invert y axis
    axs.invert_yaxis()
    #customize x-ticks
    axs.set_xticks(ticks=x_ticks, labels=x_labels,rotation=90)
    axs.tick_params(axis='both', labelsize=4)
    axs.grid(axis='x', alpha=0.5)
    return fig



def return_preprocessing():

    col1, col2, col3 = st.columns([1,2.5,1])
    with col2:


        st.title('Descriptive statistics of your data')
        option = st.selectbox(
            'Which dataset do you want to view?',
            (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
        if option == 'Select a Dataset':
            st.stop()
        dataset = pd.read_csv(option)

        st.write("""
            The dataset below shows the first 10 inputs. Based on this information, you are able to see the general outline of the dataset, e.g., the amount of columns and some values. 
            """)
   
        st.dataframe(dataset.head(10))
        st.write(f'The dataset contains a total of {len(dataset)} rows and {len(dataset.columns)} columns.')
        dtype_df = dataset.dtypes.value_counts().reset_index()

        dtype_df.columns = ['VariableType','Count']
        dtype_df['VariableType'] = dtype_df['VariableType'].astype(str)

        fig, ax = plt.subplots(figsize=(7,3))
        ax.bar(dtype_df['VariableType'],dtype_df['Count'])
        
        st.write("""
            The plot below indicates which datatypes are in the dataset, based on the different columns (variables). The barplot shows the different datatypes and how many of these columns are spotted in the dataset.
            """)

        st.pyplot(fig)

        st.write(f"""
           To get some more information regarding the dataset, we can visualize some descritive statistics. For example, we are able to see how many instances occur in the dataset, the mean and standard deviation for every variable and more summary statistics.
            The table below shows these statistics for the {option.split('/')[-1]} dataset.  
            """)
        st.dataframe(dataset.describe())

    if 'Value' in dataset.columns:
        fig = plot_columns(dataset,'act')
        st.pyplot(fig)
        fig = plot_columns(dataset,'tar')
        st.pyplot(fig)
            

        
