import streamlit as st
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

import os 
import stumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sys
import datetime

if sys.platform == 'win32':
    string_splitter = '\\'
else:
    string_splitter = '/'

# print(sys.platform, string_splitter)
def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                print(filename)
                # data = pd.read_csv(os.path.join(roots,filename))
                found_files.append(os.path.join(roots,filename))
                print(found_files)
                print(sys.platform)
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

def return_preprocessing():

    st.header('First inspect your data')
    st.markdown("""
        While the data should have already been checked with the tips from the Data Validation [tool](https://cslab-hub-data-validation-main-bx6ggw.streamlitapp.com/), we should again check if indeed our data now is correct.
        In the following dropdown box, select the dataset that you want to view.
        This dataset should be put into the data folder where this software runs from.
    """)


    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split(string_splitter)[-1], key=1)
    if option == "Select a Dataset":
        st.stop()
    
    dataset = pd.read_csv(option)
    if 'Date' in dataset.columns:
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        dataset = dataset.set_index('Date')
    if 'TIME' in dataset.columns:
        dataset['TIME'] = pd.to_datetime(dataset['TIME'])
        dataset = dataset.set_index('TIME')

    st.write("""
        The dataset below shows the first 10 inputs. Based on this information, you are able to see the general outline of the dataset, e.g., the amount of columns and some values. 
        """)
    st.table(dataset.head(10))
    st.write(f'The dataset contains a total of {len(dataset)} rows and {len(dataset.columns)} columns.')
    dtype_df = dataset.dtypes.value_counts().reset_index()

    dtype_df.columns = ['VariableType','Count']
    dtype_df['VariableType'] = dtype_df['VariableType'].astype(str)

    fig, ax = plt.subplots()
    fig.set_size_inches(7,5)
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

    

    st.markdown("""
    If the data visible here does not correspond to what you expect, go back a few steps.
    Your data should consist of seperate variables, which starts with a variable containing the notion of time.
    Each variable should be measured at the moment in time, such as shown here:
    """)
