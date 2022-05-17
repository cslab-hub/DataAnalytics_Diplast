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


def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

def return_preprocessing():

    st.title('First inspect your data')
    st.markdown("""
        
    """)

    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
    if option == "Select a Dataset":
        st.stop()
    
    dataset = pd.read_csv(option)


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
