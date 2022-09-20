import streamlit as st
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import sys 
# dataframe = pd.DataFrame(np.random.randint(80,100,size=(100, 4)))
# dataframe.columns = ['var1','var2','var3','var4']
# dataframe.to_csv('data/dataset.csv', index=False)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""


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
                # print(filename)
                # data = pd.read_csv(os.path.join(roots,filename))
                found_files.append(os.path.join(roots,filename))
                # print(found_files)
                # print(sys.platform)
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

def return_comparison():
    st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        div.block-container {padding-top:0rem;}
    </style>
    """, unsafe_allow_html=True)
    
    st.header('Compare two time series')
    st.markdown("""
        To check what happened within every individual time serie and to compare them with each other, we can plot them on this tab and see what happened throughout the process.
        """)
    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown('### Input')
        option = st.selectbox(
            'Which dataset do you want to view?',
            # ['Select dataset',(i for i in data)], format_func= lambda x:  str(x).split('/')[-1], key=1)

            (i for i in data), format_func= lambda x:  str(x).split(string_splitter)[-1], key=1)
        if option == "Select a Dataset":
            st.stop()

        plot = pd.read_csv(option)
        print(f'dataset shapes = {plot.shape}')
        plot = plot.select_dtypes(include='number')
        print(f'dataset shapes after = {plot.shape}')
        data_columns = [i for i in plot.columns]
        data_columns.insert(0, 'Select Variable')
            
        option2 = st.selectbox(
            'Which variable do you want to view?',
            (i for i in plot.columns), key=3)
        
        st.markdown('### Output')
        fig, ax = plt.subplots()
        ax.plot(plot[option2])
        st.pyplot(fig)
        
        
    with col2:
        
        st.markdown('### Input')
        option3 = st.selectbox(
            'Which dataset do you want to view?',
            # ['Select dataset',(i for i in data)], format_func= lambda x:  str(x).split('/')[-1], key=1)

            (i for i in data), format_func= lambda x:  str(x).split(string_splitter)[-1], key=4)
        if option3 == "Select a Dataset":
            st.stop()

        plot = pd.read_csv(option3)
        print(f'dataset shapes = {plot.shape}')
        plot = plot.select_dtypes(include='number')
        print(f'dataset shapes after = {plot.shape}')
        data_columns = [i for i in plot.columns]
        data_columns.insert(0, 'Select Variable')
            
        option4 = st.selectbox(
            'Which variable do you want to view?',
            (i for i in plot.columns), key=6)
        
        st.markdown('### Output')
        fig, ax = plt.subplots()
        ax.plot(plot[option4])
        st.pyplot(fig)
