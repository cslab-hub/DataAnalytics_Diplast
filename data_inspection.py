import streamlit as st
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

import os 
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
                # print(filename)
                # data = pd.read_csv(os.path.join(roots,filename))
                found_files.append(os.path.join(roots,filename))
                # print(found_files)
                # print(sys.platform)
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

def return_preprocessing():
    st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)
    st.markdown(""" ## First inspect your data""")
    
    st.markdown("""
        While the data should have already been checked with the tips from the Data Validation [tool](https://cslab-hub-data-validation-main-bx6ggw.streamlitapp.com/), we should again check if indeed our data now is correct.
        In the following dropdown box, select the dataset that you want to view.
        This dataset should be put into the data folder where this software runs from.
    """)

    st.markdown("""
    A good dataset is:
    - In the CSV format (Comma Separated Values) 
    - Has as first column a Date, which we wel automatically use as an index.
    - Does not contain too many variables (keep it below 20)

    Example of a good dataset:
    """)


    st.write(pd.DataFrame({
        'Time': ['21-12-21 10:00:00', '21-12-21 10:00:01','21-12-21 10:00:02','21-12-21 10:00:03'],
        'Sensor1': [10, 10, 11, 10],
        'Sensor2': [14,15,14,14],
        'Sensor3': [100.1,100.3,100.2,100.0],
        'Sensor4': [90.1,89.4,88.3,90]
    }).style.set_table_styles([
                {"selector":"caption",
                "props":[("text-align","center"),("caption-side","top")],
                },                
                {"selector":"td",
                "props":[("text-align","center")],
                },
                {"selector":"",
                "props":[("margin-left","auto"),("margin-right","auto")],
                }

                ]).set_caption("Table 1: Dataset.")\
                .format(precision=2)\
                .hide(axis='index')\
                .to_html()           
                , unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    For more information, we advise you to check our data validation tool that can be accessed [here](https://cslab-hub-data-validation-main-bx6ggw.streamlitapp.com/)
    """)
    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split(string_splitter)[-1], key=1)
    if option == "Select a Dataset":
        st.stop()
    
    dataset = pd.read_csv(option)
    if 'Time' in dataset.columns:
        dataset['Time'] = pd.to_datetime(dataset['Time'])
        dataset = dataset.set_index('Time')
    if 'TIME' in dataset.columns:
        dataset['TIME'] = pd.to_datetime(dataset['TIME'])
        dataset = dataset.set_index('TIME')

    st.write("""
        The dataset below shows the first 10 inputs. Based on this information, you are able to see the general outline of the dataset, e.g., the amount of columns and some values.
        """)
    st.success('Tip: Hold shift while scroling to see all variables!')
    st.dataframe(dataset.head(10))
    st.write(f'The dataset contains a total of {len(dataset)} rows and {len(dataset.columns)} columns.')

    st.markdown(""" ## Data Types""")
    dtype_df = dataset.dtypes.value_counts().reset_index()

    dtype_df.columns = ['VariableType','Count']
    dtype_df['VariableType'] = dtype_df['VariableType'].astype(str)

    fig, ax = plt.subplots()
    fig.set_size_inches(5,3)
    ax.bar(dtype_df['VariableType'],dtype_df['Count'])
    
    st.write("""
        The plot below indicates which datatypes are in the dataset, based on the different columns (variables). The barplot shows the different datatypes and how many of these columns are spotted in the dataset.
        """)
    st.markdown("""
    These are the datatypes that are common in industrial data:
    - float32, float64, int32, int64: Numbers
    - String (str), objects, datetimes: Words, Categories
    - Booleans: True or False
    
    The following are present in your dataset:
    """)

    st.pyplot(fig)

    st.write(f"""
       To get some more information regarding the dataset, we can visualize some descriptive statistics. For example, we are able to see how many instances occur in the dataset, the mean and standard deviation for every variable and more summary statistics.
        The table below shows these statistics for the {option.split(string_splitter)[-1]} dataset.  
        """)
    st.dataframe(dataset.describe())

    

    st.markdown("""
    If the data visible here does not correspond to what you expect, go back a few steps.
    Your data should consist of seperate variables, which starts with a variable containing the notion of time.
    """)
