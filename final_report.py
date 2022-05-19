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
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                print(filename)
                # data = pd.read_csv(os.path.join(roots,filename))
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

# print(data)
def return_report():

    # ls = ['']
    option = st.selectbox(
        'Which dataset do you want to view?',
        # ['',i for i in data], format_func=lambda x: 'Select an option' if x == '' else x)
        # ['Select Dataset',[i for i in data]], format_func= lambda x:  str(x).split('/')[-1], key=1)
        (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
    if option == 'Select a Dataset':
        st.stop()
    dataset = pd.read_csv(option)
    print(dataset.dtypes)

    st.write('You selected:', option)
    # dataset = dataset.select_dtypes('float64')





    st.title('Profiling')

    profile = ProfileReport(dataset, title="Pandas Profiling Report", minimal = True)
    st_profile_report(profile)
