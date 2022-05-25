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
    return found_files,cwd

data,cwd = data_loader()
data.insert(0,'Select a Dataset')

# print(data)
def return_report():
    st.header('Data Report')
    st.markdown("""The final module of our tool provides a general overview of your dataset for every specific variable.
    This overview gives you a variety of insights such as alerts in the data, a histogram that shows the distribution of values and also the 5 most common and most extreme values.
    With this information, you can get an indication of the measurements within your variables to check if the values align with the desired behavior of the variable. Also, this""")
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







    profile = ProfileReport(dataset, title="Pandas Profiling Report", minimal = True)
    st_profile_report(profile)

    option3 = st.selectbox(
        "Would you like to download this data report?",
        ["Select an option","Yes","No"])

    if option3 == "Yes":
        profile.to_file("data_report.html")
        st.markdown(f"Your report is downloaded and can be found in the folder{cwd}")


    