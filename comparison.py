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

# dataframe = pd.DataFrame(np.random.randint(80,100,size=(100, 4)))
# dataframe.columns = ['var1','var2','var3','var4']
# dataframe.to_csv('data/dataset.csv', index=False)



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

# print(data)
def return_comparison():

    col1, col2 = st.columns(2)

    with col1:
        
        option = st.selectbox(
            'Which dataset do you want to view?',
            # ['Select dataset',(i for i in data)], format_func= lambda x:  str(x).split('/')[-1], key=1)

            (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
        # if option == "Select a Dataset":
        #     st.stop()
        plot = pd.read_csv(option)
        print(plot)

        option2 = st.selectbox(
            'Which variable do you want to view?',
            (i for i in plot.columns), key=2)
        # if option2 == "Select a Dataset":
        #     st.stop()
        # fig = plt.plot(plot[option2])
        
        fig, ax = plt.subplots()
        ax.plot(plot[option2])
        st.pyplot(fig)
        
        # st.pyplot(fig)
        
        
        
        
        
    with col2:
        
        option3 = st.selectbox(
            'Which dataset do you want to view?',
            (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=3)
        plot = pd.read_csv(option3)
        print(plot)

        option4 = st.selectbox(
            'Which variable do you want to view?',
            (i for i in plot.columns), key=4)
        # fig = plt.plot(plot[option2])
        
        fig, ax = plt.subplots()
        ax.plot(plot[option4])
        st.pyplot(fig)
