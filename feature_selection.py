## Feature selection
from select import select
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 
import os

# import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

#%%


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

def return_feature_selection():
    st.title('Create Correlation plots')

    st.markdown("""Correlation is a statistical term which refers to how close two variables have a linear relationship to each other.
    Variables that have a linear relationship tell us less about our dataset, since measuring one tells you something about the other.
    In other words, if two variables have a high correlation, we can drop on of the two!
    """)

    option = st.selectbox(
    'Which dataset do you want to view?',
    # ['Select dataset',(i for i in data)], format_func= lambda x:  str(x).split('/')[-1], key=1)
    (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)

    if option == "Select a Dataset":
        st.stop()

    dataset = pd.read_csv(option)
    # dataset = dataset._get_numeric_data
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    dataset = dataset.select_dtypes(include=numerics)

    st.table(dataset.head(5))
    corr = dataset.corr().round(2)
    corr.style.background_gradient(cmap='coolwarm')
    st.table(corr.style.background_gradient(cmap='coolwarm')\
    .format(precision=2)\
    .set_table_styles([
                    {"selector":"caption",
                    "props":[("text-align","center")],
                    }

                    ], overwrite=False)\
        

        .set_caption('Table 2.'))

    st.title('PCA Analysis')
    st.markdown('''
    A technique to reduce the dimensionality of your dataset is by performing Principal Component Analysis.
    PCA uses a set of large variables by combining them together to retain as much as information as possible.
    PCA dates back to the 1990's and is one of the most widely used analysis techniques in Data Science.
    ''')

    from sklearn.preprocessing import StandardScaler # for standardizing the Data
    from sklearn.decomposition import PCA # for PCA calculation
#
    X = dataset
    option_list = [i for i in X.columns]
    option_list.insert(0,'select something or keep all variables')
    option = st.selectbox('Which variable resprents the labels for the given dataset? We will separate this variable from the rest of the dataset ', option_list, key=1)
    if option != 'select something or keep all variables':
        X = X.drop(option, axis=1)
    else:
        pass


    from sklearn.preprocessing import StandardScaler # for standardizing the Data

    sc = StandardScaler() # creating a StandardScaler object
    X_std = sc.fit_transform(X) # standardizing the data
    aim_target = 0.98
    pca = PCA(n_components = aim_target)
    X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions

    n_pcs= pca.n_components_ # get number of component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = X.columns
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    most_important_names = list(dict.fromkeys(most_important_names))
    
    for i,j in enumerate(most_important_names):
        st.write(f"{i + 1}th most important variable = {j}")