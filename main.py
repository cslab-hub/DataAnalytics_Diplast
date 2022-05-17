import streamlit as st
st.set_page_config(page_title="Di-Plast Data Analytics Tool ",layout="wide")
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

from home import *
from matrix_profile import *
from comparison import *
from data_inspection import *
# from forecast import *
from ts_classifier import *
from final_report import *
from feature_selection import *
# from EPA import *
import base64
    
# st.sidebar.markdown(Logo_html, unsafe_allow_html=True)
st.sidebar.image('images/di-plast-logo.png', use_column_width=True)
st.sidebar.title("Select a Module")
# st.sidebar.header("Each tool performs a different task.")





add_selectbox = st.sidebar.radio(
    "Choose one of the analytics options:",
    ("Home",'Data Inspection','Data Comparison',"Feature Selection","Classifier","Final Data Report"),format_func= lambda x: 'Home' if x == 'Home' else f"{x}",help="Please select one of the options that aligns with your analytics needs."
    
)         




#! Home page
if add_selectbox == 'Home':
    return_homepage()
    
    
#! Page for the file format
# if add_selectbox == 'Matrix Profile':
#     return_matrix_profile()

if add_selectbox == 'Comparison':
    return_comparison()
    
if add_selectbox == 'Data Inspection':
    return_preprocessing()

if add_selectbox == 'Feature Selection':
    return_feature_selection()

# if add_selectbox == 'EPA':
#     return_EPA()

if add_selectbox == 'Classifier':
    return_classifier()

if add_selectbox == 'Final Data Report':
    return_report()
    
# This removes the copyright of how the page is made
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)



st.sidebar.image('images/JADS_logo_RGB.png', use_column_width=True)

st.sidebar.caption("[Bug reports and suggestions welcome ](mailto:j.o.d.hoogen@jads.nl)")