import streamlit as st
st.set_page_config(page_title="Di-Plast Data Analytics Tool ",layout="wide")
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 
from home import *
from comparison import *
from data_inspection import *
from ts_classifier import *
from final_report import *
from feature_selection import *
from EPA import *
import base64



st.sidebar.image('images/di-plast-logo.png', use_column_width=True)

st.sidebar.title("Select a Module")


add_selectbox = st.sidebar.radio(
    "Choose one of the analytics options:",
    ("Home",'Data Inspection','Data Comparison',"Feature Selection","Classifier","Exploratory Pattern Analytics","Final Data Report"),format_func= lambda x: 'Home' if x == 'Home' else f"{x}",help="Please select one of the options that aligns with your analytics needs."
    
)               








#! Home page
if add_selectbox == 'Home':
    return_homepage()

if add_selectbox == 'Data Comparison':
    return_comparison()
    
if add_selectbox == 'Data Inspection':
    return_preprocessing()

if add_selectbox == 'Feature Selection':
    return_feature_selection()

if add_selectbox == 'Classifier':
    return_classifier()


if add_selectbox == 'Exploratory Pattern Analytics':
    return_EPA()


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




st.sidebar.markdown("***")

st.sidebar.image('images/JADS_logo.png', use_column_width=True)


st.sidebar.caption("[Bug reports and suggestions welcome ](mailto:j.o.d.hoogen@jads.nl)")


