## Feature selection
from select import select
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 
import os
import sys
import matplotlib.dates as mdates

# import seaborn as sns

#%%

if sys.platform == 'win32':
    string_splitter = '\\'
else:
    string_splitter = '/'


def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                # print(filename)
                # data = pd.read_csv(os.path.join(roots,filename))
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

def return_feature_selection():
    st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)
    st.header('Create Correlation plots')

    st.markdown("""Correlation is a statistical term which refers to how close two variables have a linear relationship to each other.
    Variables that have a linear relationship tell us something about how these variables move in relation to each other.
    For example, if the linear relationship is very high, it means that if one of the variables increases, the other will also increase with a similar amount.
    However, a high linear relationship means that both variables explain the dataset almost equally. Therefore, for further analysis we only take one of these variables for our task. 
    In other words, if two variables have a high correlation, we can drop on of the two! 
    This results in a more streamlined dataset which means less computational resources are needed for the task. But first, let's check the correlations between the variables of your dataset:     
    """)
    st.markdown('### Input')
    option = st.selectbox(
    'Which dataset do you want to view?',
    # ['Select dataset',(i for i in data)], format_func= lambda x:  str(x).split('/')[-1], key=1)
    (i for i in data), format_func= lambda x:  str(x).split(string_splitter)[-1], key=1)

    if option == "Select a Dataset":
        st.stop()
    st.markdown('### Output')
    dataset = pd.read_csv(option)
    print(f'dataset shapes = {dataset.shape}')
    dataset = dataset.select_dtypes(include='number')
    print(f'dataset shapes after = {dataset.shape}')
    
    
    # dataset = dataset._get_numeric_data
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    dataset = dataset.select_dtypes(include=numerics)

    st.dataframe(dataset.head(5).style.format(precision=2)\
    .set_table_styles([
                    {"selector":"caption",
                    "props":[("text-align","center")],
                    }

                    ], overwrite=False)\

        .set_caption('Table 1.'))


            

    st.markdown('''
    In the table above, we see the first 5 observations for every variable in the dataset. Based on this information, we can see the difference between the values for every variable and already make an asumption of the measurements that occurred in the dataset.
    ''')
    st.markdown('''
    Now we visualize the correlations between the variables in the dataset. 
    In Table 2, we see the correlations between the variables. A red colored surface means a high positive correlation, a blue surface indicates a negative correlation. 
    On the diagonal we see a perfect red correlation of 1, which makes sense since we see this correlation exists between the same variable on both x and y-axis.    
    Our advice: if we look at Table 2, we recommend that all variables that have a high correlation (≈ 0.9 or ≈ -0.9 and above) can be removed from the dataset. With the selector below, you are able to remove variables to see how this influences correlations.
    ''')

    option_list = [i for i in dataset.columns]
    st.markdown('### Input')
    option7 = st.multiselect('Which variable could be removed from the dataset?',option_list)

    st.markdown('### Output')
    if len(option7) == 0:
        corr = dataset.corr().round(2)
        corr.style.background_gradient(cmap='coolwarm')
        st.dataframe(corr.style.background_gradient(cmap='coolwarm')\
        .format(precision=2)\
        .set_table_styles([
                        {"selector":"caption",
                        "props":[("text-align","center")],
                        }

                        ], overwrite=False)\
            

            .set_caption('Table 2.'))

        # st.subheader('Interpet the correlations')
        # st.stop()

    if len(option7) != 0:
        dataset = dataset.drop(option7,axis=1)
        corr = dataset.corr().round(2)
        corr.style.background_gradient(cmap='coolwarm')
        st.dataframe(corr.style.background_gradient(cmap='coolwarm')\
        .format(precision=2)\
        .set_table_styles([
                        {"selector":"caption",
                        "props":[("text-align","center")],
                        }

                        ], overwrite=False)\
            

            .set_caption('Table 2.'))

    # st.subheader('Interpet the correlations')
        # st.stop()



    # corr = dataset.corr().round(2)
    # corr.style.background_gradient(cmap='coolwarm')
    # st.table(corr.style.background_gradient(cmap='coolwarm')\
    # .format(precision=2)\
    # .set_table_styles([
    #                 {"selector":"caption",
    #                 "props":[("text-align","center")],
    #                 }

    #                 ], overwrite=False)\
        

    #     .set_caption('Table 2.'))

    # st.subheader('Interpet the correlations')

    # st.markdown('''
    # In Table 2, we see the correlations between the variables. A red colored surface means a high positive correlation, A blue surface indicates a negative correlation. 
    # On the diagonal we see a perfect red correlation of 1, which makes sense since we see this correlation exists between the same variable on both x and y-axis.
    
    # Our advice: if we look at Table 2, we recommend that all variables that have a high correlation (≈ 0.9 or ≈ -0.9 and above) can be removed from the dataset.
    # ''')

    st.header('Principal Component Analysis')
    st.markdown('''
    Another method to reduce the amount of variables in your dataset (e.g., dimensionality reduction) is by performing Principal Component Analysis (PCA). 
    This technique uses a set of large variables by combining them together to retain as much as information as possible.
    PCA dates back to the 1990's and is one of the most widely used analysis techniques in Data Science.
    The module will determine a list with variables where the most important variable is on the top followed by variables with decreasing importance.
    However, keep in mind that variable could be 'so heavy' in terms of importance, that is not logical to keep it in the dataset. 
    Therefore, try to interpret if a specific variable is not too informative, because that could mean that it is so trivial that you actually would want to remove it from the dataset entirely.
    
    Let's see what PCA can tell us about the dataset. For this, we can choose to use all variables. However, if there is a clear target variable. For example, the class that you want to predict.
    We recommend to remove this one from the dataset. You can choose your preference for this in the box below:
    ''')

    from sklearn.preprocessing import StandardScaler # for standardizing the Data
    from sklearn.decomposition import PCA # for PCA calculation
#
    X = dataset
    option_list = [i for i in X.columns]
    option_list.insert(0,'select something or keep all variables')
    st.markdown('### Input')
    option2 = st.multiselect('If your dataset contains Target variable(s), which are/is it? We will separate these from the rest of the dataset',option_list)
    # if len(option2) == 0:
        # st.stop()

    if len(option2) != 0:
        X = X.drop(option2,axis=1)

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
    most_important_names2 = list(dict.fromkeys(most_important_names))
    st.markdown('### Output')
    for i,j in enumerate(most_important_names2):
        st.write(f"{i + 1}. Most important variable = {j}")
    
    st.subheader('Interpet the Principal Component Analysis')
    st.markdown(''' 
    After executing PCA, we see that certain variables are marked as most important. These variables explain most of the variance within the dataset. Therefore, it seems that something is going on in these variables. 

    Our advice: We recommend to plot the variables that have the highest importance based on PCA. In this way, you are able to assess if these variables are actually usable in the follow-up analysis. If these variables seem
    to be less relevant as PCA suggests, we recommend to remove them from the dataset and re-run the PCA again. In this way, a more interesting and more informative set of variables remain, which is essential for data analytics.
    plotting the most important variables can be done below:
    ''')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### Input')
        plot = pd.read_csv(option)
        print(f'dataset shapes = {plot.shape}')
        plot = plot.select_dtypes(include='number')
        print(f'dataset shapes after = {plot.shape}')
        plot = plot.drop(option7,axis=1)

        # print(plot)
        option4 = st.selectbox(
            'Which variable do you want to view?',
            (i for i in plot.columns), key=3)

        st.markdown('### Output')
        
        fig, ax = plt.subplots()
        ax.plot(plot[option4])
        st.pyplot(fig)
        
        # st.pyplot(fig)
        
        
        
        
        
    with col2:
        st.markdown('### Input')

        plot = pd.read_csv(option)
        print(f'dataset shapes = {plot.shape}')
        plot = plot.select_dtypes(include='number')
        print(f'dataset shapes after = {plot.shape}')
        plot = plot.drop(option7,axis=1)
        
        option6 = st.selectbox(
            'Which variable do you want to view?',
            (i for i in plot.columns), key=5)
        # fig = plt.plot(plot[option2])
        

        st.markdown('### Output')
        
        fig, ax = plt.subplots()
        ax.plot(plot[option6])
        st.pyplot(fig)

            

    
    st.markdown("""
    There is now the possiblity to download the final dataset for further analysis, you can then put the file in the 'data' folder again to use it.
    """)
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return dataset.to_csv().encode('utf-8')
    csv = convert_df(dataset)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='data/preprocessed.csv',
        mime='text/csv',
    )

    