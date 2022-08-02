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
import pmdarima as pm
from pmdarima.model_selection import train_test_split


# dataframe = pd.DataFrame(np.random.randint(0,100,size=(100, 4)))
# dataframe.columns = ['var1','var2','var3','var4']
# dataframe.to_csv('data/dataset2.csv', index=False)



def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()





# Fit your model


# make your forecasts



def return_forecast():

    col1, col2, col3 = st.columns(3)

    with col2:
        option = st.selectbox(
                'Which dataset do you want to view?',
                (i for i in data),format_func= lambda x:  str(x).split('/')[-1], key=1)
        dataset = pd.read_csv(option)

        option2 = st.selectbox(
                'Which variable do you want to predict?',
                (i for i in dataset.columns), key=1)
        variable = dataset[option2]











        def calculate_forecast():
            train, test = train_test_split(variable, train_size=0.8)
            print(train.shape)
            model = pm.auto_arima(train)


            forecasts = model.predict(test.shape[0])  # predict N steps into the future
            print(forecasts.shape)

            x = np.arange(variable.shape[0])
            
            # fig, axs = plt.subplots(1)
            # axs[0] = plt.plot(x[:train.shape[0]], train, c='blue')
            # axs[0] = plt.plot(x[train.shape[0]:], forecasts, c='green')


       

            # figure = plt.gca()


            # plt.plot(x[150:], forecasts, c='green')

            return x,train,forecasts

        x , train, forecasts = calculate_forecast()
        fig, ax = plt.subplots()
        # fig.set_size_inches(24.5, 16.5)
        ax.plot(x[:train.shape[0]], train, c='blue')
        ax.plot(x[train.shape[0]:], forecasts, c='green')    
        # ax.plot(fig)

        st.pyplot(fig)

        
        # for col in dataset.columns:
        #     dataset.loc[dataset.sample(frac=0.1, random_state=3).index, col] = np.nan
        # st.table(dataset.head(10))


        # option2 = st.selectbox(
        #         'Which action do you want to take?',
        #         ('Delete','Infer'), key=2)
        
        # if option2 == 'Delete':
        #     st.write('Result when deleting the rows with faulthy data')
            
        #     st.table(dataset.dropna().head(10))
            
        # if option2 == 'Infer':
        #     st.write('Result when inferring the values from faulthy data')
            
        #     # st.dataframe(dataset.dropna())
        #     st.table( dataset.fillna(dataset.rolling(4,min_periods=1).mean()).head(10) )