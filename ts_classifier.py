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

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import scipy as sp
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn import metrics
import seaborn as sns
import sys

if sys.platform == 'win32':
    string_splitter = '\\'
else:
    string_splitter = '/'

classifier_list = ["RandomForestClassifier","KNeighborsClassifier"]


def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                print(filename)
                found_files.append(os.path.join(roots,filename))
    return found_files

data_files = data_loader()
data_files.insert(0,'Select a Dataset')
# print(data)
def return_classifier():

    st.header("Classification Tasks")
    st.markdown("""
        One of the most used applications in the field of Data Science is Classification. The idea behind classification is to predict the class, also called target variable, of the given data points. 
        For example, spam detection by your e-mail client can be seen as a classification problem. In this case, we talk about a binary classification problem since there are only two outcomes,
        e.g., Spam (0) and Not Spam (1). A classifier, which is an algorithm used to learn from the data how the input variables relate to the specific class. Generally speaking,
        the data will be divided into a train and test set. In the e-mail use case, known spam and non-spam e-mails are used as training data. If the classifier is trained accuractely, 
        it can be used to detect unknown e-mails. The test data, which is separated in an earlier stage, is used to validate the performance of a given classifier. If the performance is good enough on 
        unseen data (the test data), we can assume that with future observations, the classifier will be able to make the distinction between classes. 

        For this tool, we are mainly interested in the performance of the classifier so that we can determine which variable contributed the most to this outcomes. Thus, when we achieve a high accuracy score,
        we can identify which variables are responsible for this particular score. With this information, you are able to do a further investigation on these variables in your production process. This could help
        with optimizing the process of increasing the uptake of recycled material. 

        As a first step, we need to identify which dataset you want to inspect, which can be chosen in the box below:
        """)


    # st.header('Choose your data for classification (if target variable exists)')



    option = st.selectbox(
        'Which dataset do you want to use for your classification problem?',
        (i for i in data_files),format_func= lambda x:  str(x).split(string_splitter)[-1], key=1)

    if option == "Select a Dataset":
        st.stop()
    dataset = pd.read_csv(option)

    st.table( dataset.head(5))
    st.markdown('''
    In the table above, we see the first 5 observations for every variable in the dataset. Based on this information, we can see the difference between the values for every variable and already make an asumption of the measurements that occurred in the dataset. Also, we might be able to identify the target variable.
    ''')

    st.markdown('''
    Now that we've seen the data format, we will choose the target variable from the dataset in the box below:
    ''')
    remove_option_list = [i for i in dataset.columns]
    to_be_removed = st.multiselect('Which variable could be removed from the dataset?',remove_option_list)
    dataset = dataset.drop(to_be_removed,axis=1)

    st.subheader('Target variable selection')
    st.markdown('''
    Now that we've seen the data format and removed unwanted variables, we will choose the target variable from the dataset in the box below:
    ''')

    # option2 = st.selectbox(
    #     'Which variable resprents the labels for the given dataset? We will separate this variable from the rest of the dataset ', 
    #     (i for i in dataset.columns), key=1)
    # target_data = dataset[option2]
    # dataset = dataset.drop(columns = [option2])

    option_list = [i for i in dataset.columns]
    option2 = st.multiselect('Which variable resprents the target variable for the given dataset? We will separate this variable from the rest of the dataset',option_list)
    if len(option2) == 0:
        st.stop()

    if len(option2) == 1:
        target_data = dataset[option2]
        dataset = dataset.drop(option2,axis=1)

    if len(option2) > 1:
        option_ifmore = st.selectbox(
        'Which variable resprents the labels for the given dataset? We will separate this variable from the rest of the dataset ', 
        (i for i in option2), key=1)
        target_data = dataset[option_ifmore]
        dataset = dataset.drop(option_ifmore,axis=1)

    st.markdown(f""" Now that we have identified which data we will use to classify the labels represented by {option2}, we can start to proceed with making a classifier. The next step is to choose the right classifier for the job.
    For this, we will give you some additional information for choosing the right classifier.

    To get a general idea of what a classifier can do, we give you the option to choose between several algorithms:
    - Decision Tree (Categorical)
    - Random Forest (Categorical)
    - Logistic Regression (Categorical)
    - Random Forest Regressor (Continuous)
                
    The first three options are Algorithms that are used for Categorical variables.
    Categorical variables are variables that can take on one of a limited set of values, belonging to a particular group.
    For example, types of plastic used: PP, PET etc.
    Continuous variables are variables that are obtained by measuring or counting something, and can therefore take on all real values.
    Examples are the pressure and temperature in an extruder machine. 
                """) 

    option3 = st.selectbox(
        'Which classifier do you want to use?',
        ('Select an Algorithm','Decision Tree', 'Random Forest','Logistic Regression','Random Forrest Regressor'), key= 1)

    x_train, x_test, y_train, y_test= train_test_split(dataset, target_data,
                                                   test_size= 0.2,
                                                   shuffle= True, #shuffle the data to avoid bias
                                                   random_state= 0)


    scaler= StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)
    st.write(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')
    st.write(f'Number of columns {len(dataset.columns)} and rows {len(dataset)}')

    
    def classification(classifier,param_grid):
        cv_classifier = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 2, verbose = 100, n_jobs = -1)
        cv_classifier.fit(scaled_x_train, y_train)        
        
        sorted(cv_classifier.cv_results_.keys())
        cv_best = cv_classifier.best_estimator_
        predictions = cv_best.predict(scaled_x_test)



        mat = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d')
        ax.set_xlabel('True label')
        ax.set_ylabel('Predicted label')

        classifier_accuracy = round(metrics.accuracy_score(y_test, predictions),4)

        st.write(f"Based on the analysis using the {option3} algorithm, the classifier was able to predict the right class with an accuracy of {classifier_accuracy} and parameter settings {cv_best}")
        # st.pyplot(fig)



        if option3 != 'Logistic Regression':

            feature_importance = cv_classifier.best_estimator_.feature_importances_
            print(feature_importance)
            f_i = list(zip(features_dataset,feature_importance)) 
            f_i.sort(key = lambda x : x[1])
            fig2, ax2 = plt.subplots()
            ax2 =  plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
            if len(features_dataset) > 10:
                ax2 = plt.tick_params(axis="y", labelsize=4)
            # ax2.set_yticklabels(fontsize=16)
            return fig2,fig
        else:
            return fig
    
    def regression(regressor,param_grid):
        from sklearn.metrics import mean_squared_error

        cv_regressor = GridSearchCV(estimator=regressor, param_grid=param_grid, cv= 2, verbose = 100, n_jobs = -1)
        cv_regressor.fit(scaled_x_train, y_train)        
        
        # sorted(cv_regressor.cv_results_.keys())
        cv_best = cv_regressor.best_estimator_
        predictions = cv_best.predict(scaled_x_test)

        feature_importance = cv_regressor.best_estimator_.feature_importances_
        # print(feature_importance)
        st.write(f'we see an MSE of ={mean_squared_error(y_test, predictions):.4f}')


        f_i = list(zip(features_dataset,feature_importance)) 
        f_i.sort(key = lambda x : x[1])
        fig2, ax2 = plt.subplots()
        ax2 =  plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
        if len(features_dataset) > 10:
            ax2 = plt.tick_params(axis="y", labelsize=4)
        return fig2

    if option3 == "Decision Tree":
        param_grid = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
        classifier = DecisionTreeClassifier()
        features_dataset = dataset.columns
        fig2,fig = classification(classifier,param_grid)
        st.pyplot(fig2)
        st.pyplot(fig)



    if option3 == "Random Forest":

        classifier = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, random_state=1) 

        param_grid = { 
            'n_estimators': range(20,100,10),
            'max_features': ['auto', 'sqrt', 'log2']
            }
        features_dataset = dataset.columns
        fig2,fig = classification(classifier,param_grid)
        st.pyplot(fig2)
        st.pyplot(fig)


    if option3 == "Logistic Regression":

        classifier = LogisticRegression()
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }

        features_dataset = dataset.columns
        fig = classification(classifier,param_grid)
        st.pyplot(fig)

    if option3 == "Random Forrest Regressor":
        from sklearn.ensemble import RandomForestRegressor

        classifier = RandomForestRegressor()
        param_grid = { 
            'n_estimators': range(20,200,10),
            'max_features': ['auto', 'sqrt', 'log2']
            }

        features_dataset = dataset.columns
        fig = regression(classifier,param_grid)
        st.pyplot(fig)


        



            
            


