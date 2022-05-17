# %%
import numpy as np
import pandas as pd
import re
import streamlit as st
from PIL import Image 
import os 
import matplotlib.pyplot as plt 
import datetime as dt
import sys 
import csv
import sd4py
import sd4py_extra
import warnings
import re
import io
import copy
import datetime


# %%


# %%

def data_loader():
    found_files = []
    found_files2 = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

def get_img_array_bytes(fig):

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=150)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=150)
    io_buf.seek(0)
    img_bytes = io_buf.getvalue()
    io_buf.close()

    return img_arr, img_bytes

# %%
def return_EPA():

    st.title('Exploratory Pattern Analytics (EPA)')
    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
    if option == 'Select a Dataset':
        st.stop()

    @st.cache
    def get_data():

        # data = pd.read_csv(option,index_col=0)
        data = pd.read_csv(option)


        assert len(np.unique(data.index)) == len(data.index), "Index column contains duplicate values"

        if data.index.dtype == 'object' or data.index.dtype.name == 'category':

            try:

                data.index = pd.to_datetime(data.index)
            
            except ValueError:

                pass
        
        return data


    '''
    ## Settings 
    '''

    dataset_production = get_data()

    st.markdown("""
    Here you should see the time variable as the first column on the left.
    IF NOT, go back a few steps and check the data validation tool again, or inspect your dataset.
    The first variable should be a timestamp variable that acts as an unique identifier of each row/observation.
    """)
    st.table(dataset_production.head(5))

    target_options = list(dataset_production.columns)
    target_options.insert(0, 'Choose the target variable')
    target = st.selectbox('Target variable: ', target_options)

    if target == 'Choose the target variable':
        st.stop()

    target_nominal = False

    if dataset_production.loc[:,target].dtype == 'object' or dataset_production.loc[:,target].dtype == 'bool' or dataset_production.loc[:,target].dtype.name == 'category':

        target_nominal = True

    value = None

    if target_nominal:

        value_options = list(np.unique(dataset_production[target]))
        value_options.insert(0, 'Choose the target value')
        value = st.selectbox('Target value: ', value_options)


    qf_options = ["Larger subgroups", "Smaller subgroups"]
    qf_options.insert(0, 'Choose the quality function')
    qf = st.selectbox('Quality function: ', qf_options)

    minsize = st.number_input("Minimum size for subgroups: ", step=1, value=10)

    jaccard_threshold = st.slider("Suppress 'duplicate' subgroups that overlap with previous subgroups by more than: ", 0.0, 1.0, 0.95)

    if target_nominal: 
        if value == 'Choose the target value':
            st.stop()
    if qf == 'Choose the quality function':
        st.stop()

    qf = {"Larger subgroups":"ps", "Smaller subgroups":"bin"}[qf]

    '''
    ## Top subgroups
    '''
    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_subgroups():

        return sd4py.discover_subgroups(dataset_production, target, target_value=value, qf=qf, k=100, minsize=minsize)

    subgroups = get_subgroups()

    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_bootstrap():

        frac = 1.0

        if len(dataset_production) > 13747: ## 13747 / log_2(l3747) = 1000

            frac = 1 / np.log2(len(dataset_production))

        else:

            frac = min(frac, 1000 / len(dataset_production))

        if target_nominal:

            subgroups_bootstrap = subgroups.to_df().merge(
                sd4py_extra.confidence_precision_recall_f1(subgroups, 
                                                        dataset_production, 
                                                        number_simulations=100,
                                                        frac=frac
                                                        )[1], 
                on="pattern")

            subgroups_bootstrap = subgroups_bootstrap.sort_values('f1_lower', ascending=False)

        else:

            subgroups_bootstrap = subgroups.to_df().merge(
                sd4py_extra.confidence_hedges_g(subgroups, 
                                                dataset_production, 
                                                number_simulations=100)[1], 
                on="pattern")

            subgroups_bootstrap = subgroups_bootstrap.sort_values('hedges_g_lower', ascending=False)

        return subgroups_bootstrap

    subgroups_bootstrap = get_bootstrap()


    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_drop_overlap(n):

        non_overlapping = []

        if jaccard_threshold < 1.0:

            for idx1 in subgroups_bootstrap.index:

                if len(non_overlapping) == 0:

                    non_overlapping.append(idx1)

                    continue

                overlapping = False
                
                indices1 = subgroups[idx1].get_indices(dataset_production)

                for idx2 in non_overlapping:
                        
                    indices2 = subgroups[idx2].get_indices(dataset_production)
                    
                    if (indices1.intersection(indices2).size / indices1.union(indices2).size) > jaccard_threshold:

                        overlapping = True

                if overlapping:

                    continue
                    
                non_overlapping.append(idx1)

                if len(non_overlapping) == n:

                    return subgroups_bootstrap.loc[non_overlapping]

            return subgroups_bootstrap.loc[non_overlapping]
        
        return subgroups_bootstrap.iloc[:n]

    subgroups_bootstrap_topn = get_drop_overlap(n=10)


    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_top10_subgroups_selection_ids():

        ids = ["*A*", "*B*", "*C*", "*D*", "*E*", "*F*", "*G*", "*H*", "*I*", "*J*"]

        subgroups_bootstrap_top10 = subgroups_bootstrap_topn.iloc[:10]
        ## This seems needless, but we actually need to create a new variable - streamlit won't allow subsequent changes (like adding the id column) to cached objects.

        subgroups_selection = subgroups[subgroups_bootstrap_top10.index]
        subgroups_bootstrap_top10.insert(0, 'id', ids[:len(subgroups_bootstrap_top10)])

        return subgroups_bootstrap_top10, subgroups_selection, ids 

    subgroups_bootstrap_top10, subgroups_selection, ids = get_top10_subgroups_selection_ids()

    st.dataframe(subgroups_bootstrap_top10)

    st.download_button(
        "Save subgroups table",
        subgroups_bootstrap_top10.to_csv(index=False).encode('utf-8'),
        file_name="{}_subgroups_table.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
        mime="text/csv",
        key='download-csv'
    )

    '''
    ## Plotting the distribution of the target value 
    '''

    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_conf_int():
            
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")
            
            return sd4py_extra.confidence_intervals(subgroups_selection, dataset_production)

    results_dict, aggregation_dict = get_conf_int()

    ## To make the subgroup names more readable
    labels = [re.sub('AND', '\nAND',key) for key in results_dict.keys()]
    labels = ['({}) {}'.format(*vals) for vals in zip(ids, labels)]

    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_boxplots():

        results_list = [results_dict[name] for name in subgroups_bootstrap_top10.pattern]

        fig = plt.figure(dpi = 150)
        
        sd4py_extra.confidence_intervals_to_boxplots(results_list[::-1], labels=labels[::-1])  ## Display is backwards by default

        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        #plt.xlabel('Proportion of Subgroup Members that Had Fault within 30 Minutes', size=12)
        plt.gca().set_title('Distribution of Mean Target Value from Bootstrapping',pad=20)
        fig.set_size_inches(17,10)
        plt.tight_layout()

        ## Convert to image to display 

        return get_img_array_bytes(fig)

    img_arr, img_bytes = get_boxplots()

    st.image(img_arr)

    st.download_button('Save boxplots', img_bytes, file_name="{}_boxplots.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mime="image/png")

    '''
    ## Overlap between subgroups
    '''

    edges_threshold = st.slider("Only draw edges when overlap is greater than: ", 0.0, 1.0, 0.25)

    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_jaccard_plot():

        fig = plt.figure(dpi=150)

        sd4py_extra.jaccard_visualisation(subgroups_selection, 
                                            dataset_production, 
                                            edges_threshold, 
                                            labels=labels)

        fig.set_size_inches(20,9)
        plt.margins(x=0.15)
        plt.gca().set_frame_on(False)
        plt.gca().set_title('Jaccard Similarity Between Subgroups', fontsize=14)
        fig.tight_layout()

        ## Convert to image to display 

        return get_img_array_bytes(fig)

    img_arr, img_bytes = get_jaccard_plot()

    st.image(img_arr)

    st.download_button('Save network diagram', img_bytes, file_name="{}_network_diagram.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mime="image/png")

    '''
    ## Focus on a specific subgroup
    '''

    chosen_sg_options = copy.deepcopy(labels)
    chosen_sg_options.insert(0, 'Choose a subgroup to visualise in more detail')
    chosen_sg = st.selectbox('Subgroup to focus on: ', chosen_sg_options)

    if chosen_sg == 'Choose a subgroup to visualise in more detail':
        st.stop()

    chosen_sg = subgroups_selection[dict(zip(labels, list(range(10))))[chosen_sg]]

    saved_figsize = plt.rcParams["figure.figsize"]

    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_subgroup_overview():

        plt.rcParams["figure.figsize"] = (20,17)

        fig = plt.figure(dpi = 150)
        fig.suptitle(re.sub('AND', '\nAND',str(chosen_sg)), y=0.95)
        plt.tight_layout()
        sd4py_extra.subgroup_overview(chosen_sg, dataset_production, axis_padding=50)

        ## Convert to image to display - so that Streamlit doesn't try to resize disasterously. 

        return get_img_array_bytes(fig)

    img_arr, img_bytes = get_subgroup_overview()

    st.image(img_arr)

    st.download_button('Save subgroup overview', img_bytes, file_name="{}_subgroup_overview.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mime="image/png")

    plt.rcParams["figure.figsize"] = saved_figsize

    if not isinstance(dataset_production.index, pd.DatetimeIndex):

        st.stop()

    '''
    ## Specific subgroup members
    '''

    chosen_member_options = copy.deepcopy(chosen_sg.get_rows(dataset_production).index.tolist())
    chosen_member_options.insert(0, 'Choose a subgroup member to inspect')
    chosen_member = st.selectbox('Subgroup member to inspect: ', chosen_member_options)

    if chosen_member == 'Choose a subgroup member to inspect':
        st.stop()

    before = st.number_input("Number of timesteps to show before member: ", step=1, value=60)
    after = st.number_input("Number of timesteps to show after member: ", step=1, value=30)

    @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    def get_most_interesting():

        most_interesting_numeric = sd4py_extra.most_interesting_columns(chosen_sg, dataset_production.drop(columns=chosen_sg.target))[0][:7]

        return most_interesting_numeric.index

    most_interesting = get_most_interesting()

    iidx = dataset_production.index.get_loc(chosen_member)

    fig = plt.figure(dpi = 150)

    sd4py_extra.time_plot(chosen_sg, dataset_production.iloc[iidx-before:iidx+after+1], 
        dataset_production[target].iloc[iidx-before:iidx+after+1],
        *[dataset_production[col].iloc[iidx-before:iidx+after+1] for col in most_interesting],
        window_size=1, use_start=True)

    fig.suptitle('Variables over time for ({})'.format(str(chosen_sg)), y=1.0, size =14)    

    fig.set_size_inches(18,20)
    plt.tight_layout()

    ## Convert to image to display

    img_arr, img_bytes = get_img_array_bytes(fig)

    st.image(img_arr)

    st.download_button('Save member time plot', img_bytes,
        file_name="{}_time_plot_member_{}.png".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
            '_'.join(str(dataset_production.index[iidx]).strip().split(' '))), 
        mime="image/png")

    # streamlit run main.py --server.headless true
