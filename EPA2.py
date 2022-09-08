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
if sys.platform == 'win32':
    string_splitter = '\\'
else:
    string_splitter = '/'

# %%

def data_loader():
    found_files = []
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

    st.header('Exploratory Pattern Analytics (EPA)')
    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split(string_splitter)[-1], key=1)
    if option == 'Select a Dataset':
        st.stop()

    @st.cache
    def get_data():

        data = pd.read_csv(option,index_col=0)
        #data = data.reset_index(drop=True)

        assert len(np.unique(data.index)) == len(data.index), "Index column contains duplicate values"

        if data.index.dtype == 'object' or data.index.dtype.name == 'category':

            try:

                data.index = pd.to_datetime(data.index)
            
            except ValueError:

                pass
        
        if re.search('[0-9]{1,2}-[0-9]{1,2}-[0-9]{1,2}preprocessed\.csv', option):

            pet_all_measurements = []

            for label in data.columns.tolist():
                
                if re.search(r'Extruder(Co){0,1}PET.+Istwert', label):

                    if re.search(r'Dosierung', label) is None:

                        pet_all_measurements.append(label)

            # data['recycled'] = data['recycled'].astype(bool)
                        
            # data = data[pet_all_measurements].join(data[['PK5.Aussenluft.Feuchte_2_OG','PK5.Aussenluft.Temperatur_2_OG','spit','recycled']])

 
        
        return data
    

    dataset_production = get_data()

    
    # # print('we actually got here')

    # n_rows = dataset_production.shape[0]
    # number_of_rows = st.slider('How many observations do you want to investigate?', 0, n_rows, int(n_rows/10))
    # # dataset_production = dataset_production.groupby('label').apply(lambda x: x.sample(n=number_of_rows)).reset_index(drop = True)
    # dataset_production = dataset_production.iloc[:number_of_rows,:]
    # # dataset_production.loc[dataset_production["label"] == 0, "label"] = 'non_recycled'
    # # dataset_production.loc[dataset_production["label"] == 1, "label"] = 'recycled'

    st.markdown(
    '''
    ## Type of analysis

    Depending on the type of analysis, different options for how to perform the analysis will be shown. 
    Please select an option below. 

    *Classification* looks at what makes one class different from others. 
    For example, distinguishing one particular product from other products, or distinguishing recyclate from virgin material,
    or distinguishing one type of outcome for a process. More generally, this is possible when the target is a non-numeric variable. 
    *High average* aims to find situations in which there is a high value for some numeric variable. 
    For example, identifying circumstances in which a quality score, or a physical property, tends to be high. 
    More generally, this is possible when the target is a numeric variable. 
    *Event detection* tries to understand events in a recording over time.
    For example, looking for faults or quality issues which occur at specific times. 
    This option is appropriate when there is a variable indicating when an event occurs. 
    '''
    )

    analysis_type = st.radio("Please select the option that best describes the analysis type", 
        ('Classification', 'High average', 'Event detection', 'Other'))

    if analysis_type == 'Event detection':
    
        assert isinstance(dataset_production.index, pd.DatetimeIndex), "Index column could not be interpreted as dates or times."

    st.markdown(
    '''
    ## Filtering

    Sometimes it is interesting to focus the analysis only on certain points in the data.
    To achieve this, filtering is possible. This step is entirely optional.
    ''')

    filtering = st.checkbox('Would you like to filter the data before analysis?')

    if filtering:

        if isinstance(dataset_production.index, pd.DatetimeIndex):

            st.markdown(
            '''
            ### Time selection

            Sometimes it is interesting to focus on a particular period within the data, 
            for example because you are analysing a manufacturing process and want to focus on the period when
            production is happening rather than periods of rest. 
            Here it is possible to provide a start and end time in order to select data to analyse. 
            A variable from the dataset can be viewed to help with this choice (e.g. a variable representing the state of the system).
            By default, the entire dataset is selected. 
            '''
            )

            preprocessing_column_options = list(dataset_production.columns)
            preprocessing_column_options.insert(0, 'Variable to display')
            preprocessing_column = st.selectbox('Variable to visualise to help choosing a time period: ', preprocessing_column_options)

            if preprocessing_column != 'Variable to display':

                if np.issubdtype(dataset_production[preprocessing_column].dtype, np.datetime64) \
                or   np.issubdtype(dataset_production[preprocessing_column].dtype, np.timedelta64) \
                or   np.issubdtype(dataset_production[preprocessing_column].dtype, np.number):
                    plt.scatter(
                        dataset_production.index.values, 
                        dataset_production[preprocessing_column]
                    )
                else:
                    plt.scatter(
                        dataset_production.index.values, 
                        dataset_production[preprocessing_column].astype(str)
                    )

                selected_period = st.slider(
                    'Choose the time period to focus on:',
                    dataset_production.index.min().floor('h').to_pydatetime(),
                    dataset_production.index.max().ceil('h').to_pydatetime(),
                    (dataset_production.index.min().floor('h').to_pydatetime(), dataset_production.index.max().ceil('h').to_pydatetime()),
                    step=datetime.timedelta(hours=1),
                    format="DD/MM/YY - hh:00"
                )

                plt.axvspan(
                    xmin=selected_period[0], 
                    xmax=selected_period[1], 
                    color='tab:green', 
                    lw=0, 
                    alpha=1 / 4
                )

                plt.gcf().set_size_inches(15,3)

                st.pyplot(plt.gcf())

                @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
                def get_selected_period():

                    return dataset_production[selected_period[0]:selected_period[1]]

                dataset_production = get_selected_period()
        
        st.markdown(
        '''
        ### Filter on variable

        Here, it is possible to filter based on the value of a variable of choice.
        The range of values to keep must be provided. Other values will be removed before analysis. 
        '''
        )
        def is_time(x):

            return np.issubdtype(dataset_production[x].dtype, np.datetime64) or np.issubdtype(dataset_production[x].dtype, np.timedelta64) 

        filtering_column_options = [col for col in dataset_production.columns if not is_time(col)]
        filtering_column_options.insert(0, 'Choose a variable')
        filtering_column = st.selectbox('Variable to use for filtering: ', filtering_column_options)

        if filtering_column != 'Choose a variable':

            if np.issubdtype(dataset_production[filtering_column].dtype, np.number):
                min_filtering = st.number_input(
                    'Only keep values greater than:', 
                    dataset_production[filtering_column].min(),
                    dataset_production[filtering_column].max(),
                    dataset_production[filtering_column].min(),
                    (dataset_production[filtering_column].max() - dataset_production[filtering_column].min()) / 20
                )
                max_filtering = st.number_input(
                    'Only keep values less than:', 
                    dataset_production[filtering_column].min(),
                    dataset_production[filtering_column].max(),
                    dataset_production[filtering_column].max(),
                    (dataset_production[filtering_column].max() - dataset_production[filtering_column].min()) / 20
                )

                @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
                def get_numeric_filtering():

                    return dataset_production[dataset_production[filtering_column].gt(min_filtering) & dataset_production[filtering_column].lt(max_filtering)]

                dataset_production = get_numeric_filtering()

            else:
                keep_values = st.multiselect("Select values to keep", np.unique(dataset_production[filtering_column].astype(str).values))
                
                @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
                def get_nonnumeric_filtering():

                    return dataset_production[dataset_production[filtering_column].astype(str).apply(lambda x: x in keep_values)]
                
                dataset_production = get_nonnumeric_filtering()


    st.markdown(
    '''
    ## Settings 
    '''
    )

    target_options = list(dataset_production.columns)
    target_options.insert(0, 'Choose the target variable')
    target = st.selectbox('Target variable: ', target_options)

    if target == 'Choose the target variable':
        st.stop()

    target_nominal = False

    if dataset_production.loc[:,target].dtype == 'object' or dataset_production.loc[:,target].dtype == 'bool' or dataset_production.loc[:,target].dtype.name == 'category':

        target_nominal = True

    value = None

    if analysis_type == 'Event detection':
    
        assert target_nominal, "With event detection, the target must be a non-numeric variable indicating when the event occurs."

    if target_nominal:

        value_options = list(np.unique(dataset_production[target]))
        value_options.insert(0, 'Choose the target value')
        value = st.selectbox('Target value: ', value_options)

    if target_nominal: 
        if value == 'Choose the target value':
            st.stop()

    if analysis_type == 'Event detection':
    
        within = st.number_input("(Optionally) also include earlier time points that happened within (number of minutes): ", step=1, value=0)

        if within > 0: 

            @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
            def get_new_target(data):
                # Remember to initialise to False!
                new_target = pd.Series(index=data.index, dtype='bool', name=target)#, name='{}=={} (within {})'.format(target, value, within))
                new_target[:] = False

                for idx in data.index:

                    if data[target][idx:idx + pd.Timedelta(within, unit='T')].eq(value).any():

                        new_target[idx] = True
                
                return dataset_production.drop(columns=[target]).join(new_target)
            
            dataset_production = get_new_target(dataset_production)
            value = True # Remember to change the target value!

    columns_to_ignore = st.multiselect(
        'Optionally choose columns to ignore (leave blank to use all columns): ', 
        [col for col in dataset_production.columns if col != target]
    )

    qf_options = ["Larger subgroups", "Smaller subgroups"]
    qf_options.insert(0, 'Choose the quality function')
    qf = st.selectbox('Quality function: ', qf_options)

    minsize = st.number_input("Minimum size for subgroups: ", step=1, value=10)

    jaccard_threshold = st.slider("Suppress 'duplicate' subgroups that overlap with previous subgroups by more than: ", 0.0, 1.0, 0.95)

    if qf == 'Choose the quality function':
        st.stop()

    qf = {"Larger subgroups":"ps", "Smaller subgroups":"bin"}[qf]

    if columns_to_ignore and len(columns_to_ignore) > 0:

        @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
        def get_drop_columns(data):

            return data.drop(columns=columns_to_ignore)

        dataset_production = get_drop_columns(dataset_production)

    st.markdown(
    '''
    ## Top patterns

    The table of results shows a list of the best patterns found, along with some measures of quality.

    For nominal targets, the precision (what proportion of the points selected by the pattern in fact belong to the target group),
    the recall (how much of the target group is selected by the pattern), and the F1-score (a combination of precision and recall)
    are provided as extra quality measures. Estimated 5% and 95% confidence intervals are shown. 
    For numeric targets, the "Hedge's G" measure is shown as an extra quality measure. This gives an indication of how large the 
    difference is between the points selected by the pattern and the rest of the dataset. Larger numbers indicate a greater difference.
    '''
    )
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

    st.markdown(
    '''
    ## Plotting the distribution of the target value 

    This visualisation shows the expected variability for the target value, meaning how much it changes across different samples 
    of measurements. This is depicted through boxes in a box plot, with wider boxes in the x-direction implying greater variability.
    How many points are selected by each pattern is also shown, with thicker boxes in the vertical direction meaning that
    a pattern selects a greater number of points on average. 
    '''
    )

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

    st.markdown(
    '''
    ## Overlap between patterns

    In this visualisation, patterns are connected to each other by how much their members overlap. 
    If two patterns select similar subsets of data (also called having similar 'subgroups'), 
    then they have a strong link between them and appear closer together. 
    Overall, this visualisation takes the form of a network diagram. 
    '''
    )

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

    st.markdown(
    '''
    ## Focus on a specific pattern/subgroup

    At this point, there may be patterns that are particularly interesting. 
    The EPA tool makes it possible to examine these in more detail. 
    This visualisation compares subgroup members (points selected by the pattern) to non-members for one specific pattern. 
    The target variable, the variables used to define the pattern (selector variables), 
    and additional variables that are most clearly different between members and non-members are shown. 
    These respectively appear in the top-left, top-right and bottom panels of the visualisation. 
    This makes it possible to see additional information about the pattern, and understand more about the circumstances in which the pattern occurs. 
    '''
    )

    chosen_sg_options = copy.deepcopy(labels)
    chosen_sg_options.insert(0, 'Choose a pattern to visualise in more detail')
    chosen_sg = st.selectbox('Pattern to focus on: ', chosen_sg_options)

    if chosen_sg == 'Choose a pattern to visualise in more detail':
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
    
    if analysis_type != 'Event detection':

        st.stop()

    st.markdown(
    '''
    ## Specific subgroup members

    Finally, if the data comes from a process that happens over time, we can focus on particular moments at which a pattern occurs, 
    to see what happens to different variables before, during, and after. 
    After selecting a single pattern, you can now select a particular moment when the pattern occurs, 
    from the drop-down list below. 
    The target variable is shown, along with the other variables that are most clearly different between subgroup members 
    and non-members. The moment at which the pattern occurs is indicated by a red rectangle in the background. 
    '''
    )

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

