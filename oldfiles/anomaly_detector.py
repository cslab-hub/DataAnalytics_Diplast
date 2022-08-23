## Anomaly Detector
import streamlit as st
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose



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

def return_anomaly():

	st.title('Detecting anomalies in the data')

	# col1,col2,col3 = st.columns(3)

# with col2:
	option = st.selectbox(
		'Which dataset do you want to view?',
		data, format_func= lambda x: str(x).split('/')[-1],key=1)
	df = pd.read_csv(option)
	mask = df.applymap(type) != bool
	d = {True: 'anomaly', False: 'no_anomaly'}
	df = df.where(mask, df.replace(d))
	# print(df)
	# plot["State"] = plot["State"].astype(str)
	# st.write(f"Dataset {data[0].split('/')[-1]} is loaded in.")
	# print(df.columns)
	option2 = st.selectbox(
		'Which variable do you want to view?',
		df.columns[df.columns.str.contains(pat = 'act')],key=3)


	# 	
	option3 = st.selectbox(
		'Which anomalies are connected to this variable?',
		df.columns[df.columns.str.contains(pat = 'anomaly')],key=3)


		# , key =3)

	option4 = st.selectbox(
		"target data",
		df.columns[df.columns.str.contains(pat = 'tar')],key=4)

	# print(option3)

	fig2,ax2 = plt.subplots(figsize=(8,5))
	colors = {'anomaly':'red', 'no_anomaly':'blue'}
	ax2.scatter(df.index,df[option2],c=df[option3].map(colors),s=2)
	ax2.plot(df.index, df[option4],c='blue')
	plt.title("Spotted Outliers")
	plt.xlabel("Time")
	ax2.set_ylabel("Temperature")

	st.pyplot(fig2)

	fig,ax1 = plt.subplots(figsize=(7,2))
	# colors = {'1.0':'green','3.0' :'red','5.0':'red','0.0':'green'}
	ax1.plot(df.index, df["State"],c='green')
	plt.xlabel("Time")
	ax1.set_ylabel("Machine State")
	ax1.set_yticks((0,1,2,3,4,5))


    # plt.title("Spotted Outliers")
	# plt.xlabel("Time")
	# ax3.set_ylabel("Machine State")
	# ax1.set_ylabel("Temperature")


	# ax1.scatter(plot.index,plot[option2],c=plot[option3].map(colors),s=10)
	# plt.plot(plot.index, plot[option2],c='blue')
	# ax3 = ax1.twinx() 
	# ax3.plot(plot.index, plot['State'],c='green',linewidth=1)


	st.pyplot(fig)

	st.markdown("""Now that we visualized the different machine states, and identified the anomalies based on a given threshold, we could also look at anomalies based on density of the data points.
	This method is called the Local Outlier Factor (LOF), which is able to to compare every individual datapoint with its neighbors. A normal value between 1 and 1.5 means that there is no anomaly. When the LOF increases,
	this would indicate that the data point seemed to behave anomalous. For implementing LOF, we first need to smoothen the time series by taking a moving average across time. Thus, the slider below indicates how the data is smoothened.
	""")

	# fig3,ax3 = plt.subplots()
	# ax3.scatter(df.index,df[option2], color = "b", s = 65)
	

	# st.pyplot(fig3)

	
	# # print(frames)
	

	# print(len(df_choice)/100)


	length = st.slider(label='choose length for smoothing the data ',min_value=0, max_value=int(len(df)/25), value=0, step=int(len(df)/1000), key=2)


	df_values = df[option2].rolling(length, on=df.index).mean()


		# option = st.selectbox(
		# 'Which dataset do you want to view?',
		# data, format_func= lambda x: str(x).split('/')[-1],key=1)
	
	# series = pd.Series(df[option2])
	# rolling = series.rolling(window=250)
	# rolling_mean = rolling.mean()
	fig4,ax4 = plt.subplots()
	ax4 = df[option2].plot(color='blue',linewidth=1)	
	ax4 = df_values.plot(color='green',linewidth=1)	
	# ax4 = df_values.plot(color='red',linewidth=1)	
	st.pyplot(fig4)
	


	

	# print(df_values)
	frames = [df_values,df[option4]]
	df_values = pd.concat(frames,axis=1)	
	df_values = df_values.dropna()

	fig5,ax5 = plt.subplots()
	#model specification
	model1 = LocalOutlierFactor(n_neighbors = 100, metric = "euclidean", contamination = 0.1)
	# # model fitting
	# print(np.all(np.isfinite(df_choice)))
	y_pred = model1.fit_predict(df_values)
	# # filter outlier index
	outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers
	# # filter outlier values
	outlier_values = df_values.iloc[outlier_index]
	# # plot data
	ax5.scatter(df_values.iloc[:,0], df_values.iloc[:,1], color = "b", s = 11)
	# # plot outlier values
	ax5.scatter(outlier_values.iloc[:,0], outlier_values.iloc[:,1], color = "r")
	ax5.grid()
	st.pyplot(fig5)



	# 	# # plot data
	# ax6.plot(df_values.index, df_values.iloc[:,0], color = "b", s = 11)
	# # # plot outlier values
	# ax6.plot(outlier_values.index, outlier_values.iloc[:,0], color = "r")
	# st.pyplot(fig6)
	df_values['pred'] = y_pred.tolist()

    # fig = plt.figure()
	fig6,ax6 = plt.subplots()
	colors = {-1:'red', 1:'blue'}
	plt.scatter(df_values.index, df_values['roll_act_temp1'],c=df_values['pred'].map(colors), s=1)
	# plt.plot(df_values.index, df_values['roll_act_temp1'])
	plt.title("Spotted Outliers")
	plt.xlabel("Time")
	plt.ylabel("Value")
	st.pyplot(fig6)

	print(np.unique(y_pred, return_counts =True))
	print(df["anomaly1"].value_counts())


	# ax2.scatter(df.index,df[option2],c=df[option3].map(colors),s=2)
	# print(df_choice.dtypes)



	# print(np.any(np.isnan(df_choice)))


	# print(np.all(np.isfinite(df_choice)))




	# print(np.isnan(df_choice.any()))

	# smoothed_list = []
	# for i in df_choice.columns:
	# 	series = pd.Series(df_choice[i])
	# 	rolling = series.rolling(window=250)
	# 	rolling_mean = rolling.mean()
	# 	fig4,ax4 = plt.subplots()
	# 	ax4 = series.plot(color='blue',linewidth=1)	
	# 	ax4 = rolling_mean.plot(color='red',linewidth=1)	
	# 	smoothed_list.append(rolling_mean)
	# 	st.pyplot(fig4)

	# print(smoothed_list)



	# st.pyplot(fig4)
	
		



	# series.plot()
	# rolling_mean.plot(color='red')
	# pyplot.show()




	# result = seasonal_decompose(df_choice, model='multiplicative')
	# ax4 = result.plot()





	





	# fig, ax1 = plt.subplots()
	# colors = {'anomaly':'red', 'no_anomaly':'blue'}
#   	ax1.scatter(option2.index, option2,c=option3.map(colors), s=10)



#       option2 = st.selectbox(
        
#           (i for i in plot.columns), key=2)

	# fig3, ax1 = plt.subplots(figsize=(20,15))

	# option = st.selectbox(
#           'Which dataset do you want to view?',
#           data, format_func= lambda x:  str(x).split('/')[-1], key=1)
#       plot = pd.read_csv(option,usecols=range(1,8))








# # ax.scatter(data.index,data['act_temp1'], s=100)

# # plt.scatter(merged_data.index, merged_data['Value_y'],c=merged_data['State'], s=15)

# ax1.plot(data.index, data['tar_temp1'],c='blue')
# ax3 = ax1.twinx() 
# ax3.plot(data.index, data['State'],c='green',linewidth=1)
# # plt.plot(data['tar_temp1'])
# plt.title("Spotted Outliers")
# plt.xlabel("Time")
# ax3.set_ylabel("Machine State")
# ax1.set_ylabel("Temperature")
# return fig3

