echo "Started the process"
call conda env remove --name data_analytics
call conda create -n data_analytics -y
call conda activate data_analytics
call conda install pip -y
call pip freeze
call pip install -r packages.txt
call pip freeze
call streamlit run main.py
cmd /k