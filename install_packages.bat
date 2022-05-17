echo "Started the process"
call conda env remove --name removethis
call conda create -n removethis -y
call conda activate removethis
call conda install pip -y
call pip freeze
call pip install -r packages.txt
call pip freeze
call streamlit run main.py
cmd /k