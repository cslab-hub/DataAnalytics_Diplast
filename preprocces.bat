echo "Started the processing"
call cd "Di-Plast Data Analytics"
call conda activate data_analytics
call python preprocess_trilux.py
cmd /k