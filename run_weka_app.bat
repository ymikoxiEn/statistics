@echo off
REM Activate virtual environment
call venvScriptsactivate

REM Install all required packages
pip install --upgrade pip
pip install streamlit pandas numpy seaborn matplotlib scikit-learn statsmodels scipy

REM Launch the Streamlit app
streamlit run weka_full_app.py

pause