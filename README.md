# Measurement Noise Filtering & Time-Series Prediction in Python

Support Materials for EQUS webinar "Measurement Noise Filtering &amp; Time-Series Prediction in Python" (24/09/2019) presented by Riddhi Gupta. 


#### Purpose:

The purpose of this webinar is to introduce techniques for time series analysis. In this webinar, I focus on the (linear) state space framework and the different ways we can use this framework to do **state estimation**, **noise filtering**, and **prediction** with time series data. These terms will be defined in due course. I assume no prior experience with time series analysis.

#### Repo Contents:

In this git repo, the main document follows the webinar presentation and is contained in:

    Measurement Noise Filtering and Time Series Prediction in Python_v4.ipynb
    
If you cannot download or run or view this Notebook on Github, the notebook can be viewed at this link:

https://nbviewer.jupyter.org/github/riddhisw/equs_timeseries_webinar/blob/master/Measurement%20Noise%20Filtering%20and%20Time%20Series%20Prediction%20in%20Python_v4.ipynb

It can take a few minutes for the equations to format themselves properly. You can toggle to view or hide the code in the second cell.

In this repo, the supporting Python codebase contains the following files, each of which comprise of a list of standalone Python functions:

    time_series_analysis_demo.py : 
        Contains the main state estimation, filtering and prediction algorithm.
    
    chirp_models.py :
        Contains functions to produce linear and geometric chirp signals. 
        

Examples in Section 3.3 of the Jupyter workbook requires access to the following Kalman Filtering Python packages: **akf, kf.** from outside of this repo. These packages can be downloaded from https://github.com/riddhisw/predictiveest/tree/riddhisw-webinar/. If the **akf** and **kf** packages are not available, then set *RUN_RESTRICTED* variable in notebook to:

    RUN_RESTRICTED = True

The codebase is written in Python 2.7 and is also Python 3.X compatible in division, print_function and and absolute import usage. 

