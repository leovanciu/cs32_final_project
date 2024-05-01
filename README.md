# Introduction
This project performs a comparison of execution time, and memory use in Python and R for generic computational operations common in statistics and machine learning. This includes generic loops and vector operations, matrix operations, and common computationally heavy statistical algorithms, including the bootstrap, Markov chain Monte Carlo, and the Support Vector Machine. For more information, see the project page https://leovanciu.github.io/cs32-project.html.

# Reproducing the analysis
Follow these steps to reproduce the project including generating the simulated data, running the algorithms, and generating the plots.
1) In this repository, run the script Data/data.r. This will generate the simulated data and store it in Data/data.csv, Data/A.csv, and Data/B.csv. Note that due to the large size of data.csv (218MB), this file could not uploaded to GitHub but the fixed seed in R should give you the same data as me.
2) Run the script Scripts/algorithms.py. This will run the algorithms in python, measure the execution time and memory, and store it in Results/Results_python.csv. You will need to have installed the libraries memory_profiler, numpy, pandas, scipy.stats, time, csv, math, sklearn, and cmdstanpy.
3) Run the script Scripts/algorithms.R. This will run the algorithms in R, measure the execution time and memory, and store it in Results/Results_R.csv. You will need to have installed the packages bench, boot, e1071, rstan, MASS, and dplyr.
4) The Rmarkdown file used to generate the plots and the results with dicussion are available at https://github.com/leovanciu/leovanciu.github.io/blob/master/cs32-project.Rmd. This file can be knit to obtain the HTML document that consists the webpage. This other repository (https://github.com/leovanciu/leovanciu.github.io/) also contains all files used to construct the website, which was forked from mmistakes/minimal-mistakes.