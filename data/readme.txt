# get r_datasets
# https://vincentarelbundock.github.io/Rdatasets/datasets.html

import statsmodels.api as sm
# "Ginzberg" is item name while "carData" is package name
df = statsmodels.datasets.get_rdataset("Ginzberg", "carData").data