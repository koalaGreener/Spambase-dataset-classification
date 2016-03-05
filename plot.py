import numpy as np
import pandas as pd
from docutils.nodes import inline
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns


def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)


# plot the mean squared error and iterations
gd = pd.read_csv('output_csv_folder/log_bgd_0.001.csv')
g = sns.FacetGrid(gd, size = 6)
g = g.map(plt.scatter, "epoch", "cost")
plt.show()




# plot the curve
curveData = pd.read_csv('output_csv_folder/log_sgd_0.01_0.6342303807996201_curve.csv')
g = sns.FacetGrid(curveData, size = 6)
g = g.map(plt.scatter, "FPR", "TPR")
plt.ylim(0.0 ,1.0)
plt.xlim(0.0 ,1.0)














