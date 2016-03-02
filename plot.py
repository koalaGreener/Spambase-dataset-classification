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

# plot the curve
curveData = pd.read_csv('output_csv_folder/SGD_ROC_Curve.csv')
g = sns.FacetGrid(curveData, size = 4)
g = g.map(plt.scatter, "FPR", "TPR")

# plot the mean squared error and iterations
gd = pd.read_csv('output_csv_folder/bgd_1_100.csv')
g = sns.FacetGrid(gd, size = 4)
g = g.map(plt.scatter, "epoch", "cost")
plt.show()