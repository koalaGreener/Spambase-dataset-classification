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

sgd = pd.read_csv('sgd_0.001_13887.csv')
g = sns.FacetGrid(sgd, size = 4)
g = g.map(plt.scatter, "epoch", "cost")
plt.show()