import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def check_residuals(resid):
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(311)
    ax.set_title("Residuals Time Series")
    ax.plot(resid)

    ax = fig.add_subplot(312)
    ax.set_title("Residuals ACF")
    plot_acf(resid, lags=40, ax=ax)

    ax = fig.add_subplot(313)
    ax.set_title("Residuals Histogram")
    ax.hist(resid, bins=40)

    plt.show()
