#!/usr/bin/env python

"""
Quick plot of WUE as a function of year

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (17.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import brewer2mpl
import seaborn as sns

def main():

    fname = "outputs/WUE.csv"
    df = pd.read_csv(fname)



    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

    golden_mean = 0.6180339887498949
    width = 9
    height = width * golden_mean
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color'] = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    ax1 = fig.add_subplot(111)

    y = []
    x = []
    sites = np.unique(df.Site)
    for site in sites:

        df_site = df[df.Site == site]
        if len(df_site > 3):

            ax1.plot(df_site.Year, df_site.WUE, ls=" ", marker="o", alpha=0.4)
            for i in range(len(df_site)):
                y.append(df_site.WUE.values[i])
                x.append(df_site.Year.values[i])

    x = np.asarray(x)
    y = np.asarray(y)
    res = stats.theilslopes(y, x, 0.90)
    lsq_res = stats.linregress(x, y)

    ax1.plot(x, res[1] + res[0] * x, 'r-')
    #ax1.plot(x, res[1] + res[2] * x, 'r--')
    #ax1.plot(x, res[1] + res[3] * x, 'r--')

    ax1.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')

    ax1.locator_params(nbins=7, axis="x")

    ax1.set_xlabel("WUE ()")
    ax1.set_ylabel("Years)")


    #odir = "/Users/%s/Dropbox/g1_leaf_ecosystem_paper/figures/figs/" % \
    #        (os.getlogin())
    #plt.savefig(os.path.join(odir, "g1_vs_lai.pdf"),
    #                bbox_inches='tight', pad_inches=0.1)

    plt.show()

if __name__ == "__main__":

    main()
