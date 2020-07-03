'''
Module for custom plots
based on the ideas gathered 
from r ggplot package,
there can exist better alternative
to these functions. 

@author : shivam pundir
@date : 3 July 2020

'''
import pandas as pd 
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns


__all__  = ['seasonal']

def seasonal(ts,col_name,figsize=(8,6)):
    '''
    Returns a fig,ax object of seasonal plot
    for the ts time series for given column
    name col_name.
    
    Parameters
    ------------
    ------------
    ts : pd.DataFrame object, with index as datetime object
    
    col_name : column name which has to be plotted
    
    figsize : tupple (optional), figure size 
    of returned fig object
    
    Rwturns
    ------------
    ------------
    fig, ax = tupple consisting of fig,ax
    '''
    f ,ax = plt.subplots(figsize=figsize)
    ((ts)
     .assign(year=ts.index.year)
     .assign(month=ts.index.month)
     .groupby(['year','month'])
     .sum().reset_index()
     .pivot_table(index='month',columns='year',values=col_name)).plot(marker='o',ax=ax)
    x_ticks = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    plt.xticks(range(1,13),x_ticks)
    plt.grid(True,alpha=.7)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Seasonal plot")
    return f,ax


def lag_plot(series,lags=5):
    '''
    returns pair plot for the time series 
    lagged upto the required number.
    Please dont overkill the purpose
    keeps lags upto = 5 -10 range
    
    Parameters
    ----------------
    ----------------
    series : numpy vector (array)
    
    Returns 
    -----------------
    -----------------
    
    fig , ax object
    '''

    ##TODO: Make a function which is line plot 
    ## rather thab correlation plot
    ## NOT SURE how to interpret that plot 
    df = pd.DataFrame(series,columns=['Yt'])
    for lag in range(1,lags+1):
        df['lag_{}'.format(lag)] = df['Yt'].shift(lag)
    return sns.pairplot(df)