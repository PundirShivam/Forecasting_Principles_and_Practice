"""
Some Helper Functions
Which make the life little
easier.

@author : shivam pundir
@date   : 8 Aug 2020
"""
__all__ = ['convert_r_dataframe_to_python']

import pandas as pd

def convert_r_dataframe_to_python(df):
    yr_ratios  = pd.np.sort((df.time - df.time.astype(int)).unique())
    df['year']  = df.time.astype(int)
    df['year_fraction'] = df.time - df.year
    df['month'] = 0
    
    for yr,adj in zip(yr_ratios,range(1,13)):
        df.loc[df.year_fraction.isin([yr]),'month'] = adj        
    df['date'] = pd.to_datetime(df.year.astype(str) + '-' + df.month.astype(str),format="%Y-%m")
    return df.drop(['year','year_fraction','month','time'],axis=1)