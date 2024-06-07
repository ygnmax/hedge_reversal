import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def create_yield_curve(zero_curve, max_days=1500):

    zero_curve['rate'] = pd.to_numeric(zero_curve['rate'], 'coerce')

    df_rate = pd.DataFrame()
    for d, group in zero_curve.groupby('date'):
        new_row = pd.DataFrame({'date': d, 'days': 1, 'rate': np.nan}, index=[0])
        group = pd.concat([new_row, group])
        group = group.sort_values(['date', 'days'])
        group = group.bfill()

        res = pd.DataFrame()
        num_days = np.arange(1, max_days + 5) 
        res['days'] = np.arange(1, max_days + 5)   
        func = interp1d(group['days'], group['rate'], bounds_error=False, fill_value=np.nan)
        res['r'] = func(num_days)
        res['date'] = d    
        df_rate = pd.concat([df_rate, res])
    

    df_rate['r'] = df_rate['r'] / 100.0  
    df_rate = df_rate.reset_index(drop = True)    
    
    return df_rate


def merge_interest(df, df_rate):
    # merge interest rate to expiry
    #----------------------------------------------------------
    # merge option price with overnight rate as short rate
    #----------------------------------------------------------    
    df_one_day = df_rate[df_rate['days'] == 1]
    df = df.merge(df_one_day[['date', 'r']], how='left', on='date')
    df = df.rename(columns={'r': 'short_rate'})
    
    #--------------------------------------------------------------------------------
    # match option price with interpolated yield curve (calender days to expiry)
    #--------------------------------------------------------------------------------
    df = df.merge(df_rate, how='left', left_on=['date', 'tau_days'], right_on=['date', 'days'])
    del df['days']
    return df