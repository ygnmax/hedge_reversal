import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def regression_1step(train, test, weights, cp_flag, delta_type='adjDelta'): 
    
    df_train = train.query("cp_flag==@cp_flag")
    df_test = test.query("cp_flag==@cp_flag")
    group = pd.merge(df_train, weights, how='left', on='date')

    y_train = df_train['y']
    x_train = df_train['x'].values.reshape(-1,1)
    y_test = df_test['y']
    x_test = df_test['x'].values.reshape(-1,1)  
    lin = LinearRegression(fit_intercept=False).fit(x_train, y_train, sample_weight=group['weight'])

    dict_coef = {}
    dict_plot = {}
    
    dict_coef['coef'] = lin.coef_[0]
    dict_coef['N_train'] = len(df_train)
    dict_coef['days_train'] = df_train['date'].nunique()
    dict_coef['N_test'] = len(df_test)
    
    dict_plot['x_train'] = x_train
    dict_plot['y_train'] = y_train
    dict_plot['w_train'] = group['weight']
    dict_plot['x_test'] = x_test
    dict_plot['y_test'] = y_test
    dict_plot['coef'] = lin.coef_[0]    
    dict_plot['delta_type'] = delta_type

    return dict_coef, dict_plot


def regression_rolling(df, delta_type='adjDelta', train_length=126, exp_weight=0.99): 
    bl = df[delta_type].isnull()
    if bl.any():
        print(f'Removed {bl.sum()} option days due to missing delta ({delta_type}).')
    df = df[~bl].copy()
    df['gross_rate'] = np.exp(df['short_rate']/252)
    df['y']= df['V1'] - df['V0'] * df['gross_rate']
    df['x'] = df[delta_type] * (df['S1'] - df['S0'] * df['gross_rate'])    
    
    weights = [exp_weight**i for i in range(train_length-1, -1, -1)]

    all_dates = df['date'].drop_duplicates().sort_values()
    n_dates = len(all_dates)

    dict_coef = {}
    dict_plot = {}
    
    for i in range(train_length, n_dates):
        train_dates = all_dates.iloc[(i-train_length):i]
        test_date = all_dates.iloc[i]
        
        df_train_rolling = df[df['date'].isin(train_dates)]
        df_test_rolling = df[df['date']==test_date]
        df_weights = pd.DataFrame({'date':train_dates, 'weight':weights})
    
        dict_coef_tmp = {}
        dict_plot_tmp = {}
        for cp_flag in ['C', 'P']:
            dict_coef_tmp[cp_flag], dict_plot_tmp[cp_flag] = regression_1step(
                df_train_rolling, df_test_rolling, df_weights, cp_flag, delta_type)
    
        test_date = str(test_date)[:10]
    
        dict_coef[test_date] = dict_coef_tmp
        dict_plot[test_date] = dict_plot_tmp

    return dict_coef, dict_plot