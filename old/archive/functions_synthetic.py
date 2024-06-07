import pandas as pd
import numpy as np
import functools as ft
import scipy as sp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import datetime 
import warnings
warnings.filterwarnings("ignore")

from functions_clean import merge_interest

def get_closest_ATM_option(df_stock, df_rate, df_raw, optype, target_maturity, stkid, ticker):    
    df = df_raw[df_raw['ContractSize'] > 0].copy()
    df = df[df['CallPut'] == optype]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df['K'] = df['Strike'] / 1000.0
    df['V0'] = (df['BestBid'] + df['BestOffer'])/2
    df['Maturity'] = df['Expiration'] - df['Date']
    df['Maturity'] = df.Maturity.dt.days
    df['tau'] = df['Maturity'] / 360

    df['IV0'] = df['ImpliedVolatility']

    df = df[['Date', 'K', 'Expiration',
           'CallPut', 'BestBid', 'BestOffer', 'LastTradeDate', 'Volume',
           'IV0', 'Delta', 'Gamma', 'Vega', 'Theta', 'OptionID', 
           'V0', 'Maturity', 'tau']]
    df = df.sort_values(by = ['Date', 'Expiration', 'CallPut', 'K'])

    df_stk = df_stock.copy()
    #     df_stk = df_stk[df_stk['SecurityID'] == stkid]
    df_stk['S0'] = df_stk['ClosePrice']

    #----------------------------------------------------
    # 2.1 merge option price with index/underlying price
    #----------------------------------------------------
    df = df.merge(df_stk[['Date', 'S0', 'AdjClosePrice', 'AdjClosePrice2', 'AdjustmentFactor', 'AdjustmentFactor2']], 
                  how = 'inner', on = 'Date')

    df['M0'] = df['S0'] / df['K']

    df = merge_interest(df, df_rate)

    df['TargetMaturity'] = target_maturity
    df = df.merge(df_rate, how = 'left', left_on = ['Date', 'TargetMaturity'], right_on =['Date', 'Days'])
    df.rename(columns = {'Rate': 'TargetRate'}, inplace = True)


    df_ATM = pd.DataFrame()
    for dt, df_tmp in df.groupby('Date'):

        df_tmp['diff'] = df_tmp['Maturity'] - target_maturity
        if target_maturity not in np.unique(df_tmp['Maturity']):

            neg_diff_list = sorted([i for i in np.unique(df_tmp['diff']) if i < 0])
            pos_diff_list = sorted([i for i in np.unique(df_tmp['diff']) if i > 0], reverse = True)
            if (len(neg_diff_list) > 0) & (len(pos_diff_list) > 0):
                neg_diff = np.max(neg_diff_list)
                pos_diff = np.min(pos_diff_list)

                df_neg_M0 = df_tmp[df_tmp['diff'].isin([neg_diff])]

                df_neg_M0_less1 = [i for i in np.unique(df_neg_M0['M0'].values) if i < 1]
                df_neg_M0_more1 = [i for i in np.unique(df_neg_M0['M0'].values) if i > 1]
                if len(df_neg_M0_less1) > 0 & len(df_neg_M0_more1) > 0:            
                    maturity_range = [neg_diff, pos_diff]



            elif (len(neg_diff_list) == 0) & (len(pos_diff_list) > 0):
                neg_diff = 0
                pos_diff = np.min(pos_diff_list)
            elif (len(neg_diff_list) > 0) & (len(pos_diff_list) == 0):
                neg_diff = np.max(neg_diff_list)
                pos_diff = 0
            else:
                print('not available')
                break

            maturity_range = [neg_diff, pos_diff]
            df_tmp = df_tmp[df_tmp['diff'].isin(maturity_range)]


            loc = np.abs(df_tmp['M0'] - 1) < 0.01
            if True in list(loc):
                idx = loc[loc == True].index
                df_ATM_one = df_tmp.loc[idx, ]
            else:
                df_neg = df_tmp[df_tmp['diff'] == neg_diff]
                if len(df_neg) > 0:
                    df_neg_M0_less1 = [i for i in np.unique(df_neg['M0'].values) if i < 1]
                    df_neg_M0_more1 = [i for i in np.unique(df_neg['M0'].values) if i > 1]
                    if (len(df_neg_M0_less1) > 0) & (len(df_neg_M0_more1) > 0):
                        moneyness_range = [np.min(df_neg_M0_more1), np.max(df_neg_M0_less1)]
                    elif (len(df_neg_M0_less1) == 0) & (len(df_neg_M0_more1) > 0):
                        moneyness_range = [np.min(df_neg_M0_more1)]
                    elif (len(df_neg_M0_less1) > 0) & (len(df_neg_M0_more1) == 0):
                        moneyness_range = [np.max(df_neg_M0_less1)]
                    df_neg = df_neg[df_neg['M0'].isin(moneyness_range)]

                df_pos = df_tmp[df_tmp['diff'] == pos_diff]
                if len(df_pos) > 0:
                    df_pos_M0_less1 = [i for i in np.unique(df_pos['M0'].values) if i < 1]
                    df_pos_M0_more1 = [i for i in np.unique(df_pos['M0'].values) if i > 1]
                    if (len(df_pos_M0_less1) > 0) & (len(df_pos_M0_more1) > 0):
                        moneyness_range = [np.min(df_pos_M0_more1), np.max(df_pos_M0_less1)]
                    elif (len(df_pos_M0_less1) == 0) & (len(df_pos_M0_more1) > 0):
                        moneyness_range = [np.min(df_pos_M0_more1)]
                    elif (len(df_pos_M0_less1) > 0) & (len(df_pos_M0_more1) == 0):
                        moneyness_range = [np.max(df_pos_M0_less1)]
                    df_pos = df_pos[df_pos['M0'].isin(moneyness_range)]

                df_ATM_one = pd.concat([df_neg, df_pos])
        else:
            df_tmp = df_tmp.loc[df_tmp['Maturity'] == target_maturity, :]

            loc = np.abs(df_tmp['M0'] - 1) < 0.01
            if True in list(loc):
                idx = loc[loc == True].index
                df_ATM_one = df_tmp.loc[idx, ]
            else:
                df_tmp_M0_less1 = [i for i in np.unique(df_tmp['M0'].values) if i < 1]
                df_tmp_M0_more1 = [i for i in np.unique(df_tmp['M0'].values) if i > 1]
                if (len(df_tmp_M0_less1) > 0) & (len(df_tmp_M0_more1) > 0):
                    moneyness_range = [np.min(df_tmp_M0_more1), np.max(df_tmp_M0_less1)]
                elif (len(df_tmp_M0_less1) == 0) & (len(df_tmp_M0_more1) > 0):
                    moneyness_range = [np.min(df_tmp_M0_more1)]
                elif (len(df_tmp_M0_less1) > 0) & (len(df_tmp_M0_more1) == 0):
                    moneyness_range = [np.max(df_tmp_M0_less1)]            
                df_ATM_one = df_tmp[df_tmp['M0'].isin(moneyness_range)]   

        df_ATM = pd.concat([df_ATM, df_ATM_one])
    df_ATM = df_ATM.reset_index(drop = True)
    df_ATM['Ticker'] = ticker
    df_ATM['SecurityID'] = stkid
    
    return df_ATM





###################
# American Pricer1
###################
def BinTree(S, u, d, n, T, r, ex_div):
    '''
    Binomial tree with dividend adjustment
    returns a list containing the binomial tree
    
    S: Stock Price
    u: np.exp(sigma * np.sqrt(t))
    d: 1.0 / u
    n: Steps of Binomial Tree
    T: Time to maturity (days)
    ex_div: Dividends, which are given in the format np.array([[time_to_ExDate, ExDate_to_Maturity, dividend],....,])
    '''   
#     print("ex_div", ex_div)
    # Creating a binomial tree with dividends adjustment
    tree = [np.array([S])]
    for i in range(n):
        tree.append(np.concatenate((tree[-1][:1]*u, tree[-1]*d))) 

    for i in range(n):
        if (len(ex_div) > 0):
            if (i * float(1/n) <= ex_div[0,0]/360):
                div = ex_div[0,2] * np.exp(-r * ex_div[0, 0]/360.)
                tree[0:i] = tree[0:i] + div
                ex_div = ex_div[1:,:]

    return tree

def VP(S, K = 35, CallPut = 'C'):
    '''
    Intrinsic value
    S: Stock Price
    K: Strike Price
    CallPut: OptType
    '''      
    if (CallPut=='C'): 
        return np.maximum(S-K, 0)
    else: 
        return np.maximum(K-S, 0)
    
def American(discount, p, tree, in_value): 
    '''
    discount: discount factor
    p: risk-neutral probability
    tree: result of binomial tree
    in_value : intrinsic value    
    '''      
    # Selecting maximum between continuation and intrinsic value
    return np.maximum(in_value, discount*(tree[:-1]*p + tree[1:]*(1-p)))

    
def GBM(am_func, intrinsic_value, S, T, r, sigma, n, ex_div):
    '''
    American Option Pricer with dividends adjustment
    am_func: function of comparing the intrinsic value and the value calulated by the binomial tree
    intrinsic_value: function of calculating intrinsic value 
    S: Stock Price
    T: Time To Maturity (days)
    r: Interest Rate
    v: Volatility
    n: Steps of Binomial Tree
    ex_div: Dividends, which are given in the format np.array([[time_to_ExDate, ExDate_to_Maturity, dividend],....,])
    '''   
    np.seterr(invalid='ignore')
    t = float(T)/n
    curr_div = ex_div.copy()
    
    u = np.exp(sigma * np.sqrt(t))
    d = 1.0/u
    p = (np.exp(r * t) - d)/(u - d)
    S0 = S    
    if len(curr_div) > 0:
        for i in range(len(curr_div)):
            S0 = S0 - curr_div[i, 2] * np.exp(-r * curr_div[i, 0]/360.)
    # Creating the binomial tree
    ptree = BinTree(S0, u, d, n, T, r, curr_div)[::-1]
    
    # Discounting through the tree with american exercise option
    result = ft.reduce(ft.partial(am_func, np.exp(-r*t), p), map(intrinsic_value, ptree))
    return result[0]

ABM = ft.partial(GBM, American)





def synthetic(raw_data, target_maturity, df_dividend):
    stkid = raw_data['SecurityID'].values[0]
    ticker = raw_data['Ticker'].values[0]
    
    data = raw_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])   
    data_out = pd.DataFrame(columns=['Date','StockPrice','CallPut','Expiration', 'Maturity', 'Strike', 'OptionPrice', 'IV', 'SecurityID', 'Ticker'])
    
    for date, df_one in data.groupby('Date'):
        S = df_one.S0.values[0]
        X = df_one.S0.values[0]
        CP = df_one.CallPut.values[0]
        T = target_maturity*1.0/360
        r = df_one.TargetRate.values[0]
        
        IV_0 = df_one.IV0[df_one.IV0>0].mean()
        if IV_0 < 0:
            dt_a = 0
            while IV_0 < 0:
                dt_a = dt_a + 1
                IV_0=data.IV0[data.Date==(date - datetime.timedelta(dt_a))].mean()
        if IV_0 < 0:
            IV_0 = 1.5
        if np.isnan(IV_0):
            IV_0 = 1.5        
        expiration = date + datetime.timedelta(days=target_maturity)
        
        
        dividends = df_dividend.loc[df_dividend['DistributionType'] == 1, ['ExDate', 'Amount', 'DeclareDate']]
        dividends = dividends.reset_index(drop = True)
        if len(dividends) > 0:
            div_idx = []
            for j in dividends.index:
                if (dividends.loc[j, 'DeclareDate'] < date) & (date < dividends.loc[j, 'ExDate']):
                    div_idx.append(j)
            dividends = dividends.loc[div_idx, :]
            if len(dividends) > 0:
                expir_date = np.unique(df_one['Date'] + pd.DateOffset(days = target_maturity))[0]
                time_to_ExDate = np.array([(t-date).days for t in dividends.ExDate])                               # time to Ex date
                ExDate_to_Maturity = np.array([(expir_date-t).days for t in dividends.ExDate])                     # Ex date to Maturity date
                div_to_expiration = np.array([time_to_ExDate, ExDate_to_Maturity, dividends.Amount]).T               # Dividend table with maturity of Ex date

                div_to_expiration = div_to_expiration[(div_to_expiration[:,0]>0) & (div_to_expiration[:,1]>0)]
            else:
                div_to_expiration = np.array([[1.0, 1.0, 0.0]])
        else:
            div_to_expiration = np.array([[1.0, 1.0, 0.0]])

        
        # try:
        if len(df_one) == 1:
            MBBO_synthetic = df_one.V0.values[0]
            iv_interpolate = df_one.IV0.values[0]
            iv_bin = df_one.IV0.values[0] 
            if np.isnan(iv_bin) == True:
                def f(x):
                    return (ABM(ft.partial(VP,K=X,CallPut=CP),S, T, r, x, 1000, div_to_expiration)-MBBO_synthetic)**2
                cons = ({'type': 'ineq', 'fun' : lambda x: np.array(x), 'jac': lambda x: np.array([1.0])})
                res = minimize(f,IV_0,constraints=cons,tol = 0.01)
                iv_bin = float(res.x)
                if np.isnan(iv_bin) == True:        
                    print(date, ' IV from binominal tree is nan, please check')                
        else:
            if (np.abs(S/X-1) < 0.01) and (target_maturity in df_one.Maturity.values):
                # print(date,1)
                MBBO_synthetic = float(df_one.loc[(target_maturity == df_one.Maturity), 'V0'].values[0])
                iv_interpolate = float(df_one.loc[(target_maturity == df_one.Maturity), 'IV0'].values[0])
            elif (target_maturity in df_one.Maturity.values) | (len(np.unique(df_one.Maturity.values)) == 1):
                # print(date,2)
                try:
                    spline = sp.interpolate.interp1d(df_one.K.values, df_one.V0.values)
                    MBBO_synthetic = float(spline(X))
                    spline2 = sp.interpolate.interp1d(df_one.K.values, df_one.IV0.values)
                    iv_interpolate = float(spline2(X))
                except:
                    MBBO_synthetic = np.mean(df_one.V0.values)
                    iv_interpolate = np.mean(df_one.IV0.values)
                    print(date)
                    print(df_one)
            elif np.abs(S/X-1) < 0.01:
                # print(date,3)
                data_2d = df_one.copy()
                try:
                    spline = sp.interpolate.interp1d(data_2d.Maturity.values, data_2d.V0.values, fill_value="extrapolate")
                    MBBO_synthetic = float(spline(target_maturity))
                    spline2 = sp.interpolate.interp1d(data_2d.Maturity.values, data_2d.IV0.values, fill_value="extrapolate")
                    iv_interpolate = float(spline2(target_maturity))
                except:
                    MBBO_synthetic = np.mean(df_one.V0.values)
                    iv_interpolate = np.mean(df_one.IV0.values)
                    print(date)
                    print(df_one)                    
            else:
                # print(date,4)
                try:
                    spline = sp.interpolate.interp2d(df_one.Maturity.values, df_one.K.values, df_one.V0.values, fill_value="extrapolate")
                    MBBO_synthetic = float(spline(target_maturity, X))
                    spline2 = sp.interpolate.interp2d(df_one.Maturity.values, df_one.K.values, df_one.IV0.values, fill_value="extrapolate")
                    iv_interpolate = float(spline2(target_maturity, X))
                except:
                    MBBO_synthetic = np.mean(df_one.V0.values)
                    iv_interpolate = np.mean(df_one.IV0.values)
                    print(date)
                    print(df_one)                      

            def f(x):
                return (ABM(ft.partial(VP,K=X,CallPut=CP),S, T, r, x, 1000, div_to_expiration)-MBBO_synthetic)**2
            cons = ({'type': 'ineq', 'fun' : lambda x: np.array(x), 'jac': lambda x: np.array([1.0])})
            res = minimize(f,IV_0,constraints=cons,tol = 0.01)
            iv_bin = float(res.x)
            
            if np.isnan(iv_bin):
                print(date, ' IV from binominal tree is nan, please check')
            
        s = pd.Series([date, S, CP, expiration, target_maturity, X, MBBO_synthetic, iv_bin, iv_interpolate, stkid, ticker],
                      index=['Date','StockPrice', 'CallPut', 'Expiration', 'Maturity', 'Strike', 'OptionPrice', 'IV', 'IV_interp', 'SecurityID', 'Ticker'])
        data_out = data_out.append(s,ignore_index=True)
        
    return data_out