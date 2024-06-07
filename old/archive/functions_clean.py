import getpass
import sys
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from scipy.interpolate import interp1d

if getpass.getuser() in ['ygnmax']:
    if sys.platform == 'linux':
        workdir = '/home/ygnmax/Dropbox/hedge_reversal/'
    if sys.platform == 'win32':
        workdir = 'C:/Users/ygnmax/Dropbox (Personal)/hedge_reversal/'
datadir = workdir + 'data/raw/WRDS/' 

def read_data(stkid):
    #########################
    ## Step 0: Read Data
    #########################    
    #---------------------------
    ## 0.1 read underlying data
    #---------------------------
    try:
        df_stock = pd.read_csv(datadir + str(stkid) + '/security_price.csv', parse_dates = ['date'], low_memory=False)
        stock_column={"secid": "SecurityID",
                "date": "Date",
                "low" : "BidLow",
                "high" : "AskHigh",
                "open" : "OpenPrice",
                "high" :  "AskHigh",
                "close" : "ClosePrice",
                "volume" : "Volume",
                "return": "TotalReturn",
                "cfadj" : "AdjustmentFactor",
                "shrout" : "SharesOutstanding",
                "cfret" : "AdjustmentFactor2" 
                }
        df_stock.rename(columns=stock_column, inplace=True)
        df_stock['AdjClosePrice'] = df_stock['ClosePrice'] * df_stock['AdjustmentFactor'] / df_stock.loc[len(df_stock)-1, 'AdjustmentFactor']
        df_stock['AdjClosePrice2'] = df_stock['ClosePrice'] * df_stock['AdjustmentFactor2'] / df_stock.loc[len(df_stock)-1, 'AdjustmentFactor2']
    except:
        df_stock = pd.DataFrame(columns = ['SecurityID', 'Date', 'BidLow', 'AskHigh', 'ClosePrice', 'Volume', 'TotalReturn', 'AdjustmentFactor', 'OpenPrice', 'SharesOutstanding', 'AdjustmentFactor2', 'AdjClosePrice', 'AdjClosePrice2'])
    #-----------------------
    ## 0.2 read option data
    #-----------------------
    try:
        df_option = pd.read_csv(datadir + str(stkid) + '/option.csv', infer_datetime_format = True, parse_dates = ['date', 'exdate', 'last_date'], low_memory=False)
        option_column={"secid": "SecurityID",
                "date": "Date",
                "exdate" : "Expiration",
                "strike_price" :  "Strike",
                "cp_flag" : "CallPut",
                "best_bid" : "BestBid",
                "best_offer" : "BestOffer",
                "impl_volatility" : "ImpliedVolatility",
                "delta" : "Delta",
                "gamma" : "Gamma",
                "vega" : "Vega",
                "theta" : "Theta",
                "volume" : "Volume",
                "open_interest": "OpenInterest",
                "last_date" : "LastTradeDate",
                "contract_size" : "ContractSize",
                "optionid" : "OptionID" 
                }
        df_option.rename(columns=option_column, inplace=True)
    except:
        df_option = pd.DataFrame(columns = ['SecurityID', 'Date', 'Symbol', 'SymbolFlag', 'Strike', 'Expiration', 'CallPut', 'BestBid', 'BestOffer', 'LastTradeDate', 'Volume', 'OpenInterest', 'SpecialSettlement', 'ImpliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta', 'OptionID', 'AdjustmentFactor', 'AMSettlement', 'ContractSize', 'ExpiryIndicator', 'ForwardPrice'])
        
    #-----------------------
    ## 0.3 read dividend data
    #-----------------------
    try:
        df_dividend = pd.read_csv(datadir + str(stkid) + '/distribution.csv', parse_dates = ['record_date', 'ex_date','declare_date','payment_date'])
        div_column={"secid": "SecurityID",
                "record_date": "RecordDate",
                "ticker" : "Ticker",
                "issuer" :  "IssuerDescription",
                "distr_type": "DistributionType",
                "ex_date": "ExDate",
                "declare_date": "DeclareDate",
                "payment_date": "PaymentDate"
                }
        df_dividend.rename(columns=div_column, inplace=True)
    except:
        df_dividend = pd.DataFrame(columns = ['SecurityID', 'RecordDate', 'ExDate', 'Amount', 'DeclareDate', 'PaymentDate', 'DistributionType', 'AdjustmentFactor'])
        
    #---------------------------
    ## 0.4 read basic information data
    #---------------------------
    try:
        df_info = pd.read_csv(datadir + str(stkid) + '/name.csv', parse_dates = ['effect_date'])
        info_column={"secid": "SecurityID",
                "ticker": "Ticker",
                "effect_date" : "Date",
                "issuer" :  "IssuerDescription"
                }
        df_info.rename(columns=info_column, inplace=True)
    except:
        df_info = pd.DataFrame(columns = ['SecurityID', 'Date', 'CUSIP', 'Ticker', 'Class', 'IssuerDescription', 'IssueDescription', 'SIC'])
    return df_stock, df_option, df_dividend, df_info
def read_dividend(stkid):
    try:
        df_dividend = pd.read_csv(datadir + str(stkid) + '/distribution.csv', parse_dates = ['record_date', 'ex_date','declare_date','payment_date'])
        div_column={"secid": "SecurityID",
                "record_date": "RecordDate",
                "ticker" : "Ticker",
                "issuer" :  "IssuerDescription",
                "distr_type": "DistributionType",
                "ex_date": "ExDate",
                "declare_date": "DeclareDate",
                "payment_date": "PaymentDate"
                }
        df_dividend.rename(columns=div_column, inplace=True)
    except:
        df_dividend = pd.DataFrame(columns = ['SecurityID', 'RecordDate', 'ExDate', 'Amount', 'DeclareDate', 'PaymentDate', 'DistributionType', 'AdjustmentFactor'])    
    return df_dividend

def clean_standardized_option(stkid, df_rate):
    try:
        df_standarized = pd.read_csv(datadir + str(stkid) + '/standardized_option.csv', parse_dates = ['date'])
        standarized_column={"secid": "SecurityID",
                "date": "Date",
                "days" : "Days",
                "forward_price" : "F0",
                "strike_price" :  "K",
                "premium": "V0",
                "impl_volatility": "IV0",
                "delta" : "Delta",
                "gamma" : "Gamma",
                "vega" : "Vega",
                "theta" : "Theta",
                "cp_flag": "CallPut"
                }
        df_standarized.rename(columns=standarized_column, inplace=True)
        df_standarized['Maturity'] = df_standarized['Days']
    except:
        df_standarized = pd.DataFrame(columns = ['SecurityID', 'Date', 'F0', 'K', 'V0', 'IV0', 'Delta', 'Gamma', 'Theta', 'Vega', 'CallPut', 'Maturity', 'F1', 'V1', 'IV1', 'r', 'Ticker'])

    if len(df_standarized) > 0:
        df = df_standarized.copy()
        #------------------------
        # Add Tomorrow Value
        #------------------------
        tmp = ['Date', 'Days', 'CallPut', 'F0', 'V0', 'IV0']
        new_cols = {
            'F0': 'F1', 
            'V0': 'V1', 
            'IV0': 'IV1'}    

        df_tmp = df[tmp].copy()
        df_tmp['Date'] -= BDay(1)

        df_tmp.rename(columns=new_cols, inplace=True)
        df_tmp.set_index(['Date', 'CallPut', 'Days'], inplace=True)
        df = df.join(df_tmp, on=['Date', 'CallPut', 'Days'])   
        
        
        
        df = pd.merge(df, df_rate, how = 'left', left_on = ['Date', 'Days'], right_on = ['Date', 'Days'])
        df.rename(columns = {'Rate': 'r'}, inplace = True)
        del df['Days']
    else:
        df = df_standarized
    
    return df


def preclean_data(df_option, df_stock, stkid, day_count = 360):
    #########################
    ## Step 1: Preclean Data
    #########################
    #--------------------------
    ## 1.1 Preclean option data
    #--------------------------
    df = df_option.copy()
    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df['LastTradeDate'] = pd.to_datetime(df['LastTradeDate'])
    df['K'] = df['Strike'] / 1000.0
    df['V0'] = (df['BestBid'] + df['BestOffer'])/2
    df['Maturity'] = df['Expiration'] - df['Date']
    df['Maturity'] = df.Maturity.dt.days
    df['tau'] = df['Maturity'] / day_count
    tau = np.busday_count(df['Date'].values.astype('datetime64[D]'), 
                      df['Expiration'].values.astype('datetime64[D]'))
    df['tau0'] = tau / day_count  
    
    df['IV0'] = df['ImpliedVolatility']
    df = df[['Date', 'K', 'Expiration',
           'CallPut', 'BestBid', 'BestOffer', 'LastTradeDate', 'Volume',
           'IV0', 'Delta', 'Gamma', 'Vega', 'Theta', 'OptionID', 
           'V0', 'Maturity', 'tau', 'tau0']]
    df = df.sort_values(by = ['Date', 'Expiration', 'CallPut', 'K'])

    #------------------------------------
    ## 1.2 Preclean index/underlying data
    #------------------------------------
    df_stk = df_stock.copy()
#     df_stk = df_stk[df_stk['SecurityID'] == stkid]
    df_stk['S0'] = df_stk['ClosePrice']
    
    #######################################################
    # Step 2: Merge Dataframes and add future volume/value
    #######################################################
    #----------------------------------------------------
    # 2.1 merge option price with index/underlying price
    #----------------------------------------------------
    df = df.merge(df_stk[['Date', 'S0', 'AdjClosePrice', 'AdjClosePrice2', 'AdjustmentFactor', 'AdjustmentFactor2']], 
                  how = 'left', on = 'Date')
    # print(df.shape)

    #------------------------
    # 2.2 Add Tomorrow Value
    #------------------------
    tmp = ['Date', 'OptionID', 'S0', 'V0', 'IV0', 'Volume']
    new_cols = {
        'S0': 'S1', 
        'V0': 'V1', 
        'IV0': 'IV1',
        'Volume': 'Volume1'}    

    df_tmp = df[tmp].copy()
    df_tmp['Date'] -= BDay(1)

    df_tmp.rename(columns=new_cols, inplace=True)
    df_tmp.set_index(['Date', 'OptionID'], inplace=True)
    df = df.join(df_tmp, on=['Date', 'OptionID'])      
    
    return df



def preclean_interest(df_zero_curve, max_days = 1500):

    yield_curve = pd.DataFrame()
    for d, group in df_zero_curve.groupby('Date'):
        group['Rate'] = pd.to_numeric(group['Rate'], 'coerce')

        new_row = pd.DataFrame({'Date': d, 'Days': 1, 'Rate': np.nan}, index =[0])
        group = pd.concat([new_row, group]).reset_index(drop = True)
        group = group.sort_values(['Date', 'Days'])
        group = group.fillna(method = 'bfill')

        yield_curve = pd.concat([yield_curve, group]).reset_index(drop = True)
    
    ## 1.3.4 Interpolate the interest rate
    num_days = np.arange(1, max_days + 5)
    df_rate = pd.DataFrame()
    holidays = []
    for key, df_group in yield_curve.groupby('Date'):
        res = pd.DataFrame()
        res['Days'] = num_days
        if len(df_group['Days']) <= 1:
            holidays.append(key.date())
            res['Rate'] = np.nan
            res['Date'] = key
            df_rate = df_rate.append(res)
        else:
            func = interp1d(df_group['Days'], df_group['Rate'], bounds_error=False, fill_value=np.nan)
            res['Rate'] = func(num_days)
            res['Date'] = key
            df_rate = df_rate.append(res)
    
    ## 1.3.5 Output divided by 100
    df_rate['Rate'] = df_rate['Rate'] / 100.0  
    df_rate = df_rate.reset_index(drop = True)    
    
    return df_rate


def merge_interest(df, df_rate):
    ###################################
    # Step 3: Interest rate to expiry
    ###################################
    #----------------------------------------------------------
    # 3.1 merge option price with overnight rate as short rate
    #----------------------------------------------------------    
    df_one_day = df_rate[df_rate['Days'] == 1]

    df = df.merge(df_one_day[['Date', 'Rate']], how = 'left', on = 'Date')
    df.rename(columns = {'Rate': 'short_rate'}, inplace = True)
    #--------------------------------------------------------------------------------
    # 3.2 match option price with interpolated yield curve (calender days to expiry)
    #--------------------------------------------------------------------------------
    df = df.merge(df_rate, how = 'left', left_on = ['Date', 'Maturity'], right_on = ['Date', 'Days'])
    df.rename(columns = {'Rate': 'r'}, inplace = True)
    del df['Days']
    return df

def prefilter_data(df_out):
    df = df_out.copy()
    #########################################
    # Step 4: Remove certain types of data
    #########################################
    #----------------------
    # 4.1 filter by volume
    #----------------------
    df = df[df['Volume'] >= 1]
#     print('After filtering by volume')
#     print(df.shape)
    #----------------------
    # 4.2 Filter by spread
    #----------------------
    # df = df[df['BestOffer'] <= 2* df['BestBid']]
#     print('After filtering by spread')
#     print(df.shape)

    df = df[df['BestBid'] > 0.049]
#     print('After filtering by spread')
#     print(df.shape)    
    
    ##############################
    # Step 5: Additional cleaning
    ##############################
    #----------------------------------------------------
    # 5.1 filter all the option with negative time value
    #----------------------------------------------------
    bl_c = (df['CallPut'] == 'C') & (df['S0'] - np.exp(-df['r'] * df['tau0']) * df['K'] >= df['V0'])
    bl_p = (df['CallPut'] == 'P') & (np.exp(-df['r'] * df['tau0']) * df['K'] - df['S0'] >= df['V0'])
    df = df.loc[~bl_c]
#     print('After filter all the CALL with negative time value')
#     print(df.shape)
    df = df.loc[~bl_p]
#     print('After filter all the PUT with negative time value')
#     print(df.shape)
    
    df['M0'] = df['S0'] / df['K'] 
    
    ###########################
    # Step 6: Normalized Price
    ###########################
    #------------------------------------
    # 6.1 Calculate the Normalized Price
    #------------------------------------
    s_divisor = df['S0']
    norm_factor = 100
    cols = ['S0', 'V0', 'K', 'S1','V1']
    cols_after = [name + '_n' for name in cols]
    df[cols_after] = df[cols] / np.expand_dims((s_divisor / norm_factor), axis=1)
    
    #####################################
    # Step7: create and delete variables
    #####################################
    df = df.reset_index(drop=True)
    df['on_ret'] = np.exp(df['short_rate'] * 1 / 360)

    #######################
    # Step8: Cleaning data
    #######################
    #-----------------------------------------
    # Filter by the Future Value / Next Trade
    #-----------------------------------------
    bl = df['V1'].notna()
    df = df.loc[bl]
#     print('After filtering by One-Step-ahead price / next trade is not available')
#     print(df.shape)


    #------------------------------
    # Filter by Implied Volatility
    #------------------------------
#     bl = df['IV0'].isna()
#     if sum(bl) > 0.5:
#         print('Check Why implvol is not available!\n\n')

#     df = df[(df['IV0'] >=0.01)]
#     print('After filtering by implied volatility')
#     print(df.shape)

    #--------------------
    # Filter by Maturity
    #--------------------
    df = df[df['Maturity'] >= 1]
#     print('After Filter by Maturity')
#     print(df.shape)

    #---------------------------------
    # Keep out-the-money options only
    #---------------------------------
    bl = ((df['CallPut'] == 'C') & (df['M0'] < 1.001)) | ((df['CallPut'] == 'P') & (df['M0'] > 0.999))
    df = df.loc[bl]
#     print('After keeping out-the-money options only')
#     print(df.shape)
    
    return df


def filter_data(df_out):
    df = df_out.copy()
    #########################################
    # Step 4: Remove certain types of data
    #########################################
    #----------------------
    # 4.1 filter by volume
    #----------------------
    df = df[df['Volume'] >= 1]
#     print('After filtering by volume')
#     print(df.shape)
    #----------------------
    # 4.2 Filter by spread
    #----------------------
    df = df[df['BestOffer'] <= 2* df['BestBid']]
#     print('After filtering by spread')
#     print(df.shape)

    df = df[df['BestBid'] > 0.049]
#     print('After filtering by spread')
#     print(df.shape)    
    
    ##############################
    # Step 5: Additional cleaning
    ##############################
    #----------------------------------------------------
    # 5.1 filter all the option with negative time value
    #----------------------------------------------------
    bl_c = (df['CallPut'] == 'C') & (df['S0'] - np.exp(-df['r'] * df['tau0']) * df['K'] >= df['V0'])
    bl_p = (df['CallPut'] == 'P') & (np.exp(-df['r'] * df['tau0']) * df['K'] - df['S0'] >= df['V0'])
    df = df.loc[~bl_c]
#     print('After filter all the CALL with negative time value')
#     print(df.shape)
    df = df.loc[~bl_p]
#     print('After filter all the PUT with negative time value')
#     print(df.shape)
    
    df['M0'] = df['S0'] / df['K'] 
    
    ###########################
    # Step 6: Normalized Price
    ###########################
    #------------------------------------
    # 6.1 Calculate the Normalized Price
    #------------------------------------
    s_divisor = df['S0']
    norm_factor = 100
    cols = ['S0', 'V0', 'K', 'S1','V1']
    cols_after = [name + '_n' for name in cols]
    df[cols_after] = df[cols] / np.expand_dims((s_divisor / norm_factor), axis=1)
    
    #####################################
    # Step7: create and delete variables
    #####################################
    df = df.reset_index(drop=True)
    df['on_ret'] = np.exp(df['short_rate'] * 1 / 360)

    #######################
    # Step8: Cleaning data
    #######################
    #-----------------------------------------
    # Filter by the Future Value / Next Trade
    #-----------------------------------------
    bl = df['V1'].notna()
    df = df.loc[bl]
#     print('After filtering by One-Step-ahead price / next trade is not available')
#     print(df.shape)


    #------------------------------
    # Filter by Implied Volatility
    #------------------------------
#     bl = df['IV0'].isna()
#     if sum(bl) > 0.5:
#         print('Check Why implvol is not available!\n\n')

#     df = df[(df['IV0'] >=0.01)]
#     print('After filtering by implied volatility')
#     print(df.shape)

    #--------------------
    # Filter by Maturity
    #--------------------
    df = df[df['Maturity'] >= 1]
#     print('After Filter by Maturity')
#     print(df.shape)

    #---------------------------------
    # Keep out-the-money options only
    #---------------------------------
    bl = ((df['CallPut'] == 'C') & (df['M0'] < 1.001)) | ((df['CallPut'] == 'P') & (df['M0'] > 0.999))
    df = df.loc[bl]
#     print('After keeping out-the-money options only')
#     print(df.shape)
    
    return df




def calc_syn_implied_div(stkid, df_input):
    ## inplied dividend
    df = df_input.copy()

    df_c = df[(df['CallPut'] == 'C') & (df['IV0'] >=0.00001)]
    df_p = df[(df['CallPut'] == 'P') & (df['IV0'] >=0.00001)]
    df_calc_rate = df_c[['Date', 'K', 'Expiration', 'V0', 'IV0']].merge(
                   df_p[['Date', 'K', 'Expiration', 'V0', 'IV0', 'S0', 'Maturity', 'short_rate', 'r']], on=['Date', 'K', 'Expiration'], 
                   suffixes=['_C', '_P'])
    df_calc_rate['diff'] = abs(df_calc_rate['S0'] - df_calc_rate['K'])
    if len(df_calc_rate) == 0:
        print('No implied dividend '+ str(stkid))

        df_implied_rate = df[['Date', 'Expiration']].drop_duplicates()
        df_implied_rate['impl_div0'] = 0.0
        df_implied_rate['impl_cdiv0'] = 0.0
    else:
        df_implied_rate = pd.DataFrame()
        i = 0
        for idx, group in df_calc_rate.groupby(['Date', 'Expiration']):
            group = group[group['diff'] == min(group['diff'])]
            implied_d = []        

            # There may be 2 rows for the smallest difference between S0 and K (above or below) 
            for j, row in group.iterrows():
                date = row.loc['Date']
                expiration = row.loc['Expiration']
                S = row['S0']
                K = row['K']
                T = row['Maturity']/360.0
                IV_0 = (row['IV0_C'] + row['IV0_P']) / 2.0
                r_0 = row['r']

                CPop = row['V0_C'] - max(S - K, 0)
                PPop = row['V0_P'] - max(K - S, 0)
                d_0 = (-(CPop - PPop - r_0 * K * T)/(S * T))
                c_d_0 = 1/T * np.log((-(CPop - PPop)-(K-S)+np.exp(r_0 * T) * K)/S)

            df_implied_rate.loc[i,'Date'] = idx[0] 
            df_implied_rate.loc[i,'Expiration'] = idx[1]
            df_implied_rate.loc[i,'impl_div0'] = d_0
            df_implied_rate.loc[i,'impl_cdiv0'] = c_d_0
            df_implied_rate.loc[i,'S0'] = np.unique(df_calc_rate.loc[(df_calc_rate['Date'] == idx[0]) & (df_calc_rate['Expiration'] == idx[1]), 'S0'].values)[0]
            i += 1
    
    df_implied_rate['impl_div0'] = df_implied_rate['impl_div0'].fillna(value=0)
    df_implied_rate['impl_cdiv0'] = df_implied_rate['impl_cdiv0'].fillna(value=0)
    df_implied_rate['Maturity'] = df_implied_rate['Expiration'] - df_implied_rate['Date']
    df_implied_rate['Maturity'] = df_implied_rate['Maturity'].dt.days
    df_out = df_input.merge(df_implied_rate[['Date', 'Expiration', 'impl_div0','impl_cdiv0']], on = ['Date', 'Expiration'], how = 'left')
    df_out['impl_div0'] = df_out['impl_div0'].fillna(value=0)
    df_out['impl_cdiv0'] = df_out['impl_cdiv0'].fillna(value=0)
    return df_out