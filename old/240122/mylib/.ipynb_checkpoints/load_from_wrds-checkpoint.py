import pandas as pd

def query_options_stock(db, secid, startdate, enddate):
   
    startyear = int(startdate[0:4])
    endyear = int(enddate[0:4])

    df = pd.DataFrame()
    for yr in list(range(startyear, endyear+1, 1)):
        query = '''
        SELECT secid, date, exdate, strike_price, cp_flag, best_bid, best_offer, impl_volatility, delta, volume, open_interest
        FROM optionm.opprcd%s
        WHERE secid=%s
        AND contract_size > 0
        ORDER BY date ASC
        ''' % (yr, secid)
        try:
            df_tmp = db.raw_sql(query, date_cols=['date', 'exdate'])
            df = pd.concat([df, df_tmp])
        except:
            print(str(secid), 'option price is not available in year', str(yr))
            pass
    df = df.reset_index(drop=True)

    df['tau_days'] = (df['exdate'] - df['date']).dt.days
    df['tau'] = df['tau_days'] / 360

    df['K'] = df['strike_price'] / 1000.0
    df['V0'] = (df['best_bid'] + df['best_offer']) / 2
    df = df.drop(['strike_price', 'best_bid', 'best_offer'], axis=1)

    enddate_stock = str(endyear+1) + enddate[4:]    # load one additional year
    query = '''
    SELECT date, close, cfret
    FROM optionm.secprd
    WHERE secid=%s
    AND date BETWEEN '%s' AND '%s'
    ORDER BY date ASC
    ''' % (secid, startdate, enddate_stock)
    df_stock = db.raw_sql(query, date_cols=['date'])  
    #df_stock['S0'] = df_stock['close'] * df_stock['cfret'] / df_stock.loc[len(df_stock)-1]['cfret']
    #df['K'] = df_stock['K'] * df_stock['cfret'] / df_stock.loc[len(df_stock)-1]['cfret']   !!! think how to do adjustments etc
    #df['V0'] = ...
    df = df.merge(df_stock, how='inner', on=['date'])
    df = df.rename(columns={'close': 'S0', 'cfret': 'adj_fac0'})
    df_stock = df_stock.drop(['close'], axis=1)
    df = df.merge(df_stock, how='inner', left_on=['exdate'], right_on=['date'], suffixes=(None, '_stock'))
    df = df.rename(columns={'cfret': 'adj_fac1'})
    df = df.drop(['date_stock'], axis=1)
    
    df['M0'] = df['S0'] / df['K']    
    df = df.sort_values(by=['date', 'exdate', 'cp_flag', 'K'])
    return df


def query_zero_curve(db, startdate, enddate):
    query = '''
    SELECT date, days, rate
    FROM optionm.zerocd
    WHERE date BETWEEN '%s' AND '%s'
    ''' % (startdate, enddate)
    zero_curve = db.raw_sql(query, date_cols=['date'])
    zero_curve['days'] = zero_curve['days'].astype('int')
    return zero_curve