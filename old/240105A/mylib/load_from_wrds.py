import pandas as pd

def query_options_stock(db, secid, startdate='2019-01-01', enddate='2022-12-31'):
   
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

    query = '''
    SELECT date, close, cfret
    FROM optionm.secprd
    WHERE secid=%s
    AND date BETWEEN '%s' AND '%s'
    ORDER BY date ASC
    ''' % (secid, startdate, enddate)
    df_stock = db.raw_sql(query, date_cols=['date'])  
    df_stock['S0'] = df_stock['close'] * df_stock['cfret'] / df_stock.loc[len(df_stock)-1]['cfret']
    df_stock = df_stock.drop(['close', 'cfret'], axis=1)
    df = df.merge(df_stock, how='inner', on=['date'])
    
    df['M0'] = df['S0'] / df['K']    
    df = df.sort_values(by=['date', 'exdate', 'cp_flag', 'K'])
    return df


def query_dividend(db, secid, startdate='2019-01-01', enddate='2022-12-31'):
    query = '''
    SELECT secid, record_date, distr_type, amount, ex_date
    FROM optionm.distrd
    WHERE secid=%s
    AND ex_date BETWEEN '%s' AND '%s'
    ''' %  (secid, startdate, enddate)
    
    df = db.raw_sql(query, date_cols=['ex_date'])     
    return df
    

def query_zero_curve(db, startdate='2019-01-01', enddate='2022-12-31'):
    query = '''
    SELECT date, days, rate
    FROM optionm.zerocd
    WHERE date BETWEEN '%s' AND '%s'
    ''' % (startdate, enddate)
    zero_curve = db.raw_sql(query, date_cols=['date'])
    zero_curve['days'] = zero_curve['days'].astype('int')
    return zero_curve