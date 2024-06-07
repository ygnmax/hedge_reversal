import pandas as pd
import numpy as np

def query_options_stock(db, secid, startdate, enddate):
   
    startyear = int(startdate[0:4])
    endyear = int(enddate[0:4])

    df = pd.DataFrame()
    for yr in list(range(startyear, endyear+1, 1)):
        query = f"""
                SELECT secid, optionid, date, exdate, strike_price, cp_flag, best_bid, best_offer, impl_volatility, delta, volume, 
                open_interest, contract_size
                FROM optionm.opprcd{yr}
                WHERE secid={secid}
                AND date BETWEEN '{startdate}' AND '{enddate}'
                AND contract_size > 0
                """
        try:
            df_tmp = db.raw_sql(query, date_cols=['date', 'exdate'])
            df = pd.concat([df, df_tmp])
        except:
            print(str(secid), 'Option price is not available in year', str(yr))
            pass
    df = df.reset_index(drop=True)
    
    df = df.rename(columns={'impl_volatility': 'IV0'})

    df['tau_days'] = (df['exdate'] - df['date']).dt.days
    df['tau'] = df['tau_days'] / 360

    df['K'] = df['strike_price'] / 1000.0
    df['V0'] = (df['best_bid'] + df['best_offer']) / 2
    df = df.drop(['strike_price', 'best_bid', 'best_offer'], axis=1)

    enddate_stock = str(endyear+1) + enddate[4:]    # load one additional year
    query = f"""
            SELECT date, close, cfret
            FROM optionm.secprd
            WHERE secid={secid}
            AND date BETWEEN '{startdate}' AND '{enddate_stock}'
            ORDER BY date ASC
            """ 
    df_stock = db.raw_sql(query, date_cols=['date'])  

    query = f'''
        SELECT ex_date, distr_type, adj_factor
        FROM optionm.distrd
        WHERE secid={secid}
        AND ex_date BETWEEN '{startdate}' AND '{enddate_stock}'
        AND distr_type in ('2', '3')
        AND currency='USD'
        AND link_secid=0
        AND frequency IS NULL
        AND approx_flag='0'
        AND cancel_flag='0'
        ''' 
    distributions = db.raw_sql(query, date_cols=['ex_date'])   
    
    df_stock['splitfactor'] = np.nan
    for _,row in distributions.iterrows():
        df_stock.loc[df_stock['date'].shift(-1) == row['ex_date'], 'splitfactor'] = 1/row['adj_factor']
    df_stock.loc[df_stock.index[-1], 'splitfactor'] = 1
    df_stock['splitfactor'] = df_stock['splitfactor'][::-1].fillna(value=1).cumprod()[::-1]
    df_stock['cfret'] = df_stock['cfret'] / df_stock['splitfactor']

    df = df.merge(df_stock, how='inner', on=['date'])
    df = df.rename(columns={'close': 'S0', 'cfret': 'adj_fac0'})
    
    df_stock = df_stock.drop(['close', 'splitfactor'], axis=1)
    df = df.merge(df_stock, how='inner', left_on=['exdate'], right_on=['date'], suffixes=(None, '_stock'))
    df = df.rename(columns={'cfret': 'adj_fac_expiration'})
    df = df.drop(['date_stock'], axis=1)

    for c in ['S0', 'K', 'V0']:
        df[c] = df[c] * df['splitfactor']

    df['M0'] = df['S0'] / df['K']    
    df['open_total_interest'] = df.groupby(['date', 'cp_flag'])['open_interest'].transform('sum')
    
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