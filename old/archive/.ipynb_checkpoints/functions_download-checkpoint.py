import pandas as pd
import numpy as np
import wrds
db = wrds.Connection(wrds_username='ygnmaxwharton')

# Get options data and security prices
def query_options(secid, startdate = '2019-01-01', enddate = '2022-12-31'):
    '''
    secid: int. Example: secid = 113993
    startdate: str. Example: startdate = '2003-01-06'
    enddate: str. Example: enddate = '2004-06-07'
    '''
    
    startyear = int(startdate[0:4])
    endyear = int(enddate[0:4])

    df_all = pd.DataFrame()
    for yr in list(range(startyear, endyear+1, 1)):
        # Get the data for the options
        sql = '''
        SELECT *
        FROM optionm.opprcd%s
        WHERE secid=%s
        ORDER BY date ASC
        ''' % (yr, secid)
        try:
            df_tmp = db.raw_sql(sql, date_cols=['date', 'exdate', 'last_date'])
            df_tmp[['secid', 'strike_price', 'optionid', 'am_settlement', 'contract_size']] = df_tmp[['secid', 'strike_price', 'optionid', 'am_settlement', 'contract_size']].astype('int', errors = 'ignore')
            df_all = pd.concat([df_all, df_tmp])
        except:
            print(str(secid), 'option price is not available in', str(yr))
            pass
    df_all = df_all.reset_index().drop(columns='index')
    return df_all

def query_StdOptions(secid, startdate = '2019-01-01', enddate = '2022-12-31'):
    '''
    secid: int. Example: secid = 113993
    startdate: str. Example: startdate = '2003-01-06'
    enddate: str. Example: enddate = '2004-06-07'
    '''
    
    startyear = int(startdate[0:4])
    endyear = int(enddate[0:4])

    df_all = pd.DataFrame()
    for yr in list(range(startyear, endyear+1, 1)):
        sql = '''
        SELECT *
        FROM optionm.stdopd%s
        WHERE secid=%s
        ORDER BY date ASC
        ''' % (yr, secid)
        try:
            df_tmp = db.raw_sql(sql, date_cols=['date'])
            df_tmp[['secid', 'days']] = df_tmp[['secid', 'days']].astype('int', errors = 'ignore')
            df_all = pd.concat([df_all, df_tmp])
        except:
            print(str(secid), 'standarized options are not available in', str(yr))
            pass   
        
    df_all = df_all.reset_index().drop(columns='index')

    return df_all

def query_hisvol(secid, startdate = '2019-01-01', enddate = '2022-12-31'):
    '''
    secid: int. Example: secid = 113993
    startdate: str. Example: startdate = '2003-01-06'
    enddate: str. Example: enddate = '2004-06-07'
    '''
    
    startyear = int(startdate[0:4])
    endyear = int(enddate[0:4])

    df_all = pd.DataFrame()
    for yr in list(range(startyear, endyear+1, 1)):
        sql = '''
        SELECT *
        FROM optionm.hvold%s
        WHERE secid=%s
        ORDER BY date ASC
        ''' % (yr, secid)
        try:
            df_tmp = db.raw_sql(sql, date_cols=['date'])
            df_tmp[['secid', 'days']] = df_tmp[['secid', 'days']].astype('int', errors = 'ignore')
            df_all = pd.concat([df_all, df_tmp])
        except:
            print(str(secid), 'historical volatility are not available in', str(yr))
            pass      
        
    df_all = df_all.reset_index().drop(columns='index')
    return df_all

def query_volsurf(secid, startdate = '2019-01-01', enddate = '2022-12-31'):
    '''
    secid: int. Example: secid = 113993
    startdate: str. Example: startdate = '2003-01-06'
    enddate: str. Example: enddate = '2004-06-07'
    '''
    
    startyear = int(startdate[0:4])
    endyear = int(enddate[0:4])

    df_all = pd.DataFrame()
    for yr in list(range(startyear, endyear+1, 1)):
        sql = '''
        SELECT *
        FROM optionm.vsurfd%s
        WHERE secid=%s
        ORDER BY date ASC
        ''' % (yr, secid)
        try:
            df_tmp = db.raw_sql(sql, date_cols=['date'])
            df_tmp[['secid', 'days']] = df_tmp[['secid', 'days']].astype('int', errors = 'ignore')
            df_all = pd.concat([df_all, df_tmp])
        except:
            print(str(secid), 'historical volatility are not available in', str(yr))
            pass      
    df_all = df_all.reset_index().drop(columns='index')
    return df_all

def query_volume(secid, startdate = '2019-01-01', enddate = '2022-12-31'): 
    sql = '''
    SELECT *
    FROM optionm.opvold
    WHERE secid=%s
    AND date BETWEEN '%s' AND '%s'
    ORDER BY date ASC
    ''' % (secid, startdate, enddate)
    output = db.raw_sql(sql, date_cols=['date'])
    if len(output) > 0:
        output['secid'] = output['secid'].astype('int')
    return output

def query_forward(secid, startdate = '2019-01-01', enddate = '2022-12-31'): 
    sql = '''
    SELECT *
    FROM optionm.fwdprd
    WHERE secid=%s
    AND date BETWEEN '%s' AND '%s'
    ORDER BY date ASC
    ''' % (secid, startdate, enddate)
    output = db.raw_sql(sql, date_cols=['date'])
    if len(output) > 0:
        output['secid'] = output['secid'].astype('int')
    return output

def query_stock(secid, startdate = '2019-01-01', enddate = '2022-12-31'):
    """
    Function for query stock price during a certain period
    output: a dataframe of stock daily trading information, including the open, high, low, close, adjustment factor, volumes
    """   
    sql = '''
    SELECT *
    FROM optionm.secprd
    WHERE secid=%s
    AND date BETWEEN '%s' AND '%s'
    ORDER BY date ASC
    ''' % (secid, startdate, enddate)

    output = db.raw_sql(sql)
    if len(output) > 0:
        output[['secid', 'volume', 'shrout']] = output[['secid', 'volume', 'shrout']].astype('int') 
        
    return output


def query_dividend(secid, startdate = '2019-01-01', enddate = '2022-12-31'):
    """
    Function for query stock distribution inforamtion during a certain period
    output: a dataframe of stock distribution information, including dividend, split, spin-off, etc.
    """     
    # get distributions
    sql = '''
    SELECT *
    FROM optionm.distrd
    WHERE secid=%s
    ''' % secid
    
    dist = db.raw_sql(sql, date_cols=['ex_date'])
    if len(dist) > 0:
        dist['secid'] = dist['secid'].astype('int')
        dist = dist[(startdate <= dist['ex_date']) & (dist['ex_date']<=enddate)].reset_index(drop = True)        
    return dist


def query_info(secid):
    """
    Function for query stock basic inforamtion during a certain period
    output: a dataframe of stock basic information, including name, ticker, SecurityID, etc.
    """    
    sql = '''
    SELECT *
    FROM optionm.secnmd
    WHERE secid='%s'
    ''' % secid
    output = db.raw_sql(sql)
    if len(output) > 0:
        output['secid'] = output['secid'].astype('int')
    return output


def download_zero_curve(startdate = '2019-01-01', enddate = '2022-12-31'):
    sql = '''
    SELECT *
    FROM optionm.zerocd
    WHERE date BETWEEN '%s' AND '%s'
    ''' % (startdate, enddate)
    zero_curve = db.raw_sql(sql, date_cols=['date'])
    zero_curve['days'] = zero_curve['days'].astype('int')
    return zero_curve