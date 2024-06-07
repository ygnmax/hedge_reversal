import pandas as pd
from mylib.rates_tools import merge_interest
from mylib.BS_formulas import bs_price

def construct_tracer(df, df_rate, cp_flag, target_tau):
    df_select = df[df['cp_flag'] == cp_flag]

    d_list = []
    close_list = []
    V_interp_list = []
    IV_interp_list = []
    
    for d, dfg in df_select.groupby('date'):
        target_K = dfg.iloc[0]['close']

        upper_left  = dfg[(dfg['tau_days'] <= target_tau) & (dfg['K'] >= target_K)] 
        upper_right = dfg[(dfg['tau_days'] >= target_tau) & (dfg['K'] >= target_K)] 
        lower_left  = dfg[(dfg['tau_days'] <= target_tau) & (dfg['K'] <= target_K)] 
        lower_right = dfg[(dfg['tau_days'] >= target_tau) & (dfg['K'] <= target_K)] 

        
        #First interpolate strikes above ATM, then below ATM
        upper_left  = upper_left[upper_left['tau_days'] == upper_left['tau_days'].max()]
        upper_left  = upper_left[upper_left['K'] == upper_left['K'].min()]        
        upper_right = upper_right[upper_right['tau_days'] == upper_right['tau_days'].min()]
        upper_right = upper_right[upper_right['K'] == upper_right['K'].min()]
        lower_left  = lower_left[lower_left['tau_days'] == lower_left['tau_days'].max()]
        lower_left  = lower_left[lower_left['K'] == lower_left['K'].max()]   
        lower_right = lower_right[lower_right['tau_days'] == lower_right['tau_days'].min()]
        lower_right = lower_right[lower_right['K'] == lower_right['K'].max()]   

        IV, K = [], []
        for l, r in zip([upper_left, lower_left], [upper_right, lower_right]):
            if r.iloc[0]['tau_days'] > l.iloc[0]['tau_days']:
                weight_right = (target_tau - l.iloc[0]['tau_days']) / (r.iloc[0]['tau_days'] - l.iloc[0]['tau_days'])
            else:
                weight_right = 1     
            # if the difference in the numerator is zero then any weight would do it as the options are the same
            
            IV.append((1 - weight_right) * l.iloc[0]['impl_volatility'] + weight_right * r.iloc[0]['impl_volatility'])
            K.append((1 - weight_right) * l.iloc[0]['K'] + weight_right * r.iloc[0]['K'])

        if K[0] > K[1]:
            weight_upper = (target_K - K[1]) / (K[0] - K[1])
        else:
            weight_upper = 1

        d_list.append(d)
        close_list.append(target_K)
        IV_interp_list.append((1-weight_upper) * IV[0] + weight_upper * IV[1]) 


    df_out = pd.DataFrame({'date': d_list, 'close': close_list, 'IV_interp': IV_interp_list})
    df_out['K'] = df_out['close']
    df_out['cp_flag'] = cp_flag
    df_out['tau_days'] = target_tau
    df_out = merge_interest(df_out, df_rate)
    df_out['V0'] = bs_price(df_out['IV_interp'], df_out['close'], df_out['K'], target_tau / 360, df_out['r'], 0, cp_flag)
    # !!!Would be better to have an American option pricer but let's leave it for the moment.
    return df_out