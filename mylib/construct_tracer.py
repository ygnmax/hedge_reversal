import pandas as pd
import numpy as np
from mylib.rates_tools import merge_interest
from mylib.BS_formulas import bs_price, bs_delta, bs_IV_vec

def construct_tracer(df, df_rate, cp_flag, target_tau_days, ALLOWED_ADJ_FACTOR_CHANGE=[0.99, 1.01]):
    #needs to be called for each secid separately
    df_select = df[(df['cp_flag'] == cp_flag) ]

    d_list = []
    S0_list = []
    IV_interp_list = []
    IV1_list = []
    S1_list = []
    tau1_list = []
    r1_list = []   
    V0_interp_list = []
    V1_interp_list = []
    
    for d, dfg in df_select.groupby('date'):
        
        target_K = dfg.iloc[0]['S0']
        
        upper_left  = dfg[(dfg['tau_days'] <= target_tau_days) & (dfg['K'] >= target_K)] 
        upper_right = dfg[(dfg['tau_days'] >= target_tau_days) & (dfg['K'] >= target_K)] 
        lower_left  = dfg[(dfg['tau_days'] <= target_tau_days) & (dfg['K'] <= target_K)] 
        lower_right = dfg[(dfg['tau_days'] >= target_tau_days) & (dfg['K'] <= target_K)] 
        

        if upper_left.empty | upper_right.empty | lower_left.empty | lower_right.empty:
            continue
        
        #First interpolate strikes above ATM, then below ATM
        upper_left  = upper_left[upper_left['tau_days'] == upper_left['tau_days'].max()]
        upper_left  = upper_left[upper_left['K'] == upper_left['K'].min()]        
        upper_right = upper_right[upper_right['tau_days'] == upper_right['tau_days'].min()]
        upper_right = upper_right[upper_right['K'] == upper_right['K'].min()]
        lower_left  = lower_left[lower_left['tau_days'] == lower_left['tau_days'].max()]
        lower_left  = lower_left[lower_left['K'] == lower_left['K'].max()]   
        lower_right = lower_right[lower_right['tau_days'] == lower_right['tau_days'].min()]
        lower_right = lower_right[lower_right['K'] == lower_right['K'].max()]       

        upper_left = upper_left.select_dtypes(include='number').mean()    # in case there are several options with same strike and expiry
        upper_right = upper_right.select_dtypes(include='number').mean()  
        lower_left = lower_left.select_dtypes(include='number').mean()  
        lower_right = lower_right.select_dtypes(include='number').mean()   
        
        adj_change_upper = upper_right['adj_fac_expiration'] / upper_left['adj_fac0']
        adj_change_lower = lower_right['adj_fac_expiration'] / lower_left['adj_fac0']
        bl = (
                (ALLOWED_ADJ_FACTOR_CHANGE[0] > min(adj_change_upper, adj_change_lower)) | 
                (max(adj_change_upper, adj_change_lower)  > ALLOWED_ADJ_FACTOR_CHANGE[1])
             )
        if bl:
            continue
            # no tracer computed due to large changes in adjustment factor
        
        IV0, K, IV1, S1, tau1, r1, V0, V1 = [], [], [], [], [], [], [], []
        for l, r in zip([upper_left, lower_left], [upper_right, lower_right]):
            if r['tau_days'] > l['tau_days']:
                weight_right = (target_tau_days - l['tau_days']) / (r['tau_days'] - l['tau_days'])
            else:
                weight_right = 1     
            # if the difference in the numerator is zero then any weight would do it as the options are the same
            
            IV0.append((1 - weight_right) * l['IV0'] + weight_right * r['IV0'])
            K.append((1 - weight_right) * l['K'] + weight_right * r['K'])
            IV1.append((1 - weight_right) * l['IV1'] + weight_right * r['IV1'])
            # the following 3 should be the same across all four options but in case it's not ...
            S1.append((1 - weight_right) * l['S1'] + weight_right * r['S1'])  
            tau1.append((1 - weight_right) * l['tau1'] + weight_right * r['tau1'])
            r1.append((1 - weight_right) * l['r1'] + weight_right * r['r1'])
            V0.append((1 - weight_right) * l['V0'] + weight_right * r['V0'])
            V1.append((1 - weight_right) * l['V1'] + weight_right * r['V1'])

        if K[0] > K[1]:
            weight_upper = (target_K - K[1]) / (K[0] - K[1])
        else:
            weight_upper = 1

        d_list.append(d)
        S0_list.append(target_K)
        IV_interp_list.append((1-weight_upper) * IV0[0] + weight_upper * IV0[1]) 
        IV1_list.append((1-weight_upper) * IV1[0] + weight_upper * IV1[1]) 
        S1_list.append((1-weight_upper) * S1[0] + weight_upper * S1[1]) 
        tau1_list.append((1-weight_upper) * tau1[0] + weight_upper * tau1[1]) 
        r1_list.append((1-weight_upper) * r1[0] + weight_upper * r1[1]) 
        V0_interp_list.append((1-weight_upper) * V0[0] + weight_upper * V0[1])
        V1_interp_list.append((1-weight_upper) * V1[0] + weight_upper * V0[1])
        
    df_out = pd.DataFrame({'date': d_list, 'S0': S0_list, 'S1': S1_list, 'IV0': IV_interp_list})
    df_out['K'] = df_out['S0']
    df_out['cp_flag'] = cp_flag
    df_out['tau_days'] = target_tau_days
    df_out['tau'] = target_tau_days / 360
    df_out = merge_interest(df_out, df_rate)

    # !!!Would be better to have an American option pricer but let's leave it for the moment.
    df_out['V0'] = bs_price(df_out['IV0'], df_out['S0'], df_out['K'], target_tau_days / 360, df_out['r'], 0, cp_flag)
    bl = df_out['IV0'].isna()
    df_out.loc[bl, 'V0'] = np.array(V0_interp_list)[bl]
    df_out.loc[bl, 'IV0'] = bs_IV_vec(
        df_out.loc[bl, 'V0'], df_out.loc[bl, 'S0'], df_out.loc[bl, 'K'], df_out.loc[bl, 'tau'], df_out.loc[bl, 'r'], df_out.loc[bl, 'cp_flag'])
    df_out['V0_interpolated'] = bl
    
    
    df_out['V1'] = bs_price(np.array(IV1_list), np.array(S1_list), df_out['K'], np.array(tau1_list), np.array(r1_list), 0, cp_flag)    
    bl = df_out['V1'].isna()
    df_out.loc[bl, 'V1'] = np.array(V1_interp_list)[bl] 
    df_out['V1_interpolated'] = bl    

    
    df_out['normDelta'] = bs_delta(df_out['IV0'], df_out['K'], df_out['K'], df_out['tau'], df_out['r'], 0, cp_flag)

    return df_out