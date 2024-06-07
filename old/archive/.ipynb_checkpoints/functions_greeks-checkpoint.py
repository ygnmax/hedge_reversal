import getpass
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
import functools as ft
from scipy.optimize import minimize

def calc_d1(vol, S, K, tau, r, q):
    ix_int = (tau < 1e-6)
    d1 = np.zeros(len(S))
    
    if np.sum(ix_int) == 0:    
        d1[~ix_int] = (np.log(S[~ix_int] / K[~ix_int]) + (r[~ix_int] - q[~ix_int] + vol[~ix_int] ** 2 / 2) * tau[~ix_int]) / np.sqrt(vol[~ix_int] ** 2 * tau[~ix_int])    
    else:
        d1[~ix_int] = (np.log(S[~ix_int] / K[~ix_int]) + (r[~ix_int] - q[~ix_int] + vol[~ix_int] ** 2 / 2) * tau[~ix_int]) / np.sqrt(vol[~ix_int] ** 2 * tau[~ix_int])
        d1[ix_int] = (np.log(S[ix_int] / K[ix_int]) + (r[ix_int] - q[ix_int] + vol[ix_int] ** 2 / 2) * tau[ix_int]) / np.sqrt(vol[ix_int] ** 2 * (tau[ix_int] + 1e-10))
    
    return d1


def bs_call_delta(vol, S, K, tau, r, q):
    delta = np.exp(-q * tau) * norm.cdf(calc_d1(vol, S, K, tau, r, q))
    return delta


def bs_put_delta(vol, S, K, tau, r, q):
    return np.exp(-q * tau) * (bs_call_delta(vol, S, K, tau, r, q) - 1.)

def bs_gamma(vol, S, K, tau, r, q):
    """
    Calls and puts have same gamma.
    """

    d1 = calc_d1(vol, S, K, tau, r, q)
    ix_int = (tau < 1e-6)
    gamma = np.zeros(len(S))
    if np.sum(ix_int) == 0:   
        gamma[~ix_int] = np.exp(-q[~ix_int] * tau[~ix_int]) * norm.pdf(d1[~ix_int]) / (S[~ix_int] * np.sqrt(tau[~ix_int]) * vol[~ix_int])
    else:
        gamma[ix_int] = np.exp(-q[ix_int] * tau[ix_int]) * norm.pdf(d1[ix_int]) / (S[ix_int] * np.sqrt(tau[ix_int] + 1e-10) * vol[ix_int])
        gamma[~ix_int] = np.exp(-q[~ix_int] * tau[~ix_int]) * norm.pdf(d1[~ix_int]) / (S[~ix_int] * np.sqrt(tau[~ix_int]) * vol[~ix_int])

    return gamma


def bs_vega(vol, S, K, tau, r, q):
    d1 = calc_d1(vol, S, K, tau, r, q)
    vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)

    return vega




###################
# American Pricer1
###################
def BinTree(S, u, d, n, T, r):
    '''
    Binomial tree with dividend adjustment
    returns a list containing the binomial tree
    
    S: Stock Price
    u: np.exp(sigma * np.sqrt(t))
    d: 1.0 / u
    n: Steps of Binomial Tree
    T: Time to maturity (days)
    '''   
#     print("ex_div", ex_div)
    # Creating a binomial tree with dividends adjustment
    tree = [np.array([S])]
    for i in range(n):
        tree.append(np.concatenate((tree[-1][:1]*u, tree[-1]*d)))
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

    
def GBM(am_func, intrinsic_value, S, T, r, sigma, n, q):
    '''
    American Option Pricer with dividends adjustment
    am_func: function of comparing the intrinsic value and the value calulated by the binomial tree
    intrinsic_value: function of calculating intrinsic value 
    S: Stock Price
    T: Time To Maturity (days)
    r: Interest Rate
    v: Volatility
    n: Steps of Binomial Tree
    q: Dividends yield, percentage of S
    '''   
    np.seterr(invalid='ignore')
    t = float(T)/n
    
    u = np.exp(sigma * np.sqrt(t))
    d = 1.0/u
    p = (np.exp((r-q) * t) - d)/(u - d)
    S0 = S
    # Creating the binomial tree
    ptree = BinTree(S0, u, d, n, T, r)[::-1]
    
    # Discounting through the tree with american exercise option
    result = ft.reduce(ft.partial(am_func, np.exp(-r*t), p), map(intrinsic_value, ptree))
    return result[0]

ABM = ft.partial(GBM, American)




dict_IV = {}
def cal_iv_core(idx, group):
    dict_IV_rows = {} 
    for j, row in group.iterrows():
        if row.impl_cdiv_median == 0:
            # print(j, ' dividend rate is 0')
            dict_IV_rows[j] = row.IV0
        else:
            date = row.Date
            S = row.S0
            X = row.K
            T = row.Maturity/360.0
            r_0 = row.r
            q_0 = row.impl_cdiv_median
            IV0 = row.IV0            
            if np.isnan(IV0):
                IV0 = 1.5
            if row.CallPut == 'C':    
                def f(x):
                    return (ABM(ft.partial(VP, K=X, CallPut='C'), S, T, r_0, x, 1000, q_0)-row.V0)**2
            else:
                def f(x):
                    return (ABM(ft.partial(VP,K=X, CallPut='P'), S, T, r_0, x, 1000, q_0)-row.V0)**2
            # Optimizing
            cons = ({'type': 'ineq', 'fun' : lambda x: np.array(x), 'jac': lambda x: np.array([1.0])})
            res = minimize(f, np.array([IV0]), constraints=cons, tol = 0.0001)
            dict_IV_rows[j] = res.x[0]
        
    df_IV_rows = pd.DataFrame(data = dict_IV_rows.items(), columns = ['index', 'IV0_c'])
    df_IV_rows = df_IV_rows.set_index(['index'])
    df_IV_rows_out = group.merge(df_IV_rows, left_index = True, right_index = True, how = 'left')
    dict_IV[idx] = df_IV_rows_out
    
dict_delta = {}    
def cal_delta_core(idx, group):
    dict_V_self = {} 
    dict_V_self_up = {} 
    dict_V_self_down = {} 
    for j, row in group.iterrows():
        date = row.Date
        S = row.S0
        X = row.K
        T = row.Maturity/360.0
        IV0 = row.IV0_c
        r_0 = row.r
        q_0 = row.impl_cdiv_median

        epsilon = 0.01
        S_epsilon_up = S * (1+epsilon)
        S_epsilon_down = S * (1-epsilon)

        if row.CallPut == 'C':
            V_self = ABM(ft.partial(VP,K=X,CallPut='C'), S, T, r_0, IV0, 1000, q_0)                 
            V_self_epsilon_up = ABM(ft.partial(VP,K=X,CallPut='C'), S_epsilon_up, T, r_0, IV0, 1000, q_0) 
            V_self_epsilon_download = ABM(ft.partial(VP,K=X,CallPut='C'), S_epsilon_down, T, r_0, IV0, 1000, q_0) 
            dict_V_self[j] = V_self
            dict_V_self_up[j] = V_self_epsilon_up
            dict_V_self_down[j] = V_self_epsilon_download
        else:
            V_self = ABM(ft.partial(VP,K=X,CallPut='P'), S, T, r_0, IV0, 1000, q_0)                 
            V_self_epsilon_up = ABM(ft.partial(VP,K=X,CallPut='P'), S_epsilon_up, T, r_0, IV0, 1000, q_0) 
            V_self_epsilon_download = ABM(ft.partial(VP,K=X,CallPut='P'), S_epsilon_down, T, r_0, IV0, 1000, q_0) 
            dict_V_self[j] = V_self
            dict_V_self_up[j] = V_self_epsilon_up
            dict_V_self_down[j] = V_self_epsilon_download
    
    df_V_self = pd.DataFrame(data = dict_V_self.items(), columns = ['index', 'V_self'])
    df_V_self_epsilon_up = pd.DataFrame(data = dict_V_self_up.items(), columns = ['index', 'V_up'])
    df_V_self_epsilon_down = pd.DataFrame(data = dict_V_self_down.items(), columns = ['index', 'V_down'])
    
    df_V_self = df_V_self.set_index(['index'])
    df_V_self_epsilon_up = df_V_self_epsilon_up.set_index(['index'])
    df_V_self_epsilon_down = df_V_self_epsilon_down.set_index(['index'])
    
    df_V_out = group.merge(df_V_self, left_index = True, right_index = True, how = 'left')
    df_V_out = df_V_out.merge(df_V_self_epsilon_up, left_index = True, right_index = True, how = 'left')
    df_V_out = df_V_out.merge(df_V_self_epsilon_down, left_index = True, right_index = True, how = 'left')
    dict_delta[idx] = df_V_out