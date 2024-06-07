import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from concurrent.futures import ThreadPoolExecutor


def compute_d1(vol, S, K, tau, r, q):
    return (np.log(S / K) + (r - q + vol** 2 / 2) * tau) / np.maximum(np.sqrt(vol ** 2 * tau), 1e-8)


def bs_price(vol, S, K, tau, r, q, cp_flag):
    d1 = compute_d1(vol, S, K, tau, r, q)
    d2 = d1 - np.sqrt(vol**2 * tau)
    price = np.where(cp_flag == 'C', S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2),
                     -S * np.exp(-q * tau) * norm.cdf(-d1) + K * np.exp(-r * tau) * norm.cdf(-d2))
    return price
    

def bs_delta(vol, S, K, tau, r, q, cp_flag):
    d1 = compute_d1(vol, S, K, tau, r, q)
    delta = norm.cdf(d1) - np.where(cp_flag == 'P', 1, 0)
    return delta


def bs_IV(V0=None, S0=None, K=None, tau=None, r=None, cp_flag=None, **kargs):
    try:
        vol = brentq(
            lambda x: V0 - bs_price(vol=x, S=S0, K=K, tau=tau,  r=r, q=0, cp_flag=cp_flag),
            0.0001, 1000.)
    except ValueError:
        vol = np.nan
    return vol

def bs_IV_vec(V0, S0, K, tau, r, cp_flag):
    inputs = (V0, S0, K, tau, r, cp_flag)
    
    # in parallel
    with ThreadPoolExecutor() as executor:
        vols = list(executor.map(lambda args: bs_IV(*args), zip(*inputs)))
    
    return np.array(vols)