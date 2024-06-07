import numpy as np
from scipy.stats import norm


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
