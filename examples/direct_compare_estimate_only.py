#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:34:57 2021

@author: huawang
"""
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from old_edgeworth_eps_delta import *
from eps_delta_edgeworth import * 
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_mu_poisson
from prv_accountant import Accountant
import numpy as np
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition 
from autodp import mechanism_zoo, transformer_zoo
from prv_accountant import Accountant



def rdp(step, sigma, delta, prob):
    mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
    subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=True) # by default this is using poisson sampling
      
    SubsampledGaussian_mech = subsample(mech,prob,improved_bound_flag=True)
    compose = transformer_zoo.Composition()
    mech = compose([SubsampledGaussian_mech],[step])
    rdp_total = mech.RenyiDP
    noisysgd = Mechanism()
    noisysgd.propagate_updates(rdp_total,type_of_update='RDP')
    return mech.get_approxDP(delta=delta)


#%%

## The Figure 1(b)'s parameters:
#delta = 1e-7
epoch = 2
prob = 1e-2
num_examples = 100000
batchsize = num_examples * prob
sigma = 0.8
number_lst = points = [20, 50, 100, 200, 500, 1000, 2000]
total_epoches = 1 / prob
batches = [batchsize * i for i in points]
delta = 0.015

key = (delta, prob, sigma)

filename = "data/direct_compare/Figure 1(b).pickle"
if os.path.isfile(filename):
    with open(filename, "rb") as f:
        cache_list = pickle.load(f)
else:
    cache_list = {}
# load caches to save time if possible
if key in cache_list:
    cache = cache_list[key]
    eps_gdp = cache["eps_gdp"]
    eps_ew_est = cache["eps_ew_est"]
    eps_low = cache["eps_low"]
    eps_upper = cache["eps_upper"]
    eps_rdp = cache["eps_rdp"]
    eps_ew_est2 = cache["eps_ew_est2"]
    eps_ew_est3 = cache["eps_ew_est3"]
else:
    cache = {}
    # GDP:
    print("\n\nNow calculating GDP...")
    eps_gdp = [compute_eps_poisson(epoch * pt / total_epoches , sigma, num_examples, batchsize, delta) for pt in points]
    # FFT:
    print("\n\nNow calculating FFT...")
    accountant = Accountant(
     	noise_multiplier=sigma,
     	sampling_probability=prob,
     	delta=delta,
     	eps_error=0.1,
    max_compositions = 2500
    )
    results = [accountant.compute_epsilon(num_compositions=pt) for pt in points]
    eps_low, _, eps_upper = [item[0] for item in results], [item[1] for item in results], [item[2] for item in results]
    # RDP:
    print("\n\nNow calculating RDP...")
    eps_rdp = [rdp(step, sigma, delta, prob) for step in points]
    # Edgeworth:
    print("\n\nNow calculating Edgeworth...")
    sgd = GaussianSGD(sigma = sigma, p = prob, order = 1)
    eps_ew_est = [sgd.approx_eps_from_delta_edgeworth(delta, n, method = "estimate") for n in points]
    eps_ew_est2 = DP_SGD_eps_list_at_delta(sigma, points, prob, delta, order = 2)
    eps_ew_est3 = DP_SGD_eps_list_at_delta(sigma, points, prob, delta, order = 3)
    # save
    cache["eps_gdp"] = eps_gdp
    cache["eps_low"] = eps_low
    cache["eps_upper"] = eps_upper
    cache["eps_rdp"] = eps_rdp
    cache["eps_ew_est"] = eps_ew_est
    cache["eps_ew_est2"] = eps_ew_est2
    cache["eps_ew_est3"] = eps_ew_est2
    cache_list[delta] = cache

#%%
    ## save it to pickle
    cache_list[key] = cache
    with open(filename, "wb") as f:
        pickle.dump(cache_list, f)


#%%  
# Plot
figure(figsize=(4, 4))
plt.plot(number_lst, eps_low, label = "FFT_LOW", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_upper, label = "FFT_UPP", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_gdp, label = "GDP")
#plt.plot(number_lst, eps_ew_est, label = "EW_EST")
#plt.plot(number_lst, eps_rdp, label = "RDP")
plt.plot(number_lst, eps_ew_est2, label = "EW_EST")
# plt.plot(number_lst, eps_ew_est2, label = "3rd_EW_EST")
plt.legend(fontsize=10)
plt.ylabel(r"$\epsilon$", fontsize=15, rotation=90)
plt.xlabel("m", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


#plt.title(r"$\delta$" + f" = {delta}")
#plt.title("Eps as function of iterations.")

plt.savefig(f"figs/direct_compare/Figure1(b)data_delta={delta}_no_RDP.pdf", format='pdf',  bbox_inches = 'tight')
plt.show()

figure(figsize=(4, 4))
plt.plot(number_lst, eps_low, label = "FFT_LOW", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_upper, label = "FFT_UPP", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_gdp, label = "GDP")
#plt.plot(number_lst, eps_ew_est, label = "EW_EST")
plt.plot(number_lst, eps_ew_est2, label = "EW_EST")
plt.plot(number_lst, eps_rdp, label = "RDP")
#plt.plot(number_lst, eps_ew_est2, label = "3rd_EW_EST")

plt.legend(fontsize=10)
plt.ylabel(r"$\epsilon$", fontsize=15, rotation=90)
plt.xlabel("m", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title(r"$\delta$" + f" = {delta}")
#plt.title("Eps as function of iterations.")

plt.savefig(f"figs/direct_compare/Figure1(b)data_delta={delta}.pdf", format='pdf',  bbox_inches = 'tight')
plt.show()



#%%

epoch = 1
num_examples = 54384336
batchsize = 24000
prob = 0.0005
sigma = 1
number_lst = points = [200, 500, 1000, 2000, 5000, 10000, 20000]#[1000, 2500, 5000, 10000, 25000, 50000, 100000]
total_epoches = 1 / prob
batches = [batchsize * i for i in points]
delta = 0.00001

key = (delta, prob, sigma)


filename = "data/direct_compare/FB setting.pickle"

if os.path.isfile(filename):
    with open(filename, "rb") as f:
        cache_list = pickle.load(f)
else:
    cache_list = {}
# load caches to save time if possible
if key in cache_list:
    cache = cache_list[key]
    eps_gdp = cache["eps_gdp"]
    #eps_ew_est = cache["eps_ew_est"]
    eps_low = cache["eps_low"]
    eps_upper = cache["eps_upper"]
    eps_rdp = cache["eps_rdp"]
    eps_ew_est2 = cache["eps_ew_est2"]
    eps_ew_est3 = cache["eps_ew_est3"]
else:
    cache = {}
    # GDP:
    print("\n\nNow calculating GDP...")
    eps_gdp = [compute_eps_poisson(epoch * pt / total_epoches , sigma, num_examples, batchsize, delta) for pt in points]
    # FFT:
    print("\n\nNow calculating FFT...")
    accountant = Accountant(
     	noise_multiplier=sigma,
     	sampling_probability=prob,
     	delta=delta,
     	eps_error=0.002,
    max_compositions = 250000
    )
    results = [accountant.compute_epsilon(num_compositions=pt) for pt in points]
    eps_low, _, eps_upper = [item[0] for item in results], [item[1] for item in results], [item[2] for item in results]
    # RDP:
    print("\n\nNow calculating RDP...")
    eps_rdp = [rdp(step, sigma, delta, prob) for step in points]
    # Edgeworth:
    print("\n\nNow calculating Edgeworth...")
    #sgd = GaussianSGD(sigma = sigma, p = prob, order = 1)
    #eps_ew_est = [sgd.approx_eps_from_delta_edgeworth(delta, n, method = "estimate") for n in points]
    eps_ew_est2 = DP_SGD_eps_list_at_delta(sigma, points, prob, delta, order = 2)
    eps_ew_est3 = DP_SGD_eps_list_at_delta(sigma, points, prob, delta, order = 3)
    # save
    cache["eps_gdp"] = eps_gdp
    cache["eps_low"] = eps_low
    cache["eps_upper"] = eps_upper
    cache["eps_rdp"] = eps_rdp
    #cache["eps_ew_est"] = eps_ew_est
    cache["eps_ew_est2"] = eps_ew_est2
    cache["eps_ew_est3"] = eps_ew_est2
    cache_list[delta] = cache

#%%
    ## save it to pickle
    cache_list[key] = cache
    with open(filename, "wb") as f:
        pickle.dump(cache_list, f)


#%%  
# Plot
figure(figsize=(4, 4))
plt.plot(number_lst, eps_low, label = "FFT_LOW", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_upper, label = "FFT_UPP", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_gdp, label = "GDP")
#plt.plot(number_lst, eps_rdp, label = "RDP")
#plt.plot(number_lst, eps_ew_est, label = "EW_EST")
plt.plot(number_lst, eps_ew_est2, label = "EW_EST")
plt.plot(number_lst, eps_rdp, label = "RDP")
# plt.plot(number_lst, eps_ew_est2, label = "3rd_EW_EST")

plt.legend(fontsize=10, loc = 6)
plt.ylabel(r"$\epsilon$", fontsize=15, rotation=90)
plt.xlabel("m", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#plt.title(r"$\delta$" + f" = {delta}")
#plt.title("Eps as function of iterations.")

plt.savefig(f"figs/direct_compare/FB_usage_data_delta={delta}_no_RDP.pdf", format='pdf',  bbox_inches = 'tight')
plt.show()

figure(figsize=(4, 4))

plt.plot(number_lst, eps_low, label = "FFT_LOW", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_upper, label = "FFT_UPP", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_gdp, label = "GDP")
#plt.plot(number_lst, eps_rdp, label = "RDP")
#plt.plot(number_lst, eps_ew_est, label = "EW_EST")
plt.plot(number_lst, eps_ew_est2, label = "EW_EST")
#plt.plot(number_lst, eps_ew_est2, label = "3rd_EW_EST")

plt.legend(fontsize=10)
plt.ylabel(r"$\epsilon$", fontsize=15, rotation=90)
plt.xlabel("m", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title(r"$\delta$" + f" = {delta}")
#plt.title("Eps as function of iterations.")

plt.savefig(f"figs/direct_compare/FB_usage_data_delta={delta}.pdf", format='pdf',  bbox_inches = 'tight')
plt.show()



