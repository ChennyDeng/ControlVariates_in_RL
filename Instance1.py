# -*- codeing = utf-8 -*-
# @Time :11/09/2021 14:54
# @Author :Chenny
# @Site :
# @File :Instance 1.py
# @Software :PyCharm

import numpy as np
from numpy.lib.shape_base import kron
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import bisect
from scipy.stats import entropy
from scipy.stats.morestats import probplot

from tqdm import tqdm


# ########################## Plotting functions #########################
# Getting Average regret and Confidence interval
def accumulative_regret_error(regret):
    time_horizon = [0]
    samples = len(regret[0]) # number of rounds
    runs = len(regret)       # number of runs
    batch = samples / 10     # number of batches; results in 11 candidates time step

    # Time horizon
    t = 0
    while True:
        t += 1
        if time_horizon[-1] + batch > samples:
            if time_horizon[-1] != samples:
                time_horizon.append(time_horizon[-1] + samples % batch)
            break
        time_horizon.append(time_horizon[-1] + batch)

    # Mean batch regret of R runs
    avg_batched_regret = []
    for r in range(runs):
        # reset for each run
        count = 0
        accumulative_regret = 0
        batch_regret = [0]
        for s in range(samples):
            # Calculate the accumulative regret at each round
            count += 1
            accumulative_regret += regret[r][s]

            # Only record the accumulative regret if it meets these two conditions
            # condition 1: time step = a multiple of the given batch number
            if count == batch:
                batch_regret.append(accumulative_regret)
                count = 0
        # condition 2: time step = last round
        if samples % batch != 0:
            batch_regret.append(accumulative_regret)

        # Record the regret of all experiments at candidate rounds
        avg_batched_regret.append(batch_regret)
    # Calculate the average regret over all experiments
    regret = np.mean(avg_batched_regret, axis=0)

    # Confidence interval
    conf_regret = []
    freedom_degree = runs - 1
    for r in range(len(avg_batched_regret[0])):
        # Compute the CI
        conf_regret.append(ss.t.ppf(0.95, freedom_degree) *
                           ss.sem(np.array(avg_batched_regret)[:, r])) # ss.sem: Compute the standard error of the mean of the values in the input array.
    return time_horizon, regret, conf_regret


# Regret Plotting
def regret_plotting(regret, cases, plotting_info):
    colors = list("gbcmryk")
    shape = ['--^', '--v', '--H', '--d', '--+', '--*']

    # Scatter Error bar with scatter plot
    for c in range(cases):
        horizon, batched_regret, error = accumulative_regret_error(np.array(regret)[:, c])
        # np.array(algos_regret)[:, c].shape  # (runs,rounds)
        # np.array(algos_regret).shape        # (runs,cases,rounds)

        # plot the CI with the selected color
        plt.errorbar(horizon, batched_regret, error, color=colors[c])
        # plot the average cumulative regret at the candidate rounds over the experiments with the selected color and selected shape
        plt.plot(horizon, batched_regret, colors[c] + shape[c], label=plotting_info[4][c])

    plt.rc('font', size=10)  # controls default text sizes
    plt.title(plotting_info[2])
    plt.legend(loc='upper left', numpoints=1)  # Location of the legend
    plt.xlabel(plotting_info[0], fontsize=15)
    plt.ylabel(plotting_info[1], fontsize=15)

    plt.savefig(plotting_info[3], bbox_inches='tight')

    plt.close()


# #######################################################################


# #############################  Algorithms #############################
# UCB1 algorithm
def ucb1(mu, omega, sigma, sigma_w, T):
    '''
     Upper Confidence Bound 1 Algorithm for Multi-Armed Bandit Problem
     Inputs
     ============================================
     mu: mean for V
     omega: mean for W
     sigma: sd for V
     sigma_w: sd for W
     T: number of rounds (int)
     '''

    K = len(mu)  # Number of arms
    arm_rewards = np.zeros(K)  # Collected rewards for each arm
    num_pulls = np.zeros(K)  # Record the number of rounds that selected for each arm
    max_mean_reward = max(mu + omega)  # Maximum mean reward - find the optimal arm

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm once
    for k in range(K):
        arm_rewards[k] += np.random.normal(mu[k], sigma[k], 1)[0] + np.random.normal(omega[k], sigma_w[k], 1)[0]
        # Update the number of arm pulls
        num_pulls[k] += 1
        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])

    # Remaining Rounds
    for t in range(K, T):
        # Calculating the UCBs for each arm
        arm_ucb = arm_rewards / num_pulls + np.sqrt( (2*np.log(t)) / num_pulls)
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Updating variables
        # Update the total collected rewards for the selected arm
        arm_rewards[I_t] += np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        # Update number of pulls for the selected arm
        num_pulls[I_t] += 1

        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])

    # Returning instantaneous regret
    return instantaneous_regret


# UCB based algorithm with Control Variate
def ucb_cv(mu, omega, sigma, sigma_w, T):

    '''
    UCB-CV Algorithm for MAB-CV Problem
    Inputs
    ============================================
    mu: mean for V
    omega: mean for W
    sigma: sd for V
    sigma_w: sd for W
    T: number of rounds (int)
    '''

    K = len(mu)  # Number of arms
    arm_rewards = np.zeros(K)          # Collected rewards for arms
    arm_rewards_squared = np.zeros(K)  # Collected squares of rewards for arms
    arm_cv = np.zeros(K)               # Collected CV for arms
    arm_cv_squared = np.zeros(K)       # Collected squares of CV for arms
    rewards_times_cv = np.zeros(K)     # Collected product of reward and CV for arms
    num_pulls = np.zeros(K)            # Number of arm pulls
    max_mean_reward = max(mu + omega)  # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm 3 (number of CV +2) times
    for k in range(3 * K):
        # Samples
        k = k % K
        random_sample = np.random.normal(mu[k], sigma[k], 1)[0]
        arm_cv_value = np.random.normal(omega[k], sigma_w[k], 1)[0]
        arm_reward = random_sample + arm_cv_value

        # Update all variables
        # Update the total collected rewards for the selected arm
        arm_rewards[k] += arm_reward
        # Update the total collected squares of rewards for the selected arm
        arm_rewards_squared[k] += (arm_reward ** 2)
        # Update the total collected CV for the selected arm
        arm_cv[k] += arm_cv_value
        # Update the total collected squares of CV for the selected arm
        arm_cv_squared[k] += (arm_cv_value ** 2)
        # Update the total collected rewards*CV for the selected arm
        rewards_times_cv[k] += (arm_reward * arm_cv_value)
        # Update the number of arm pulls
        num_pulls[k] += 1

        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])

    # Remaining Rounds
    for t in range(3 * K, T):

        # 1. Compute the optimal coefficients: alpha
        # Estimated mean rewards of arms
        mu_est = arm_rewards / num_pulls
        # Estimated CV of arms
        cv_est = arm_cv / num_pulls
        # Computing the denominator of alpha (var part): sum of squares of centered CV values of arms
        denominator_alpha = arm_cv_squared + (num_pulls*(omega**2)) - (2*omega*arm_cv)
        # Computing the numerator of alpha (cov part)
        numerator_alpha = rewards_times_cv - (omega*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*omega)
        alpha = numerator_alpha / denominator_alpha

        # 2. Compute the mean estimator of the new observations: mu_cv_est
        mu_cv_est = mu_est + (alpha*omega) - (alpha*cv_est)

        # 3. Compute the sample variance of the mean reward estimator: V_mu_cv_est
        # Computing sum of squares of the new observations
        lcv_rewards_squared = arm_rewards_squared + (alpha*alpha*denominator_alpha) + (2*alpha*omega*arm_rewards) - (2*alpha*rewards_times_cv)
        # Computing sample variance of arms
        sample_var = (1/(num_pulls-2)) * (lcv_rewards_squared - (num_pulls*mu_cv_est*mu_cv_est))  #simplified
        # Compute Z_m/m: Multiplier of sample variance to get variance of new estimator
        var_mult = (1/num_pulls) / (1.0 - ( ((arm_cv-(num_pulls*omega))**2) / (num_pulls*denominator_alpha) ))
        # Compute the value of V_mu_cv_est
        V_mu_cv_est = var_mult * sample_var

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with s-2 d.f.
        V_ts1 = ss.t.ppf(1-(1/(t**2)), num_pulls-2)

        # 5. Arm selection based on UCB-CV
        # Calculating the UCBs for each arm
        arm_ucb = mu_cv_est + (V_ts1 * np.sqrt(V_mu_cv_est))
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm and observe its related reward
        random_sample = np.random.normal(mu[I_t], sigma[I_t], 1)[0]
        arm_cv_value = np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        arm_reward = random_sample + arm_cv_value

        # 7. Update all variables as before
        arm_rewards[I_t] += arm_reward
        arm_rewards_squared[I_t] += (arm_reward**2)
        arm_cv[I_t] += arm_cv_value
        arm_cv_squared[I_t] += (arm_cv_value**2)
        rewards_times_cv[I_t] += (arm_reward * arm_cv_value)
        num_pulls[I_t] += 1

        # 8. Regret: record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])

    # Returning instantaneous regret
    return instantaneous_regret

# UCB based algorithm with Linear Zero-Variance Control Variate
def ucb_lzvcv(mu, omega, sigma, sigma_w, T):

    '''
    UCB-LZVCV Algorithm for MAB Problem
    Inputs
    ============================================
    mu: mean for V
    omega: mean for W
    sigma: sd for V
    sigma_w: sd for W
    T: number of rounds (int)
    '''

    K = len(mu)  # Number of arms
    arm_rewards = np.zeros(K)  # Collected rewards for arms
    arm_rewards_squared = np.zeros(K)  # Collected squares of rewards for arms
    arm_cv = np.zeros(K)  # Constructed CV for arms
    arm_cv_squared = np.zeros(K)  # Constructed squares of CV for arms
    rewards_times_cv = np.zeros(K)  # Collected product of reward and CV for arms
    num_pulls = np.zeros(K)  # Number of arm pulls
    max_mean_reward = max(mu + omega)  # Maximum mean reward
    cv_mean = np.zeros(K)

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm 3(number of CV+2) times
    for k in range(3*K):
        # Generates samples
        k = k % K
        np.random.seed(0)
        # Observes the reward samples from environment
        arm_reward = np.random.normal(mu[k], sigma[k], 1)[0] + np.random.normal(omega[k], sigma_w[k], 1)[0]
        # Construct the LZVCV from evaluations
        arm_reward_mean = mu[k] + omega[k]
        arm_reward_var = sigma[k]**2 + sigma_w[k]**2
        arm_cv_value = -1/2 * (-arm_reward + arm_reward_mean) / arm_reward_var

        # Update all variables
        # Update the total collected rewards for the selected arm
        arm_rewards[k] += arm_reward
        # Update the total collected squares of rewards for the selected arm
        arm_rewards_squared[k] += (arm_reward**2)
        # Update the total collected CV for the selected arm
        arm_cv[k] += arm_cv_value
        # Update the total collected squares of CV for the selected arm
        arm_cv_squared[k] += (arm_cv_value**2)
        # Update the total collected rewards*CV for the selected arm
        rewards_times_cv[k] += (arm_reward * arm_cv_value)
        # Update the number of arm pulls
        num_pulls[k] += 1

        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])

    # Remaining Rounds
    for t in range(3*K, T):

        # 1. Compute the optimal coefficients: alpha
        # Estimated mean rewards of arms
        mu_est = arm_rewards / num_pulls
        # Estimated CV of arms
        cv_est = arm_cv / num_pulls
        # Computing the denominator of alpha (var part): sum of squares of centered CV values of arms
        denominator_alpha = arm_cv_squared + (num_pulls*(cv_est**2)) - (2.0*cv_est*arm_cv)
        denominator_alpha = arm_cv_squared + (num_pulls*(cv_mean**2)) - (2.0*cv_mean*arm_cv)
        # Computing the numerator of alpha (cov part)
        numerator_alpha = rewards_times_cv - (cv_est*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*cv_est)
        numerator_alpha = rewards_times_cv - (cv_mean*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*cv_mean)
        # Computing the optimal alpha value for each arm
        alpha = numerator_alpha / denominator_alpha

        # 2. Compute the mean reward estimator: mu_lzv_est
        # Estimated mean of new estimator
        mu_lzv_est = mu_est - (alpha * cv_est)

        # 3. Compute the sample variance of the mean reward estimator: V_mu_lzv_est
        # Computing sum of squares of new samples
        lzv_rewards_squared = arm_rewards_squared + (alpha*alpha*arm_cv_squared) - (2*alpha*rewards_times_cv)
        # Computing the estimated sample variance of the new samples
        sample_var = (1/(num_pulls-2)) * (lzv_rewards_squared - (num_pulls*mu_lzv_est*mu_lzv_est))  # simplifies
        # Compute Z_m
        Zm = 1 / (1 - ((arm_cv**2) / (num_pulls * arm_cv_squared)))
        # Compute the value of V_mu_lzv_est
        V_mu_lzv_est = Zm * sample_var / num_pulls

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with m-2 d.f.
        V_tm1 = ss.t.ppf(1-(1/(t**2)), num_pulls-2)

        # 5. Arm selection based on UCB-CV
        # Calculating the UCBs for each arm
        arm_ucb = mu_lzv_est + (V_tm1 * np.sqrt(V_mu_lzv_est))
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm and observe its related reward
        arm_reward = np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        arm_reward_mean = mu[I_t] + omega[I_t]
        arm_reward_var = sigma[I_t]**2 + sigma_w[I_t]**2
        arm_cv_value = -1/2 * (-arm_reward + arm_reward_mean) / arm_reward_var

        # 7. Update all variables as before
        arm_rewards[I_t] += arm_reward
        arm_rewards_squared[I_t] += (arm_reward ** 2)
        arm_cv[I_t] += arm_cv_value
        arm_cv_squared[I_t] += (arm_cv_value ** 2)
        rewards_times_cv[I_t] += (arm_reward * arm_cv_value)
        num_pulls[I_t] += 1

        # 8. Regret: record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])

    # Returning instantaneous regret
    return instantaneous_regret

# UCB based algorithm with Quadratic Zero-Variance Control Variate
def ucb_qzvcv(mu, omega, sigma, sigma_w, T):

    '''
    UCB-QZVCV Algorithm for MAB Problem
    Inputs
    ============================================
    mu: mean for V
    omega: mean for W
    sigma: sd for V
    sigma_w: sd for W
    T: number of rounds (int)
    '''

    K = len(mu)  # Number of arms
    arm_rewards = np.zeros(K)          # Collected rewards for arms
    arm_rewards_squared = np.zeros(K)  # Collected squares of rewards for arms
    arm_cv_z = np.zeros(K)             # Constructed z for arms
    arm_cv_u = np.zeros(K)             # Constructed u for arms
    arm_cv_z_squared = np.zeros(K)     # Constructed squares of z for arms
    arm_cv_u_squared = np.zeros(K)     # Constructed squares of u for arms
    rewards_times_cv_z = np.zeros(K)   # Collected product of reward and z for arms
    rewards_times_cv_u = np.zeros(K)   # Collected product of reward and u for arms
    cv_z_times_cv_u = np.zeros(K)      # Constructed z*u
    num_pulls = np.zeros(K)            # Number of arm pulls
    max_mean_reward = max(mu + omega)  # Maximum mean reward
    cv_z_mean = np.zeros(K)            # Expected mean of z
    cv_u_mean = np.zeros(K)            # Expected mean of u

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm 4(number of CV+2) times
    for k in range(4*K):
        # Generates samples
        k = k % K
        np.random.seed(0)
        # Observe the reward samples from environment
        arm_reward = np.random.normal(mu[k], sigma[k], 1)[0] + np.random.normal(omega[k], sigma_w[k], 1)[0]
        # arm_reward = np.random.normal(mu[k], sigma[k]) + np.random.normal(omega[k], sigma_w[k])
        # Construct the QZVCV from evaluations
        arm_reward_mean = mu[k] + omega[k]
        arm_reward_var = sigma[k]**2 + sigma_w[k]**2
        arm_cv_z_value = -1/2 * (-arm_reward + arm_reward_mean) / arm_reward_var
        arm_cv_u_value = 1/(2*arm_reward_var) * (arm_reward*arm_reward - arm_reward*arm_reward_mean - arm_reward_var)
        # print("arm_reward",arm_reward,"arm_reward_mean",arm_reward_mean)
        # print(arm_cv_z_value,arm_cv_u_value)

        # Update all variables
        # Update the total collected rewards for the selected arm
        arm_rewards[k] += arm_reward
        # Update the total collected squares of rewards for the selected arm
        arm_rewards_squared[k] += (arm_reward**2)
        # Update the total constructed QZVCV for the selected arm
        arm_cv_z[k] += arm_cv_z_value
        arm_cv_u[k] += arm_cv_u_value
        # Update the total constructed squares of QZVCV for the selected arm
        arm_cv_z_squared[k] += (arm_cv_z_value**2)
        arm_cv_u_squared[k] += (arm_cv_u_value**2)
        # Update the total collected rewards*QZVCV for the selected arm
        rewards_times_cv_z[k] += (arm_reward * arm_cv_z_value)
        rewards_times_cv_u[k] += (arm_reward * arm_cv_u_value)
        # Update z*u for the selected arm
        cv_z_times_cv_u[k] += (arm_cv_z_value * arm_cv_u_value)
        # Update the number of arm pulls
        num_pulls[k] += 1

        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])

    # Remaining Rounds
    for t in range(3*K, T):

        # 1. Compute the optimal coefficients vector: alpha
        # Estimated mean rewards of arms
        mu_est = arm_rewards / num_pulls
        # Estimated z of arms
        cv_z_est = arm_cv_z / num_pulls
        # Estimated u of arms
        cv_u_est = arm_cv_u / num_pulls
        # print(mu_est,cv_z_est,cv_u_est)

        # Computing the denominator of c (var part): sum of squares of centered z values of arms
        # denominator_c = arm_cv_z_squared + (num_pulls*(cv_z_est**2)) - (2.0*cv_z_est*arm_cv_z)
        denominator_c = arm_cv_z_squared + (num_pulls*(cv_z_mean**2)) - (2*cv_z_mean*arm_cv_z)
        # Computing the numerator of c (cov part)
        # numerator_c = rewards_times_cv - (cv_est*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*cv_est)
        numerator_c = rewards_times_cv_z - (cv_z_mean*arm_rewards) - (mu_est*arm_cv_z) + (num_pulls*mu_est*cv_z_mean)
        # Computing the optimal c value for each arm
        c = numerator_c / denominator_c

        # Computing the denominator of d (var part): sum of squares of centered u values of arms
        # denominator_d = arm_cv_u_squared + (num_pulls*(cv_u_est**2)) - (2.0*cv_u_est*arm_cv_u)
        denominator_d = arm_cv_u_squared + (num_pulls*(cv_u_mean**2)) - (2*cv_u_mean*arm_cv_u)
        # Computing the numerator of d (cov part)
        # numerator_d = rewards_times_cv - (cv_est*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*cv_est)
        numerator_d = rewards_times_cv_u - (cv_u_mean*arm_rewards) - (mu_est*arm_cv_u) + (num_pulls*mu_est*cv_u_mean)
        # Computing the optimal c value for each arm
        d = numerator_d / denominator_d
        alpha = np.array([[c], [d]])

        # # Compute S_HiHi & S_FiHi
        # S_HiHi = np.array([[arm_cv_z_squared-num_pulls*cv_z_est*cv_z_est, cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est],
        #                    [cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est, arm_cv_u_squared-num_pulls*cv_u_est*cv_u_est]])
        # S_FiHi = np.array([[rewards_times_cv_z-num_pulls*cv_z_est*mu_est],
        #                                      [rewards_times_cv_u-num_pulls*cv_u_est*mu_est]])
        # # det_S = (arm_cv_z_squared-num_pulls*cv_z_est*cv_z_est)*(arm_cv_u_squared-num_pulls*cv_u_est*cv_u_est)-(cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est)*(cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est)
        # det_S = (arm_cv_z_squared-num_pulls*cv_z_mean*cv_z_mean)*(arm_cv_u_squared-num_pulls*cv_u_mean*cv_u_mean)-(cv_z_times_cv_u-num_pulls*cv_z_mean*cv_u_mean)*(cv_z_times_cv_u-num_pulls*cv_z_mean*cv_u_mean)
        #
        # print("det_S",det_S)
        # print("1",arm_cv_z_squared-num_pulls*cv_z_mean*cv_z_mean)
        # print("2",arm_cv_u_squared-num_pulls*cv_u_mean*cv_u_mean)
        # print("3",cv_z_times_cv_u-num_pulls*cv_z_mean*cv_u_mean)
        #
        # S_HiHi_inv = (num_pulls-1) * 1/det_S * np.array([[arm_cv_u_squared-num_pulls*cv_u_est*cv_u_est, -(cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est)],
        #                                   [-(cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est), arm_cv_z_squared-num_pulls*cv_z_est*cv_z_est]])
        # matrix = np.array([[(arm_cv_u_squared-num_pulls*cv_u_est*cv_u_est)*(rewards_times_cv_z-num_pulls*cv_z_est*mu_est)-(-(cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est))*(rewards_times_cv_u-num_pulls*cv_u_est*mu_est)],
        #           [(-(cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est))*(rewards_times_cv_z-num_pulls*cv_z_est*mu_est)+(rewards_times_cv_u-num_pulls*cv_u_est*mu_est)*(arm_cv_z_squared-num_pulls*cv_z_est*cv_z_est)]])
        # # S_HiHi = 1\(num_pulls-1) * np.array([[arm_cv_z_squared-num_pulls*cv_z_est*cv_z_est, cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est], [cv_z_times_cv_u-num_pulls*cv_z_est*cv_u_est, arm_cv_u_squared-num_pulls*cv_u_est*cv_u_est]])
        # # S_FiHi = 1\(num_pulls-1) * np.array([[rewards_times_cv_z-num_pulls*cv_z_est*mu_est],
        # #                                      [rewards_times_cv_u-num_pulls*cv_u_est*mu_est]])
        # # Compute alpha
        # alpha = 1/det_S * matrix

        # 2. Compute the mean reward estimator: mu_qzv_est
        # Construct the estimated mean vector
        cv_est = np.array([[cv_z_est], [cv_u_est]])
        # Estimated mean of new estimator
        mu_qzv_est = mu_est - c * cv_z_est - d * cv_u_est

        # 3. Compute the sample variance of the mean reward estimator: V_mu_qzv_est
        det_HTH = (num_pulls*arm_cv_z_squared*arm_cv_u_squared) + 2*(arm_cv_z*cv_z_times_cv_u*arm_cv_u) - (arm_cv_u**2 * arm_cv_z_squared) - (num_pulls*cv_z_times_cv_u*cv_z_times_cv_u) - (arm_cv_u_squared * arm_cv_z**2)
        qzv_rewards_squared = arm_rewards_squared - (2*arm_rewards*(c*arm_cv_z)) + (c*arm_cv_z)**2 - (2*arm_rewards*d*arm_cv_u) + (2*c*arm_cv_z*d*arm_cv_u) + (d*arm_cv_u)**2
        sample_var = (1/(num_pulls-3)) * (qzv_rewards_squared - (num_pulls*mu_qzv_est*mu_qzv_est))  # simplifies
        print("sample_var",sample_var)
        print("det_HTH",det_HTH)
        V_mu_qzv_est = 1/det_HTH * (arm_cv_z_squared*arm_cv_u_squared-cv_z_times_cv_u*cv_z_times_cv_u) * sample_var
        print("V_mu_qzv_est",V_mu_qzv_est)

        # # Computing sum of squares of new samples
        # qzv_rewards_squared = arm_rewards_squared - (2*arm_rewards*(alpha @ cv_est)) + ((alpha @ cv_est)*(alpha @ cv_est))
        # # Computing the estimated sample variance of the new samples
        # sample_var = (1/(num_pulls-3)) * (qzv_rewards_squared - (num_pulls*mu_qzv_est*mu_qzv_est))  # simplifies
        # # Compute Z_m
        # centered_cv_est = cv_est - np.array([[cv_z_mean], [cv_u_mean]])
        # Zm = 1 + (centered_cv_est.T @ S_HiHi_inv @ centered_cv_est) / (1-1/num_pulls)
        # # Compute the value of V_mu_lzv_est
        # V_mu_qzv_est = Zm * sample_var / num_pulls

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with m-3 d.f.
        V_tm2 = ss.t.ppf(1-(1/(t**2)), num_pulls-3)

        # 5. Arm selection based on UCB-CV
        # Calculating the UCBs for each arm
        arm_ucb = mu_qzv_est + (V_tm2 * np.sqrt(V_mu_qzv_est))
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm, observe its related reward and construct the QZVCV from evaluations
        arm_reward = np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        arm_reward_mean = mu[I_t] + omega[I_t]
        arm_reward_var = sigma[I_t]**2 + sigma_w[I_t]**2
        arm_cv_z_value = -1/2 * (-arm_reward + arm_reward_mean) / arm_reward_var
        arm_cv_u_value = 1/(2*arm_reward_var) * (arm_reward*arm_reward - arm_reward*arm_reward_mean - arm_reward_var)
        # print("arm_reward",arm_reward,"arm_reward_mean",arm_reward_mean)
        # print(arm_cv_z_value,arm_cv_u_value)

        # 7. Update all variables as before
        arm_rewards[I_t] += arm_reward
        arm_rewards_squared[I_t] += (arm_reward**2)
        arm_cv_z[I_t] += arm_cv_z_value
        arm_cv_u[I_t] += arm_cv_u_value
        arm_cv_z_squared[I_t] += (arm_cv_z_value**2)
        arm_cv_u_squared[I_t] += (arm_cv_u_value**2)
        rewards_times_cv_z[I_t] += (arm_reward * arm_cv_z_value)
        rewards_times_cv_u[I_t] += (arm_reward * arm_cv_u_value)
        cv_z_times_cv_u[I_t] += (arm_cv_z_value * arm_cv_u_value)
        num_pulls[I_t] += 1

        # 8. Regret: record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])

    # Returning instantaneous regret
    return instantaneous_regret


# #######################################################################


# ############################## Main Code ##############################
# ######## Dataset details ########
samples = 50000       # Number of rounds
runs = 100            # Number of experiments
np.random.seed(2021)   # Set seed to get the same result

# ######## Problem Instance 1 ########

# # Number of arms
# arms = 10
#
# # Parameter settings for mean & variance
# max_arm_mean = 0.06 * arms
# arm_gap = 0.05
# max_cv_mean = 0.08*arms
# cv_gap = 0.05
# arm_sd = 0.01
# cv_sd = 0.01


# # Number of arms
# arms = 10
#
# # Parameter settings for mean & variance
# max_arm_mean = 30
# arm_gap = 2
# max_cv_mean = 35
# cv_gap = 3
# arm_sd = 4
# cv_sd = 4

# Number of arms
arms = 10

# Parameter settings for mean & variance
max_arm_mean = 10
arm_gap = 0.5
max_cv_mean = 15
cv_gap = 0.5
arm_sd = 2
cv_sd = 2


# Mean & sd vector
arms_mean = np.zeros(arms)
arms_sd = np.zeros(arms)
cvs_mean = np.zeros(arms)
cvs_sd = np.zeros(arms)
for k in range(arms):
    arms_mean[k] = max_arm_mean - (k*arm_gap)
    arms_sd[k] = arm_sd
    cvs_mean[k] = max_cv_mean - (k*cv_gap)
    cvs_sd[k] = cv_sd

# Run each algorithm t=runs time
cases = ['UCB1','UCB-CV','UCB-LZVCV','UCB-QZVCV']
cases = ['UCB_QZVCV']
cases = ['UCB1','UCB-CV','UCB-LZVCV']
total_cases = len(cases)
algos_regret = []
for _ in tqdm(range(runs)):
    run_regret = []
    iter_regret = []
    for c in range(total_cases):
        if cases[c] == 'UCB1':
            iter_regret = ucb1(arms_mean, cvs_mean, arms_sd, cvs_sd, samples)

        elif cases[c] == 'UCB-CV':
            iter_regret = ucb_cv(arms_mean, cvs_mean, arms_sd, cvs_sd, samples)

        elif cases[c] == 'UCB-LZVCV':
            iter_regret = ucb_lzvcv(arms_mean, cvs_mean, arms_sd, cvs_sd, samples)

        elif cases[c] == 'UCB-QZVCV':
            iter_regret = ucb_qzvcv(arms_mean, cvs_mean, arms_sd, cvs_sd, samples)

        run_regret.append(iter_regret)

    algos_regret.append(run_regret)

# ########## Plotting parameters ##########
xlabel = "Rounds"
ylabel = "Regret"
file_to_save = "Instance1_" + str(cases) + "_" + str(max_arm_mean) + "_" + str(max_cv_mean) + "_" + str(arm_sd)+ "_" + str(cv_sd) +".png"
title = "Comparison of Algorithms - Average Regret After 100 Experiments"
save_to_path = "plots/experiments/"
location_to_save = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases, samples]

# Regret Plotting
regret_plotting(algos_regret, total_cases, plotting_parameters)
