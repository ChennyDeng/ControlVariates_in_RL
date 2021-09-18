# -*- codeing = utf-8 -*-
# @Time :11/09/2021 14:54
# @Author :Chenny
# @Site :
# @File :Instance 1.py
# @Software :PyCharm

import numpy as np
from numpy.lib.shape_base import kron
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
    shape = ['--^', '--v', '--H', '--d', '--*', '--s', '--o']

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
def ucb1(mean_reward, df, T):
    '''
     Upper Confidence Bound 1 Algorithm for Multi-Armed Bandit Problem
     Inputs
     ============================================
     mean_reward: reward mean for each arm
     df: degrees of freedom
     T: number of rounds (int)
     '''

    K = len(mean_reward)  # Number of arms
    arm_rewards = np.zeros(K)  # Collected rewards for each arm
    num_pulls = np.zeros(K)  # Record the number of rounds that selected for each arm
    max_mean_reward = max(mean_reward)  # Maximum mean reward - find the optimal arm

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm once
    for k in range(K):
        arm_rewards[k] += (np.random.standard_t(df,1) + mean_reward[k])
        # Update the number of arm pulls
        num_pulls[k] += 1
        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mean_reward[k])

    # Remaining Rounds
    for t in range(K, T):
        # Calculating the UCBs for each arm
        arm_ucb = arm_rewards/num_pulls + np.sqrt((2*np.log(t)) / num_pulls)
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Updating variables
        # Update the total collected rewards for the selected arm
        arm_rewards[I_t] += (np.random.standard_t(df, 1) + mean_reward[I_t])
        # Update number of pulls for the selected arm
        num_pulls[I_t] += 1

        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - mean_reward[I_t])

    # Returning instantaneous regret
    return instantaneous_regret


# UCB based algorithm with Control Variate
def ucb_cv(mu, omega, df, T):

    '''
    UCB-CV Algorithm for MAB-CV Problem
    Inputs
    ============================================
    mu: mean for V
    omega: mean for CV
    df: degrees of freedom
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

        random_sample = np.random.standard_t(df, 1)
        arm_cv_value = random_sample + omega[k]
        arm_reward = mu[k] + arm_cv_value

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
        cv_rewards_squared = arm_rewards_squared + (alpha*alpha*denominator_alpha) + (2*alpha*omega*arm_rewards) - (2*alpha*rewards_times_cv)
        # Computing sample variance of arms
        sample_var = (1/(num_pulls-2)) * (cv_rewards_squared - (num_pulls*mu_cv_est*mu_cv_est))  #simplified
        # Compute Z_m/m: Multiplier of sample variance to get variance of new estimator
        var_mult = (1/num_pulls) / (1.0 - ( ((arm_cv-(num_pulls*omega))**2) / (num_pulls*denominator_alpha) ))
        # Compute the value of V_mu_cv_est
        V_mu_cv_est = var_mult * sample_var

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with s-2 d.f.
        V_ts1 = ss.t.ppf(1-(1/(t**2)), num_pulls-2)

        # 5. Arm selection based on UCB-CV
        # Calculating the UCBs for each arm
        arm_ucb = mu_cv_est + (V_ts1 * np.sqrt(np.abs(V_mu_cv_est)))
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm and observe its related reward
        random_sample = np.random.standard_t(df, 1)
        arm_cv_value = random_sample + omega[I_t]
        arm_reward = mu[I_t] + arm_cv_value

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
def ucb_lzvcv(arm_reward_mean, df, T):

    '''
    UCB-LZVCV Algorithm for MAB Problem
    Inputs
    ============================================
    arm_reward_mean: mean reward for each arm
    df: degrees of freedom
    T: number of rounds (int)
    '''

    K = len(arm_reward_mean)  # Number of arms
    arm_rewards = np.zeros(K)  # Collected rewards for arms
    arm_rewards_squared = np.zeros(K)  # Collected squares of rewards for arms
    arm_cv = np.zeros(K)  # Constructed CV for arms
    arm_cv_squared = np.zeros(K)  # Constructed squares of CV for arms
    rewards_times_cv = np.zeros(K)  # Collected product of reward and CV for arms
    num_pulls = np.zeros(K)  # Number of arm pulls
    max_mean_reward = max(arm_reward_mean)  # Maximum mean reward
    cv_mean = np.zeros(K)

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm 3(number of CV+2) times
    for k in range(3*K):
        # Generates samples
        k = k % K
        # Observes the reward samples from environment
        tt = np.random.standard_t(df, 1)
        arm_reward = tt + arm_reward_mean[k]
        # Construct the LZVCV from evaluations
        arm_cv_value = tt * (df+1)/ (2*(tt*tt+df))

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
        instantaneous_regret.append(max_mean_reward - arm_reward_mean[k])

    # Remaining Rounds
    for t in range(3*K, T):

        # 1. Compute the optimal coefficients: alpha
        # Estimated mean rewards of arms
        mu_est = arm_rewards / num_pulls
        # Estimated CV of arms
        lzv_est = arm_cv / num_pulls
        # Computing the denominator of alpha (var part): sum of squares of centered CV values of arms
        # denominator_alpha = arm_cv_squared + (num_pulls*(lzv_est**2)) - (2.0*lzv_est*arm_cv)
        denominator_alpha = arm_cv_squared + (num_pulls*(cv_mean**2)) - (2.0*cv_mean*arm_cv)
        # Computing the numerator of alpha (cov part)
        # numerator_alpha = rewards_times_cv - (lzv_est*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*lzv_est)
        numerator_alpha = rewards_times_cv - (cv_mean*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*cv_mean)
        # Computing the optimal alpha value for each arm
        alpha = numerator_alpha / denominator_alpha

        # 2. Compute the mean reward estimator: mu_lzv_est
        # Estimated mean of new estimator
        mu_lzv_est = mu_est - (alpha * lzv_est)

        # 3. Compute the sample variance of the mean reward estimator: V_mu_lzv_est
        # Computing sum of squares of new samples
        lzv_rewards_squared = arm_rewards_squared + (alpha*alpha*arm_cv_squared) - (2*alpha*rewards_times_cv)
        # Computing the estimated sample variance of the new samples
        sample_var = (1/(num_pulls-2)) * (lzv_rewards_squared - (num_pulls*mu_lzv_est*mu_lzv_est))  # simplifies
        # Compute the multiplier of sample variance to get the unbiased variance estimator for mu_lzv_est
        var_mult = (1/num_pulls) / (1 - (arm_cv**2) / (num_pulls * arm_cv_squared))
        # Compute the value of V_mu_lzv_est
        V_mu_lzv_est = var_mult * sample_var

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with m-2 d.f.
        V_tm1 = ss.t.ppf(1-(1/(t**2)), num_pulls-2)

        # 5. Arm selection based on UCB-CV
        # Calculating the UCBs for each arm
        arm_ucb = mu_lzv_est + (V_tm1 * np.sqrt(V_mu_lzv_est))
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm and observe its related reward
        # Observes the reward samples from environment
        tt = np.random.standard_t(df, 1)
        arm_reward = tt + arm_reward_mean[I_t]
        # Construct the LZVCV from evaluations
        arm_cv_value = tt * (df+1)/ (2*(tt*tt+df))

        # 7. Update all variables as before
        arm_rewards[I_t] += arm_reward
        arm_rewards_squared[I_t] += (arm_reward ** 2)
        arm_cv[I_t] += arm_cv_value
        arm_cv_squared[I_t] += (arm_cv_value ** 2)
        rewards_times_cv[I_t] += (arm_reward * arm_cv_value)
        num_pulls[I_t] += 1

        # 8. Regret: record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - arm_reward_mean[I_t])

    # Returning instantaneous regret
    return instantaneous_regret

# UCB based algorithm with Quadratic Zero-Variance Control Variate
def ucb_qzvcv(arm_reward_mean, df, T):

    '''
    UCB-QZVCV Algorithm for MAB Problem
    Inputs
    ============================================
    arm_reward_mean: mean reward for each arm
    df: degrees of freedom
    T: number of rounds (int)
    '''

    K = len(arm_reward_mean)  # Number of arms
    arm_rewards = np.zeros(K)  # Collected rewards for arms
    arm_rewards_seq = np.zeros(K)  # Collected sequare of rewards for arms
    arm_cv = np.zeros(K)  # Collected CV for arms
    arm_cv_seq = np.zeros(K)  # Collected sequare of CV for arms
    arm_cross_terms = np.zeros(K)  # Collected product of reward and CV for arms
    num_pulls = np.zeros(K)  # Number of arm pulls
    max_mean_reward = max(arm_reward_mean)  # Maximum mean reward
    cv1_mean = np.zeros(K)
    cv2_mean = np.zeros(K)
    arm_cv2 = np.zeros(K)
    arm_cv2_seq = np.zeros(K)
    cv_cross_cv2 = np.zeros(K)
    arm_cross2_terms = np.zeros(K)
    cv_mean_matrix = np.array([cv1_mean, cv2_mean])

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm once
    for k in range(4 * K):
        # Samples
        k = k % K
        # Observes the reward samples from environment
        tt = np.random.standard_t(df, 1)
        arm_reward = tt + arm_reward_mean[k]
        # Construct the QZVCV from evaluations
        arm_cv_value = tt * (df+1)/ (2*(tt*tt+df))
        arm_cv2_value = tt*tt*(df+1) / (2*(tt*tt+df)) -1/2

        # Update all variables
        arm_rewards[k] += arm_reward
        arm_rewards_seq[k] += (arm_reward ** 2)
        arm_cv[k] += arm_cv_value
        arm_cv_seq[k] += (arm_cv_value ** 2)
        arm_cross_terms[k] += (arm_reward * arm_cv_value)
        num_pulls[k] += 1
        arm_cv2[k] += arm_cv2_value
        arm_cv2_seq[k] += (arm_cv2_value ** 2)
        cv_cross_cv2[k] += (arm_cv_value * arm_cv2_value)
        arm_cross2_terms[k] += (arm_reward * arm_cv2_value)

        # Regret
        instantaneous_regret.append(max_mean_reward-arm_reward_mean[k])

    # Remaining Rounds
    for t in range(4 * K, T):

        # 1. Compute beta: beta
        # Estimated mean rewards of arms
        mu_est = arm_rewards / num_pulls
        cv_est = arm_cv / num_pulls
        cv2_est = arm_cv2 / num_pulls
        cv_est_matrix = np.array([cv_est, cv2_est])  # shape:2*K

        # Compute the inverse of S_WiWi
        var_cv1 = 1 / (num_pulls - 1) * (arm_cv_seq - num_pulls * cv_est * cv_est)
        cov_cv1_cv2 = 1 / (num_pulls - 1) * (cv_cross_cv2 - num_pulls * cv_est * cv2_est)
        var_cv2 = 1 / (num_pulls - 1) * (arm_cv2_seq - num_pulls * cv2_est * cv2_est)
        S_WiWi = np.array([[var_cv1, cov_cv1_cv2], [cov_cv1_cv2, var_cv2]])
        S_WiWi_inv = np.zeros(shape=(2, 2, K))
        for i in range(K):
            S_WiWi_inv[:, :, i] = np.linalg.inv(S_WiWi[:, :, i])

        # Compute S_XiWi
        cov_cv1_reward = 1 / (num_pulls - 1) * (arm_cross_terms - num_pulls * cv_est * mu_est)
        cov_cv2_reward = 1 / (num_pulls - 1) * (arm_cross2_terms - num_pulls * cv2_est * mu_est)
        S_XiWi = np.array([cov_cv1_reward, cov_cv2_reward])

        # Compute beta
        beta = np.einsum('ijk,jk->ik', S_WiWi_inv, S_XiWi)

        # 2. Compute mean for the new estimators: mu_cv_est
        mu_cv_est = mu_est + np.einsum('ij,ij->j', beta, (cv_mean_matrix - cv_est_matrix))

        # 3. Compute the unbiased variance estimator for mu_cv_est
        # Computing squares of centered cv values of arms
        cv_centered_seq = arm_cv_seq + (num_pulls * (cv1_mean ** 2)) - (2.0 * cv1_mean * arm_cv)
        cv2_centered_seq = arm_cv2_seq + (num_pulls * (cv2_mean ** 2)) - (2.0 * cv2_mean * arm_cv2)
        # Computing sum of sequare of new observation
        B = beta[0, :] * beta[0, :] * cv_centered_seq + beta[1, :] * beta[1, :] * cv2_centered_seq + 2 * beta[0,:] * beta[1,:] * (num_pulls * cv1_mean * cv2_mean - cv1_mean * arm_cv2 - cv2_mean * arm_cv + cv_cross_cv2)
        cv_rewards_seq = arm_rewards_seq + B + 2 * arm_rewards * beta[0, :] * cv1_mean - 2 * arm_cross_terms * beta[0,:] + 2 * arm_rewards * beta[1,:] * cv2_mean - 2 * arm_cross2_terms * beta[1,:]
        # Computing sample variance of arms
        sample_var = (1 / (num_pulls - 3)) * (cv_rewards_seq - (num_pulls * mu_cv_est * mu_cv_est))
        # Multiplier of sample variance to get variance of new estimator
        cv_est_minus_cv_mean_matrix = np.array([cv_est - cv1_mean, cv2_est - cv2_mean])
        numerator = np.einsum('ik,ijk,jk->k', cv_est_minus_cv_mean_matrix, S_WiWi_inv, cv_est_minus_cv_mean_matrix)
        var_mult = (1.0 / num_pulls) * (1 + (numerator / (1 - 1 / num_pulls)))

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with m-3 d.f.
        V_tm2 = ss.t.ppf(1 - (1 / (t ** 2)), num_pulls - 3)

        # 5. Arm selection based on UCB-QZVCV
        # Calculating the UCBs for each arm
        arm_ucb = mu_cv_est + (V_tm2 * np.sqrt(np.abs(var_mult * sample_var)))
        # Selecting arm with maximum UCB index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm, observe its related reward and construct the QZVCV from the original data
        # Observes the reward samples from environment
        tt = np.random.standard_t(df, 1)
        arm_reward = tt + arm_reward_mean[I_t]
        # Construct the QZVCV from evaluations
        arm_cv_value = tt * (df+1)/ (2*(tt*tt+df))
        arm_cv2_value = tt*tt*(df+1) / (2*(tt*tt+df)) -1/2

        # Update all variables
        arm_rewards[I_t] += arm_reward
        arm_rewards_seq[I_t] += (arm_reward ** 2)
        arm_cv[I_t] += arm_cv_value
        arm_cv_seq[I_t] += (arm_cv_value ** 2)
        arm_cross_terms[I_t] += (arm_reward * arm_cv_value)
        num_pulls[I_t] += 1
        arm_cv2[I_t] += arm_cv2_value
        arm_cv2_seq[I_t] += (arm_cv2_value ** 2)
        cv_cross_cv2[I_t] += (arm_cv_value * arm_cv2_value)
        arm_cross2_terms[I_t] += (arm_reward * arm_cv2_value)

        # Regret
        instantaneous_regret.append(max_mean_reward-arm_reward_mean[I_t])

    # Returning instantaneous regret
    return instantaneous_regret

# #######################################################################


# ############################## Main Code ##############################
# ######## Dataset details ########
samples = 50000       # Number of rounds
runs = 100            # Number of experiments
np.random.seed(100)   # Set seed to get the same result

# ######## Problem Instance 2 ########

# Number of arms
arms = 10

# Parameter settings for mean & df
max_arm_mean = 0.06 * arms
arm_gap = 0.05
max_cv_mean = 0.08*arms
cv_gap = 0.05
df = 10


# Mean & sd vector
arms_mean = np.zeros(arms)
cvs_mean = np.zeros(arms)
for k in range(arms):
    arms_mean[k] = max_arm_mean - (k*arm_gap)
    cvs_mean[k] = max_cv_mean - (k*cv_gap)
arms_reward_mean = arms_mean+cvs_mean

# Run Algorithm
cases = ['UCB1','UCB-CV','UCB-LZVCV','UCB-QZVCV']
# cases = ['UCB-CV','UCB-LZVCV','UCB-QZVCV']
# cases = ['UCB-QZVCV']
total_cases = len(cases)
algos_regret = []
for _ in tqdm(range(runs)):
    run_regret = []
    iter_regret = []
    for c in range(total_cases):
        if cases[c] == 'UCB1':
            iter_regret = ucb1(arms_reward_mean, df, samples)

        elif cases[c] == 'UCB-CV':
            iter_regret = ucb_cv(arms_mean, cvs_mean, df, samples)

        elif cases[c] == 'UCB-LZVCV':
            iter_regret = ucb_lzvcv(arms_reward_mean, df, samples)

        elif cases[c] == 'UCB-QZVCV':
            iter_regret = ucb_qzvcv(arms_reward_mean, df, samples)

        run_regret.append(iter_regret)

    algos_regret.append(run_regret)

# ########## Plotting parameters ##########
xlabel = "Rounds"
ylabel = "Regret"
file_to_save = "Instance2.png"
title = "Student's t-Distribution - Average Regret After 100 Experiments"
save_to_path = "Figures/"
location_to_save = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases, samples]

# Regret Plotting
regret_plotting(algos_regret, total_cases, plotting_parameters)
