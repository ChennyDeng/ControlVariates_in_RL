# -*- codeing = utf-8 -*-
# @Time :11/09/2021 14:54
# @Author :Chenny
# @Site :
# @File :Instance4.py
# @Software :PyCharm

import numpy as np
from numpy.lib.shape_base import kron
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import bisect
from scipy.stats import entropy
from scipy.stats.morestats import probplot
from tqdm import tqdm


# #############################  Algorithms #############################
# UCB1 algorithm
def ucb1(arms_reward_mean, arms_reward_var, T):
    '''
     Upper Confidence Bound 1 Algorithm for Multi-Armed Bandit Problem
     Inputs
     ============================================
     arms_reward_mean: mean reward for each arm
     arms_reward_var: variance of the reward for each arm
     T: number of rounds (int)
     '''

    K = len(arms_reward_mean)  # Number of arms
    arm_rewards = np.zeros(K)  # Collected rewards for each arm
    num_pulls = np.zeros(K)  # Record the number of rounds that selected for each arm
    max_mean_reward = max(arms_reward_mean)  # Maximum mean reward - find the optimal arm

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm once
    for k in range(K):
        arm_rewards[k] += np.random.normal(arms_reward_mean[k], np.sqrt(arms_reward_var[k]), 1)[0]
        # Update the number of arm pulls
        num_pulls[k] += 1
        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - arms_reward_mean[k])

    # Remaining Rounds
    for t in range(K, T):
        # Calculating the UCBs for each arm
        arm_ucb = arm_rewards/num_pulls + np.sqrt((2*np.log(t)) / num_pulls)
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Updating variables
        # Update the total collected rewards for the selected arm
        arm_rewards[I_t] += np.random.normal(arms_reward_mean[I_t], np.sqrt(arms_reward_var[I_t]), 1)[0]
        # Update number of pulls for the selected arm
        num_pulls[I_t] += 1

        # Record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - arms_reward_mean[I_t])

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

# UCB with Linear Zero-Variance Control Variate
def ucb_lzvcv(arms_reward_mean, arms_reward_var, T):

    '''
    UCB-LZVCV Algorithm for MAB Problem
    Inputs
    ============================================
    arms_reward_mean: mean reward for each arm
    arms_reward_var: variance of the reward for each arm
    T: number of rounds (int)
    '''

    K = len(arms_reward_mean)                      # Number of arms
    arm_rewards = np.zeros(K)                      # Collected rewards for arms
    arm_rewards_squared = np.zeros(K)              # Collected squares of rewards for arms
    arm_cv = np.zeros(K)                           # Constructed CV for arms
    arm_cv_squared = np.zeros(K)                   # Constructed squares of CV for arms
    rewards_times_cv = np.zeros(K)                 # Collected product of reward and CV for arms
    num_pulls = np.zeros(K)                        # Number of arm pulls
    max_mean_reward = max(arms_reward_mean)        # Maximum mean reward
    cv_mean = np.zeros(K)

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm 3(number of CV+2) times
    for k in range(3*K):
        # Generates samples
        k = k % K
        # Observes the reward samples from environment
        arm_reward_mean = arms_reward_mean[k]
        arm_reward_var = arms_reward_var[k]
        arm_reward = np.random.normal(arm_reward_mean, np.sqrt(arm_reward_var), 1)[0]
        # Construct the LZVCV from evaluations
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
        instantaneous_regret.append(max_mean_reward - arm_reward_mean)

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
        arm_ucb = mu_lzv_est + (V_tm1 * np.sqrt(np.abs(V_mu_lzv_est)))
        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm and observe its related reward
        arm_reward_mean = arms_reward_mean[I_t]
        arm_reward_var = arms_reward_var[I_t]
        arm_reward = np.random.normal(arm_reward_mean, np.sqrt(arm_reward_var), 1)[0]
        arm_cv_value = -1/2 * (-arm_reward + arm_reward_mean) / arm_reward_var

        # 7. Update all variables as before
        arm_rewards[I_t] += arm_reward
        arm_rewards_squared[I_t] += (arm_reward ** 2)
        arm_cv[I_t] += arm_cv_value
        arm_cv_squared[I_t] += (arm_cv_value ** 2)
        rewards_times_cv[I_t] += (arm_reward * arm_cv_value)
        num_pulls[I_t] += 1

        # 8. Regret: record the instantaneous regret of current round
        instantaneous_regret.append(max_mean_reward - arm_reward_mean)

    # Returning instantaneous regret
    return instantaneous_regret

# UCB with Quadratic Zero-Variance Control Variate
def ucb_qzvcv(arms_reward_mean, arms_reward_var, T):

    '''
    UCB-QZVCV Algorithm for MAB Problem
    Inputs
    ============================================
    arms_reward_mean: mean reward for each arm
    arms_reward_var: variance of the reward for each arm
    T: number of rounds (int)
    '''

    K = len(arms_reward_mean)                       # Number of arms
    arm_rewards = np.zeros(K)                       # Collected rewards for arms
    arm_rewards_squared = np.zeros(K)               # Collected squares of rewards for arms
    arm_cv = np.zeros(K)                            # Constructed CV1 for arms
    arm_cv1_squared = np.zeros(K)                   # Constructed squares of CV1 for arms
    reward_times_cv1 = np.zeros(K)                  # Collected product of reward and CV1 for arms
    num_pulls = np.zeros(K)                         # Number of arm pulls
    max_mean_reward = max(arms_reward_mean)         # Maximum mean reward
    cv1_mean = np.zeros(K)                          # Mean for the constructed CV1
    cv2_mean = np.zeros(K)                          # Mean for the constructed CV2
    arm_cv2 = np.zeros(K)                           # Constructed CV2 for arms
    arm_cv2_squared = np.zeros(K)                   # Constructed squares of CV2 for arms
    cv1_times_cv2 = np.zeros(K)                     # Collected product of CV1 and CV2 for arms
    reward_times_cv2 = np.zeros(K)                  # Collected product of reward and CV2 for arms
    cv_mean_matrix = np.array([cv1_mean, cv2_mean]) # Mean matrix for the constructedZVCV

    # Stores instantaneous regret of each round
    instantaneous_regret = []

    # Initialization: Sampling each arm once
    for k in range(4 * K):
        # Samples
        k = k % K
        arm_reward_mean = arms_reward_mean[k]
        arm_reward_var = arms_reward_var[k]
        arm_reward = np.random.normal(arm_reward_mean, np.sqrt(arm_reward_var), 1)[0]
        arm_cv1_value = -1/2 * (-arm_reward + arm_reward_mean) / arm_reward_var
        arm_cv2_value = 1/(2*arm_reward_var) * (arm_reward*arm_reward - arm_reward*arm_reward_mean - arm_reward_var)

        # Update all variables
        arm_rewards[k] += arm_reward
        arm_rewards_squared[k] += (arm_reward ** 2)
        arm_cv[k] += arm_cv1_value
        arm_cv1_squared[k] += (arm_cv1_value ** 2)
        reward_times_cv1[k] += (arm_reward * arm_cv1_value)
        num_pulls[k] += 1
        arm_cv2[k] += arm_cv2_value
        arm_cv2_squared[k] += (arm_cv2_value ** 2)
        cv1_times_cv2[k] += (arm_cv1_value * arm_cv2_value)
        reward_times_cv2[k] += (arm_reward * arm_cv2_value)

        # Regret
        instantaneous_regret.append(max_mean_reward-arm_reward_mean)

    # Remaining Rounds
    for t in range(4 * K, T):

        # 1. Compute alpha: alpha
        # 1.1 Estimated mean rewards of arms
        mu_est = arm_rewards / num_pulls
        cv_est = arm_cv / num_pulls
        cv2_est = arm_cv2 / num_pulls
        cv_est_matrix = np.array([cv_est, cv2_est])  # shape:2*K

        # 1.2 Compute the inverse of S_WiWi
        var_cv1 = 1 / (num_pulls - 1) * (arm_cv1_squared - num_pulls * cv_est * cv_est)
        cov_cv1_cv2 = 1 / (num_pulls - 1) * (cv1_times_cv2 - num_pulls * cv_est * cv2_est)
        var_cv2 = 1 / (num_pulls - 1) * (arm_cv2_squared - num_pulls * cv2_est * cv2_est)
        S_WiWi = np.array([[var_cv1, cov_cv1_cv2], [cov_cv1_cv2, var_cv2]])
        S_WiWi_inv = np.zeros(shape=(2, 2, K))
        for i in range(K):
            S_WiWi_inv[:, :, i] = np.linalg.inv(S_WiWi[:, :, i])

        # 1.3 Compute S_XiWi
        cov_cv1_reward = 1 / (num_pulls - 1) * (reward_times_cv1 - num_pulls * cv_est * mu_est)
        cov_cv2_reward = 1 / (num_pulls - 1) * (reward_times_cv2 - num_pulls * cv2_est * mu_est)
        S_XiWi = np.array([cov_cv1_reward, cov_cv2_reward])

        # 1.4 Compute beta using S_WiWi_inv and S_XiWi
        alpha = np.einsum('ijk,jk->ik', S_WiWi_inv, S_XiWi)
        # ALTERNATIVELY
        # alpha_ = np.zeros(shape=(2, K))
        # for i in range(K):
        #     alpha_[:, i] = S_WiWi_inv[:, :, i] @ S_XiWi[:, i]

        # 2. Compute mean for the new estimators: mu_qcv_est
        # Estimated mean of new estimator
        mu_qcv_est = mu_est + np.einsum('ij,ij->j', alpha, (cv_mean_matrix - cv_est_matrix))
        # ALTERNATIVELY
        # mu_qcv_est = mu_est + (alpha[0, :] * cv1_mean) - (alpha[0, :] * cv_est) + (alpha[1, :] * cv2_mean) - (
        #             alpha[1, :] * cv2_est)

        # 3. Compute the unbiased variance estimator for mu_qcv_est: V_mu_qcv_est
        # 3.1 Compute the sample variance of arms
        # Computing squares of centered cv values of arms
        cv1_centered_squared = arm_cv1_squared + (num_pulls * (cv1_mean ** 2)) - (2.0 * cv1_mean * arm_cv)
        cv2_centered_squared = arm_cv2_squared + (num_pulls * (cv2_mean ** 2)) - (2.0 * cv2_mean * arm_cv2)
        # Computing sum of squares of new observation
        B = alpha[0, :] * alpha[0, :] * cv1_centered_squared + alpha[1, :] * alpha[1, :] * cv2_centered_squared + 2 * alpha[0,:] * alpha[1,:] * (num_pulls * cv1_mean * cv2_mean - cv1_mean * arm_cv2 - cv2_mean * arm_cv + cv1_times_cv2)
        cv_rewards_squared = arm_rewards_squared + B + 2 * arm_rewards * alpha[0, :] * cv1_mean - 2 * reward_times_cv1 * alpha[0,:] + 2 * arm_rewards * alpha[1,:] * cv2_mean - 2 * reward_times_cv2 * alpha[1,:]
        # Computing sample variance of arms
        sample_var = (1 / (num_pulls - 3)) * (cv_rewards_squared - (num_pulls * mu_qcv_est * mu_qcv_est))

        # 3.2 Multiplier of sample variance to get variance of new estimator
        cv_est_minus_cv_mean_matrix = np.array([cv_est - cv1_mean, cv2_est - cv2_mean])
        numerator = np.einsum('ik,ijk,jk->k', cv_est_minus_cv_mean_matrix, S_WiWi_inv, cv_est_minus_cv_mean_matrix)
        var_mult = (1.0 / num_pulls) * (1 + (numerator / (1 - 1 / num_pulls)))

        # 3.3 Compute the unbiased variance estimator for V_mu_qcv_est
        V_mu_qcv_est = var_mult * sample_var

        # 4. Calculate the 100(1-1/t^2)th percentile value of the t-distribution with m-3 d.f.
        V_tm2 = ss.t.ppf(1 - (1 / (t ** 2)), num_pulls - 3)

        # 5. Arm selection based on UCB-QZVCV
        # Calculating the UCBs for each arm
        arm_ucb = mu_qcv_est + (V_tm2 * np.sqrt(np.abs(V_mu_qcv_est)))
        # Selecting arm with maximum UCB index value
        I_t = np.argmax(arm_ucb)

        # 6. Samples: play the selected arm, observe its related reward and construct the QZVCV from the original data
        arm_reward_mean = arms_reward_mean[I_t]
        arm_reward_var = arms_reward_var[I_t]
        arm_reward = np.random.normal(arm_reward_mean, np.sqrt(arm_reward_var), 1)[0]
        arm_cv1_value = -1 / 2 * (-arm_reward + arm_reward_mean) / arm_reward_var
        arm_cv2_value = 1 / (2 * arm_reward_var) * (arm_reward * arm_reward - arm_reward * arm_reward_mean - arm_reward_var)

        # 7. Update all variables
        arm_rewards[I_t] += arm_reward
        arm_rewards_squared[I_t] += (arm_reward ** 2)
        arm_cv[I_t] += arm_cv1_value
        arm_cv1_squared[I_t] += (arm_cv1_value ** 2)
        reward_times_cv1[I_t] += (arm_reward * arm_cv1_value)
        num_pulls[I_t] += 1
        arm_cv2[I_t] += arm_cv2_value
        arm_cv2_squared[I_t] += (arm_cv2_value ** 2)
        cv1_times_cv2[I_t] += (arm_cv1_value * arm_cv2_value)
        reward_times_cv2[I_t] += (arm_reward * arm_cv2_value)

        # 8. Regret
        instantaneous_regret.append(max_mean_reward-arm_reward_mean)

    # Returning instantaneous regret
    return instantaneous_regret

# #######################################################################


# ############################## Main Code ##############################
# ######## Dataset details ########
samples = 50000       # Number of rounds
runs = 100            # Number of experiments
np.random.seed(100)   # Set seed to get the same result

# ######## Problem Instance 4 ########

# List of Arm numbers
arm_no_list = [10, 15 , 20, 25, 30]

# Construct the mean & sd vector with varying K
arms_mean_list = []
arms_sd_list = []
cvs_mean_list = []
cvs_sd_list = []
arms_reward_mean_list = []
arms_reward_var_list = []
legend = []

for i in range(len(arm_no_list)):
    legend.append("K="+str(arm_no_list[i]))
    arms = arm_no_list[i]
    # Parameter settings for mean & variance
    max_arm_mean = 0.06 * arms
    arm_gap = 0.05
    max_cv_mean = 0.08 * arms
    cv_gap = 0.05
    arm_sd = 0.1
    cv_sd = 0.1

    # Mean & sd vector
    arms_mean = np.zeros(arms)
    arms_sd = np.zeros(arms)
    cvs_mean = np.zeros(arms)
    cvs_sd = np.zeros(arms)
    for k in range(arms):
        arms_mean[k] = max_arm_mean - (k * arm_gap)
        arms_sd[k] = arm_sd
        cvs_mean[k] = max_cv_mean - (k * cv_gap)
        cvs_sd[k] = cv_sd

    arms_reward_mean = arms_mean + cvs_mean
    arms_reward_var = arms_sd * arms_sd + cvs_sd * cvs_sd

    arms_mean_list.append(arms_mean)
    arms_sd_list.append(arms_sd)
    cvs_mean_list.append(cvs_mean)
    cvs_sd_list.append(cvs_sd)
    arms_reward_mean_list.append(arms_reward_mean)
    arms_reward_var_list.append(arms_reward_var)


# Algorithm Running
cases = ['UCB1','UCB-CV','UCB-LZVCV','UCB-QZVCV']
varying_K = legend
UCB1_runs = []
UCBCV_runs = []
UCBLZVCV_runs = []
UCBQZVCV_runs = []
for _ in tqdm(range(runs)):
    UCB1 = []
    UCBCV = []
    UCBLZVCV = []
    UCBQZVCV = []
    for k in range(len(varying_K)):
        for c in range(len(cases)):
            if cases[c] == 'UCB1':
                iter_regret = ucb1(arms_reward_mean_list[k], arms_reward_var_list[k], samples)
                total_regret_after_T = np.sum(iter_regret)
                UCB1.append(total_regret_after_T)
            elif cases[c] == 'UCB-CV':
                iter_regret = ucb_cv(arms_mean_list[k], cvs_mean_list[k], arms_sd_list[k], cvs_sd_list[k], samples)
                total_regret_after_T = np.sum(iter_regret)
                UCBCV.append(total_regret_after_T)
            elif cases[c] == 'UCB-LZVCV':
                iter_regret = ucb_lzvcv(arms_reward_mean_list[k], arms_reward_var_list[k], samples)
                total_regret_after_T = np.sum(iter_regret)
                UCBLZVCV.append(total_regret_after_T)
            elif cases[c] == 'UCB-QZVCV':
                iter_regret = ucb_qzvcv(arms_reward_mean_list[k], arms_reward_var_list[k], samples)
                total_regret_after_T = np.sum(iter_regret)
                UCBQZVCV.append(total_regret_after_T)
    UCB1_runs.append(UCB1)
    UCBCV_runs.append(UCBCV)
    UCBLZVCV_runs.append(UCBLZVCV)
    UCBQZVCV_runs.append(UCBQZVCV)

# Average regret at time step T after all runs
UCB1_average = np.mean(UCB1_runs, axis=0)
UCBCV_average = np.mean(UCBCV_runs, axis=0)
UCBLZVCV_average = np.mean(UCBLZVCV_runs, axis=0)
UCBQZVCV_average = np.mean(UCBQZVCV_runs, axis=0)

# Confidence interval
conf_regret_UCB1 = []
conf_regret_UCBCV = []
conf_regret_UCBLZVCV = []
conf_regret_UCBQZVCV = []
dof = runs-1
for i in range(len(UCB1_average)):
    conf_regret_UCB1.append(ss.t.ppf(0.95, dof) *
                       ss.sem(np.array(UCB1_runs)[:, i]))
    conf_regret_UCBCV.append(ss.t.ppf(0.95, dof) *
                       ss.sem(np.array(UCBCV_runs)[:, i]))
    conf_regret_UCBLZVCV.append(ss.t.ppf(0.95, dof) *
                       ss.sem(np.array(UCBLZVCV_runs)[:, i]))
    conf_regret_UCBQZVCV.append(ss.t.ppf(0.95, dof) *
                       ss.sem(np.array(UCBQZVCV_runs)[:, i]))

CI_list = [conf_regret_UCB1,conf_regret_UCBCV,conf_regret_UCBLZVCV,conf_regret_UCBQZVCV]
Average_list = [UCB1_average,UCBCV_average,UCBLZVCV_average,UCBQZVCV_average]

# Define function: Regret plotting with varying K
def regret_with_varing_K(arm_no_list,CI_list,Average_list,plotting_info):
    colors = list("gbcmryk")
    shape = ['--^', '--v', '--H', '--d', '--*', '--s', '--o']
    for i in range(len(CI_list)):
        # plot the CI with the selected color
        plt.errorbar(arm_no_list, Average_list[i], CI_list[i], color=colors[i])
        # plot the average cumulative regret at the candidate rounds over the experiments with the selected color and selected shape
        plt.plot(arm_no_list, Average_list[i], colors[i] + shape[i], label=plotting_info[4][i])
        plt.xticks(arm_no_list, arm_no_list)
    plt.rc('font', size=10)  # controls default text sizes
    plt.legend(loc='upper left', numpoints=1)  # Location of the legend
    plt.xlabel(plotting_info[0], fontsize=15)
    plt.ylabel(plotting_info[1], fontsize=15)
    plt.title(plotting_info[2])
    plt.savefig(plotting_info[3], bbox_inches='tight')
    plt.close()

# ########## Plotting parameters ##########
xlabel = "Number of Arms K"
ylabel = "Regret at T = 50000"
file_to_save = "Instance4.png"
title = "Regret with Varying Number of Arms"
save_to_path = "Figures/"
location_to_save = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases]

# Regret Plotting
regret_with_varing_K(arm_no_list, CI_list, Average_list, plotting_parameters)