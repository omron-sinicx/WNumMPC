import numpy as np
import json
import sys
from scipy import stats

with open('datas/maru_results/slow-7-100-wnum-margin.json', 'r') as file:
    res_wnum = json.load(file)
with open('datas/maru_results/slow-7-100-vanilla-margin.json', 'r') as file:
    res_normal = json.load(file)

success_cases = [t for t in range(len(res_wnum["success_times"])) if t % 2 == 0
                    and np.isnan(res_wnum['success_times'][t]) == False and np.isnan(res_normal['success_times'][t]) == False]
etg_wnum = np.array(res_wnum['extra_time_to_goals']).reshape(100,7)[success_cases].mean(axis=1)
etg_normal = np.array(res_normal['extra_time_to_goals']).reshape(100,7)[success_cases].mean(axis=1)

print('ave_extra_time_to_goals: {}'.format(np.mean(etg_wnum)))
print('ave_extra_time_to_goals: {}'.format(np.mean(etg_normal)))

# Perform Wilcoxon signed-rank test for crossing cases
stat_opp, pval_opp = stats.wilcoxon(etg_wnum, etg_normal)
print('\nWilcoxon signed-rank test results for crossing cases:')
print(f'p-value = {pval_opp:.3e}')

success_cases = [t for t in range(len(res_wnum["success_times"])) if t % 2 == 1
                    and np.isnan(res_wnum['success_times'][t]) == False and np.isnan(res_normal['success_times'][t]) == False]
etg_wnum = np.array(res_wnum['extra_time_to_goals']).reshape(100,7)
etg_normal = np.array(res_normal['extra_time_to_goals']).reshape(100,7)
print('ave_extra_time_to_goals: {}'.format(np.mean(etg_wnum[success_cases,:])))
print('ave_extra_time_to_goals: {}'.format(np.mean(etg_normal[success_cases,:])))

# Perform Wilcoxon signed-rank test for random cases
etg_wnum_random = etg_wnum[success_cases].mean(axis=1)
etg_normal_random = etg_normal[success_cases].mean(axis=1)
stat_gen, pval_gen = stats.wilcoxon(etg_wnum_random, etg_normal_random)
print('\nWilcoxon signed-rank test results for random cases:')
print(f'p-value = {pval_gen:.3e}')

