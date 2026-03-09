import numpy as np
from scipy.stats import wilcoxon, binomtest
import json
import sys

sim = False

files = [["wnum_iros.json", "wnum_iros-2.json", "wnum_iros-3.json", "wnum_iros-4.json"], ["mean_iros.json", "mean_iros-2.json", "mean_iros-3.json", "mean_iros-4.json"],["vanilla_iros.json", "vanilla_iros-2.json", "vanilla_iros-3.json", "vanilla_iros-4.json"]]
data_dir = "./datas/maru_results/"

if sim:
    print("sim")
    data_dir = "./datas/sim_results/"
    files = [["wnum_iros_sim.json", "wnum_iros_sim-2.json", "wnum_iros_sim-3.json", "wnum_iros_sim-4.json"], ["mean_iros_sim.json", "mean_iros_sim-2.json", "mean_iros_sim-3.json", "mean_iros_sim-4.json"],["vanilla_iros_sim.json", "vanilla_iros_sim-2.json", "vanilla_iros_sim-3.json", "vanilla_iros_sim-4.json"]]
else:
    print("real")
etgs = [[[],[]],[[],[]],[[],[]]]
success_casess = [[[],[]],[[],[]],[[],[]]]
success= [[[],[]],[[],[]],[[],[]]]

for j in range(3):

    with open(data_dir + files[j][0], 'r') as file:
        res = json.load(file)
    for k in range(1,4):
        with open(data_dir + files[j][k], 'r') as file:
            res2 = json.load(file)
            for key in res.keys():
                res[key].extend(res2[key])

    for i in range(2):
        N = len(res['path_lengths'])//2
        collision = len([t for t in res["collision_cases"] if t % 2 == i])
        timeout = len([t for t in res["timeout_cases"] if t % 2 == i])
        success[j][i] = np.isfinite(res['success_times'][i::2])
        success_cases = [t for t in range(len(res["success_times"])) if t % 2 == i and np.isnan(res['success_times'][t]) == False]
        etg = np.array(res['extra_time_to_goals']).reshape(2*N,7)
        etgs[j][i] = etg
        success_casess[j][i] = success_cases
        #print(etg[success_cases].mean(axis=1).mean())
    print(files[j][0].split('_')[0])
    print(f'Crossing: success rate = {success[j][0].mean()}, etg={etgs[j][0][success_casess[j][0]].mean(axis=1).mean()}')
    print(f'Random  : success rate = {success[j][1].mean()}, etg={etgs[j][1][success_casess[j][1]].mean(axis=1).mean()}')

print('=============== Comparison Test ===============')
for j,j0 in [(1,0),(2,0),(1,2)]:
    name = files[j][0].split('_')[0]
    name0 = files[j0][0].split('_')[0]
    print(f'{name} vs {name0}')
    for i in range(2):
        print('---' + ('Crossing' if i==0 else 'Random') + '---')
        # McNemar検定: success[j][i] vs success[0][i]（インデックス対応の2値列、片方が他方より確率が高いか）
        a = np.sum(success[j][i] & success[j0][i])   # 両方成功
        b = np.sum(success[j][i] & ~success[j0][i])  # 0のみ成功
        c = np.sum(~success[j][i] & success[j0][i])  # 1のみ成功
        d = np.sum(~success[j][i] & ~success[j0][i]) # 両方失敗
        n_discordant = b + c
        print("McNemar ({name} vs {name0})".format(name=name, name0=name0))
        print("(0_only_success, 1_only_success) = ({}, {}), n_discordant = {}".format(b, c, n_discordant))
        print("success rate: {:.4f}, {:.4f}".format(success[j][i].mean(), success[j0][i].mean()))
        if n_discordant == 0:
            print("no discordant pairs")
        else:
            # 帰無仮説: 両群の成功率は等しい → b は Binomial(b+c, 0.5)
            # alternative='greater': success[j0][0] の方が確率が高い (b > c の方向)
            # alternative='less': success[j][0] の方が確率が高い
            res_two = binomtest(b, n_discordant, p=0.5, alternative='two-sided')
            res_0_gt_1 = binomtest(b, n_discordant, p=0.5, alternative='greater')  # [j0][0] > [j][0]
            res_1_gt_0 = binomtest(b, n_discordant, p=0.5, alternative='less')     # [j][0] > [j0][0]
            print("two-sided p-value (rate difference): {}".format(res_two.pvalue))
            print("one-sided p-value ({name} is higher): {}".format(res_0_gt_1.pvalue, name=name))
            print("one-sided p-value ({name0} is higher): {}".format(res_1_gt_0.pvalue, name0=name0))

        both_success_cases = [t for t in success_casess[j][i] if t in success_casess[0][i]]
        
        etg_j = etgs[j][i][both_success_cases].reshape(-1)
        etg_0 = etgs[0][i][both_success_cases].reshape(-1)

        print('Wilcoxon ({name} vs {name0})'.format(name=name, name0=name0))

        print(f'etg: {name} = {etg_j.mean()}, {name0} = {etg_0.mean()}')
        stat_j, pval_j = wilcoxon(etg_j, etg_0, alternative='two-sided')
        print(f'two-sided stat: {stat_j}, pval: {pval_j}')
        stat_j, pval_j = wilcoxon(etg_j, etg_0, alternative='less')
        print(f'one-sided ({name} is lower) stat: {stat_j}, pval: {pval_j}')
        stat_j, pval_j = wilcoxon(etg_j, etg_0, alternative='greater')
        print(f'one-sided ({name0} is lower) stat: {stat_j}, pval: {pval_j}')