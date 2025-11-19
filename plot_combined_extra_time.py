import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, json
from scipy import stats
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

# First plot (from plot_success_rates.py)
short_names = ['orca', 'cadrl', 'vanilla', "mean", 'wnum']
methods = ['ORCA', 'CADRL', "Vanilla MPC", "T-MPC", 'WNumMPC (Ours)']
extra_time = {}
for instance_type in ['opp', 'gen']:
    extra_time[instance_type] = {}
    for i in range(4):
        extra_time[instance_type][i] = {}
        n = 2 * i + 2
        for short_name, method in zip(short_names, methods):
            file_name = './datas/eval_results/{}-100-{}-{}.json'.format(n, short_name, instance_type)
            print(file_name)
            with open(file_name, 'r') as file:
                res = json.load(file)
            extra_time[instance_type][i][method] = np.nanmean(res['extra_time_to_goals'])

gap = 0.2
bar_width = 0.2
spaces = [0.0] + [gap + bar_width * 5 for _ in range(7)]
position = np.cumsum(spaces)


sum_gap: float = 0.0
plt.rcParams["image.cmap"] = "viridis"
for id, key in enumerate(methods):
    data = []
    for i in range(4):
        for instance_type in ['gen', 'opp']:
            data.append(extra_time[instance_type][i][key])
    ax1.bar(position+sum_gap, data, width=0.2, label=key, color=cm.viridis(1.0-id/(len(methods))))
    sum_gap += gap

x_max = position[-1] + 5.5*gap
ax1.set_xlim(-1.5*gap, x_max)
ax1.set_xticks(position+2.0*gap)

ax1.set_xticklabels(["N=3\nRandom", "N=3\nCrossing", "N=5\nRandom", "N=5\nCrossing", "N=7\nRandom", "N=7\nCrossing", "N=9\nRandom", "N=9\nCrossing"])
ax1.set_ylabel("Extra Time to Goals")
ax1.set_title("Holonomic Simulation")

for ytick in ax1.get_yticks():
    ax1.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Second plot (from plot_success_rates_maru.py)
short_names = ['vanilla', 'mean', 'wnum']
methods = ["Vanilla MPC", 'T-MPC', 'WNumMPC (Ours)']
extra_time = {}
for short_name, method in zip(short_names, methods):
    # Load data from maru_results
    file_name_maru = './datas/maru_results/slow-7-100-{}-margin.json'.format(short_name)
    print(file_name_maru)
    with open(file_name_maru, 'r') as file:
        res_maru = json.load(file)
    assert len(res_maru['path_lengths']) % 2 == 0
    N_maru = len(res_maru['path_lengths']) // 2
    extra_time[method] = {'opp': [], 'gen': []}
    etg = np.array(res_maru['extra_time_to_goals']).reshape(2*N_maru,7)
    success_cases_opp = [t for t in range(len(res_maru["success_times"])) if t % 2 == 0 and np.isnan(res_maru['success_times'][t]) == False]
    success_cases_gen = [t for t in range(len(res_maru["success_times"])) if t % 2 == 1 and np.isnan(res_maru['success_times'][t]) == False]

    extra_time[method]['opp'] = np.mean(etg[success_cases_opp])
    extra_time[method]['gen'] = np.mean(etg[success_cases_gen])
    extra_time[method]['opp_data'] = etg[success_cases_opp]  # Store raw data for statistical test
    extra_time[method]['gen_data'] = etg[success_cases_gen]  # Store raw data for statistical test

    # Load data from sim_results
    file_name_sim = './datas/sim_results/slow-7-100-{}-margin.json'.format(short_name)
    print(file_name_sim)
    if not os.path.exists(file_name_sim):
        raise FileNotFoundError(f"Sim environment data not found: {file_name_sim}")
    
    with open(file_name_sim, 'r') as file:
        res_sim = json.load(file)
    assert len(res_sim['path_lengths']) % 2 == 0
    N_sim = len(res_sim['path_lengths']) // 2
    etg = np.array(res_sim['extra_time_to_goals']).reshape(2*N_maru,7)
    success_cases_opp = [t for t in range(len(res_sim["success_times"])) if t % 2 == 0 and np.isnan(res_sim['success_times'][t]) == False]
    success_cases_gen = [t for t in range(len(res_sim["success_times"])) if t % 2 == 1 and np.isnan(res_sim['success_times'][t]) == False]
    extra_time[method]['opp_sim'] = np.mean(etg[success_cases_opp])
    extra_time[method]['gen_sim'] = np.mean(etg[success_cases_gen])
    extra_time[method]['opp_sim_data'] = etg[success_cases_opp]  # Store raw data for statistical test
    extra_time[method]['gen_sim_data'] = etg[success_cases_gen]  # Store raw data for statistical test
    
position = np.arange(4)  # 4 positions for Random, Crossing, Random(Sim), Crossing(Sim)
gap = 1./4

sum_gap: float = 0.0
plt.rcParams["image.cmap"] = "viridis"
for id, key in enumerate(methods):
    data = []
    # Add sim results
    data.append(extra_time[key]['gen_sim'])  # Random(Sim)
    data.append(extra_time[key]['opp_sim'])  # Crossing(Sim)
    # Add maru results
    data.append(extra_time[key]['gen'])  # Random
    data.append(extra_time[key]['opp'])  # Crossing
    
    ax2.bar(position+sum_gap, data, width=gap, label=key, color=cm.viridis(1.0-(2+id)/5))
    sum_gap += gap

# Perform Wilcoxon signed-rank test
p_test: bool = False
if p_test:
    print("\nWilcoxon signed-rank test results:")
    print("Simulation environment:")
    stat_sim_opp, pval_sim_opp = stats.wilcoxon(extra_time['Normal MPC']['opp_sim_data'], extra_time['WNumMPC (Ours)']['opp_sim_data'])
    stat_sim_gen, pval_sim_gen = stats.wilcoxon(extra_time['Normal MPC']['gen_sim_data'], extra_time['WNumMPC (Ours)']['gen_sim_data'])
    print(f"Crossing (Sim): p-value = {pval_sim_opp:.3e}")
    print(f"Random (Sim): p-value = {pval_sim_gen:.3e}")

    print("\nReal-world environment:")
    stat_real_opp, pval_real_opp = stats.wilcoxon(extra_time['Normal MPC']['opp_data'], extra_time['WNumMPC (Ours)']['opp_data'])
    stat_real_gen, pval_real_gen = stats.wilcoxon(extra_time['Normal MPC']['gen_data'], extra_time['WNumMPC (Ours)']['gen_data'])
    print(f"Crossing (Real): p-value = {pval_real_opp:.3e}")
    print(f"Random (Real): p-value = {pval_real_gen:.3e}")

ax2.set_xlim(-1.5*gap, 3+2.5*gap)
ax2.set_xticks(position + 0.5*gap)
ax2.set_xticklabels(["Random\n(Sim)", "Crossing\n(Sim)", "Random\n(Real)", "Crossing\n(Real)"])
ax2.set_ylabel("Extra Time to Goals (s)")
ax2.set_title("Differential Wheeled Robots")

for ytick in ax2.get_yticks():
    ax2.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Create a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), ncol=5)

# Adjust the margins
plt.subplots_adjust(bottom=0.24, top=0.9, left=0.05, right=0.99, wspace=0.12)

#plt.show()
plt.savefig("./figures/Combined_Extra_Time.png")
plt.close() 
