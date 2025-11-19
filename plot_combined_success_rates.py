import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, json
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


matplotlib.rcParams["hatch.linewidth"] = 1.0


# Create figure with two subplots side by side
# Adjust width ratio to make bar widths consistent (8:2 = 4:1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

# First plot (from plot_success_rates.py)
short_names = ['orca', 'cadrl', 'vanilla', "mean", 'wnum']
methods = ['ORCA', 'CADRL', "Vanilla MPC", "T-MPC", 'WNumMPC (Ours)']
success_rate = {}
timeout_rate = {}
for instance_type in ['opp', 'gen']:
    success_rate[instance_type] = {}
    timeout_rate[instance_type] = {}
    for i in range(4):
        success_rate[instance_type][i] = {}
        timeout_rate[instance_type][i] = {}
        n = 2 * i + 2
        for short_name, method in zip(short_names, methods):
            file_name = './datas/eval_results/{}-100-{}-{}.json'.format(n, short_name, instance_type)
            print(file_name)
            with open(file_name, 'r') as file:
                res = json.load(file)
            N = len(res['path_lengths'])
            success_rate[instance_type][i][method] = (N-len(res["collision_cases"])-len(res["timeout_cases"]))/N
            timeout_rate[instance_type][i][method] = len(res["timeout_cases"])/N

# Adjust position range to match the second plot's margin
gap = 0.2
bar_width = 0.2
spaces = [0.0] + [gap + bar_width * 5 for _ in range(7)]
position = np.cumsum(spaces)

sum_gap: float = 0.0
plt.rcParams["image.cmap"] = "viridis"
for id, key in enumerate(methods):
    data = []
    data_timeout = []
    for i in range(4):
        for instance_type in ['gen', 'opp']:
            data.append(100*success_rate[instance_type][i][key])
            data_timeout.append(100*timeout_rate[instance_type][i][key])
    ax1.bar(position+sum_gap, data, width=0.2, label=key, color=cm.viridis(1.0-id/(len(methods))))
    timeout_bars = ax1.bar(position+sum_gap, data_timeout, width=0.2, bottom=data, color="white", lw=0.0, edgecolor=cm.viridis(1.0-id/(len(methods))), hatch="//////")
    sum_gap += gap

x_max = position[-1] + 5.5*gap
ax1.set_xlim(-1.5*gap, x_max)
ax1.set_xticks(position+2.0*gap)

ax1.set_xticklabels(["N=3\nRandom", "N=3\nCrossing", "N=5\nRandom", "N=5\nCrossing", "N=7\nRandom", "N=7\nCrossing", "N=9\nRandom", "N=9\nCrossing"])
ax1.set_ylabel("Success Rate (%)")
ax1.set_ylim(0, 100)
ax1.set_title("Holonomic Simulation")

for ytick in ax1.get_yticks():
    ax1.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Second plot (from plot_success_rates_maru.py)
short_names = ['vanilla', 'mean', 'wnum']
methods = ["Vanilla MPC", 'T-MPC', 'WNumMPC (Ours)']
success_rate = {}
for short_name, method in zip(short_names, methods):
    file_name = './datas/maru_results/slow-7-100-{}-margin.json'.format(short_name)
    print(file_name)
    with open(file_name, 'r') as file:
        res = json.load(file)
    assert len(res['path_lengths']) % 2 == 0
    N = len(res['path_lengths']) // 2
    success_rate[method] = {'opp': N, 'gen': N, 'opp-sim': N, 'gen-sim': N}
    timeout_rate[method] = {'opp': 0, 'gen': 0, 'opp-sim': 0, 'gen-sim': 0}
    for case in res['collision_cases'] + res['timeout_cases']:
        if case % 2 == 0:
            success_rate[method]['opp'] -= 1
        else:
            success_rate[method]['gen'] -= 1
    for case in res['timeout_cases']:
        if case % 2 == 0:
            timeout_rate[method]['opp'] += 1
        else:
            timeout_rate[method]['gen'] += 1
    success_rate[method]['opp'] /= N
    success_rate[method]['gen'] /= N
    timeout_rate[method]['opp'] /= N
    timeout_rate[method]['gen'] /= N

for short_name, method in zip(short_names, methods):
    file_name = './datas/sim_results/slow-7-100-{}-margin.json'.format(short_name)
    print(file_name)
    with open(file_name, 'r') as file:
        res = json.load(file)
    assert len(res['path_lengths']) % 2 == 0
    N = len(res['path_lengths']) // 2
    for case in res['collision_cases'] + res['timeout_cases']:
        if case % 2 == 0:
            success_rate[method]['opp-sim'] -= 1
        else:
            success_rate[method]['gen-sim'] -= 1
    for case in res['timeout_cases']:
        if case % 2 == 0:
            timeout_rate[method]['opp-sim'] += 1
        else:
            timeout_rate[method]['gen-sim'] += 1
    success_rate[method]['opp-sim'] /= N
    success_rate[method]['gen-sim'] /= N
    timeout_rate[method]['opp-sim'] /= N
    timeout_rate[method]['gen-sim'] /= N

position = np.arange(4)
gap = 1./4

sum_gap: float = 0.0
for id, key in enumerate(methods):
    data = []
    data_timeout = []
    # Add sim results
    data.append(100*success_rate[key]['gen-sim'])  # Random(Sim)
    data.append(100*success_rate[key]['opp-sim'])  # Crossing(Sim)
    # Add maru results
    data.append(100*success_rate[key]['gen'])  # Random
    data.append(100*success_rate[key]['opp'])  # Crossing

    data_timeout.append(100*timeout_rate[key]['gen-sim'])
    data_timeout.append(100*timeout_rate[key]['opp-sim'])
    data_timeout.append(100*timeout_rate[key]['gen'])
    data_timeout.append(100*timeout_rate[key]['opp'])
    
    ax2.bar(position+sum_gap, data, width=gap, label=key, color=cm.viridis(1.0-(2+id)/5))
    timeout_bars = ax2.bar(position+sum_gap, data_timeout, width=gap, bottom=data, color="white", lw=0.0, edgecolor=cm.viridis(1.0-(2+id)/5), hatch="//////")
    sum_gap += gap

ax2.set_xlim(-1.5*gap, 3+2.5*gap)
ax2.set_xticks(position+gap/2)
ax2.set_xticklabels(["Random\n(Sim)", "Crossing\n(Sim)", "Random\n(Real)", "Crossing\n(Real)"])
ax2.set_ylabel("Success Rate (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Differential Wheeled Robots")

for ytick in ax2.get_yticks():
    ax2.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Create a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), ncol=5)

# Adjust the margins
plt.subplots_adjust(bottom=0.24, top=0.9, left=0.05, right=0.99, wspace=0.12)

#plt.show()
plt.savefig("./figures/Combined_Success_Rate.png")
plt.close() 