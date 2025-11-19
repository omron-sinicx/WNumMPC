# Winding-Number-Aware Navigation 

This repository contains the code for the paper "Winding-Number-Aware Navigation" by ... .
Conf2024, [project_page], [paper].

The framework of our codes is based on the work ([sriyash421/Pred2Nav](https://github.com/sriyash421/Pred2Nav)),
and the codes for CADRL are based on the work ([vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav)). 

This `main` branch contains the code for the holonomic simulation experiment.
For the code used in the real-robot experiment, please see `maru_exp` branch of this repository.

<br>

## Tested Environment
- Manjaro Linux 24.0.4
- Python 3.11.9

<br>

## Installation

```shell
# create venv
python -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
pip install git+https://github.com/sybrenstuvel/Python-RVO2.git  # for ORCA
```

<br>

## Getting started
if you want to run other policy, you can change the robot_policy in the config/experiment_param.yaml file.
- available policies: ORCA, CADRL, VanillaMPC, MeanMPC (T-MPC), WNumMPC (proposed method),
  - default: WNumMPC
- Agent Counts: 3, 5, 7, 9
  - default: 9
- Placement Generation: random (gen), crossing (opp)

As a specific rule of placement:
if Agent count = 9 and placement generation = gen, 
set eval_states: ./datas/eval_states/8-100-gen.npy.

<br>


### Run WNumMPC Policy
To execute the trained policy, run the following script. 
If you want to change the settings, modify experiment_param.yaml.
```shell
python eval_policy.py
```

<br>

### Training WNumMPC Policy
To run the policy training, execute the following script.
If you want to change the settings, modify experiment_param.yaml and h32/h64.yaml.
```shell
python train_ppo.py
```


<br>


## Citation
TODO

<br>


### (メモ) 各コードの説明
- 実行関係のcode
  - train_ppo.py: PPOでWnumMPCを学習するコード
  - optimize_param.py: WnumMPCのパラメータをoptunaするコード
  - eval_policy.py: 学習したWnumMPCを評価するコード
    - experiment_param.yamlのrobot_policyを変更すればCADRL/ORCAも動かせる
  - train_cadrl.py : CADRLを学習するコード
    - experiment_param.yamlのrobot_policy = CADRLにしてから実行


- config
  - config/experiment_param.yaml: envの設定ファイル
    - 方策やAgent数を変更出来る (H: humanの人数(全体のAgent数はH+1), robot_policy: 方策)
    - plot_trajectory = True にするとエピソードごとの軌跡を描画する
  - その他のconfigはhydraで管理している


- WnumMPC
  - crowd_nav/policy/wnum_mpc.py: WnumMPCのラッパー
  - crowd_nav/policy/wnum_mpc_utils/cv_predictor.py: WnumMPCのコストや入力計算の実装
  - crowd_nav/policy/wnum_mpc_utils/nn_module.py: WnumMPCのNNの実装

    
- CADRL
  - crowd_nav/policy/cadrl.py: CADRLの実装
  - crowd_nav/policy/cadrl_utils/以下: CADRLで使う物の実装


- ORCA
  - crowd_nav/policy/orca.py: ORCAのラッパー
  - 本体: Python-RVO2

