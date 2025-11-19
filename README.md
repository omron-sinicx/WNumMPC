# Winding-Number-Aware Navigation 

This repository contains the code for the paper "Winding-Number-Aware Navigation" by ... .
Conf2024, [project_page], [paper].

The framework of our codes is based on the work ([sriyash421/Pred2Nav](https://github.com/sriyash421/Pred2Nav)),
and the codes for CADRL are based on the work ([vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav)). 

This `maru_exp` branch contains the codes for the real-robot experiment.
For the codes used in the holonomic simulation experiment, please see `main` branch of this repository.


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
- available policies: VanillaMPC, MeanMPC (T-MPC), WNumMPC (proposed method),
  - default: WNumMPC

Whether you run the policy with real maru robots or in the simulate environment is determined `use_maru` (default: False) in the config/experiment_param.yaml file.

If you use the real maru robots, the serial ports for connecting maru cradles must be specified with `serial_port` (default: ["/dev/ttyUSB0", "/dev/ttyUSB1"]) in config/config.py file.

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


- maru (submodule)
  - maruを動かすためのコード

- config
  - config/experiment_param.yaml: envの設定ファイル
    - 方策やAgent数を変更出来る (H: humanの人数(全体のAgent数はH+1), robot_policy: 方策)
    - plot_trajectory = True にするとエピソードごとの軌跡を描画する
  - その他のconfigはhydraで管理している


- WnumMPC
  - crowd_nav/policy/wnum_mpc.py: WnumMPCのラッパー
  - crowd_nav/policy/wnum_mpc_utils/cv_predictor.py: WnumMPCのコストや入力計算の実装
  - crowd_nav/policy/wnum_mpc_utils/nn_module.py: WnumMPCのNNの実装
