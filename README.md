# Winding-Number-Aware Navigation 

This repository contains the code for the paper "Symmetry-Breaking in Multi-Agent Navigation: Winding Number-Aware MPC with a Learned Topological Strategy".

The framework of our codes is based on the work ([sriyash421/Pred2Nav](https://github.com/sriyash421/Pred2Nav)),
and the codes for CADRL are based on the work ([vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav)). 

This `maru_exp` branch contains the code for the real-robot experiment.
For the code used in the holonomic simulation experiment, please see `main` branch of this repository.


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
