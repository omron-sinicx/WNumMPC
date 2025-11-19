import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor, TensorDictSequential
from torchrl.data import BoundedTensorSpec
from crowd_nav.policy.wnum_mpc_utils.wnum_utils import convert_trajectory, WNumPolicyObservation
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.utils import ExplorationType


class WNumNetworkCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WNumNetworkActor(nn.Module):
    def __init__(self, input_size, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 * out_size),
            NormalParamExtractor(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WNumNNSelector:
    def __init__(self, training_param: DictConfig, input_size: int, out_size: int, human_num: int) -> None:
        self.nn_param: DictConfig = training_param.nn_param
        self.input_size: int = input_size
        self.out_size: int = out_size
        self.device: torch.device = torch.device("cpu")

        # model setting
        self.nn_model: WNumNetworkActor = WNumNetworkActor(input_size, self.nn_param.hidden_size, 2*human_num).to(self.device)
        self.policy_module: TensorDictModule = TensorDictSequential(
            TensorDictModule(self.nn_model, ["observation"], ["loc", "scale"]),
        )
        self.model: ProbabilisticActor = ProbabilisticActor(
            module=self.policy_module,
            spec=BoundedTensorSpec(-torch.ones(2*human_num), torch.ones(2*human_num)),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": -1.0,
                "max": 1.0,
                "upscale": 2.0,
                # "tanh_loc": True
            },
            return_log_prob=True,  # importance sampling
            default_interaction_type=ExplorationType.RANDOM,
        )

        self.human_num = human_num

    def eval(self) -> None:
        self.model.eval()

    def enable_train(self) -> None:
        self.model.train(True)

    def select_target_winding_number(self, observation: WNumPolicyObservation) -> tuple[np.ndarray | torch.Tensor, torch.Tensor | None, torch.Tensor]:
        input_data: torch.Tensor = convert_trajectory(observation)
        w_num_dist: TensorDict = self.model.forward(TensorDict({"observation": input_data}, []))
        w_num: torch.Tensor = w_num_dist["action"].detach().numpy()
        log_probs: torch.Tensor = w_num_dist["sample_log_prob"].detach()
        w_num_id = None
        return w_num, w_num_id, log_probs

