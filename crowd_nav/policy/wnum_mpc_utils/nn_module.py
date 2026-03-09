import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor, TensorDictSequential
from torchrl.data import BoundedTensorSpec
from crowd_nav.policy.wnum_mpc_utils.wnum_utils import WNumPolicyObservation, convert_actor_observation
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.utils import ExplorationType


class WNumNetworkCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: TensorDict) -> torch.Tensor:
        # flatten (batch_size, ...) -> (batch_size, input_size)
        batch_size = x.batch_size
        self_states = x["self_states"].view(batch_size + (-1,))      # (batch_size, 5)
        others_states = x["others_states"].view(batch_size + (-1,))  # (batch_size, human_num*9)
        input_tensor = torch.cat([self_states, others_states], dim=-1)

        return self.model(input_tensor)


class WNumNetworkActor(nn.Module):
    def __init__(self, input_size, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 * out_size),
            NormalParamExtractor(),
        )

    def forward(self, x: TensorDict) -> torch.Tensor:
        # flatten (batch_size, ...) -> (batch_size, input_size)
        batch_size = x.batch_size
        converted_self_states = x["converted_self_states"].view(batch_size + (-1,))  # (batch_size, 5)
        converted_other_states = x["converted_other_states"].view(batch_size + (-1,))  # (batch_size, human_num*7)
        input_tensor = torch.cat([converted_self_states, converted_other_states], dim=-1)

        return self.model(input_tensor)


class WNumNNSelector:
    def __init__(self, training_param: DictConfig, input_size: int, out_size: int, human_num: int) -> None:
        self.nn_param: DictConfig = training_param.nn_param
        self.input_size: int = input_size  # (H*7+5)
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
                "low": -1.0,
                "high": 1.0,
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
        input_dict: TensorDict = convert_actor_observation(observation)
        w_num_dist: TensorDict = self.model.forward(TensorDict({"observation": input_dict}, []).to(self.device))
        w_num: torch.Tensor = w_num_dist["action"].detach().numpy()
        log_probs: torch.Tensor = w_num_dist["sample_log_prob"].detach()
        w_num_id = None
        return w_num, w_num_id, log_probs
