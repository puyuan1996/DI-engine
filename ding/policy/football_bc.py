from typing import Tuple, List, Dict, Any
from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_train_sample, get_nstep_return_data
from ding.policy import BehaviourCloningPolicy
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
import torch
from collections import namedtuple
import os

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


@POLICY_REGISTRY.register('football_bc')
class FootballBCPolicy(BehaviourCloningPolicy):

    def _forward_learn(self, data: dict) -> dict:
        return super()._forward_learn(data)

    def _forward_collect(self, data: dict, eps: float):
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        ret = super()._process_transition(obs, model_output, timestep)
        ret['next_obs'] = timestep.obs
        return ret

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        super()._get_train_sample(data)
        data = get_nstep_return_data(data, 1, gamma=0.99)
        return get_train_sample(data, unroll_len=1)

    def _forward_eval(self, data: dict) -> dict:
        if isinstance(data, dict):
            data_id = list(data.keys())
            data = default_collate(list(data.values()))
            if self._cuda:
                data = to_device(data, self._device)
            self._eval_model.eval()
            with torch.no_grad():
                output = self._eval_model.forward(data)
            if self._cuda:
                output = to_device(output, 'cpu')
            output = default_decollate(output)
            return {i: d for i, d in zip(data_id, output)}
        return self._model(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'football_naive_q', ['dizoo.gfootball.model.football_q_network']
