from typing import Dict, Any, Callable
from collections import namedtuple
import torch
from ding.torch_utils import to_device
import gym
import numpy as np


class PolicyFactory:
    r"""
    Overview:
        Pure random policy. Only used for initial sample collecting if `cfg.policy.random_collect_size` > 0.
    """

    @staticmethod
    def get_random_policy(
            policy: 'BasePolicy',  # noqa
            action_space: 'gym.spaces.Space' = None,  # noqa
            forward_fn: Callable = None,
    ) -> None:
        assert not (action_space is None and forward_fn is None)
        random_collect_function = namedtuple(
            'random_collect_function', [
                'forward',
                'process_transition',
                'get_train_sample',
                'reset',
                'get_attribute',
            ]
        )

        def forward(data: Dict[int, Any], *args, **kwargs) -> Dict[int, Any]:

            actions = {}
            for env_id in data:
                if isinstance(action_space, gym.spaces.Discrete) or isinstance(action_space, gym.spaces.Box):
                    actions[env_id] = {'action': action_space.sample()}
                elif isinstance(action_space, gym.spaces.MultiDiscrete):
                    action = action_space.sample()
                    action = [torch.LongTensor([v]) for v in action]
                    actions[env_id] = {'action': action}
                elif 'global_state' in data[env_id].keys():
                    # for smac
                    logit = np.ones_like(data[env_id]['action_mask'])
                    logit[data[env_id]['action_mask'] == 0.0] = -1e8
                    dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(logit))
                    actions[env_id] = {'action': np.array(dist.sample()), 'logit': np.array(logit)}
                elif isinstance(action_space, list):
                    # for gfootball
                    actions[env_id] = {
                        'action': np.array([action_space_agent.sample() for action_space_agent in action_space]),
                        'logit': np.ones([len(action_space), action_space[0].n])
                    }
                else:
                    if isinstance(action_space[0], gym.spaces.Discrete) and isinstance(action_space[1], gym.spaces.Box):
                        # for gym_hybrid
                        action_sample = {
                            'action_type': action_space[0].sample(),
                            'action_args': action_space[1].sample()
                        }
                        actions[env_id] = {'action': action_sample}

                    else:
                        # for go-bigger
                        if isinstance(action_space[0], gym.spaces.Box) and isinstance(action_space[1],
                                                                                      gym.spaces.Discrete):
                            action_sample_player1 = np.concatenate(
                                [action_space[0].sample(),
                                 np.array([action_space[1].sample()])]
                            )
                            action_sample_player2 = np.concatenate(
                                [action_space[0].sample(),
                                 np.array([action_space[1].sample()])]
                            )
                            action_sample_player3 = np.concatenate(
                                [action_space[0].sample(),
                                 np.array([action_space[1].sample()])]
                            )
                            actions[env_id] = {
                                'action': np.stack(
                                    [action_sample_player1, action_sample_player2, action_sample_player3]
                                )
                            }


            return actions

        def reset(*args, **kwargs) -> None:
            pass

        if action_space is None:
            return random_collect_function(
                forward_fn, policy.process_transition, policy.get_train_sample, reset, policy.get_attribute
            )
        elif forward_fn is None:
            return random_collect_function(
                forward, policy.process_transition, policy.get_train_sample, reset, policy.get_attribute
            )
