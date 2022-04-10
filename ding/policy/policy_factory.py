# from typing import Dict, Any, Callable
# from collections import namedtuple

# from ding.torch_utils import to_device


# class PolicyFactory:
#     r"""
#     Overview:
#         Pure random policy. Only used for initial sample collecting if `cfg.policy.random_collect_size` > 0.
#     """

#     @staticmethod
#     def get_random_policy(
#             policy: 'BasePolicy',  # noqa
#             action_space: 'gym.spaces.Space' = None,  # noqa
#             forward_fn: Callable = None,
#     ) -> None:
#         assert not (action_space is None and forward_fn is None)
#         random_collect_function = namedtuple(
#             'random_collect_function', [
#                 'forward',
#                 'process_transition',
#                 'get_train_sample',
#                 'reset',
#                 'get_attribute',
#             ]
#         )

#         def forward(data: Dict[int, Any], *args, **kwargs) -> Dict[int, Any]:

#             actions = {}
#             for env_id in data:
#                 # For continuous env, action is limited in [-1, 1] for model output.
#                 # Env would scale it to its original action range.
#                 actions[env_id] = {
#                     'action': discrete_random_action(min, max, shape)
#                     if discrete else continuous_random_action(-1, 1, shape)
#                 }
#                 if 'global_state' in data[env_id].keys():
                    # for smac
                    # logit = np.ones_like(data[env_id]['action_mask'])
                    # logit[data[env_id]['action_mask'] == 0.0] = -1e8
                    # import torch
                    # dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(logit))
                    # actions[env_id] = {
                    #     'action': np.array(dist.sample()), 'logit': np.array(logit)
                    # }

#             return actions

#         def reset(*args, **kwargs) -> None:
#             pass

#         if action_space is None:
#             return random_collect_function(
#                 forward_fn, policy.process_transition, policy.get_train_sample, reset, policy.get_attribute
#             )
#         elif forward_fn is None:
#             return random_collect_function(
#                 forward, policy.process_transition, policy.get_train_sample, reset, policy.get_attribute
#             )


from typing import Dict, Any, Callable
from collections import namedtuple
from ding.torch_utils import to_device
import numpy as np
from ding.torch_utils import to_ndarray, to_list

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
                if not isinstance(action_space,list):
                    actions[env_id] = {'action': action_space.sample()} 
                # elif 'global_state' in data[env_id].keys():
                #     # for smac
                #     logit = np.ones_like(data[env_id]['action_mask'])
                #     logit[data[env_id]['action_mask'] == 0.0] = -1e8
                #     import torch
                #     dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(logit))
                #     actions[env_id] = {
                #         'action': np.array(dist.sample()), 'logit': np.array(logit)
                #     }
                else:
                    # for gfootball
                    actions[env_id] = {'action': np.array([action_space_agent.sample()  for action_space_agent in action_space]), 
                    'logit': np.ones([len(action_space),action_space[0].n])}

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