from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder
from copy import deepcopy


@MODEL_REGISTRY.register('maqac_nsw')
class MAQACNSW(nn.Module):
    r"""
    Overview:
        The MAQAC not share weight model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            agent_num: int = 8,  # TODO(pu):
    ) -> None:
        r"""
        Overview:
            Init the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - twin_critic (:obj:`bool`): Whether include twin critic.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details.
        """
        super(MAQACNSW, self).__init__()
        self.agent_num = agent_num
        agent_obs_shape: int = squeeze(agent_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.actor_agent = nn.Sequential(
            nn.Linear(agent_obs_shape, actor_head_hidden_size), activation,
            DiscreteHead(
                actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
            )
        )
        self.actor = nn.ModuleList()
        for _ in range(self.agent_num):
            self.actor.append(
                deepcopy(self.actor_agent)
            )

        self.twin_critic = twin_critic
        if self.twin_critic:
            self.critic_agent = nn.Sequential(
                nn.Linear(global_obs_shape, critic_head_hidden_size), activation,
                DiscreteHead(
                    critic_head_hidden_size,
                    action_shape,
                    critic_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            )
            self.critic = nn.ModuleList()
            self.critic_1 = nn.ModuleList()
            self.critic_2 = nn.ModuleList()

            for _ in range(self.agent_num):
                self.critic_1.append(
                    deepcopy(self.critic_agent)
                )
            for _ in range(self.agent_num):
                self.critic_2.append(
                    deepcopy(self.critic_agent)
                )
            self.critic.append(
                self.critic_1
            )
            self.critic.append(
                self.critic_2
            )
        # else:
        #     self.critic = nn.Sequential(
        #         nn.Linear(global_obs_shape, critic_head_hidden_size), activation,
        #         DiscreteHead(
        #             critic_head_hidden_size,
        #             action_shape,
        #             critic_head_layer_num,
        #             activation=activation,
        #             norm_type=norm_type
        #         )
        #     )

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use bbservation and action tensor to predict output.
            Parameter updates with QAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.
            Forward with ``'compute_critic'``, inputs (`Dict`) Necessary Keys:
                - ``obs``, ``action`` encoded tensors.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.
                Forward with ``'compute_actor'``, Necessary Keys (either):
                    - action (:obj:`torch.Tensor`): Action tensor with same size as input ``x``.
                    - logit (:obj:`torch.Tensor`): Action's probabilities.
                Forward with ``'compute_critic'``, Necessary Keys:
                    - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Actor Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
        Critic Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``global_obs_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = actor_head_hidden_size``
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of forward pass encoder and head.
        ReturnsKeys (either):
            - action (:obj:`torch.Tensor`): Continuous action tensor with same size as ``action_shape``.
            - logit (:obj:`torch.Tensor`):
                Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - logit (:obj:`list`): 2 elements, mu and sigma, each is the shape of :math:`(B, N0)`.
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, B is batch size.
        Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])
        """
        action_mask = inputs['obs']['action_mask']

        # share weight:
        # x = self.actor(inputs['obs']['agent_state'])

        # not share weight:
        output = []
        for agent_num in range(self.agent_num):
            # inputs['obs']['agent_state'] shape: batch_size, agent_num, agent_obs_shape
            # batch_size, 1, agent_obs_shape -> batch_size, action_shape
            output.append(self.actor[agent_num](inputs['obs']['agent_state'][:, agent_num, :])['logit'])
        # {list: agent_num}, {Tensor: (batch_size, action_shape)} -> (batch_size, agent_num, action_shape)
        x = {'logit': torch.stack(output, dim=1)}

        return {'logit': x['logit'], 'action_mask': action_mask}

    def compute_critic(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - ``obs``, ``action`` encoded tensors.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Q-value output.
        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.
        """

        if self.twin_critic:
            # share weight:
            # x = [m(inputs['obs']['global_state'])['logit'] for m in self.critic]

            # not share weight:
            x = []
            for critic in self.critic:
                output = []
                for agent_num in range(self.agent_num):
                    # inputs['obs']['global_state'] shape: batch_size, agent_num, global_obs_shape
                    # batch_size, 1, global_obs_shape -> batch_size, action_shape
                    output.append(critic[agent_num](inputs['obs']['global_state'][:, agent_num, :])['logit'])
                output = torch.stack(output, dim=1)
                x.append(output)
        # else:
        #     x = self.critic(inputs['obs']['global_state'])['logit']
        return {'q_value': x}

