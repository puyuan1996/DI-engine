from typing import Union, Dict, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder
from copy import deepcopy


@MODEL_REGISTRY.register('mavac_nsw')
class MAVACNSW(nn.Module):
    r"""
    Overview:
        The MAVAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            actor_head_hidden_size: int = 128,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 128,
            critic_head_layer_num: int = 1,
            action_space: str = 'discrete',
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            agent_num: int = 8,  # TODO(pu):
    ) -> None:
        r"""
        Overview:
            Init the VAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - share_encoder (:obj:`bool`): Whether share encoder.
            - continuous (:obj:`bool`): Whether collect continuously.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
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
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(MAVACNSW, self).__init__()
        self.agent_num = agent_num
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape
        # Encoder Type
        if isinstance(agent_obs_shape, int) or len(agent_obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(agent_obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(agent_obs_shape)
            )
        if isinstance(global_obs_shape, int) or len(global_obs_shape) == 1:
            global_encoder_cls = FCEncoder
        elif len(global_obs_shape) == 3:
            global_encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(global_obs_shape)
            )

        # self.actor_encoder_agent = encoder_cls(
        #     agent_obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
        # )
        # self.critic_encoder_agent = global_encoder_cls(
        #     global_obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
        # )
        # Head Type
        # self.critic_head_agent = RegressionHead(
        #     critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        # )
        #
        # actor_head_cls = DiscreteHead
        # self.actor_head_agent = actor_head_cls(
        #     actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
        # )

        self.critic_head_agent = nn.Sequential(
                        nn.Linear(global_obs_shape, critic_head_hidden_size), activation,
                        RegressionHead(
                            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
                        )
                    )
        actor_head_cls = DiscreteHead
        self.actor_head_agent = nn.Sequential(
                        nn.Linear(agent_obs_shape, actor_head_hidden_size), activation,
                        actor_head_cls(
                            actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type
                        )
                    )

        # share weight:
        # self.actor = [self.actor_encoder, self.actor_head]
        # self.critic = [self.critic_encoder, self.critic_head]

        # not share weight:
        # self.actor_encoder = nn.ModuleList()
        # self.critic_encoder = nn.ModuleList()

        self.actor_encoder = nn.Sequential()
        self.critic_encoder = nn.Sequential()

        self.actor_head = nn.ModuleList()
        self.critic_head = nn.ModuleList()

        # for _ in range(self.agent_num):
        #     self.actor_encoder.append(
        #         deepcopy(self.actor_encoder_agent)
        #     )
        for _ in range(self.agent_num):
            self.actor_head.append(
                deepcopy(self.actor_head_agent)
            )
        # for _ in range(self.agent_num):
        #     self.critic_encoder.append(
        #         deepcopy(self.critic_encoder_agent)
        #     )
        for _ in range(self.agent_num):
            self.critic_head.append(
                deepcopy(self.critic_head_agent)
            )
        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]

        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Parameter updates with VAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'`` or ``'compute_critic'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Forward with ``'compute_actor'``, Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N corresponding ``hidden_size``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Actor Examples:
            >>> model = VAC(64,128)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 128])

        Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> critic_outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)

        Actor-Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = actor_head_hidden_size``
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
        """
        action_mask = x['action_mask']
        x = x['agent_state']

        # share weight:
        # x = self.actor_encoder(x)
        # x = self.actor_head(x)

        # not share weight:
        # output = []
        # for agent_num in range(self.agent_num):
        #     # inputs['obs']['agent_state'] shape: batch_size, agent_num, agent_obs_shape
        #     # batch_size, 1, agent_obs_shape -> batch_size, action_shape
        #     output.append(self.actor_encoder[agent_num](x[:, agent_num, :]))
        # # {list: agent_num}, {Tensor: (batch_size, action_shape)} -> (batch_size, agent_num, action_shape)
        # x = torch.stack(output, dim=1)
        x = self.actor_encoder(x)

        output = []
        for agent_num in range(self.agent_num):
            # inputs['obs']['agent_state'] shape: batch_size, agent_num, agent_obs_shape
            # batch_size, 1, agent_obs_shape -> batch_size, action_shape
            output.append(self.actor_head[agent_num](x[:, agent_num, :])['logit'])
        # {list: agent_num}, {Tensor: (batch_size, action_shape)} -> (batch_size, agent_num, action_shape)
        x = {'logit': torch.stack(output, dim=1)}

        logit = x['logit']
        logit[action_mask == 0.0] = -99999999
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`Dict`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = critic_head_hidden_size``
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> critic_outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
        """
        # share weight:
        # x = self.critic_encoder(x['global_state'])
        # x = self.critic_head(x)

        # not share weight:
        # output = []
        # for agent_num in range(self.agent_num):
        #     # inputs['obs']['agent_state'] shape: batch_size, agent_num, agent_obs_shape
        #     # batch_size, 1, agent_obs_shape -> batch_size, action_shape
        #     output.append(self.critic_encoder[agent_num](x['global_state'][:, agent_num, :]))
        # # {list: agent_num}, {Tensor: (batch_size, action_shape)} -> (batch_size, agent_num, action_shape)
        # x = torch.stack(output, dim=1)
        x = self.critic_encoder(x['global_state'])

        output = []
        for agent_num in range(self.agent_num):
            # inputs['obs']['agent_state'] shape: batch_size, agent_num, agent_obs_shape
            # batch_size, 1, agent_obs_shape -> batch_size, action_shape
            output.append(self.critic_head[agent_num](x[:, agent_num, :])['pred'])
        # {list: agent_num}, {Tensor: (batch_size, action_shape)} -> (batch_size, agent_num, action_shape)
        x = {'pred': torch.stack(output, dim=1)}

        return {'value': x['pred']}

    def compute_actor_critic(self, x: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The encoded embedding tensor.

        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
            - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])


        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder.
            Returning the combination dictionry.

        """
        logit = self.compute_actor(x)['logit']
        value = self.compute_critic(x)['value']
        action_mask = x['action_mask']
        return {'logit': logit, 'value': value}
