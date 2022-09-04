from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict, Optional
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from ding.torch_utils import one_hot, to_tensor
from ding.model.common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder
from ding.torch_utils import Adam, to_device, unsqueeze, ContrastiveLoss

class ExponentialMovingAverage(nn.Module):

    def __init__(self, decay: float, shape: Tuple[int]) -> None:
        super(ExponentialMovingAverage, self).__init__()
        self.decay = decay
        self.shape = shape
        self.reset()

    # @torch.no_grad
    def update(self, value: torch.Tensor) -> None:
        # i.e. self.hidden = (1.0 - self.decay) * value + self.decay * self.hidden
        self.count.add_(1)
        self.hidden -= (1.0 - self.decay) * (self.hidden - value)
        # ema correction
        # NOTE
        self.average = self.hidden / (1 - torch.pow(self.decay, self.count))

    @property
    def value(self) -> torch.Tensor:
        return self.average

    def reset(self, shape: Optional[Tuple] = None) -> None:
        if shape is None:
            shape = self.shape
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('hidden', torch.zeros(*shape))
        self.register_buffer('average', torch.zeros(*shape))


class VectorQuantizer(nn.Module):
    """
    Overview:
        The Variational AutoEncoder (VQ-VAE).
        Reference:
            [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
            self,
            embedding_num: int,
            embedding_dim: int = 128,
            beta: float = 0.25,
            is_ema: bool = False,
            is_ema_target: bool = False,
            eps_greedy_nearest: bool = False,
            embedding_table_onehot: bool = False,
    ):
        super(VectorQuantizer, self).__init__()
        self.K = embedding_num
        self.D = embedding_dim
        self.beta = beta
        self.is_ema = is_ema
        self.is_ema_target = is_ema_target
        self.eps_greedy_nearest = eps_greedy_nearest
        self.embedding_table_onehot = embedding_table_onehot

        self.embedding = nn.Embedding(self.K, self.D)
        if self.embedding_table_onehot:
            self.embedding.weight.data = one_hot(torch.arange(self.D), self.D)  # B, K
        else:
            self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        if self.is_ema:
            self.ema_N = ExponentialMovingAverage(0.99, (self.K,))
            self.ema_m = ExponentialMovingAverage(0.99, (self.K, self.D))

    def train(self, encoding: torch.Tensor, eps=0.05) -> torch.Tensor:
        """
        Overview:
            input the encoding of actions, calculate the vqvae loss
        Arguments:
            - encoding shape: (B,D)
        """
        device = encoding.device
        encoding_shape = encoding.shape
        if len(encoding_shape) == 3:
            # for multi-agent case, e.g. gobigger, 
            # if encoding shape is (B,A,D), where B is batch_size, A is agent_num, D is encoding_dim
            encoding = encoding.view(-1, encoding_shape[-1])
        quantized_index = self.encode(encoding)
        # print('torch.unique(quantized_index):', torch.unique(quantized_index))
        if self.eps_greedy_nearest:
            for i in range(encoding.shape[0]):
                if np.random.random() < eps:
                    quantized_index[i] = torch.randint(0, self.K, [1])
        quantized_one_hot = one_hot(quantized_index, self.K)  # B, K
        quantized_embedding = torch.matmul(quantized_one_hot, self.embedding.weight)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(encoding, quantized_embedding.detach())

        #  VQ-VAE dictionary updates with Exponential Moving Averages
        if self.is_ema:
            # (K, )
            self.ema_N.update(quantized_one_hot.sum(dim=0))
            # (B,K)->(K,B)  * (B,D)
            delta_m = torch.matmul(quantized_one_hot.permute(1, 0), encoding)  # K, D
            self.ema_m.update(delta_m)

            N = self.ema_N.value
            total = N.sum()
            normed_N = (N + 1e-5) / (total + self.K * 1e-5) * total
            target_embedding_value = self.ema_m.value / normed_N.unsqueeze(1)

            if self.is_ema_target:
                embedding_loss = F.mse_loss(self.embedding.weight, target_embedding_value)
            else:
                embedding_loss = torch.zeros(1, device=device)
                self.embedding.weight.data = target_embedding_value
        else:
            if not self.embedding_table_onehot:
                embedding_loss = F.mse_loss(quantized_embedding, encoding.detach())
            else:
                embedding_loss = torch.tensor(0.)

        vq_loss = commitment_loss * self.beta + embedding_loss
        # straight-through estimator for passing gradient from decoder, add the residue back to the encoding
        quantized_embedding = encoding + (quantized_embedding - encoding).detach()

        return quantized_index, quantized_embedding, vq_loss, embedding_loss, commitment_loss

    def encode(self, encoding: torch.Tensor) -> torch.Tensor:
        # # Method 2: Compute L2 distance between encoding and embedding weights
        # dist = torch.sum(flat_encoding ** 2, dim=1, keepdim=True) + \
        #        torch.sum(self.embedding.weight ** 2, dim=1) - \
        #        2 * torch.matmul(flat_encoding, self.embedding.weight.t())
        # # Get the encoding that has the min distance
        # quantized_index = torch.argmin(dist, dim=1).unsqueeze(1)

        # .sort()[1] take the index after sorted, [:,0] take the nearest index
        # return torch.cdist(encoding, self.embedding.weight, p=2).sort()[1][:, 0]
        """
        Overview:
            input the encoding of actions, find the nearest encoding index in embedding table
        Arguments:
            - encoding shape: (B,D) self.embedding.weight shape:(K,D)
            - return shape: (B), the nearest encoding index in embedding table
        """

        return torch.cdist(encoding, self.embedding.weight, p=2).min(dim=-1)[1]

    def decode(self, quantized_index: torch.Tensor) -> torch.Tensor:
        # Convert to one-hot encodings
        quantized_one_hot = one_hot(quantized_index, self.K)
        # Quantize the encoding
        quantized_embedding = torch.matmul(quantized_one_hot, self.embedding.weight)
        return quantized_embedding


class ActionVQVAE(nn.Module):
    """
        Overview:
            The ``ActionVQVAE`` used to do action representation learning.
        Interfaces:
            ``__init__``, ``train``, ``train_with_obs``, ``encode``, ``decode``.
    """

    def __init__(
            self,
            action_shape: Union[int, Dict],
            embedding_num: int,
            embedding_dim: int = 128,
            hidden_dims: List = [256],
            beta: float = 0.25,
            vq_loss_weight: float = 1,
            is_ema: bool = False,
            is_ema_target: bool = False,
            eps_greedy_nearest: bool = False,
            cont_reconst_l1_loss: bool = False,
            cont_reconst_smooth_l1_loss: bool = False,
            categorical_head_for_cont_action: bool = False,
            threshold_categorical_head_for_cont_action: bool = False,
            categorical_head_for_cont_action_threshold: int = 0.9,
            n_atom: int = 51,
            gaussian_head_for_cont_action: bool = False,
            embedding_table_onehot: bool = False,
            vqvae_return_weight: bool = False,
            obs_regularization: bool = False,
            obs_shape: int = None,
            predict_loss_weight: float = 1,
            mask_pretanh: bool = False,
            recons_loss_cont_weight: float = 1,
            q_contrastive_regularizer: bool = False,
    ) -> None:
        super(ActionVQVAE, self).__init__()

        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.obs_regularization = obs_regularization
        self.hidden_dims = hidden_dims
        self.vq_loss_weight = vq_loss_weight
        self.predict_loss_weight = predict_loss_weight
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.act = nn.ReLU()
        self.cont_reconst_l1_loss = cont_reconst_l1_loss
        self.cont_reconst_smooth_l1_loss = cont_reconst_smooth_l1_loss
        self.categorical_head_for_cont_action = categorical_head_for_cont_action
        self.threshold_categorical_head_for_cont_action = threshold_categorical_head_for_cont_action
        self.categorical_head_for_cont_action_threshold = categorical_head_for_cont_action_threshold
        self.n_atom = n_atom
        self.gaussian_head_for_cont_action = gaussian_head_for_cont_action
        self.embedding_table_onehot = embedding_table_onehot
        self.vqvae_return_weight = vqvae_return_weight
        self.mask_pretanh = mask_pretanh
        self.recons_loss_cont_weight = recons_loss_cont_weight
        self.q_contrastive_regularizer = q_contrastive_regularizer

        if self.q_contrastive_regularizer==True:
             self.q_contrastive_regularizer = ContrastiveLoss(self.embedding_dim, self.embedding_dim, encode_shape=64)

        """Encoder"""
        # action encode head
        if isinstance(self.action_shape, int):  # continuous action
            self.encode_action_head = nn.Sequential(nn.Linear(self.action_shape, self.hidden_dims[0]), self.act)
        elif isinstance(self.action_shape, dict):  # hybrid action
            # input action: concat(continuous action, one-hot encoding of discrete action)
            self.encode_action_head = nn.Sequential(
                nn.Linear(
                    self.action_shape['action_type_shape'] + self.action_shape['action_args_shape'], self.hidden_dims[0]
                ), self.act
            )
        if self.obs_regularization:
            # encode: obs head
            self.encode_obs_head = nn.Sequential(nn.Linear(self.obs_shape, self.hidden_dims[0]), self.act)

            # decode: residual prediction head
            self.decode_prediction_head_layer1 = nn.Sequential(nn.Linear(self.hidden_dims[0], self.hidden_dims[0]),
                                                               nn.ReLU())
            self.decode_prediction_head_layer2 = nn.Linear(self.hidden_dims[0], self.obs_shape)

        self.encode_common = nn.Sequential(nn.Linear(self.hidden_dims[0], self.hidden_dims[0]), nn.ReLU())
        self.encode_mu_head = nn.Linear(self.hidden_dims[0], self.embedding_dim)

        modules = [self.encode_action_head, self.encode_common, self.encode_mu_head]
        self.encoder = nn.Sequential(*modules)

        # VQ layer
        self.vq_layer = VectorQuantizer(embedding_num, embedding_dim, beta, is_ema, is_ema_target, eps_greedy_nearest,
                                        embedding_table_onehot)

        """Decoder"""
        self.decoder = nn.Sequential(
            *[
                nn.Linear(self.embedding_dim, self.hidden_dims[0]),
                self.act,
                nn.Linear(self.hidden_dims[0], self.hidden_dims[0]),
                self.act,
            ]
        )
        self.decode_action_head = nn.Sequential(nn.Linear(self.embedding_dim, hidden_dims[0]), self.act)
        self.decode_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())

        if isinstance(self.action_shape, int):  # continuous action
            if self.categorical_head_for_cont_action or self.threshold_categorical_head_for_cont_action:
                # self.recons_action_head = nn.Sequential(nn.Linear(self.hidden_dims[0], self.action_shape*self.n_atom), nn.Tanh())
                self.recons_action_head = nn.Sequential(nn.Linear(self.hidden_dims[0], self.action_shape * self.n_atom))
            elif self.gaussian_head_for_cont_action:
                self.recons_action_head = ReparameterizationHead(
                    self.hidden_dims[0],
                    self.action_shape,
                    1,
                    sigma_type='conditioned',
                    activation=nn.ReLU(),
                    norm_type=None
                )
            else:
                self.recons_action_head = nn.Sequential(nn.Linear(self.hidden_dims[0], self.action_shape), nn.Tanh())

        elif isinstance(self.action_shape, dict):  # hybrid action
            # input action: concat(continuous action, one-hot encoding of discrete action)

            if self.categorical_head_for_cont_action or self.threshold_categorical_head_for_cont_action:
                # self.recons_action_cont_head = nn.Sequential(
                #     nn.Linear(self.hidden_dims[0], self.action_shape['action_args_shape']*self.n_atom), nn.Tanh()
                # )
                self.recons_action_cont_head = nn.Sequential(
                    nn.Linear(self.hidden_dims[0], self.action_shape['action_args_shape'] * self.n_atom))
            elif self.gaussian_head_for_cont_action:
                self.recons_action_cont_head = ReparameterizationHead(
                    self.hidden_dims[0],
                    self.action_shape['action_args_shape'],
                    1,
                    sigma_type='conditioned',
                    activation=nn.ReLU(),
                    norm_type=None
                )
            else:
                if self.mask_pretanh:
                    self.recons_action_cont_head_pretanh = nn.Sequential(
                        nn.Linear(self.hidden_dims[0], self.action_shape['action_args_shape'])
                    )
                else:
                    self.recons_action_cont_head = nn.Sequential(
                        nn.Linear(self.hidden_dims[0], self.action_shape['action_args_shape']), nn.Tanh()
                    )

            # self.recons_action_disc_head = nn.Sequential(
            #     nn.Linear(self.hidden_dims[0], self.action_shape['action_type_shape']), nn.ReLU()
            # )
            # TODO
            self.recons_action_disc_head = nn.Sequential(
                nn.Linear(self.hidden_dims[0], self.action_shape['action_type_shape'])
            )

    def _get_action_embedding(self, data: Dict) -> torch.Tensor:
        """
         Overview:
             maps the given action to ``action_embedding``. If action is continuous, use the original action,
             if is hybrid action, use the concatenate(``action_args``, one_hot(``action_type``)).
        """
        if isinstance(self.action_shape, int):  # continuous action
            action_embedding = data['action']
        elif isinstance(self.action_shape, dict):  # hybrid action
            action_disc_onehot = one_hot(data['action']['action_type'], num=self.action_shape['action_type_shape'])
            action_embedding = torch.cat([action_disc_onehot, data['action']['action_args']], dim=-1)
        return action_embedding

    def _recons_action(
            self,
            action_decoding: torch.Tensor,
            target_action: Union[torch.Tensor, Dict[str, torch.Tensor]] = None,
            weight: Union[torch.Tensor, Dict[str, torch.Tensor]] = None,
            threshold_phase: bool = False,
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
         Overview:
             maps the given action_decoding to original action ``recons_action``
             and calculate the ``recons_loss`` if given `` target_action``.
        """
        sigma = None  # debug
        mask=None
        recons_action_probs_left_mask_proportion=0
        recons_action_probs_right_mask_proportion=0
        if isinstance(self.action_shape, int):
            # continuous action
            if self.categorical_head_for_cont_action:
                support = torch.linspace(-1, 1, self.n_atom).to(action_decoding.device)  # TODO
                recons_action_logits = self.recons_action_head(action_decoding).view(-1, self.action_shape, self.n_atom)
                recons_action_probs = F.softmax(recons_action_logits, dim=-1)  # TODO
                recons_action = torch.sum(recons_action_probs * support, dim=-1)
                # recons_action_logits: tanh 
                # recons_action = torch.mean(recons_action_logits * support, dim=-1)
            elif self.threshold_categorical_head_for_cont_action:
                support = torch.linspace(-1, 1, self.n_atom).to(action_decoding.device)  # shape: (self.n_atom)
                recons_action_logits = self.recons_action_head(action_decoding).view(-1, self.action_shape, self.n_atom)
                recons_action_probs = F.softmax(recons_action_logits, dim=-1)  # shape: (B,A, self.n_atom)
                
                recons_action = torch.sum(recons_action_probs * support, dim=-1) # shape: (B,A)
                
                if target_action is None:
                    # collect or eval phase

                    if threshold_phase:
                        # TODO(pu): for construct some extreme action
                        # prob=[p1,p2,p3,p4], support=[s1,s2,s3,s4], if pi>threshold, then recons_action=support[i]
                        # shape: (B,A)
                        recons_action_left_lt_threshold_mask = recons_action_probs[:,:,0].ge(self.categorical_head_for_cont_action_threshold) 
                        recons_action_right_lt_threshold_mask = recons_action_probs[:,:,-1].ge(self.categorical_head_for_cont_action_threshold) 

                        if recons_action_left_lt_threshold_mask.sum()>0 or recons_action_right_lt_threshold_mask.sum()>0:
                            recons_action_probs_left_lt_threshold =  recons_action_probs[:,:,0].masked_select(recons_action_left_lt_threshold_mask) 
                            recons_action_probs_right_lt_threshold =  recons_action_probs[:,:,-1].masked_select(recons_action_right_lt_threshold_mask) 

                            # straight-through estimator for passing gradient from recons_action_probs_lt_threshold
                            recons_action[recons_action_left_lt_threshold_mask] = (recons_action_probs_left_lt_threshold + (1-recons_action_probs_left_lt_threshold ).detach()) *  support[0]
                            recons_action[recons_action_right_lt_threshold_mask] = (recons_action_probs_right_lt_threshold + (1-recons_action_probs_right_lt_threshold ).detach()) *  support[-1]

                            # statistics
                            recons_action_probs_left_mask_proportion = recons_action_left_lt_threshold_mask.sum()/ (recons_action_left_lt_threshold_mask.shape[0]* recons_action_left_lt_threshold_mask.shape[1])
                            recons_action_probs_right_mask_proportion = recons_action_right_lt_threshold_mask.sum()/ (recons_action_right_lt_threshold_mask.shape[0]* recons_action_right_lt_threshold_mask.shape[1])
                            print('='*20)
                            print('recons_action_probs_left_mask_proportion:', recons_action_probs_left_mask_proportion, 'recons_action_probs_right_mask_proportion:', recons_action_probs_right_mask_proportion)
                            print('='*20)


                    # recons_action_max = torch.max(recons_action_probs, dim=-1)[0]  # shape: (B,A)
                    # recons_action_max_index = torch.max(recons_action_probs, dim=-1)[1] # shape: (B,A)
                    
                    # # TODO(pu): for construct some extreme action
                    # # prob=[p1,p2,p3,p4], support=[s1,s2,s3,s4], if pi>threshold, then recons_action=support[i]
                    # # shape: (B,A)
                    # recons_action_lt_threshold_mask = recons_action_max.ge(self.categorical_head_for_cont_action_threshold)

                    # if recons_action_lt_threshold_mask.sum()>0:
                    #     recons_action_probs_lt_threshold = recons_action_max.masked_select(recons_action_lt_threshold_mask ) 
                    #     recons_action_probs_index_lt_threshold = recons_action_max_index.masked_select(recons_action_lt_threshold_mask )  # shape: (B,A)

                    #     # straight-through estimator for passing gradient from recons_action_probs_lt_threshold
                    #     recons_action[recons_action_lt_threshold_mask] = (recons_action_probs_lt_threshold + (1-recons_action_probs_lt_threshold ).detach())*  support[recons_action_probs_index_lt_threshold]

            elif self.gaussian_head_for_cont_action:
                mu_sigma_dict = self.recons_action_head(action_decoding)
                (mu, sigma) = (mu_sigma_dict['mu'], mu_sigma_dict['sigma'])
                # print(mu, sigma)
                dist = Independent(Normal(mu, sigma), 1)
                recons_action_sample = dist.rsample()
                recons_action = torch.tanh(recons_action_sample)
            else:
                recons_action = self.recons_action_head(action_decoding)

        elif isinstance(self.action_shape, dict):
            # hybrid action
            if self.categorical_head_for_cont_action:
                support = torch.linspace(-1, 1, self.n_atom).to(action_decoding.device)  # TODO
                recons_action_logits = self.recons_action_cont_head(action_decoding).view(-1, self.action_shape[
                    'action_args_shape'], self.n_atom)
                recons_action_probs = F.softmax(recons_action_logits, dim=-1)  # TODO
                recons_action_cont = torch.sum(recons_action_probs * support, dim=-1)
                # recons_action_logits: tanh 
                # recons_action_connt = torch.mean(recons_action_logits * support, dim=-1)
            elif self.gaussian_head_for_cont_action:
                mu_sigma_dict = self.recons_action_head(action_decoding)
                (mu, sigma) = (mu_sigma_dict['mu'], mu_sigma_dict['sigma'])
                dist = Independent(Normal(mu, sigma), 1)
                recons_action_sample = dist.rsample()
                recons_action_cont = torch.tanh(recons_action_sample)
            else:
                if self.mask_pretanh:
                    x = self.recons_action_cont_head_pretanh(action_decoding)
                    mask = x.ge(-1.2) & x.le(1.2)  # TODO
                    recons_action_cont = torch.tanh(x)
                else:
                    recons_action_cont = self.recons_action_cont_head(action_decoding)
        

            recons_action_disc_logit = self.recons_action_disc_head(action_decoding)
            recons_action_disc = torch.argmax(recons_action_disc_logit, dim=-1)
            recons_action = {
                'action_args': recons_action_cont,
                'action_type': recons_action_disc,
                'logit': recons_action_disc_logit,
                'mask': mask
            }

        if target_action is None:
            return recons_action, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion 
        else:
            if isinstance(self.action_shape, int):  # continuous action
                if self.cont_reconst_l1_loss:
                    recons_loss = F.l1_loss(recons_action, target_action)
                    recons_loss_none_reduction = F.l1_loss(recons_action, target_action, reduction='none').mean(-1)
                elif self.cont_reconst_smooth_l1_loss:
                    recons_loss = F.smooth_l1_loss(recons_action, target_action)
                    recons_loss_none_reduction = F.smooth_l1_loss(recons_action, target_action, reduction='none').mean(
                        -1)
                else:
                    if self.vqvae_return_weight and weight is not None:
                        recons_loss = (weight.reshape(-1, 1) * F.mse_loss(recons_action, target_action,
                                                                          reduction='none')).mean()
                    else:
                        recons_loss = F.mse_loss(recons_action, target_action)
                    recons_loss_none_reduction = F.mse_loss(recons_action, target_action, reduction='none').mean(-1)

            elif isinstance(self.action_shape, dict):  # hybrid action
                if self.cont_reconst_l1_loss:
                    recons_loss_cont = F.l1_loss(recons_action['action_args'],
                                                 target_action['action_args']
                                                 .view(-1, target_action['action_args'].shape[-1]))
                    recons_loss_cont_none_reduction = F.l1_loss(recons_action['action_args'],
                                                                target_action['action_args']
                                                                .view(-1, target_action['action_args'].shape[-1]),
                                                                reduction='none').mean(-1)
                elif self.cont_reconst_smooth_l1_loss:
                    recons_loss_cont = F.smooth_l1_loss(recons_action['action_args'],
                                                        target_action['action_args']
                                                        .view(-1, target_action['action_args'].shape[-1]))
                    recons_loss_cont_none_reduction = F.smooth_l1_loss(recons_action['action_args'],
                                                                       target_action['action_args']
                                                                       .view(-1,
                                                                             target_action['action_args'].shape[-1]),
                                                                       reduction='none').mean(-1)
                else:
                    if self.mask_pretanh:
                        # statistic the percent of mask
                        mask_percent = 1 - mask.sum().item() / (recons_action['action_args'].shape[0] * recons_action['action_args'].shape[1])
                        print('mask_percent:',mask_percent)
                        recons_loss_cont = F.mse_loss(recons_action['action_args'].masked_select(mask), target_action['action_args']
                                                  .view(-1, target_action['action_args'].shape[-1]).masked_select(mask))
                    else:
                        recons_loss_cont = F.mse_loss(recons_action['action_args'], target_action['action_args']
                                                    .view(-1, target_action['action_args'].shape[-1]))
                    recons_loss_cont_none_reduction = F.mse_loss(recons_action['action_args'],
                                                                 target_action['action_args']
                                                                 .view(-1, target_action['action_args'].shape[-1]),
                                                                 reduction='none').mean(-1)

                recons_loss_disc = F.cross_entropy(recons_action['logit'], target_action['action_type'].view(-1))
                recons_loss_disc_none_reduction = F.cross_entropy(recons_action['logit'],
                                                                  target_action['action_type'].view(-1),
                                                                  reduction='none').mean(-1)

                # here view(-1) is to be compatible with multi_agent case, e.g. gobigger
                # TODO(pu): relative weight
                # recons_loss_cont_weight = 0.1
                recons_loss = self.recons_loss_cont_weight * recons_loss_cont + recons_loss_disc
                recons_loss_none_reduction = recons_loss_cont_none_reduction + recons_loss_disc_none_reduction

            return recons_action, recons_loss, recons_loss_none_reduction, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion 

    def train(self, data: Dict, warmup: bool = False) -> Dict[str, torch.Tensor]:
        """
         Overview:
             The train method when don't use obs regularization.
        """
        action_embedding = self._get_action_embedding(data)
        encoding = self.encoder(action_embedding)
        quantized_index, quantized_embedding, vq_loss, embedding_loss, commitment_loss = self.vq_layer.train(encoding)
        action_decoding = self.decoder(quantized_embedding)
        if self.vqvae_return_weight and not warmup:
            recons_action, recons_loss, recons_loss_none_reduction, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion  \
                = self._recons_action(action_decoding,
                                      data['action'], data[
                                          'return_normalization'])
        else:
            recons_action, recons_loss, recons_loss_none_reduction, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion \
                = self._recons_action(action_decoding,
                                      data['action'])

        total_vqvae_loss = recons_loss + self.vq_loss_weight * vq_loss
        return {
            'quantized_index': quantized_index,
            'recons_loss_none_reduction': recons_loss_none_reduction,  # use for rl priority in dqn_vqvae
            'sigma': sigma,
            'total_vqvae_loss': total_vqvae_loss,
            'recons_loss': recons_loss,
            'vq_loss': vq_loss,
            'embedding_loss': embedding_loss,
            'commitment_loss': commitment_loss,
            'recons_action_probs_left_mask_proportion':recons_action_probs_left_mask_proportion, 
            'recons_action_probs_right_mask_proportion':recons_action_probs_right_mask_proportion,
        }

    def train_with_obs(self, data: Dict, warmup: bool = False) -> Dict[str, torch.Tensor]:
        """
         Overview:
             The train method when use obs regularization.
         Arguments:
             - data (:obj:`Dict`): Dict containing keyword:
                 - action (:obj:`torch.Tensor`): the original action
                 - obs (:obj:`torch.Tensor`): observation
                 - true_residual (:obj:`torch.Tensor`): the true observation residual, i.e. o_{t+1}-o_{t}
         Returns:
             - outputs (:obj:`Dict`): Dict containing keyword:
                 - quantized_index (:obj:`torch.Tensor`): the latent action.
                 and loss statistics
         """
        action_embedding = self._get_action_embedding(data)

        # encode
        action_embedding = self.encode_action_head(action_embedding)
        obs_embedding = self.encode_obs_head(data['obs'])
        action_obs_embedding_dot = action_embedding * obs_embedding
        action_obs_embedding = self.encode_common(action_obs_embedding_dot)
        encoding = self.encode_mu_head(action_obs_embedding)

        if self.q_contrastive_regularizer:
            data['q_value']
            encoding


        quantized_index, quantized_embedding, vq_loss, embedding_loss, commitment_loss = self.vq_layer.train(encoding)

        # decode
        action_decoding = self.decode_action_head(quantized_embedding)
        action_obs_decoding = action_decoding * obs_embedding

        action_obs_decoding_common = self.decode_common(action_obs_decoding)

        prediction_residual_tmp = self.decode_prediction_head_layer1(action_obs_decoding_common)
        prediction_residual = self.decode_prediction_head_layer2(prediction_residual_tmp)

        if self.vqvae_return_weight and not warmup:
            recons_action, recons_loss, recons_loss_none_reduction, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion \
                = self._recons_action(action_obs_decoding_common,
                                      data['action'], data[
                                          'return_normalization'])
        else:
            recons_action, recons_loss, recons_loss_none_reduction, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion\
                = self._recons_action(action_obs_decoding_common,
                                      data['action'])
        predict_loss = F.mse_loss(prediction_residual, data['true_residual'])

        total_vqvae_loss = recons_loss + self.vq_loss_weight * vq_loss + self.predict_loss_weight * predict_loss
        return {
            'quantized_index': quantized_index,
            'recons_loss_none_reduction': recons_loss_none_reduction,  # use as rl priority in dqn_vqvae
            'sigma': sigma,
            'total_vqvae_loss': total_vqvae_loss,
            'recons_loss': recons_loss,
            'vq_loss': vq_loss,
            'embedding_loss': embedding_loss,
            'commitment_loss': commitment_loss,
            'predict_loss': predict_loss,
            'recons_action_probs_left_mask_proportion':recons_action_probs_left_mask_proportion, 
            'recons_action_probs_right_mask_proportion':recons_action_probs_right_mask_proportion,
        }

    def encode(self, data: Dict) -> torch.Tensor:
        """
         Overview:
             Maps the given action and obs onto the latent action space.
         Arguments:
             - data (:obj:`Dict`): Dict containing keyword:
                 - action (:obj:`torch.Tensor`): the original action
                 - obs (:obj:`torch.Tensor`): observation
         Returns:
             - outputs (:obj:`Dict`): Dict containing keyword:
                 - quantized_index (:obj:`torch.Tensor`): the latent action.
         Shapes:
             - action (:obj:`torch.Tensor`): :math:`(B, A)`, where B is batch size and A is ``action_shape``
             - obs (:obj:`torch.Tensor`): :math:`(B, O)`, where B is batch size and O is ``obs_shape``
         """
        with torch.no_grad():
            action_embedding = self._get_action_embedding(data)
            if self.obs_regularization:
                action_embedding = self.encode_action_head(action_embedding)
                obs_embedding = self.encode_obs_head(data['obs'])

                action_obs_embedding_dot = action_embedding * obs_embedding
                action_obs_embedding = self.encode_common(action_obs_embedding_dot)
                encoding = self.encode_mu_head(action_obs_embedding)
            else:
                encoding = self.encoder(action_embedding)

            quantized_index = self.vq_layer.encode(encoding)
            return quantized_index

    def decode(self, data) -> Dict:
        """
        Overview:
            Maps the given quantized_index (latent action) and obs onto the original action space.
            Using the method ``self.encode_obs_head(obs)`` to get the obs_encoding.
        Arguments:
            - data (:obj:`Dict`): Dict containing keyword:
                - quantized_index (:obj:`torch.Tensor`): the sampled latent action
                - obs (:obj:`torch.Tensor`): observation
        Returns:
            - outputs (:obj:`Dict`): Dict containing keyword:
                - recons_action (:obj:`torch.Tensor`): reconstruction_action.
                - prediction_residual (:obj:`torch.Tensor`): prediction_residual.
        Shapes:
            - quantized_index (:obj:`torch.Tensor`): :math:`(B, L)`, where B is batch size and L is ``latent_action_size``
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`, where B is batch size and O is ``obs_shape``
        """

        with torch.no_grad():
            quantized_embedding = self.vq_layer.decode(data['quantized_index'])
            if self.obs_regularization:
                obs_encoding = self.encode_obs_head(data['obs'])
                action_decoding = self.decode_action_head(quantized_embedding)
                action_obs_decoding = action_decoding * obs_encoding

                action_obs_decoding_common = self.decode_common(action_obs_decoding)

                recons_action, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion = self._recons_action(action_obs_decoding_common, threshold_phase=data['threshold_phase'])

                prediction_residual_tmp = self.decode_prediction_head_layer1(action_obs_decoding_common)
                prediction_residual = self.decode_prediction_head_layer2(prediction_residual_tmp)

                return {'recons_action': recons_action, 
                'recons_action_probs_left_mask_proportion':recons_action_probs_left_mask_proportion, 
                'recons_action_probs_right_mask_proportion':recons_action_probs_right_mask_proportion,
                'prediction_residual': prediction_residual}

            else:
                action_decoding = self.decoder(quantized_embedding)
                recons_action, sigma, recons_action_probs_left_mask_proportion, recons_action_probs_right_mask_proportion = self._recons_action(action_decoding, threshold_phase=data['threshold_phase'])
                return {'recons_action': recons_action, 
                'recons_action_probs_left_mask_proportion':recons_action_probs_left_mask_proportion, 
                'recons_action_probs_right_mask_proportion':recons_action_probs_right_mask_proportion}
