"""Note the following vae model is borrowed from https://github.com/AntixK/PyTorch-VAE"""

import torch
from torch.nn import functional as F
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
import collections, numpy
Tensor = TypeVar('torch.tensor')


class EMA():

    def __init__(self, decay):
        self.decay = decay
        self.variables = {}

    def register(self, name, val):
        # self.variables[name] = val.clone()
        self.variables[name] = val.clone().detach()  # NOTE

    def get(self, name):
        return self.variables[name]

    @torch.no_grad()
    def update(self, name, x):
        assert name in self.variables
        self.variables[name] = (1.0 - self.decay) * x + self.decay * self.variables[name]


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, is_ema=False):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.is_ema = is_ema

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        if self.is_ema:
            self.ema_N = EMA(0.99)
            self.ema_m = EMA(0.99)
            for i in range(self.K):
                self.ema_N.register('N' + f'{i}', torch.zeros(1, device=torch.device('cuda')))  # TODO(pu)
                self.ema_m.register('m' + f'{i}', torch.zeros(self.K, device=torch.device('cuda')))
                # self.ema_N.register('N'+f'{i}', torch.zeros(1, device=torch.device('cpu')))
                # self.ema_m.register('m'+f'{i}', torch.zeros(self.K, device=torch.device('cpu')))

    def forward(self, encoding: Tensor) -> Tensor:
        encoding_shape = encoding.shape  # [A x D]
        flat_encoding = encoding.view(-1, self.D)

        # # Method 2: Compute L2 distance between encoding and embedding weights
        # dist = torch.sum(flat_encoding ** 2, dim=1, keepdim=True) + \
        #        torch.sum(self.embedding.weight ** 2, dim=1) - \
        #        2 * torch.matmul(flat_encoding, self.embedding.weight.t())
        # # Get the encoding that has the min distance
        # quantized_index = torch.argmin(dist, dim=1).unsqueeze(1)

        quantized_index = torch.cdist(flat_encoding, self.embedding.weight, p=2).sort()[1][:, 0]
        # .sort()[1] take the index after sorted, [:,0] take the nearest index
        quantized_index = quantized_index.unsqueeze(1)

        # print(collections.Counter(quantized_index.squeeze().cpu().numpy()).items())

        # Convert to one-hot encodings
        device = encoding.device
        encoding_one_hot = torch.zeros(quantized_index.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, quantized_index, 1)

        # Quantize the encoding
        quantized_embedding = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_embedding = quantized_embedding.view(encoding_shape)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_embedding.detach(), encoding)

        if self.is_ema:
            #  VQ-VAE dictionary updates with Exponential Moving Averages
            for i, N_i in collections.Counter(quantized_index.squeeze().cpu().numpy()).items():
                N_i_index_list = numpy.where(quantized_index.squeeze().cpu().numpy() == i)[0]
                self.ema_m.update('m' + f'{i}', encoding[N_i_index_list].sum(dim=0))  # m_i
                self.ema_N.update('N' + f'{i}', N_i)  #   # N_i
                self.embedding.weight.data[i] = self.ema_m.get('m' + f'{i}') / self.ema_N.get('N' + f'{i}')  # TODO(pu)
            vq_loss = commitment_loss * self.beta

        else:
            embedding_loss = F.mse_loss(quantized_embedding, encoding.detach())
            vq_loss = commitment_loss * self.beta + embedding_loss

        # straight-through estimator
        # Add the residue back to the encoding
        quantized_embedding = encoding + (quantized_embedding - encoding).detach()

        return quantized_index, quantized_embedding, vq_loss

    def inference(self, encoding: Tensor) -> Tensor:
        encoding_shape = encoding.shape  # [A x D]
        flat_encoding = encoding.view(-1, self.D)

        quantized_index = torch.cdist(flat_encoding, self.embedding.weight, p=2).sort()[1][:, 0]
        # .sort()[1] take the index after sorted, [:,0] take the nearest index
        quantized_index = quantized_index.unsqueeze(1)

        return quantized_index


class VQVAE(nn.Module):

    def __init__(
            self,
            action_dim: int,
            embedding_dim: int,
            num_embeddings: int,
            hidden_dims: List = None,
            beta: float = 0.25,
            is_ema: bool = False,
            img_size: int = 64,
            **kwargs
    ) -> None:
        super(VQVAE, self).__init__()

        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.is_ema = is_ema

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        ### Encoder
        self.encode_action_head = nn.Sequential(nn.Linear(self.action_dim, hidden_dims[0]), nn.ReLU())
        self.encode_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        self.encode_mu_head = nn.Linear(hidden_dims[0], self.embedding_dim)
        modules = [self.encode_action_head, self.encode_common, self.encode_mu_head]
        self.encoder = nn.Sequential(*modules)

        ### VQ layer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta, self.is_ema)

        ### Decoder
        self.decode_action_head = nn.Sequential(nn.Linear(self.embedding_dim, hidden_dims[0]), nn.ReLU())
        self.decode_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        # TODO(pu): tanh
        self.decode_reconst_action_head = nn.Sequential(nn.Linear(hidden_dims[0], self.action_dim), nn.Tanh())
        modules = [self.decode_action_head, self.decode_common, self.decode_reconst_action_head]
        self.decoder = nn.Sequential(*modules)

    def train_without_obs(self, data):
        encoding = self.encoder(data['action'])
        quantized_index, quantized_embedding, vq_loss = self.vq_layer(encoding)

        recons_action = self.decoder(quantized_embedding)
        recons_loss = F.mse_loss(recons_action, data['action'])
        total_vqvae_loss = recons_loss + vq_loss

        return {'total_vqvae_loss': total_vqvae_loss, 'recons_loss': recons_loss, 'vq_loss': vq_loss}

    def inference_without_obs(self, data):
        encoding = self.encoder(data['action'])
        quantized_index = self.vq_layer.inference(encoding)

        return {'quantized_index': quantized_index}

    def decode_without_obs(self, quantized_index: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        # Convert to one-hot encodings
        device = quantized_index.device
        quantized_index = quantized_index.view(-1, 1)
        encoding_one_hot = torch.zeros(quantized_index.size(0), self.vq_layer.K, device=device)
        encoding_one_hot.scatter_(1, quantized_index, 1)
        # Quantize the encoding
        quantized_embedding = torch.matmul(encoding_one_hot, self.vq_layer.embedding.weight)
        recons_action = self.decoder(quantized_embedding)

        return {'recons_action': recons_action}
