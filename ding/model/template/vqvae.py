"""Note the following vae model is borrowed from https://github.com/AntixK/PyTorch-VAE"""

import torch
from torch.nn import functional as F
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        # latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape  # [A x D]
        self.latents_shape = latents_shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

        return encoding_inds, quantized_latents, vq_loss  # [B x D x H x W]


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                 action_dim: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.action_dim = action_dim
        # self.obs_dim = obs_dim
        # self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Build Encoder: for image
        # modules = []
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size=4, stride=2, padding=1),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim
        #
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels,
        #                   kernel_size=3, stride=1, padding=1),
        #         nn.LeakyReLU())
        # )
        #
        # for _ in range(6):
        #     modules.append(ResidualLayer(in_channels, in_channels))
        # modules.append(nn.LeakyReLU())
        #
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, embedding_dim,
        #                   kernel_size=1, stride=1),
        #         nn.LeakyReLU())
        # )
        # self.encoder = nn.Sequential(*modules)

        # action
        self.encode_action_head = nn.Sequential(nn.Linear(self.action_dim, hidden_dims[0]), nn.ReLU())
        # obs
        # self.encode_obs_head = nn.Sequential(nn.Linear(self.obs_dim, hidden_dims[0]), nn.ReLU())

        self.encode_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        self.encode_mu_head = nn.Linear(hidden_dims[0], self.embedding_dim)
        modules = [self.encode_action_head, self.encode_common, self.encode_mu_head]
        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder: for image
        # modules = []
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(embedding_dim,
        #                   hidden_dims[-1],
        #                   kernel_size=3,
        #                   stride=1,
        #                   padding=1),
        #         nn.LeakyReLU())
        # )
        #
        # for _ in range(6):
        #     modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        #
        # modules.append(nn.LeakyReLU())
        #
        # hidden_dims.reverse()
        #
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=4,
        #                                stride=2,
        #                                padding=1),
        #             nn.LeakyReLU())
        #     )
        #
        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(hidden_dims[-1],
        #                            out_channels=3,
        #                            kernel_size=4,
        #                            stride=2, padding=1),
        #         nn.Tanh()))
        #
        # self.decoder = nn.Sequential(*modules)

        # Build Decoder
        # self.condition_obs = nn.Sequential(nn.Linear(self.obs_dim, hidden_dims[0]), nn.ReLU())
        self.decode_action_head = nn.Sequential(nn.Linear(self.embedding_dim, hidden_dims[0]), nn.ReLU())
        # self.decode_action_head = nn.Sequential(nn.Linear(self.embedding_dim*self.action_dim, hidden_dims[0]), nn.ReLU())

        self.decode_common = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        # TODO(pu): tanh
        self.decode_reconst_action_head = nn.Sequential(nn.Linear(hidden_dims[0], self.action_dim), nn.Tanh())
        # self.decode_reconst_action_head = nn.Linear(hidden_dims[0], self.action_dim)

        # residual prediction
        # self.decode_prediction_head_layer1 = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU())
        # self.decode_prediction_head_layer2 = nn.Linear(hidden_dims[0], self.obs_dim)
        modules = [self.decode_action_head, self.decode_common, self.decode_reconst_action_head]
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def decode_with_obs(self, encoding_inds: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        # Convert to one-hot encodings
        # device = latents.device
        device = torch.device('cpu')
        encoding_inds_shape = encoding_inds.shape  # 5,2

        encoding_inds = encoding_inds.view(-1, 1)
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.vq_layer.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.vq_layer.embedding.weight)  # [BHW, D]
        # quantized_latents = quantized_latents.view(encoding_inds_shape)  # [B x H x W x D]

        recons_action = self.decode(quantized_latents)
        return recons_action

    def forward(self, input, **kwargs) -> dict:
        encoding = self.encode(input['action'])[0]
        encoding_inds, quantized_inputs, vq_loss = self.vq_layer(encoding)
        # quantized_inputs=quantized_inputs.view(-1,64*2)
        return {'encoding_inds': encoding_inds,
                'quantized_inputs': quantized_inputs,
                'recons_action': self.decode(quantized_inputs),
                # 'prediction_residual': self.decode(z)[1],
                'input': input['action'],
                'vqloss': vq_loss}

        # import matplotlib.pyplot as plt
        # xx, yy = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
        # action_samples= np.array([xx.ravel(), yy.ravel()]).reshape(40000,2)
        # encoding = policy._vqvae_model.encode(torch.Tensor(action_samples))[0]
        # encoding_inds, quantized_inputs, vq_loss = policy._vqvae_model.vq_layer(encoding)
        # x = xx
        # y = yy
        # c = encoding_inds
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # sc = ax.scatter(x, y, c=c, marker='o')
        # ax.set_title('K=4 latent action')
        # fig.colorbar(sc)
        # plt.show()


    def loss_function(self,
                      args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """

        recons_action = args['recons_action']
        # prediction_residual = args['prediction_residual']
        original_action = args['original_action']
        # true_residual = args['true_residual']
        # predict_weight = kwargs['predict_weight']
        vq_loss = args['vqloss']

        recons_loss = F.mse_loss(recons_action, original_action)

        loss = recons_loss + vq_loss
        # return {'loss': loss, 'reconstruction_loss': recons_loss, 'vq_loss': vq_loss, 'predict_loss': predict_loss}
        return {'loss': loss, 'reconstruction_loss': recons_loss, 'vq_loss': vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
