import torch.nn.functional as F
from torch import nn

from ding.torch_utils.network.nn_module import conv2d_block
from ding.utils.registry_factory import MODEL_REGISTRY


@MODEL_REGISTRY.register('gomoku_model')
class GomokuModel(nn.Module):
    """policy-value network module"""

    def __init__(self, model_cfg):
        super(GomokuModel, self).__init__()
        self.cfg = model_cfg
        self.input_channels = self.cfg.get('input_channels',3)
        self.board_size = self.cfg.get('board_size',15)

        # encoder part
        self.encoder = nn.Sequential(
            conv2d_block(in_channels=3, out_channels=32, kernel_size=self.input_channels, stride=1, padding=1, activation=nn.ReLU()),
            conv2d_block(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()),
            conv2d_block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()),
        )

        # action policy head
        self.policy_head = nn.Sequential(
            conv2d_block(in_channels=128, out_channels=4, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()),
            nn.Flatten(),
            nn.Linear(4 * self.board_size * self.board_size, self.board_size * self.board_size)
        )

        # state value layers
        self.value_head = nn.Sequential(
            conv2d_block(in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, state_input):
        # common layers
        encoded_embedding = self.encoder(state_input)
        # action policy layers
        logit = self.policy_head(encoded_embedding)
        # state value layers
        value = self.value_head(encoded_embedding)
        return logit, value

    def compute_prob_value(self, state_batch):
        logits, values = self.forward(state_batch)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs()
        return probs, values

    def compute_logp_value(self, state_batch):
        logits, values = self.forward(state_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, values


if __name__ == '__main__':
    import torch
    from easydict import EasyDict

    board_size = 19
    input_channels = 3
    batch_size = 4
    cfg = EasyDict(board_size=board_size,
                   input_channels= input_channels)

    inputs = torch.randn(batch_size, input_channels, board_size, board_size)
    model = GomokuModel(cfg)
    print(model(inputs))
