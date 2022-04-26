import pytest
import torch
from ding.model.template import ActionVQVAE, VectorQuantizer
from ding.model.template.action_vqvae import ExponentialMovingAverage
from ding.torch_utils import is_differentiable


@pytest.mark.unittest
def test_ema():
    ema = ExponentialMovingAverage(0.99, shape=(4, 3))
    assert ema.average.eq(torch.zeros_like(ema.average)).all()
    for i in range(20):
        value = torch.rand(4, 3)
        ema.update(value)
        assert ema.average.ne(torch.zeros_like(ema.average)).all()
        assert ema.count.item() == 1 + i
        assert ema.value.shape == (4, 3)
    state_dict = ema.state_dict()
    assert set(state_dict.keys()) == set(['count', 'hidden', 'average'])


@pytest.mark.unittest
def test_vq_layer():
    # no ema case
    B, D = 3, 32
    model = VectorQuantizer(4, D, is_ema=False)
    encoding = torch.randn(B, D).requires_grad_(True)
    quantized_index, quantized_embedding, vq_loss, embedding_loss, commitment_loss = model.train(encoding)
    assert quantized_index.shape == (B, )
    assert quantized_index.dtype == torch.long
    assert quantized_embedding.shape == (B, D)
    assert encoding.grad is None
    is_differentiable(vq_loss, model)
    assert encoding.grad.shape == (B, D)

    encoding = torch.randn(B, D).requires_grad_(True)
    index = model.encode(encoding)
    assert index.shape == (B, )

    embedding = model.decode(index)
    assert embedding.shape == (B, D)

    # ema case
    model = VectorQuantizer(4, D, is_ema=True)
    encoding = torch.randn(B, D).requires_grad_(True)
    quantized_index, quantized_embedding, vq_loss, embedding_loss, commitment_loss = model.train(encoding)
    assert quantized_index.shape == (B, )
    assert quantized_index.dtype == torch.long
    assert quantized_embedding.shape == (B, D)
    assert embedding_loss.item() == 0.
    assert encoding.grad is None
    # ema method update embedding table without grad
    commitment_loss.backward()
    assert encoding.grad.shape == (B, D)


@pytest.mark.unittest
def test_action_vqvae():
    B, D = 3, 32
    model = ActionVQVAE(
        {
            'action_type_shape': 6,
            'action_args_shape': 8
        },
        4,
        D,
        is_ema=True,
        is_ema_target=True,
        eps_greedy_nearest=True
    )
    print(model)
    action = {'action_type': torch.randint(0, 6, size=(B, )), 'action_args': torch.tanh(torch.randn(B, 8))}
    inputs = {'action': action}
    output = model.train(inputs)
    print(output)
    is_differentiable(output['total_vqvae_loss'], model)
    index = model.encode(inputs)
    assert index.shape == (B, )
    recons_action = model.decode(index)['recons_action']
    assert recons_action['action_type'].shape == (B, )
    assert recons_action['action_args'].shape == (B, 8)
    assert recons_action['logit'].shape == (B, 6)
