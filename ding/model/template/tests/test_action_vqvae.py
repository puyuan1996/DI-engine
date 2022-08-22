import pytest
import torch
from ding.model.template import VQVAE, ActionVQVAE, VectorQuantizer
from ding.model.template.action_vqvae import ExponentialMovingAverage
from ding.torch_utils import is_differentiable
import torch.nn.functional as F


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

@pytest.mark.unittest
def test_threshold_categorical_head_for_cont_action():
    B=5
    action_shape=3
    n_atom=11
    categorical_head_for_cont_action_threshold=0.5
    recons_action_probs_left_mask_proportion = 0
    recons_action_probs_right_mask_proportion = 0
    support = torch.linspace(-1, 1, n_atom)  # shape: (n_atom)
    recons_action_logits = torch.randn(B, action_shape, n_atom).requires_grad_(True)
    recons_action_probs = F.softmax(recons_action_logits, dim=-1)  # shape: (B,A, n_atom)
    
    recons_action = torch.sum(recons_action_probs * support, dim=-1) # shape: (B,A)

    assert recons_action.shape == (B, action_shape)
    print('recons_action_probs: ',recons_action_probs)

    # TODO(pu): for construct some extreme action
    # prob=[p1,p2,p3,p4], support=[s1,s2,s3,s4], if pi>threshold, then recons_action=support[i]
    # shape: (B,A)
    recons_action_left_lt_threshold_mask = recons_action_probs[:,:,0].ge(categorical_head_for_cont_action_threshold) 
    recons_action_right_lt_threshold_mask = recons_action_probs[:,:,-1].ge(categorical_head_for_cont_action_threshold)
    assert recons_action_left_lt_threshold_mask.shape == (B, action_shape)
    assert recons_action_right_lt_threshold_mask.shape == (B, action_shape)


    if recons_action_left_lt_threshold_mask.sum()>0 or recons_action_right_lt_threshold_mask.sum()>0:
        recons_action_probs_left_lt_threshold =  recons_action_probs[:,:,0].masked_select(recons_action_left_lt_threshold_mask ) 
        recons_action_probs_right_lt_threshold =  recons_action_probs[:,:,-1].masked_select(recons_action_right_lt_threshold_mask ) 

        # straight-through estimator for passing gradient from recons_action_probs_lt_threshold
        recons_action[recons_action_left_lt_threshold_mask] = (recons_action_probs_left_lt_threshold + (1-recons_action_probs_left_lt_threshold ).detach())*  support[0]
        recons_action[recons_action_right_lt_threshold_mask] = (recons_action_probs_right_lt_threshold + (1-recons_action_probs_right_lt_threshold ).detach())*  support[-1]

        # statistics
        recons_action_probs_left_mask_proportion = recons_action_left_lt_threshold_mask.sum()/ (recons_action_left_lt_threshold_mask.shape[0]* recons_action_left_lt_threshold_mask.shape[1])
        recons_action_probs_right_mask_proportion = recons_action_right_lt_threshold_mask.sum()/ (recons_action_right_lt_threshold_mask.shape[0]* recons_action_right_lt_threshold_mask.shape[1])
    
    print('recons_action_probs_left_mask_proportion:',recons_action_probs_left_mask_proportion, 'recons_action_probs_right_mask_proportion:',recons_action_probs_right_mask_proportion)
    fake_loss = recons_action.sum()
    assert recons_action_logits.grad is None
    fake_loss.backward()
    assert recons_action_logits.grad.shape == (B, action_shape, n_atom)


# test = test_threshold_categorical_head_for_cont_action()