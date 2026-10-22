import torch

from latent_drifting.drifting.bilateral_kernel import bilateral_drift
from latent_drifting.drifting.cfg import cfg_unconditional_weight, sample_cfg_alpha
from latent_drifting.drifting.loss import DriftingLoss


def test_identical_distributions_have_zero_drift():
    torch.manual_seed(3)
    query = torch.randn(6, 4, 8)
    samples = torch.randn(5, 4, 8)
    drift = bilateral_drift(query, samples, samples, temperature=0.05)
    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-6)


def test_bilateral_field_is_antisymmetric_under_distribution_swap():
    torch.manual_seed(4)
    query = torch.randn(5, 3, 6)
    positive = torch.randn(4, 3, 6)
    negative = torch.randn(4, 3, 6)
    forward = bilateral_drift(query, positive, negative, temperature=0.2)
    reverse = bilateral_drift(query, negative, positive, temperature=0.2)
    assert torch.allclose(forward, -reverse, atol=2e-6, rtol=2e-5)


def test_chunked_kernel_matches_large_chunk():
    torch.manual_seed(5)
    query = torch.randn(7, 5, 8)
    positive = torch.randn(6, 5, 8)
    negative = torch.randn(7, 5, 8)
    dense = bilateral_drift(query, positive, negative, 0.05, chunk_size=128)
    chunked = bilateral_drift(query, positive, negative, 0.05, chunk_size=2)
    assert torch.allclose(dense, chunked, atol=1e-6, rtol=1e-5)


def test_self_exclusion_with_zero_weight_unconditional_samples_is_finite():
    query = torch.randn(4, 2, 8)
    positive = torch.randn(3, 2, 8)
    unconditionals = torch.randn(2, 2, 8)
    negatives = torch.cat((query, unconditionals))
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    drift = bilateral_drift(
        query,
        positive,
        negatives,
        0.05,
        negative_weights=weights,
        self_exclusion_count=query.shape[0],
    )
    assert torch.isfinite(drift).all()


def test_cfg_weight_and_sampler_bounds():
    assert cfg_unconditional_weight(1.0, 64, 32) == 0.0
    assert cfg_unconditional_weight(2.0, 64, 32) == 63 / 32
    values = sample_cfg_alpha(256, 1.0, 4.0, power=5.0)
    assert torch.all((values >= 1.0) & (values <= 4.0))


def test_drifting_loss_backpropagates_to_prediction_only():
    prediction = torch.randn(4, 3, 8, requires_grad=True)
    positives = torch.randn(5, 3, 8)
    objective = DriftingLoss(
        temperatures=(0.05,),
        normalize_features=False,
        normalize_drift=False,
        chunk_size=2,
    )
    result = objective({"toy": prediction}, {"toy": positives})
    result.loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert positives.grad is None
