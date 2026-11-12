import math

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler

from diffuselab.schedules import DIFFUSERS_NAME, make_betas, make_schedule, predict_x0, q_sample


@pytest.mark.parametrize("kind", ["linear", "cosine"])
@pytest.mark.parametrize("T", [1000, 200])
def test_matches_diffusers(kind, T):
    ours = make_schedule(kind, T)
    ref = DDPMScheduler(num_train_timesteps=T, beta_schedule=DIFFUSERS_NAME[kind])
    np.testing.assert_allclose(ours.betas, ref.betas.double().numpy(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(ours.alpha_bar, ref.alphas_cumprod.double().numpy(), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_alpha_bar_monotone_and_bounded(kind):
    s = make_schedule(kind)
    assert np.all(np.diff(s.alpha_bar) < 0)
    assert 0 < s.alpha_bar[-1] < s.alpha_bar[0] < 1
    assert np.all(np.diff(s.log_snr) < 0)


def test_linear_endpoints():
    b = make_betas("linear")
    assert b[0] == pytest.approx(1e-4) and b[-1] == pytest.approx(0.02)


def test_cosine_is_clipped_and_ends_nearly_pure_noise():
    s = make_schedule("cosine")
    assert s.betas.max() == pytest.approx(0.999)
    assert s.alpha_bar[-1] < 1e-3  # almost no signal left


def test_cosine_keeps_more_signal_in_the_middle_than_linear():
    lin, cos = make_schedule("linear"), make_schedule("cosine")
    assert cos.alpha_bar[500] > lin.alpha_bar[500]


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        make_betas("quadratic")


def test_snr_definition():
    s = make_schedule("linear")
    np.testing.assert_allclose(s.snr, s.alpha_bar / (1 - s.alpha_bar))
    assert s.abar_at(-1) == 1.0


def test_forward_then_predict_x0_is_identity_with_true_noise():
    s = make_schedule("cosine")
    rng = np.random.default_rng(0)
    x0, eps = rng.standard_normal((2, 3, 8, 8))
    for t in (0, 10, 500, 999):
        xt = q_sample(x0, t, eps, s)
        np.testing.assert_allclose(predict_x0(xt, eps, s.alpha_bar[t]), x0, atol=1e-9)


def test_q_sample_variance_matches_theory():
    s = make_schedule("linear")
    rng = np.random.default_rng(1)
    x0 = np.zeros(200_000)
    xt = q_sample(x0, 300, rng.standard_normal(x0.shape), s)
    assert xt.var() == pytest.approx(1 - s.alpha_bar[300], rel=0.02)
    assert math.isfinite(xt.mean())


def test_predict_x0_torch():
    x = torch.ones(2, 2)
    assert torch.allclose(predict_x0(x, torch.zeros(2, 2), 0.25), torch.full((2, 2), 2.0))
