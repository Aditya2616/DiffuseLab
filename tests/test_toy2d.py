import json
from pathlib import Path

import numpy as np
import pytest

from diffuselab.schedules import make_schedule
from diffuselab.toy2d import (
    DATASETS,
    ddim_step,
    ddpm_step,
    ideal_denoiser,
    moons,
    sample_toy,
    trailing_timesteps,
)

VECTORS = Path(__file__).parent / "vectors" / "toy2d.json"


def test_single_point_dataset_denoises_to_that_point():
    data = np.array([[0.3, -0.7]])
    sch = make_schedule("linear")
    x_t = np.random.default_rng(0).standard_normal((5, 2))
    x0, _ = ideal_denoiser(x_t, 400, data, sch)
    np.testing.assert_allclose(x0, np.tile(data, (5, 1)))


def test_eps_is_consistent_with_x0():
    data = moons(200, 0)
    sch = make_schedule("cosine")
    x_t = np.random.default_rng(1).standard_normal((7, 2))
    x0, eps = ideal_denoiser(x_t, 300, data, sch)
    ab = sch.alpha_bar[300]
    np.testing.assert_allclose(np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps, x_t, atol=1e-12)


def test_at_low_noise_denoiser_snaps_to_nearest_data_point():
    data = np.array([[0.0, 0.0], [2.0, 0.0]])
    sch = make_schedule("linear")
    x0, _ = ideal_denoiser(np.array([[1.8, 0.1]]), 2, data, sch)
    np.testing.assert_allclose(x0, [[2.0, 0.0]], atol=1e-6)


def test_ddim_step_to_clean_returns_x0_hat():
    sch = make_schedule("linear")
    x_t = np.array([[0.4, -0.2]])
    data = moons(50, 0)
    x0, eps = ideal_denoiser(x_t, 49, data, sch)
    np.testing.assert_allclose(ddim_step(x_t, 49, -1, eps, sch), x0, atol=1e-10)


def test_ddpm_last_step_has_no_noise():
    sch = make_schedule("linear")
    x = np.ones((3, 2))
    eps = np.zeros((3, 2))
    a = ddpm_step(x, 0, eps, sch, np.ones((3, 2)))
    b = ddpm_step(x, 0, eps, sch, np.zeros((3, 2)))
    np.testing.assert_array_equal(a, b)


def test_trailing_timesteps_match_diffusers():
    from diffuselab.schedulers import make_scheduler

    s = make_scheduler("ddim")
    for steps in (10, 20, 50):
        s.set_timesteps(steps)
        assert trailing_timesteps(1000, steps) == [int(t) for t in s.timesteps]


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_reverse_process_lands_on_the_data(name, sampler):
    data = DATASETS[name](150, 0)
    steps = 50 if sampler == "ddim" else 1000
    final = sample_toy(data, "linear", steps, sampler, n=100, seed=1)[-1]
    # every sample should sit close to some data point (the ideal denoiser memorises the set)
    d = np.sqrt(((final[:, None] - data[None]) ** 2).sum(-1)).min(1)
    assert np.median(d) < 0.1


def test_vectors_file_is_current():
    """tests/vectors/toy2d.json is what the browser tests (web/tests) compare against; regenerate it
    with `python scripts/make_vectors.py` and this test confirms it still matches the Python reference."""
    v = json.loads(VECTORS.read_text())
    for kind in ("linear", "cosine"):
        sch = make_schedule(kind)
        ref = v["schedule"][kind]
        for t, ab, beta in zip(ref["t"], ref["abar"], ref["beta"], strict=True):
            assert sch.alpha_bar[t] == pytest.approx(ab, rel=1e-12)
            assert sch.betas[t] == pytest.approx(beta, rel=1e-12)
    data = np.array(v["denoiser"]["data"])
    sch = make_schedule("cosine")
    x0, eps = ideal_denoiser(np.array(v["denoiser"]["x_t"]), v["denoiser"]["t"], data, sch)
    np.testing.assert_allclose(x0, v["denoiser"]["x0"], atol=1e-12)
    np.testing.assert_allclose(eps, v["denoiser"]["eps"], atol=1e-12)


def test_ddim_eta_zero_is_deterministic_and_eta_one_adds_noise():
    sch = make_schedule("linear")
    x = np.ones((4, 2))
    eps = np.full((4, 2), 0.3)
    z = np.random.default_rng(0).standard_normal((4, 2))
    np.testing.assert_array_equal(ddim_step(x, 600, 550, eps, sch, 0.0, z), ddim_step(x, 600, 550, eps, sch))
    assert not np.allclose(ddim_step(x, 600, 550, eps, sch, 1.0, z), ddim_step(x, 600, 550, eps, sch))


def test_ddim_eta_one_on_consecutive_steps_is_ddpm():
    """DDIM with eta = 1 and t_prev = t - 1 is exactly the DDPM ancestral step (the DDIM paper's observation)."""
    sch = make_schedule("cosine")
    x = np.random.default_rng(1).standard_normal((5, 2))
    eps = np.random.default_rng(2).standard_normal((5, 2))
    z = np.random.default_rng(3).standard_normal((5, 2))
    np.testing.assert_allclose(ddim_step(x, 400, 399, eps, sch, 1.0, z), ddpm_step(x, 400, eps, sch, z), atol=1e-10)
