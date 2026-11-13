import numpy as np
import pytest

from diffuselab.metrics import generate, to_uint8_rgb


def test_to_uint8_rgb_range_and_shape():
    x = np.array([[[[-1.0, 0.0], [1.0, 2.0]]]], np.float32)
    u = to_uint8_rgb(x)
    assert u.shape == (1, 3, 2, 2) and u.dtype.is_floating_point is False
    assert u[0, 0].flatten().tolist() == [0, 128, 255, 255] and (u[0, 0] == u[0, 2]).all()


@pytest.mark.parametrize("name,steps", [("ddim", 5), ("dpmpp", 5), ("euler", 5), ("ddpm", 8)])
def test_generate_is_deterministic_and_finds_the_oracle_target(oracle, name, steps):
    model, c = oracle
    a = generate(model, "linear", name, steps, n=6, seed=1, batch=3, size=8)
    b = generate(model, "linear", name, steps, n=6, seed=1, batch=3, size=8)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (6, 1, 8, 8) and np.isfinite(a).all()
    if name != "ddpm":  # the deterministic solvers reach the oracle's single target image
        np.testing.assert_allclose(a, np.broadcast_to(c.numpy(), a.shape), atol=0.05)
    other = generate(model, "linear", name, steps, n=6, seed=2, batch=3, size=8)
    assert not np.array_equal(a, other) or name != "ddpm"
