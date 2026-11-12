import numpy as np
import pytest

from diffuselab.recorder import Trajectory
from diffuselab.sampler import initial_noise, sample
from diffuselab.schedulers import SCHEDULERS, make_scheduler


@pytest.mark.parametrize("name", ["ddim", "dpmpp"])
def test_oracle_recovers_target_with_deterministic_solvers(oracle, name):
    model, c = oracle
    tr = sample(model, name, steps=20, seed=3)
    np.testing.assert_allclose(tr.x0[:-1], np.broadcast_to(c.numpy(), tr.x0[:-1].shape), atol=1e-3)
    np.testing.assert_allclose(tr.final, c.numpy(), atol=5e-3)


def test_oracle_ddpm_final_is_close_to_target(oracle):
    model, c = oracle
    tr = sample(model, "ddpm", steps=1000, seed=1, record_every=100)
    assert np.abs(tr.final - c.numpy()).max() < 0.2  # stochastic, but the oracle pulls it to c


def test_trajectory_shapes_and_conventions(oracle):
    model, _ = oracle
    tr = sample(model, "ddim", steps=10, seed=0)
    n = tr.n_frames
    assert n == 11
    assert tr.x_t.shape == tr.eps.shape == tr.x0.shape == (n, 1, 8, 8)
    assert tr.t[0] == 999 and tr.t[-1] == -1 and np.all(np.diff(tr.t) < 0)
    assert tr.abar[-1] == 1.0 and np.all(np.diff(tr.abar) > 0)
    np.testing.assert_array_equal(tr.x0[-1], tr.x_t[-1])
    np.testing.assert_allclose(tr.snr[:-1], tr.abar[:-1] / (1 - tr.abar[:-1]))


def test_first_frame_is_the_seeded_noise(oracle):
    model, _ = oracle
    tr = sample(model, "ddim", steps=5, seed=7)
    np.testing.assert_allclose(tr.x_t[0], initial_noise((1, 1, 8, 8), 7)[0].numpy())


def test_same_seed_is_deterministic_and_different_seeds_differ(oracle):
    model, _ = oracle
    a = sample(model, "ddpm", steps=50, seed=2)
    b = sample(model, "ddpm", steps=50, seed=2)
    c = sample(model, "ddpm", steps=50, seed=3)
    np.testing.assert_array_equal(a.final, b.final)
    assert not np.allclose(a.final, c.final)


def test_all_schedulers_share_the_same_starting_noise(oracle):
    model, _ = oracle
    starts = [sample(model, s, steps=10, seed=5).x_t[0] for s in SCHEDULERS]
    for s in starts[1:]:
        np.testing.assert_array_equal(starts[0], s)


def test_record_every_keeps_first_and_last(oracle):
    model, _ = oracle
    tr = sample(model, "ddpm", steps=100, seed=0, record_every=30)
    assert tr.t[0] == 999 and tr.t[-2] == 9 and tr.t[-1] == -1
    assert tr.n_frames == len({0, 30, 60, 90, 99}) + 1


def test_dpm_solver_state_does_not_leak_between_runs(oracle):
    model, _ = oracle
    a = sample(model, "dpmpp", steps=10, seed=1)
    sample(model, "dpmpp", steps=10, seed=2)
    b = sample(model, "dpmpp", steps=10, seed=1)
    np.testing.assert_array_equal(a.final, b.final)


def test_on_frame_callback_sees_every_kept_frame(oracle):
    model, _ = oracle
    seen = []
    tr = sample(model, "ddim", steps=6, seed=0, on_frame=lambda k, f: seen.append((k, f["t"])))
    assert [k for k, _ in seen] == list(range(tr.n_frames))
    assert seen[-1][1] == -1


def test_runs_a_real_unet_on_cpu(tiny_unet):
    tr = sample(tiny_unet, "ddim", steps=4, seed=0)
    assert tr.x_t.shape == (5, 1, 16, 16)
    assert np.isfinite(tr.x_t).all() and np.isfinite(tr.eps).all()


def test_cosine_schedule_changes_the_run(oracle):
    model, _ = oracle
    from conftest import OracleModel

    lin = sample(model, "ddim", steps=10, seed=0, kind="linear")
    cos = sample(OracleModel(model.c, "cosine"), "ddim", steps=10, seed=0, kind="cosine")
    assert not np.allclose(lin.abar, cos.abar)


def test_unknown_scheduler():
    with pytest.raises(ValueError):
        make_scheduler("euler")


def test_trajectory_roundtrip(tmp_path, oracle):
    model, _ = oracle
    tr = sample(model, "ddim", steps=5, seed=0)
    tr.save(tmp_path / "t.npz")
    back = Trajectory.load(tmp_path / "t.npz")
    np.testing.assert_array_equal(back.x_t, tr.x_t)
    np.testing.assert_array_equal(back.t, tr.t)
    assert back.scheduler == "ddim" and back.seed == 0 and back.meta["steps"] == 5
    assert back.select([0, -1]).n_frames == 2
