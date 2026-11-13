import numpy as np
import pytest
import torch

from diffuselab.models import load_checkpoint
from diffuselab.sampler import sample
from diffuselab.train import train


def blobs(n=256, seed=0):
    """Tiny synthetic 'dataset': a bright square on a dark background at random positions."""
    rng = np.random.default_rng(seed)
    x = -np.ones((n, 1, 32, 32), np.float32)
    for i in range(n):
        r, c = rng.integers(4, 20, 2)
        x[i, 0, r : r + 8, c : c + 8] = 1.0
    return x


def test_training_smoke_writes_a_loadable_checkpoint(tmp_path):
    info = train(
        "cosine",
        "tiny",
        epochs=1,
        batch=32,
        device="cpu",
        out_dir=tmp_path,
        max_steps=6,
        images=blobs(192),
        log=lambda *_: None,
    )
    assert info["steps"] == 6 and info["kind"] == "cosine" and np.isfinite(info["final_epoch_loss"])
    net, meta = load_checkpoint(tmp_path / "fmnist_cosine_tiny.pt", device="cpu")
    assert meta["steps"] == 6
    tr = sample(net, "ddim", steps=3, seed=0, kind="cosine")
    assert tr.x_t.shape[-3:] == (1, 16, 16)  # the tiny preset trains at 16x16
    assert np.isfinite(tr.final).all()


@pytest.mark.slow
def test_loss_actually_goes_down(tmp_path):
    info = train(
        "linear", "tiny", epochs=3, batch=64, device="cpu", out_dir=tmp_path, images=blobs(512), log=lambda *_: None
    )
    losses = info["loss_per_epoch"]
    assert losses[-1] < losses[0]
    assert torch.isfinite(torch.tensor(losses)).all()
