import gzip
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from diffuselab import data as data_mod
from diffuselab.export import EPS_RANGE, dequantise, image_grid, quantise, sprite, write_run
from diffuselab.forward import noise_trajectory
from diffuselab.sampler import sample
from diffuselab.schedules import make_schedule
from diffuselab.server import create_app


def idx_bytes(arr: np.ndarray) -> bytes:
    return bytes([0, 0, 8, arr.ndim]) + b"".join(int(d).to_bytes(4, "big") for d in arr.shape) + arr.tobytes()


def test_idx_roundtrip_and_validation():
    a = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    np.testing.assert_array_equal(data_mod.parse_idx(idx_bytes(a)), a)
    with pytest.raises(ValueError):
        data_mod.parse_idx(b"\x00\x00\x09\x01\x00\x00\x00\x01\x00")
    with pytest.raises(ValueError):
        data_mod.parse_idx(idx_bytes(a)[:-5])


def test_fetch_rejects_a_corrupt_file(tmp_path):
    for name in data_mod.FILES:
        (tmp_path / name).write_bytes(gzip.compress(b"not the real file"))
    with pytest.raises(RuntimeError, match="checksum"):
        data_mod.fetch(tmp_path)
    assert not (tmp_path / next(iter(data_mod.FILES))).exists()  # the corrupt file was removed


def test_quantise_roundtrip_error_is_below_one_level():
    x = np.linspace(-1, 1, 1001)
    assert np.abs(dequantise(quantise(x)) - x).max() <= 1 / 127.5
    e = np.linspace(-3, 3, 1001)
    assert np.abs(dequantise(quantise(e, EPS_RANGE), EPS_RANGE) - e).max() <= 3 / 127.5


def test_sprite_layout_and_export(tmp_path, oracle):
    model, c = oracle
    tr = sample(model, "ddim", steps=6, seed=0)
    img = sprite(tr)
    assert img.size == (tr.n_frames * 8, 3 * 8)
    info = write_run(tr, tmp_path, "oracle", quality=100)
    meta = json.loads((tmp_path / "runs" / f"{info['id']}.json").read_text())
    assert meta["frames"] == tr.n_frames and len(meta["t"]) == tr.n_frames and meta["channels"] == 1
    assert meta["t"][-1] == -1 and meta["snr"][-1] is None
    decoded = np.asarray(Image.open(tmp_path / "runs" / f"{info['id']}.webp").convert("RGB"))
    last_x0 = decoded[16:24, -8:, 0].astype(np.float32) / 127.5 - 1
    assert np.abs(last_x0 - np.clip(c[0].numpy(), -1, 1)).max() < 0.08  # lossy WebP, final x0 is the oracle image


def test_image_grid_shape():
    g = image_grid(np.zeros((10, 1, 32, 32), np.float32), cols=4)
    assert g.size == (4 * 32, 3 * 32)


def test_forward_trajectory_matches_closed_form():
    x0 = np.random.default_rng(0).uniform(-1, 1, (1, 8, 8))
    tr = noise_trajectory(x0, "linear", n_frames=11, seed=3)
    sch = make_schedule("linear")
    for k, t in enumerate(tr.t):
        ab = sch.alpha_bar[t]
        np.testing.assert_allclose(tr.x_t[k], np.sqrt(ab) * x0 + np.sqrt(1 - ab) * tr.eps[k], atol=1e-6)
    assert tr.t[0] == 0 and tr.t[-1] == 999
    assert np.corrcoef(tr.x_t[0].ravel(), x0.ravel())[0, 1] > 0.99
    assert abs(np.corrcoef(tr.x_t[-1].ravel(), x0.ravel())[0, 1]) < 0.5


@pytest.fixture
def client(oracle):
    model, _ = oracle

    def loader(key):
        if key.startswith("fashion"):
            raise FileNotFoundError("no checkpoint")
        return model, 1, "linear"

    return TestClient(create_app(loader=loader, docs_dir="does-not-exist"))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_schedules_endpoint(client):
    r = client.get("/api/schedules", params={"kind": "cosine", "T": 100})
    j = r.json()
    assert len(j["beta"]) == 100 and j["alpha_bar"][0] > j["alpha_bar"][-1]
    assert client.get("/api/schedules", params={"kind": "quadratic"}).status_code == 422


@pytest.mark.parametrize("scheduler", ["ddim", "euler"])
def test_run_lifecycle_streams_frames(client, scheduler):
    r = client.post("/api/runs", json={"model": "cifar10", "scheduler": scheduler, "steps": 5, "seed": 1})
    assert r.status_code == 202
    rid = r.json()["run_id"]
    with client.stream("GET", f"/api/runs/{rid}/stream") as s:
        body = "".join(s.iter_text())
    events = [e for e in body.split("\n\n") if e.strip()]
    assert events[0].startswith("event: meta") and events[-1].startswith("event: done")
    frames = [json.loads(e.split("data: ", 1)[1]) for e in events if e.startswith("event: frame")]
    assert len(frames) == 6 and frames[0]["t"] >= 900 and frames[-1]["t"] == -1
    assert len(frames[0]["data"]) > 0 and frames[-1]["snr"] is None


def test_validation_and_unknown_run(client):
    assert client.post("/api/runs", json={"steps": 0}).status_code == 422
    assert client.post("/api/runs", json={"steps": 5000}).status_code == 422
    assert client.post("/api/runs", json={"scheduler": "heun"}).status_code == 422
    assert client.get("/api/runs/nope/stream").status_code == 404
    assert client.post("/api/runs", json={"model": "fashion-linear"}).status_code == 404




def test_responses_carry_the_live_marker(client):
    assert client.get("/api/health").headers['X-DiffuseLab'] == 'live'

