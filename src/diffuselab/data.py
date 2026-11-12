"""Fashion-MNIST download and parsing (plain numpy, no torchvision)."""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
}
DATA_DIR = Path("data")
CLASSES = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
HOLDOUT = 5000  # last 5k training images are never trained on (used as a memorisation control)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dest: Path = DATA_DIR) -> Path:
    """Download the files once, verifying their SHA-256. Raises if a checksum does not match."""
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        p = dest / name
        if not p.exists():
            urllib.request.urlretrieve(BASE + name, p)  # noqa: S310 (fixed https URL)
        if sha256(p) != digest:
            p.unlink()
            raise RuntimeError(f"checksum mismatch for {name}; file removed, try again")
    return dest


def parse_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX array (the format MNIST-style datasets use)."""
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0 or raw[2] != 0x08:
        raise ValueError("not a uint8 IDX file")
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    arr = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if arr.size != int(np.prod(shape)):
        raise ValueError("IDX size does not match header")
    return arr.reshape(shape)


def load_fashion(dest: Path = DATA_DIR, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return (images float32 in [-1,1] shape (N,1,size,size), labels). 28x28 is zero-padded to 32x32."""
    fetch(dest)
    imgs = parse_idx(gzip.decompress((dest / "train-images-idx3-ubyte.gz").read_bytes()))
    labels = parse_idx(gzip.decompress((dest / "train-labels-idx1-ubyte.gz").read_bytes()))
    pad = (size - 28) // 2
    x = np.pad(imgs, ((0, 0), (pad, size - 28 - pad), (pad, size - 28 - pad)), constant_values=0)
    x = x.astype(np.float32) / 127.5 - 1.0
    return x[:, None], labels
