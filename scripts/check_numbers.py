"""Keep every number in README.md identical to the results files.

    python scripts/check_numbers.py --check    # exit 1 if the README is out of sync (a fast test runs this)
    python scripts/check_numbers.py --write    # regenerate the README blocks and inline values

The README holds two kinds of generated text:

* blocks between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->`` (the result tables), and
* inline values written as ``<!--v:key-->text<!--/v-->`` (numbers quoted in prose).

Both are rendered from docs/data/*.json (training.json, convergence.json, benchmark.json, benchmark_cpu.json,
cpu_timing.json, index.json), the training presets in src/diffuselab/train.py and the image files under docs/.
The UI reads the same JSON files, so the README, the UI and the data cannot drift apart without this check failing.
Only the standard library is needed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
README = ROOT / "README.md"

NAMES = {"ddpm": "DDPM", "ddim": "DDIM", "dpmpp": "DPM-Solver++", "euler": "Euler (Karras)"}


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def preset_tiny() -> dict:
    """PRESETS["tiny"] read from train.py without importing torch."""
    tree = ast.parse((ROOT / "src" / "diffuselab" / "train.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PRESETS":
            presets = {k.value: {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords} for k, v in zip(node.value.keys, node.value.values, strict=True)}
            return presets["tiny"]
    raise SystemExit("PRESETS not found in train.py")


def image_megabytes() -> float:
    """Total size of the binary images under docs/ (text files are left out: their size depends on line endings)."""
    total = sum(p.stat().st_size for p in (ROOT / "docs").rglob("*") if p.suffix in {".png", ".webp"})
    return total / 1e6


def n(v, d=0) -> str:
    return f"{v:,.{d}f}"


def values() -> dict[str, str]:
    t, conv, bg, bc, cpu, idx = (load(x) for x in ("training.json", "convergence.json", "benchmark.json", "benchmark_cpu.json", "cpu_timing.json", "index.json"))
    lin, cos = t["runs"]["linear"], t["runs"]["cosine"]
    for key in ("params", "steps", "epochs_run", "batch", "lr", "ema_decay", "seed", "train_images", "gpu", "torch", "device"):
        assert lin[key] == cos[key], f"the two runs differ in {key}: they must share their settings"
    tiny = preset_tiny()
    per_epoch = lin["steps"] // lin["epochs_run"]
    rows = {(r["scheduler"], r["steps"]): r["seconds_median"] for r in bg["rows"]}
    rows_cpu = {(r["scheduler"], r["steps"]): r["seconds_median"] for r in bc["rows"]}
    conv_by = {r["steps"]: r for r in conv["rows"]}
    below = [t["nn_check"][k]["generated_below_heldout_p1"] for k in t["nn_check"]]
    nn_claim = (
        "No generated sample is closer to a training image than the closest 1% of held-out real images are."
        if all(b == 0 for b in below)
        else "Some generated samples are closer to a training image than the closest 1% of held-out real images (see the table in `training.json`)."
    )
    by_model: dict[str, set[int]] = {}
    for r in idx:
        by_model.setdefault(r["model"], set()).add(r["seed"])
    samplers = {r["scheduler"] for r in idx}
    lr = f"{lin['lr']:.0e}".replace("e-0", "e-")
    return {
        "params": n(lin["params"]),
        "params_m": f"{lin['params'] / 1e6:.1f}M",
        "steps": n(lin["steps"]),
        "epochs": str(lin["epochs_run"]),
        "steps_per_epoch": n(per_epoch),
        "batch": str(lin["batch"]),
        "lr": lr,
        "ema": str(lin["ema_decay"]),
        "seed": str(lin["seed"]),
        "train_images": n(lin["train_images"]),
        "holdout": n(t["holdout"]),
        "gpu": lin["gpu"],
        "torch": lin["torch"],
        "lin_s": n(lin["seconds"]),
        "lin_min": f"{lin['seconds'] / 60:.1f}",
        "lin_loss": f"{lin['final_epoch_loss']:.4f}",
        "cos_s": n(cos["seconds"]),
        "cos_min": f"{cos['seconds'] / 60:.1f}",
        "cos_loss": f"{cos['final_epoch_loss']:.4f}",
        "full_minutes": f"{round(lin['seconds'] / 60)} to {round(cos['seconds'] / 60)}",
        "nn_n": str(t["nn_check"]["linear"]["n"]),
        "heldout_median": f"{t['nn_check']['linear']['heldout_to_train_median']:.2f}",
        "nn_claim": nn_claim,
        "lin_nn": f"{t['nn_check']['linear']['generated_to_train_median']:.2f}",
        "cos_nn": f"{t['nn_check']['cosine']['generated_to_train_median']:.2f}",
        "conv_seeds": str(conv["n_seeds"]),
        "gpu_repeats": str(len(bg["rows"][0]["seconds_all"])),
        "cpu_repeats": str(len(bc["rows"][0]["seconds_all"])),
        "dpm20_rmse": f"{conv_by[20]['dpmpp']:.3f}",
        "ddim50_rmse": f"{conv_by[50]['ddim']:.3f}",
        "dpm20_cost_pct": f"{100 * rows[('dpmpp', 20)] / rows[('ddim', 50)]:.0f}%",
        "bench_torch": bg["torch"],
        "bench_gpu": bg["gpu"],
        "cpu_bench_torch": bc["torch"],
        "tiny_epochs": str(tiny["epochs"]),
        "tiny_steps": n(per_epoch * tiny["epochs"]),
        "tiny_widths": "/".join(str(w) for w in tiny["widths"]),
        "cpu_steps": str(cpu["steps"]),
        "cpu_seconds": n(cpu["seconds"]),
        "cpu_per_step": f"{cpu['seconds'] / cpu['steps']:.1f}",
        "cpu_threads": str(cpu["threads"]),
        "cpu_load": str(cpu["cpu_load_percent"]),
        "tiny_cpu_hours": f"{per_epoch * tiny['epochs'] * cpu['seconds'] / cpu['steps'] / 3600:.1f}",
        "n_runs": str(len(idx)),
        "n_samplers": str(len(samplers)),
        "cifar_seeds": str(len(by_model["cifar10"])),
        "fashion_seeds": str(len(by_model["fashion-linear"])),
        "image_mb": f"{image_megabytes():.1f}",
        "ddpm_gpu_s": f"{rows[('ddpm', 1000)]:.1f}",
        "ddpm_cpu_s": f"{rows_cpu[('ddpm', 1000)]:.0f}",
    }


def blocks() -> dict[str, str]:
    t, conv, bg, bc = (load(x) for x in ("training.json", "convergence.json", "benchmark.json", "benchmark_cpu.json"))
    lin, cos = t["runs"]["linear"], t["runs"]["cosine"]
    out = {}
    out["training"] = "\n".join(
        [
            "| schedule | steps | wall time (CUDA) | final-epoch loss |",
            "|---|---|---|---|",
            f"| linear | {n(lin['steps'])} | {n(lin['seconds'])} s ({lin['seconds'] / 60:.1f} min), GPU mostly to itself | {lin['final_epoch_loss']:.4f} |",
            f"| cosine | {n(cos['steps'])} | {n(cos['seconds'])} s ({cos['seconds'] / 60:.1f} min), GPU shared with other jobs | {cos['final_epoch_loss']:.4f} |",
        ]
    )
    names = list(t["samplers"]["linear"])
    nn = ["| sampler | linear: median NN distance | cosine: median NN distance |", "|---|---|---|"]
    for key in names:
        s, st = key.split("-")
        label = f"{NAMES[s]}, {st} steps"
        nn.append(f"| {label} | {t['samplers']['linear'][key]['generated_to_train_median']:.2f} | {t['samplers']['cosine'][key]['generated_to_train_median']:.2f} |")
    out["nn"] = "\n".join(nn)
    cv = ["| steps | DDIM | DPM-Solver++ | Euler (Karras) |", "|---|---|---|---|"]
    cv += [f"| {r['steps']} | {r['ddim']:.4f} | {r['dpmpp']:.4f} | {r['euler']:.4f} |" for r in conv["rows"]]
    out["convergence"] = "\n".join(cv)
    cpu = {(r["scheduler"], r["steps"]): r["seconds_median"] for r in bc["rows"]}
    bt = ["| sampler | network calls | GPU (s) | CPU (s) |", "|---|---|---|---|"]
    for r in bg["rows"]:
        bt.append(f"| {NAMES[r['scheduler']]} | {r['steps']} | {r['seconds_median']:.2f} | {cpu[(r['scheduler'], r['steps'])]:.1f} |")
    out["benchmark"] = "\n".join(bt)
    return out


def render(text: str) -> str:
    v, b = values(), blocks()
    for name, body in b.items():
        text, k = re.subn(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", lambda m, body=body: m.group(1) + body + m.group(2), text, flags=re.S)
        if k != 1:
            raise SystemExit(f"README needs exactly one BEGIN/END marker pair for '{name}' (found {k})")

    def sub(m: re.Match) -> str:
        if m.group(1) not in v:
            raise SystemExit(f"README uses an unknown inline value '{m.group(1)}'")
        return f"<!--v:{m.group(1)}-->{v[m.group(1)]}<!--/v-->"

    return re.sub(r"<!--v:(\w+)-->.*?<!--/v-->", sub, text)


def main(argv: list[str]) -> int:
    mode = "--write" if "--write" in argv else "--check"
    cur = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    new = render(cur)
    if mode == "--write":
        README.write_text(new, encoding="utf-8", newline="\n")
        print("README.md regenerated" if new != cur else "README.md already in sync")
        return 0
    if new == cur:
        print("README numbers match the results files")
        return 0
    import difflib

    sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True), new.splitlines(True), "README.md", "from results files", n=0))
    print("\nREADME.md is out of sync with docs/data. Run: python scripts/check_numbers.py --write")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
